# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ideogram 4 CPU unit tests for the native port.

    pytest tests/diffusion/models/ideogram4/test_ideogram4.py -v

No checkpoint and no GPU are needed: these tests cover the parts that carry
reference parity (config mapping, latent norm, timestep schedule, FP8 weight
routing, the CFG loop) and the sequence-parallel mask contract.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.model_metadata import get_diffusion_model_metadata
from vllm_omni.diffusion.models.ideogram4 import constants
from vllm_omni.diffusion.models.ideogram4.config import Ideogram4Config, Ideogram4TextConfig
from vllm_omni.diffusion.models.ideogram4.fp8_linear import (
    FP8_E4M3_MAX,
    FP8_WEIGHT_DTYPE,
    Fp8Linear,
    is_fp8_state_dict,
    load_streamed_weights,
    quantize_weight_to_fp8,
    swap_linears_to_fp8,
)
from vllm_omni.diffusion.models.ideogram4.latent_norm import LATENT_SCALE, LATENT_SHIFT, get_latent_norm
from vllm_omni.diffusion.models.ideogram4.pipeline_ideogram4 import (
    DEFAULT_PRESET,
    DIT_COPY_MODES,
    Ideogram4Pipeline,
    get_ideogram4_post_process_func,
)
from vllm_omni.diffusion.models.ideogram4.scheduler import (
    PRESETS,
    LogitNormalSchedule,
    SamplerParameters,
    get_schedule_for_resolution,
    make_step_intervals,
)
from vllm_omni.diffusion.models.ideogram4.sp_layout import ShardLayout, rank_concat_segment_mask_or_none
from vllm_omni.diffusion.models.ideogram4.text_encoder import build_text_attention_mask
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _bare_pipeline() -> Ideogram4Pipeline:
    """Pipeline instance without components, for helpers that need no weights."""
    pipeline = Ideogram4Pipeline.__new__(Ideogram4Pipeline)
    nn.Module.__init__(pipeline)
    pipeline.od_config = OmniDiffusionConfig()
    pipeline.compute_dtype = torch.float32
    pipeline._execution_device = torch.device("cpu")
    pipeline.vae_scale_factor = 8
    pipeline.tokenizer_max_length = constants.MAX_TEXT_TOKENS
    return pipeline


class TestReferenceParity:
    def test_transformer_config_maps_checkpoint_names(self) -> None:
        # The checkpoint uses different names than the dataclass fields.
        config = Ideogram4Config.from_hf_config(
            {
                "num_attention_heads": 18,
                "attention_head_dim": 256,
                "num_layers": 34,
                "intermediate_size": 12288,
                "adaln_dim": 512,
                "in_channels": 128,
                "llm_features_dim": 53248,
            }
        )
        assert config.num_heads == 18
        assert config.head_dim == 256
        # emb_dim is not stored in the checkpoint and is derived.
        assert config.emb_dim == 18 * 256
        assert config.adaln_dim == 512
        assert config.in_channels == 128
        assert config.llm_features_dim == 53248
        assert config.mrope_section == (24, 20, 20)

    def test_text_config_reads_rope_parameters(self) -> None:
        config = Ideogram4TextConfig.from_hf_config(
            {
                "vocab_size": 151936,
                "hidden_size": 4096,
                "num_hidden_layers": 36,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "intermediate_size": 12288,
                "rms_norm_eps": 1e-6,
                "rope_parameters": {"rope_theta": 5_000_000.0, "mrope_section": [24, 20, 20]},
            }
        )
        assert config.num_kv_heads == 8
        # head_dim falls back to hidden_size // num_heads.
        assert config.head_dim == 128
        assert config.rope_theta == 5_000_000.0

    def test_latent_norm_constants(self) -> None:
        shift, scale = get_latent_norm()
        assert len(LATENT_SHIFT) == 128 and len(LATENT_SCALE) == 128
        assert shift.shape == (128,) and shift.dtype == torch.float32
        assert scale.shape == (128,) and scale.dtype == torch.float32
        # Broadcasts against a [B, L, 128] latent.
        latents = torch.zeros(1, 4, 128)
        assert (latents * scale + shift).shape == (1, 4, 128)

    def test_schedule_runs_from_noise_to_data(self) -> None:
        schedule = LogitNormalSchedule(mean=0.0, std=1.5)
        intervals = make_step_intervals(8)
        assert intervals.shape == (9,)
        assert float(intervals[0]) == 0.0
        assert float(intervals[-1]) == 1.0

        start = float(schedule(intervals[0]))
        end = float(schedule(intervals[-1]))
        # schedule(0) is the noisiest timestep and schedule(1) the final one.
        assert start == pytest.approx(1.0)
        assert end < start
        values = [float(schedule(t)) for t in intervals]
        assert all(values[i] >= values[i + 1] for i in range(len(values) - 1))

    def test_resolution_shifts_schedule_mean(self) -> None:
        schedule = get_schedule_for_resolution((1024, 1024), known_mean=0.0, std=1.5)
        assert schedule.mean == pytest.approx(0.0 + 0.5 * 2 * torch.log(torch.tensor(4.0)).item())
        assert schedule.std == 1.5

    def test_presets_match_reference_configs(self) -> None:
        assert PRESETS["V4_QUALITY_48"] == SamplerParameters(
            num_steps=48, guidance_schedule=(3.0,) * 3 + (7.0,) * 45, mu=0.0, std=1.5
        )
        assert PRESETS["V4_DEFAULT_20"].num_steps == 20
        assert PRESETS["V4_TURBO_12"].mu == 0.5
        # Loop-index order: index 0 is the last, most polished step, so the
        # polish steps come first in the tuple.
        assert PRESETS["V4_QUALITY_48"].guidance_schedule[0] == 3.0
        assert PRESETS["V4_QUALITY_48"].guidance_schedule[-1] == 7.0

    def test_sampler_parameters_rejects_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="guidance_schedule has length 2, expected num_steps=3"):
            SamplerParameters(num_steps=3, guidance_schedule=(7.0, 7.0), mu=0.0)

    def test_default_preset_is_the_quality_config(self) -> None:
        assert DEFAULT_PRESET == "V4_QUALITY_48"


class TestFp8Weights:
    def test_quantization_roundtrip(self) -> None:
        weight = torch.randn(8, 16)
        quantized, scale = quantize_weight_to_fp8(weight)
        assert quantized.dtype == FP8_WEIGHT_DTYPE
        assert scale.dtype == torch.float32
        # Row-wise symmetric: the largest entry of each row maps to the FP8 max.
        amax = weight.abs().amax(dim=1)
        torch.testing.assert_close(amax / scale, torch.full_like(amax, FP8_E4M3_MAX), atol=1e-3, rtol=1e-3)
        dequantized = quantized.to(torch.float32) * scale.unsqueeze(1)
        torch.testing.assert_close(dequantized, weight, atol=0.05, rtol=0.02)

    def test_fp8_linear_keeps_buffers_and_forward_dequantizes(self) -> None:
        layer = Fp8Linear(16, 8, compute_dtype=torch.float32)
        weight = torch.randn(8, 16)
        quantized, scale = quantize_weight_to_fp8(weight)
        assert layer.load_param("weight", quantized)
        assert layer.load_param("weight_scale", scale)
        assert layer.quantized
        # A blanket .to(dtype) must not cast the FP8 weight.
        layer.to(torch.bfloat16)
        assert layer.weight.dtype == FP8_WEIGHT_DTYPE
        assert layer.weight_scale.dtype == torch.float32
        assert not any(param.requires_grad for param in layer.parameters())

        x = torch.randn(2, 16)
        expected = torch.nn.functional.linear(x, quantized.to(torch.float32) * scale.unsqueeze(1))
        torch.testing.assert_close(layer.forward(x), expected, atol=1e-6, rtol=1e-6)

    def test_dense_checkpoint_loads_through_the_same_layer(self) -> None:
        layer = Fp8Linear(4, 2, bias=False, compute_dtype=torch.float32)
        weight = torch.randn(2, 4)
        assert layer.load_param("weight", weight)
        assert not layer.quantized
        x = torch.randn(1, 4)
        torch.testing.assert_close(layer.forward(x), torch.nn.functional.linear(x, weight), atol=1e-6, rtol=1e-6)

    def test_swap_only_layers_with_a_scale_entry(self) -> None:
        model = torch.nn.Module()
        model.quantized = torch.nn.Linear(4, 2)
        model.dense = torch.nn.Linear(4, 2)
        state_dict = {
            "quantized.weight": torch.randn(2, 4),
            "quantized.weight_scale": torch.ones(2),
            "dense.weight": torch.randn(2, 4),
        }
        assert is_fp8_state_dict(state_dict)
        assert swap_linears_to_fp8(model, state_dict, compute_dtype=torch.float32) == 1
        assert isinstance(model.quantized, Fp8Linear)
        assert isinstance(model.dense, torch.nn.Linear)

    def test_load_streamed_weights_routes_scales_and_skips_unexpected(self) -> None:
        model = torch.nn.Module()
        model.quantized = Fp8Linear(4, 2, compute_dtype=torch.float32)
        model.norm = torch.nn.LayerNorm(4)
        model.register_buffer("running_mean", torch.zeros(4), persistent=True)

        weight, scale = quantize_weight_to_fp8(torch.randn(2, 4))
        loaded = load_streamed_weights(
            model,
            [
                ("quantized.weight", weight),
                ("quantized.weight_scale", scale.to(torch.bfloat16)),
                ("norm.weight", torch.ones(4, dtype=torch.bfloat16)),
                ("running_mean", torch.ones(4, dtype=torch.bfloat16)),
                ("visual.patch_embed.weight", torch.randn(4)),
            ],
            torch.bfloat16,
        )

        assert loaded == {
            "quantized.weight",
            "quantized.weight_scale",
            "norm.weight",
            "running_mean",
        }
        assert model.quantized.weight.dtype == FP8_WEIGHT_DTYPE
        # Scales stay float32 even when the pipeline runs in bf16.
        assert model.quantized.weight_scale.dtype == torch.float32
        assert model.norm.weight.dtype == torch.bfloat16
        assert model.running_mean.dtype == torch.bfloat16

        with pytest.raises(ValueError, match="Unexpected checkpoint key"):
            load_streamed_weights(model, [("visual.patch_embed.weight", torch.randn(4))], torch.bfloat16,
                                  skip_unexpected=False)

    def test_dtype_for_overrides_the_cast_for_a_component(self) -> None:
        model = torch.nn.Module()
        model.vae = torch.nn.Module()
        model.vae.register_buffer("decoder_conv_weight", torch.zeros(2, 2), persistent=True)
        load_streamed_weights(
            model,
            [("vae.decoder_conv_weight", torch.randn(2, 2, dtype=torch.bfloat16))],
            torch.bfloat16,
            dtype_for=lambda name: torch.float32 if name.startswith("vae.") else torch.bfloat16,
        )
        assert model.vae.decoder_conv_weight.dtype == torch.float32


class TestSequenceParallelMask:
    def test_shard_layout_pads_at_the_end(self) -> None:
        layout = ShardLayout(original_seq_len=10, world_size=3, rank=1)
        assert layout.padded_seq_len == 12
        assert layout.local_seq_len == 4
        assert layout.padding_size == 2
        assert layout.bounds(0) == (0, 4)
        assert layout.bounds(1) == (4, 8)
        assert layout.bounds(2) == (8, 12)

    def test_resolve_falls_back_to_an_unsharded_layout(self, mocker) -> None:
        mocker.patch(
            "vllm_omni.diffusion.forward_context.get_sp_shard_original_seq_len",
            return_value=None,
        )
        layout = ShardLayout.resolve("sequence", local_seq_len=5)
        assert layout.world_size == 1
        assert layout.rank == 0
        assert layout.padded_seq_len == 5

    def test_mask_describes_the_rank_concatenated_sequence(self, mocker) -> None:
        mocker.patch(
            "vllm_omni.diffusion.distributed.parallel_state.get_sequence_parallel_world_size",
            return_value=2,
        )
        mocker.patch(
            "vllm_omni.diffusion.distributed.parallel_state.get_sequence_parallel_rank",
            return_value=1,
        )
        # Two samples: each keeps its own segment, so the mask stays block diagonal.
        segment_ids = torch.tensor([[1, 1, 1], [2, 2, 2]])
        mask = rank_concat_segment_mask_or_none(segment_ids, "sequence", like=torch.zeros(2, 2, 1), required=True)
        assert mask is not None
        assert mask.shape == (2, 1, 4, 4)
        assert mask.dtype == torch.bool
        for b in range(2):
            for i in range(4):
                for j in range(4):
                    # Position 3 is the auto-padded slot; it only sees itself.
                    expected = (i < 3 and j < 3) or (i == 3 and j == 3)
                    assert mask[b, 0, i, j] == expected, f"sample {b} keeps its own block diagonal"

    def test_padding_positions_are_never_attended(self, mocker) -> None:
        mocker.patch(
            "vllm_omni.diffusion.distributed.parallel_state.get_sequence_parallel_world_size",
            return_value=2,
        )
        mocker.patch(
            "vllm_omni.diffusion.distributed.parallel_state.get_sequence_parallel_rank",
            return_value=0,
        )
        segment_ids = torch.tensor([[1, 1, 1]])
        mask = rank_concat_segment_mask_or_none(segment_ids, "sequence", like=torch.zeros(1, 2, 1), required=True)
        assert mask is not None
        assert mask.shape == (1, 1, 4, 4)
        # The sequence is padded to 4, so position 3 is the padding slot.
        padding = {3}
        for i in range(4):
            for j in range(4):
                expected = (i in padding) == (j in padding)
                assert mask[0, 0, i, j] == expected

    def test_all_valid_mask_keeps_the_fast_path(self, mocker) -> None:
        mocker.patch(
            "vllm_omni.diffusion.distributed.parallel_state.get_sequence_parallel_world_size",
            return_value=1,
        )
        mocker.patch(
            "vllm_omni.diffusion.distributed.parallel_state.get_sequence_parallel_rank",
            return_value=0,
        )
        segment_ids = torch.tensor([[1, 1, 1]])
        assert rank_concat_segment_mask_or_none(segment_ids, "sequence", like=torch.zeros(1, 3, 1)) is None
        full = rank_concat_segment_mask_or_none(segment_ids, "sequence", like=torch.zeros(1, 3, 1), required=True)
        assert full is not None and full.all()

    def test_negative_branch_passes_its_own_length(self, mocker) -> None:
        # The image-only CFG branch is shorter than the recorded positive split.
        mocker.patch(
            "vllm_omni.diffusion.distributed.parallel_state.get_sequence_parallel_world_size",
            return_value=2,
        )
        mocker.patch(
            "vllm_omni.diffusion.distributed.parallel_state.get_sequence_parallel_rank",
            return_value=0,
        )
        layout = ShardLayout.resolve("sequence", local_seq_len=3, original_seq_len=6)
        assert layout.original_seq_len == 6
        assert layout.padded_seq_len == 6

    def test_sp_plan_keeps_segment_ids_replicated(self) -> None:
        from vllm_omni.diffusion.models.ideogram4.ideogram4_transformer import (  # the plan lives on the transformer
            SHARD_GROUP,
            Ideogram4Transformer2DModel,
        )

        assert SHARD_GROUP == "sequence"
        split = Ideogram4Transformer2DModel._sp_plan["sequence_prepare"]
        assert split[0].shard_group == SHARD_GROUP
        assert split[1].shard_group == SHARD_GROUP
        # segment_ids (2) and adaln_input (3) must stay replicated so the mask
        # and the modulation see the whole sequence.
        assert 2 not in split
        assert 3 not in split
        assert Ideogram4Transformer2DModel._sp_plan["final_layer"].gather_dim == 1


class TestTextAttentionMask:
    def test_combines_causality_and_padding(self) -> None:
        padding = torch.tensor([[1, 1, 0, 1]])
        mask = build_text_attention_mask(padding)
        assert mask.shape == (1, 1, 4, 4)
        for i in range(4):
            for j in range(4):
                if j > i:
                    assert not mask[0, 0, i, j], "future positions are not visible"
                elif padding[0, j] == 0:
                    assert not mask[0, 0, i, j], "padding keys are not visible"
                else:
                    assert mask[0, 0, i, j]

    def test_padding_rows_do_not_produce_nan_softmax(self) -> None:
        mask = build_text_attention_mask(torch.tensor([[1, 0, 1]]))
        row_sums = mask[0, 0].sum(dim=-1)
        assert (row_sums > 0).all(), "every query row must attend to at least one key"
        # A padding query attends only to itself, and its output is zeroed by the caller.
        assert mask[0, 0, 1].tolist() == [False, True, False]

    def test_rejects_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match="attention_mask must be \\(B, L\\)"):
            build_text_attention_mask(torch.ones(1, 3, 3, dtype=torch.long))


class TestPipelineContracts:
    def test_config_value_precedence(self, monkeypatch) -> None:
        from vllm_omni.diffusion.models.ideogram4.pipeline_ideogram4 import _config_value

        monkeypatch.setenv("IDEOGRAM4_DIT_COPIES", "keep")
        od_config = OmniDiffusionConfig(additional_config={"dit_copies": "clone"})
        assert _config_value(od_config, "dit_copies", "IDEOGRAM4_DIT_COPIES", "clone") == "clone"
        od_config.custom_pipeline_args = {"dit_copies": "keep"}
        assert _config_value(od_config, "dit_copies", "IDEOGRAM4_DIT_COPIES", "clone") == "keep"
        monkeypatch.delenv("IDEOGRAM4_DIT_COPIES")
        od_config.custom_pipeline_args = {}
        od_config.additional_config = {}
        assert _config_value(od_config, "dit_copies", "IDEOGRAM4_DIT_COPIES", "clone") == "clone"

    def test_dit_copies_default_and_validation(self, monkeypatch) -> None:
        from vllm_omni.diffusion.models.ideogram4.pipeline_ideogram4 import _resolve_dit_copies

        assert DIT_COPY_MODES == ("clone", "keep")
        assert _resolve_dit_copies(OmniDiffusionConfig()) == "clone"
        monkeypatch.setenv("IDEOGRAM4_DIT_COPIES", "keep")
        assert _resolve_dit_copies(OmniDiffusionConfig()) == "keep"
        with pytest.raises(ValueError, match="dit_copies must be one of"):
            _resolve_dit_copies(OmniDiffusionConfig(custom_pipeline_args={"dit_copies": "share"}))

    def test_clone_mode_shares_one_dit_and_keep_mode_loads_two(self) -> None:
        # Instance _dit_modules decide which modules the SP hooks are applied to;
        # in clone mode the plan must run once, so only "transformer" is listed.
        pipeline = _bare_pipeline()
        pipeline.transformer = SimpleNamespace(name="transformer")
        pipeline.unconditional_transformer = pipeline.transformer
        pipeline.dit_copies = "clone"
        assert pipeline._dit_modules == ["transformer"]
        assert pipeline.unconditional_transformer is pipeline.transformer
        assert pipeline._negative_model(OmniDiffusionSamplingParams()) is pipeline.transformer

        pipeline.dit_copies = "keep"
        pipeline.unconditional_transformer = SimpleNamespace(name="unconditional_transformer")
        assert pipeline._dit_modules == ["transformer", "unconditional_transformer"]
        assert pipeline._negative_model(OmniDiffusionSamplingParams()) is pipeline.unconditional_transformer

    def test_request_override_cannot_create_a_second_copy(self) -> None:
        pipeline = _bare_pipeline()
        pipeline.transformer = SimpleNamespace(name="transformer")
        pipeline.unconditional_transformer = pipeline.transformer
        pipeline.dit_copies = "clone"
        sp = OmniDiffusionSamplingParams(extra_args={"dit_copies": "keep"})
        # The server never allocated a second DiT, so the request value is
        # honoured only as a warning and the shared copy is used.
        assert pipeline._negative_model(sp) is pipeline.transformer
        with pytest.raises(ValueError, match="dit_copies must be one of"):
            pipeline._negative_model(OmniDiffusionSamplingParams(extra_args={"dit_copies": "two"}))

    def test_sampling_plan_uses_preset_and_request_overrides(self) -> None:
        pipeline = _bare_pipeline()
        num_steps, gw, schedule = pipeline._resolve_sampling_plan(OmniDiffusionSamplingParams(), (1024, 1024))
        preset = PRESETS[DEFAULT_PRESET]
        assert num_steps == preset.num_steps
        assert gw == list(preset.guidance_schedule)
        assert schedule.mean == pytest.approx(preset.mu + 0.5 * torch.log(torch.tensor(4.0)).item())

        sp = OmniDiffusionSamplingParams(
            num_inference_steps=3,
            extra_args={"preset": "V4_TURBO_12", "guidance_schedule": [7.0, 3.0, 3.0], "mu": 1.0, "std": 2.0},
        )
        num_steps, gw, schedule = pipeline._resolve_sampling_plan(sp, (512, 512))
        assert num_steps == 3
        assert gw == [7.0, 3.0, 3.0]
        assert schedule.mean == 1.0 and schedule.std == 2.0

        with pytest.raises(ValueError, match="guidance_schedule has 2 entries but num_inference_steps is 4"):
            pipeline._resolve_sampling_plan(
                OmniDiffusionSamplingParams(num_inference_steps=4, extra_args={"guidance_schedule": [7.0, 3.0]}),
                (512, 512),
            )

    def test_extract_prompts(self) -> None:
        pipeline = _bare_pipeline()
        assert pipeline._extract_prompts(["a caption"]) == (["a caption"], None)
        prompts = [{"prompt": "a caption", "negative_prompt": "blurry"}, {"prompt": "another"}]
        prompt, negative = pipeline._extract_prompts(prompts)
        assert prompt == ["a caption", "another"]
        assert negative == ["blurry"]
        assert pipeline._extract_prompts([]) == (None, None)

    def test_build_inputs_packs_text_then_image(self, mocker) -> None:
        pipeline = _bare_pipeline()
        tokens = torch.tensor([10, 11, 12])
        mocker.patch.object(pipeline, "_tokenize", return_value=(tokens, 3))
        inputs = pipeline._build_inputs(["a caption"], 32, 16)

        patch = constants.PATCH_SIZE * pipeline.vae_scale_factor
        assert patch == 16
        grid_h, grid_w = 2, 1
        max_text_tokens = 3
        total = max_text_tokens + grid_h * grid_w
        assert inputs["grid_h"] == grid_h and inputs["grid_w"] == grid_w
        assert inputs["num_image_tokens"] == grid_h * grid_w
        assert inputs["token_ids"].shape == (1, total)
        # No left padding: the text block starts at offset 0 and image slots stay 0.
        assert inputs["token_ids"][0].tolist() == [10, 11, 12, 0]
        assert inputs["indicator"][0].tolist() == [3, 3, 3, 2]
        assert inputs["segment_ids"][0].tolist() == [1, 1, 1, 1]
        # Image positions carry the mRoPE offset; text positions are the token index.
        image_pos = inputs["position_ids"][0, max_text_tokens:]
        assert image_pos[0].tolist() == [constants.IMAGE_POSITION_OFFSET, 0, 0]
        assert image_pos[1].tolist() == [constants.IMAGE_POSITION_OFFSET, 1, 0]
        assert inputs["text_position_ids"][0, :max_text_tokens].tolist() == [[0, 0, 0], [1, 1, 1], [2, 2, 2]]

    def test_build_inputs_rejects_sizes_that_are_not_patch_multiples(self, mocker) -> None:
        pipeline = _bare_pipeline()
        mocker.patch.object(pipeline, "_tokenize", return_value=(torch.tensor([1]), 1))
        with pytest.raises(ValueError, match="multiples of 16"):
            pipeline._build_inputs(["a caption"], 30, 16)

    def test_post_process_denormalizes_the_minus_one_to_one_range(self, tmp_path) -> None:
        (tmp_path / "vae").mkdir()
        (tmp_path / "vae" / "config.json").write_text(json.dumps({"block_out_channels": [128, 256, 512, 512]}))
        post_process = get_ideogram4_post_process_func(OmniDiffusionConfig(model=str(tmp_path)))

        dark = post_process(torch.full((1, 3, 16, 16), -1.0))
        bright = post_process(torch.full((1, 3, 16, 16), 1.0))
        assert dark.min() == 0 and dark.max() == 0
        assert bright.min() == 255 and bright.max() == 255

        with pytest.raises(ValueError, match="expected payload\\['image'\\]"):
            post_process({"payload": {"text": "x"}, "metadata": {}})

    def test_metadata_marks_streaming_unsupported(self) -> None:
        # One image per request means a streaming chat request has nothing to emit.
        metadata = get_diffusion_model_metadata("Ideogram4Pipeline")
        assert metadata.final_output_type == "image"
        assert metadata.supports_streaming is False
        assert get_diffusion_model_metadata("UnknownPipeline").supports_streaming is True


class TestCFGLoopParity:
    def test_loop_matches_the_reference_formulation(self, mocker) -> None:
        pipeline = _bare_pipeline()
        num_steps = 4
        max_text_tokens = 3
        num_image_tokens = 4
        channels = 8
        gw_per_step = [3.0, 7.0, 7.0, 7.0]
        schedule = LogitNormalSchedule(mean=0.0, std=1.5)
        intervals = make_step_intervals(num_steps)
        sigmas = torch.tensor([float(schedule(t)) for t in intervals], dtype=torch.float32)

        z0 = torch.randn(1, num_image_tokens, channels, generator=torch.Generator().manual_seed(0))
        pos_img = torch.full((1, num_image_tokens, channels), 0.5)
        neg_img = torch.full((1, num_image_tokens, channels), -0.25)

        def fake_predict_noise(model, **kwargs):
            return pos_img if model is pipeline.transformer else neg_img

        mocker.patch.object(pipeline, "predict_noise", side_effect=fake_predict_noise)
        pipeline.transformer = SimpleNamespace(name="transformer")
        pipeline.unconditional_transformer = pipeline.transformer

        result = pipeline.diffuse(
            latents=z0,
            llm_features=torch.zeros(1, num_image_tokens, 4),
            position_ids=torch.zeros(1, max_text_tokens + num_image_tokens, 3, dtype=torch.long),
            segment_ids=torch.ones(1, max_text_tokens + num_image_tokens, dtype=torch.long),
            indicator=torch.full((1, max_text_tokens + num_image_tokens), constants.OUTPUT_IMAGE_INDICATOR),
            max_text_tokens=max_text_tokens,
            num_steps=num_steps,
            gw_per_step=gw_per_step,
            sigmas=sigmas,
            negative_model=pipeline.unconditional_transformer,
        )

        # Reference form: for i in range(num_steps - 1, -1, -1) with
        # t = schedule(step_intervals[i + 1]), s = schedule(step_intervals[i]).
        z_ref = z0.clone()
        for i in range(num_steps - 1, -1, -1):
            t_val = float(schedule(intervals[i + 1]))
            s_val = float(schedule(intervals[i]))
            gw = gw_per_step[i]
            v = gw * pos_img + (1.0 - gw) * neg_img
            z_ref = z_ref + v * (s_val - t_val)
        torch.testing.assert_close(result, z_ref, atol=1e-6, rtol=1e-6)

    def test_guidance_weight_below_one_skips_the_negative_branch(self, mocker) -> None:
        pipeline = _bare_pipeline()
        calls = []

        def fake_predict_noise(model, **kwargs):
            calls.append(kwargs["model"])
            return torch.zeros(1, 1, 2)

        mocker.patch.object(pipeline, "predict_noise", side_effect=fake_predict_noise)
        pipeline.transformer = SimpleNamespace(name="transformer")
        pipeline.unconditional_transformer = SimpleNamespace(name="unconditional_transformer")

        pipeline.diffuse(
            latents=torch.zeros(1, 1, 2),
            llm_features=torch.zeros(1, 1, 2),
            position_ids=torch.zeros(1, 2, 3, dtype=torch.long),
            segment_ids=torch.ones(1, 2, dtype=torch.long),
            indicator=torch.full((1, 2), constants.OUTPUT_IMAGE_INDICATOR),
            max_text_tokens=1,
            num_steps=1,
            gw_per_step=[1.0],
            sigmas=torch.tensor([1.0, 0.0]),
            negative_model=pipeline.unconditional_transformer,
        )
        assert calls == [pipeline.transformer]

    def test_rejects_schedule_length_mismatch(self, mocker) -> None:
        pipeline = _bare_pipeline()
        mocker.patch.object(pipeline, "predict_noise", side_effect=lambda model, **k: torch.zeros(1, 1, 2))
        with pytest.raises(ValueError, match="Expected 2 schedule values, got 3"):
            pipeline.diffuse(
                latents=torch.zeros(1, 1, 2),
                llm_features=torch.zeros(1, 1, 2),
                position_ids=torch.zeros(1, 2, 3, dtype=torch.long),
                segment_ids=torch.ones(1, 2, dtype=torch.long),
                indicator=torch.full((1, 2), constants.OUTPUT_IMAGE_INDICATOR),
                max_text_tokens=1,
                num_steps=1,
                gw_per_step=[7.0],
                sigmas=torch.tensor([1.0, 0.5, 0.0]),
                negative_model=pipeline.transformer,
            )

    def test_forward_returns_one_output_per_request(self, mocker) -> None:
        # The runner requires list[DiffusionOutput] whenever a batch has more
        # than one request, so per-request generation is the contract.
        pipeline = _bare_pipeline()
        outputs = [SimpleNamespace(name="output_a"), SimpleNamespace(name="output_b")]
        mocker.patch.object(pipeline, "generate", side_effect=outputs)
        batch = SimpleNamespace(requests=[SimpleNamespace(request_id="a"), SimpleNamespace(request_id="b")])
        assert pipeline.forward(batch) == outputs

    def test_decode_rearranges_patch_channels_into_the_latent_grid(self, mocker) -> None:
        pipeline = _bare_pipeline()
        shift, scale = get_latent_norm()
        pipeline.register_buffer("latent_shift", shift, persistent=False)
        pipeline.register_buffer("latent_scale", scale, persistent=False)
        pipeline.vae = SimpleNamespace(dtype=torch.float32, decoder=lambda x: x)

        grid_h, grid_w = 2, 1
        latents = torch.arange(grid_h * grid_w * 128, dtype=torch.float32).reshape(1, grid_h * grid_w, 128)
        out = pipeline._decode(latents, grid_h, grid_w)
        assert out.output.shape == (1, 32, grid_h * constants.PATCH_SIZE, grid_w * constants.PATCH_SIZE)
        # A patch is read as (row, col, channel) and written as (channel, row, col).
        first_patch = (latents[0, 0] * scale + shift).reshape(32, 2, 2)
        torch.testing.assert_close(out.output[0, :, 0, 0], first_patch[:, 0, 0])
        torch.testing.assert_close(out.output[0, :, 0, 1], first_patch[:, 0, 1])
        torch.testing.assert_close(out.output[0, :, 1, 0], first_patch[:, 1, 0])
        assert out.output.min() >= -1.0 and out.output.max() <= 1.0
