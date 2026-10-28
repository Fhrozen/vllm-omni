# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""L1 registration + pipeline + deploy-config tests for WeVisOne-VL (CPU, no weights)."""

from pathlib import Path

import pytest
import yaml

from vllm_omni.config.config_factory import _match_pipeline_key
from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
from vllm_omni.config.stage_config import StageExecutionType
from vllm_omni.model_executor.models.registry import _OMNI_MODELS
from vllm_omni.model_executor.models.wevisone.config import Qwen3VLSegConfig
from vllm_omni.model_executor.models.wevisone.pipeline import (
    IM_END_TOKEN_ID,
    WEVISONE_VL_PIPELINE,
    WEVISONE_VL_TEXT_ONLY_PIPELINE,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Values recorded from the shipped ``mask_decoder.pt`` cfg. They are asserted
# against the real sidecar in the local_model parity test; here they pin the
# defaults so a change cannot silently drift from the released checkpoint.
SHIPPED_SIDECAR_CFG = {
    "architecture_version": "wevisone_vl_v1",
    "vision_feature_layers": [5, 11, 17, 23],
    "hidden_dim": 192,
    "mask_feature_dim": 64,
    "num_fourier_freqs": 32,
    "msda_encoder_layers": 3,
    "msda_num_levels": 4,
    "msda_num_points": 4,
    "transformer_layers": 6,
    "transformer_heads": 8,
    "transformer_ffn_dim": 768,
    "transformer_dropout": 0.0,
    "group_norm_groups": 32,
    "stem_mid_dim": 128,
    "alpha_cascade_dim": 48,
    "alpha_cascade_blocks": 2,
    "cnn_image_size": 1024,
}


def _deploy_yaml(name: str) -> dict:
    path = Path(__file__).parents[4] / "vllm_omni" / "deploy" / name
    assert path.is_file(), f"missing deploy config: {path}"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_both_pipelines_registered() -> None:
    assert OMNI_PIPELINES["wevisone_vl"] is WEVISONE_VL_PIPELINE
    assert OMNI_PIPELINES["wevisone_vl_text_only"] is WEVISONE_VL_TEXT_ONLY_PIPELINE
    # Checkpoints report the base Qwen3-VL architecture, so the key comes from
    # the model path segment; the canonical key must survive the name match.
    assert _match_pipeline_key("tencent/WeVisOne-VL-4B", OMNI_PIPELINES) == "wevisone_vl"


def test_mask_pipeline_shape() -> None:
    pipeline = WEVISONE_VL_PIPELINE
    assert len(pipeline.stages) == 2
    assert pipeline.default_deploy_config_name == "wevisone_vl.yaml"

    vlm, decoder = pipeline.stages
    assert vlm.execution_type == StageExecutionType.LLM_AR
    assert vlm.input_sources == ()
    assert vlm.owns_tokenizer and vlm.requires_multimodal_data
    # Plain "text" would drop the feature payload, so the compound type is the
    # handoff contract.
    assert vlm.engine_output_type == "text+segmentation"
    # Both stages are final outputs: stage 0 carries the box text, stage 1 the
    # masks. Mask i answers box i, so the client needs both halves.
    assert vlm.final_output and vlm.final_output_type == "text"
    assert vlm.custom_process_next_stage_input_func.endswith("vlm2pixel_decoder_full_payload")

    constraints = vlm.sampling_constraints
    assert constraints["skip_special_tokens"] is False
    assert constraints["stop_token_ids"] == [IM_END_TOKEN_ID]
    assert constraints["detokenize"] is True

    assert decoder.execution_type == StageExecutionType.LLM_GENERATION
    assert decoder.input_sources == (0,)
    assert decoder.requires_full_payload_input
    assert decoder.final_output and decoder.final_output_type == "image"
    assert decoder.model_arch == "WeVisOnePixelDecoder"


def test_text_only_pipeline_is_single_stage() -> None:
    (stage,) = WEVISONE_VL_TEXT_ONLY_PIPELINE.stages
    assert stage.execution_type == StageExecutionType.LLM_AR
    assert stage.final_output and stage.final_output_type == "text"
    assert stage.engine_output_type == "text"
    assert WEVISONE_VL_TEXT_ONLY_PIPELINE.default_deploy_config_name == "wevisone_vl_text_only.yaml"
    # Boxes must still survive decoding for text-only tasks, which is why
    # skip_special_tokens stays off even without a second stage.
    assert WEVISONE_VL_TEXT_ONLY_PIPELINE.stages[0].sampling_constraints["skip_special_tokens"] is False


def test_architectures_registered() -> None:
    assert _OMNI_MODELS["WeVisOneVLForConditionalGeneration"] == (
        "wevisone",
        "wevisone_vl",
        "WeVisOneVLForConditionalGeneration",
    )
    assert _OMNI_MODELS["WeVisOnePixelDecoder"] == ("wevisone", "pixel_decoder", "WeVisOnePixelDecoder")


def test_decoder_defaults_match_the_shipped_sidecar_cfg() -> None:
    # Read through ``from_dict``, which is the shape the sidecar arrives in.
    assert Qwen3VLSegConfig.from_dict(SHIPPED_SIDECAR_CFG) == Qwen3VLSegConfig()
    # ``from_dict`` ignores unknown keys, so a newer checkpoint cfg does not
    # break an older decoder build.
    drifted = Qwen3VLSegConfig.from_dict({"hidden_dim": 256, "not_a_field": 1})
    assert drifted.hidden_dim == 256
    assert drifted.cnn_image_size == 1024
    assert Qwen3VLSegConfig.from_dict(None) == Qwen3VLSegConfig()


def test_deploy_yaml_keeps_the_handoff_contract() -> None:
    config = _deploy_yaml("wevisone_vl.yaml")
    # Chunked handoff would split a <box> span across payloads.
    assert config["async_chunk"] is False
    vlm, decoder = config["stages"]
    assert vlm["default_sampling_params"]["stop_token_ids"] == [IM_END_TOKEN_ID]
    assert vlm["default_sampling_params"]["skip_special_tokens"] is False
    # A processor-cache hit returns image hashes only, so stage 1 would never
    # see the pixel canvas the decoder needs.
    assert vlm["mm_processor_cache_gb"] == 0
    assert vlm["limit_mm_per_prompt"] == {"image": 1}
    assert decoder["dtype"] == "float32"
    assert decoder["stage_id"] == 1
    assert config["async_chunk"] is False

    text_only = _deploy_yaml("wevisone_vl_text_only.yaml")
    assert len(text_only["stages"]) == 1
    assert text_only["stages"][0]["default_sampling_params"]["skip_special_tokens"] is False


def test_deploy_yaml_pins_the_architectures_the_processor_needs() -> None:
    # vLLM resolves the multimodal processor from hf_config.architectures, not
    # from the stage's model_arch, so this pin is what makes the WeVisOne
    # processor run and the pixel canvas reach stage 1.
    overrides = _deploy_yaml("wevisone_vl.yaml")["stages"][0]["hf_overrides"]
    assert overrides == {"architectures": ["WeVisOneVLForConditionalGeneration"]}
    # model_type has to stay qwen3_vl for the backbone's own config checks.
    assert "model_type" not in overrides
