# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Native vLLM-Omni pipeline for Ideogram 4.

The released checkpoint is a Diffusers-style folder, but ``Ideogram4Pipeline``
is not a real Diffusers class, so the pipeline is implemented natively and never
imports the reference package. Semantics that differ from an ordinary Diffusers
pipeline:

* Conditioning comes from Qwen3-VL hidden states at a fixed set of taps, stacked
  and concatenated into ``llm_features`` — not a pooled text embedding.
* CFG is asymmetric: the positive branch sees text + image tokens, the negative
  branch is the unconditional transformer seeing image tokens only with zero
  conditioning. A ``negative_prompt`` in a request is therefore not used.
* The timestep schedule is a resolution-aware logit-normal transform of uniform
  step intervals, and the guidance weight varies per step. Both are resolved
  before the loop, so the loop itself is a plain flow-matching Euler step.
* Latents are normalized with per-channel shift/scale before VAE decode, and the
  patch grid is unpatchified back to an image.

The pipeline returns ``[-1, 1]`` float tensors; ``get_ideogram4_post_process_func``
turns them into images.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Iterable
from typing import Any, ClassVar

import torch
from torch import nn
from transformers import AutoConfig, AutoTokenizer

from diffusers.image_processor import VaeImageProcessor

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_flux2 import (
    DistributedAutoencoderKLFlux2,
)
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import prefetch_subfolders
from vllm_omni.diffusion.model_loader.weight_utils import download_weights_from_hf_specific
from vllm_omni.diffusion.models.ideogram4.config import Ideogram4Config, Ideogram4TextConfig
from vllm_omni.diffusion.models.ideogram4.constants import (
    IMAGE_POSITION_OFFSET,
    LLM_TOKEN_INDICATOR,
    MAX_TEXT_TOKENS,
    OUTPUT_IMAGE_INDICATOR,
    PATCH_SIZE,
    SEQUENCE_PADDING_INDICATOR,
)
from vllm_omni.diffusion.models.ideogram4.fp8_linear import load_streamed_weights
from vllm_omni.diffusion.models.ideogram4.ideogram4_transformer import Ideogram4Transformer2DModel
from vllm_omni.diffusion.models.ideogram4.latent_norm import get_latent_norm
from vllm_omni.diffusion.models.ideogram4.scheduler import (
    PRESETS,
    LogitNormalSchedule,
    get_schedule_for_resolution,
    make_step_intervals,
)
from vllm_omni.diffusion.models.ideogram4.text_encoder import Ideogram4Qwen3VLTextEncoder
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.utils.size_utils import normalize_min_aligned_size
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

logger = logging.getLogger(__name__)

# Request-level knobs accepted through ``extra_args``. ``guidance_schedule`` is
# given in loop-index order (index 0 is the last, most polished step), matching
# the reference presets.
DECLARED_EXTRA_ARGS = frozenset({"guidance_schedule", "mu", "std", "preset"})

DIT_COPY_MODES = ("clone", "keep")

DEFAULT_PRESET = "V4_QUALITY_48"


def _validate_copy_mode(value: Any) -> str:
    mode = str(value).lower()
    if mode not in DIT_COPY_MODES:
        raise ValueError(f"dit_copies must be one of {DIT_COPY_MODES}, got {mode!r}.")
    return mode


def _config_value(od_config: OmniDiffusionConfig, key: str, env_key: str, default: Any = None) -> Any:
    """Read a serve-time knob from ``--custom-pipeline-args``, ``additional_config``, or the environment."""
    custom = od_config.custom_pipeline_args or {}
    additional = od_config.additional_config or {}
    if key in custom:
        return custom[key]
    if key in additional:
        return additional[key]
    return os.environ.get(env_key, default)


def _resolve_dit_copies(od_config: OmniDiffusionConfig) -> str:
    return _validate_copy_mode(_config_value(od_config, "dit_copies", "IDEOGRAM4_DIT_COPIES", "clone"))


def get_ideogram4_post_process_func(
    od_config: OmniDiffusionConfig,
):
    model_name = od_config.model
    if os.path.exists(model_name):
        model_path = model_name
    else:
        model_path = download_weights_from_hf_specific(model_name, None, ["*"])
    vae_config_path = os.path.join(model_path, "vae/config.json")
    with open(vae_config_path) as f:
        vae_config = json.load(f)
        # The Ideogram 4 AE downsamples by 8 and each latent channel is a
        # PATCH_SIZE x PATCH_SIZE patch, so a post-processed pixel block is 16.
        block_out_channels = vae_config.get("block_out_channels") or [128, 256, 512, 512]
        vae_scale_factor = 2 ** (len(block_out_channels) - 1)

    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * PATCH_SIZE)

    def post_process_func(
        images: torch.Tensor | dict[str, Any],
    ):
        if isinstance(images, dict) and isinstance(images.get("payload"), dict):
            payload = dict(images["payload"])
            image_payload = payload.get("image")
            if image_payload is None:
                raise ValueError("Ideogram 4 postprocess expected payload['image'] in output envelope.")
            payload["image"] = image_processor.postprocess(image_payload)
            metadata = images.get("metadata") or {}
            return {
                "payload": payload,
                "metadata": metadata if isinstance(metadata, dict) else {},
            }
        return image_processor.postprocess(images)

    return post_process_func


class Ideogram4Pipeline(
    nn.Module,
    CFGParallelMixin,
    ProgressBarMixin,
    DiffusionPipelineProfilerMixin,
    SupportsComponentDiscovery,
):
    """Text-to-image pipeline for Ideogram 4.

    Conditioning is text-only, so the pipeline does not implement
    ``SupportImageInput`` and reference images are rejected by the API.
    """

    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]
    _callback_tensor_inputs: ClassVar[list[str]] = ["latents", "prompt_embeds"]

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.prefix = prefix
        self.compute_dtype = od_config.dtype
        self.model = od_config.model
        self._execution_device = get_local_device()
        local_files_only = os.path.exists(self.model)

        self.dit_copies = _resolve_dit_copies(od_config)

        prefetch_subfolders(
            self.model,
            ["scheduler", "text_encoder", "tokenizer", "vae"],
            local_files_only=local_files_only,
        )

        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=self.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            ),
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=self.model,
                subfolder="text_encoder",
                revision=None,
                prefix="text_encoder.",
                fall_back_to_pt=True,
                # The vision tower is not used by Ideogram 4's conditioning path;
                # skipping it saves roughly 1.2 GB of weights.
                weight_name_patterns=("language_model.*",),
            ),
        ]

        hf_text_config = AutoConfig.from_pretrained(
            self.model, subfolder="text_encoder", local_files_only=local_files_only
        )
        hf_text_config = getattr(hf_text_config, "text_config", hf_text_config)
        self.text_encoder = Ideogram4Qwen3VLTextEncoder(
            config=Ideogram4TextConfig.from_hf_config(hf_text_config),
            compute_dtype=self.compute_dtype,
            prefix="text_encoder",
        )

        transformer_config = Ideogram4Config.from_hf_config(od_config.tf_model_config.params)
        self.transformer = Ideogram4Transformer2DModel(
            config=transformer_config,
            od_config=od_config,
            prefix="transformer",
            compute_dtype=self.compute_dtype,
        )

        # ``clone`` shares one copy of the DiT between the conditional and
        # unconditional branches, so the sequence-parallel hooks must run once;
        # ``keep`` materializes a second copy and both are hooked. These are
        # instance attributes on purpose: the ClassVar declarations above are the
        # defaults for the shared case.
        if self.dit_copies == "clone":
            self.unconditional_transformer = self.transformer
            self._dit_modules = ["transformer"]
        else:
            self.unconditional_transformer = Ideogram4Transformer2DModel(
                config=transformer_config,
                od_config=od_config,
                prefix="unconditional_transformer",
                compute_dtype=self.compute_dtype,
            )
            self._dit_modules = ["transformer", "unconditional_transformer"]
            self.weights_sources.append(
                DiffusersPipelineLoader.ComponentSource(
                    model_or_path=self.model,
                    subfolder="transformer",
                    revision=None,
                    prefix="unconditional_transformer.",
                    fall_back_to_pt=True,
                )
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model, subfolder="tokenizer", local_files_only=local_files_only
        )

        shift, scale = get_latent_norm()
        self.register_buffer("latent_shift", shift, persistent=False)
        self.register_buffer("latent_scale", scale, persistent=False)

        self.vae = DistributedAutoencoderKLFlux2.from_pretrained(
            self.model, subfolder="vae", local_files_only=local_files_only
        ).to(self._execution_device)

        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * PATCH_SIZE)
        self.tokenizer_max_length = MAX_TEXT_TOKENS
        self.default_sample_size = 128

        for component in (self.text_encoder, self.transformer, self.unconditional_transformer):
            if not any(param.is_meta for param in component.parameters()):
                component.to(self._execution_device)

        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # ``Fp8Linear`` keeps FP8 weights and per-row scales, so streamed tensors
        # are routed to the owning module rather than dequantized at load time.
        return load_streamed_weights(
            self,
            weights,
            self.compute_dtype,
            dtype_for=lambda name: self.vae.dtype if name.startswith("vae.") else self.compute_dtype,
        )

    def _extract_prompts(self, prompts: list) -> tuple[list[str] | None, list[str] | None]:
        prompt = [p if isinstance(p, str) else (p.get("prompt") or "") for p in prompts] or None
        negative_prompt = [
            p.get("negative_prompt") for p in prompts if isinstance(p, dict) and p.get("negative_prompt")
        ] or None
        return prompt, negative_prompt

    def _tokenize(self, prompt: str) -> tuple[torch.Tensor, int]:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        token_ids = encoded["input_ids"][0]
        num_text_tokens = int(token_ids.shape[0])
        if num_text_tokens > self.tokenizer_max_length:
            raise ValueError(
                f"Prompt is too long for Ideogram 4: {num_text_tokens} tokens > {self.tokenizer_max_length}."
            )
        return token_ids, num_text_tokens

    def _build_inputs(self, prompts: list[str], height: int, width: int) -> dict[str, Any]:
        patch = PATCH_SIZE * self.vae_scale_factor
        if height % patch or width % patch:
            raise ValueError(f"height and width must be multiples of {patch} (got {height}x{width}).")
        grid_h, grid_w = height // patch, width // patch
        num_image_tokens = grid_h * grid_w
        device = self._execution_device

        tokenized = [self._tokenize(prompt) for prompt in prompts]
        batch_size = len(tokenized)
        max_text_tokens = max(num_text for _, num_text in tokenized)
        total_seq_len = max_text_tokens + num_image_tokens

        token_ids = torch.zeros(batch_size, total_seq_len, dtype=torch.long, device=device)
        pad_position = torch.full(
            (batch_size, total_seq_len, 3), SEQUENCE_PADDING_INDICATOR, dtype=torch.long, device=device
        )
        text_position_ids, position_ids = pad_position.clone(), pad_position.clone()
        segment_ids = torch.full(
            (batch_size, total_seq_len), SEQUENCE_PADDING_INDICATOR, dtype=torch.long, device=device
        )
        indicator = torch.full(
            (batch_size, total_seq_len), SEQUENCE_PADDING_INDICATOR, dtype=torch.long, device=device
        )

        h_idx = torch.arange(grid_h, device=device).view(-1, 1).expand(grid_h, grid_w).reshape(-1)
        w_idx = torch.arange(grid_w, device=device).view(1, -1).expand(grid_h, grid_w).reshape(-1)
        image_pos = torch.stack([torch.zeros_like(h_idx), h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET

        for b, (tokens, num_text) in enumerate(tokenized):
            offset = max_text_tokens - num_text
            text_span = slice(offset, offset + num_text)
            image_span = slice(offset + num_text, offset + num_text + num_image_tokens)
            token_ids[b, text_span] = tokens
            text_pos = torch.arange(num_text, device=device)
            stacked_text_pos = torch.stack([text_pos, text_pos, text_pos], dim=1)
            text_position_ids[b, text_span] = stacked_text_pos
            position_ids[b, text_span] = stacked_text_pos
            position_ids[b, image_span] = image_pos
            indicator[b, text_span] = LLM_TOKEN_INDICATOR
            indicator[b, image_span] = OUTPUT_IMAGE_INDICATOR
            segment_ids[b, offset : offset + num_text + num_image_tokens] = 1

        return {
            "token_ids": token_ids,
            "text_position_ids": text_position_ids,
            "position_ids": position_ids,
            "segment_ids": segment_ids,
            "indicator": indicator,
            "num_image_tokens": num_image_tokens,
            "grid_h": grid_h,
            "grid_w": grid_w,
            "max_text_tokens": max_text_tokens,
        }

    @torch.no_grad()
    def _encode_text(
        self, token_ids: torch.Tensor, text_position_ids: torch.Tensor, indicator: torch.Tensor
    ) -> torch.Tensor:
        attention_mask = (indicator == LLM_TOKEN_INDICATOR).to(torch.long)
        pos_2d = text_position_ids[..., 0].contiguous()
        taps = self.text_encoder(token_ids, pos_2d, attention_mask)
        stacked = torch.stack(taps, dim=0)
        stacked = torch.permute(stacked, (1, 2, 3, 0)).reshape(stacked.shape[1], stacked.shape[2], -1)
        text_mask = attention_mask.to(stacked.dtype).unsqueeze(-1)
        return (stacked * text_mask).to(torch.float32)

    def predict_noise(self, model: nn.Module, **kwargs) -> torch.Tensor:
        # ``CFGParallelMixin`` assumes ``predict_noise`` returns a tuple so that
        # ``output_slice`` can prefix-slice it. Ideogram 4 returns a plain tensor
        # and the image-token slice is taken by the caller, so keep it plain.
        return model(**kwargs)

    def _negative_model(self, sampling_params) -> nn.Module:
        """Pick the DiT used by the unconditional CFG branch.

        The startup default comes from ``--custom-pipeline-args``, ``additional_config``
        or ``IDEOGRAM4_DIT_COPIES``. A request may override it, but only when both
        copies are loaded (``keep`` mode): in ``clone`` mode the second copy was never
        allocated, so the request value cannot be honoured.
        """
        requested = _validate_copy_mode(sampling_params.extra_args.get("dit_copies") or self.dit_copies)
        if requested != self.dit_copies and self.unconditional_transformer is self.transformer:
            logger.warning(
                "dit_copies=%s requested but this server loaded one shared DiT; using the shared copy.",
                requested,
            )
            return self.transformer
        return self.unconditional_transformer

    def _resolve_sampling_plan(
        self, sampling_params, image_resolution: tuple[int, int]
    ) -> tuple[int, list[float], LogitNormalSchedule]:
        preset = PRESETS.get(str(sampling_params.extra_args.get("preset") or DEFAULT_PRESET))
        num_steps = int(sampling_params.num_inference_steps or (preset.num_steps if preset else 128))
        raw_schedule = sampling_params.extra_args.get("guidance_schedule")
        if raw_schedule:
            gw_per_step = [float(g) for g in raw_schedule]
            if len(gw_per_step) != num_steps:
                raise ValueError(
                    f"guidance_schedule has {len(gw_per_step)} entries but num_inference_steps is {num_steps}."
                )
        else:
            scale = float(sampling_params.true_cfg_scale or (max(preset.guidance_schedule) if preset else 7.0))
            gw_per_step = [scale] * num_steps
        mu = float(sampling_params.extra_args.get("mu") or (preset.mu if preset else 0.5))
        std = float(sampling_params.extra_args.get("std") or (preset.std if preset else 1.0))
        return num_steps, gw_per_step, get_schedule_for_resolution(image_resolution, known_mean=mu, std=std)

    @torch.no_grad()
    def diffuse(
        self,
        latents: torch.Tensor,
        llm_features: torch.Tensor,
        position_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        indicator: torch.Tensor,
        max_text_tokens: int,
        num_steps: int,
        gw_per_step: list[float],
        sigmas: torch.Tensor,
        negative_model: nn.Module,
    ) -> torch.Tensor:
        device = latents.device
        batch_size = latents.shape[0]
        if sigmas.shape[0] != num_steps + 1:
            raise ValueError(f"Expected {num_steps + 1} schedule values, got {sigmas.shape[0]}.")

        # Text slots carry no latent: they are zero-filled so the packed sequence
        # has one contiguous image block per sample.
        text_z_padding = torch.zeros(batch_size, max_text_tokens, latents.shape[-1], dtype=torch.float32, device=device)
        neg_position_ids = position_ids[:, max_text_tokens:]
        neg_segment_ids = segment_ids[:, max_text_tokens:]
        neg_indicator = indicator[:, max_text_tokens:]
        neg_llm_features = torch.zeros(
            batch_size, latents.shape[1], llm_features.shape[-1], dtype=llm_features.dtype, device=device
        )

        z = latents
        with self.progress_bar(total=num_steps) as progress_bar:
            for k in range(num_steps):
                # ``sigmas`` is in the usual diffusers order: sigmas[0] is the
                # starting (noisiest) timestep and sigmas[num_steps] the final one.
                t_val = float(sigmas[num_steps - k])
                s_val = float(sigmas[num_steps - k - 1])
                gw = gw_per_step[num_steps - 1 - k]
                t = torch.full((batch_size,), t_val, dtype=torch.float32, device=device)

                positive_kwargs = {
                    "model": self.transformer,
                    "llm_features": llm_features,
                    "x": torch.cat([text_z_padding, z], dim=1),
                    "t": t,
                    "position_ids": position_ids,
                    "segment_ids": segment_ids,
                    "indicator": indicator,
                }
                negative_kwargs = {
                    "model": negative_model,
                    "llm_features": neg_llm_features,
                    "x": z,
                    "t": t,
                    "position_ids": neg_position_ids,
                    "segment_ids": neg_segment_ids,
                    "indicator": neg_indicator,
                }

                # The negative branch is a genuinely unconditional model, so CFG
                # applies whenever the per-step guidance weight is above 1.
                do_true_cfg = gw > 1.0
                pred = self.predict_noise_maybe_with_cfg(
                    do_true_cfg, gw, positive_kwargs, negative_kwargs if do_true_cfg else None
                )
                v = pred[:, max_text_tokens:]
                z = z + v * (s_val - t_val)
                progress_bar.update()

        return z

    def _decode(self, latents: torch.Tensor, grid_h: int, grid_w: int) -> DiffusionOutput:
        latents = latents * self.latent_scale + self.latent_shift
        batch_size = latents.shape[0]
        ae_channels = int(latents.shape[-1] // (PATCH_SIZE * PATCH_SIZE))
        latents = (
            latents.view(batch_size, grid_h, grid_w, PATCH_SIZE, PATCH_SIZE, ae_channels)
            .permute(0, 5, 1, 3, 2, 4)
            .contiguous()
            .view(batch_size, ae_channels, grid_h * PATCH_SIZE, grid_w * PATCH_SIZE)
        )
        latents = latents.to(self.vae.dtype)
        image = self.vae.decoder(latents).float().clamp(-1.0, 1.0)
        return DiffusionOutput(
            output=image,
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    def forward(self, req: DiffusionRequestBatch) -> list[DiffusionOutput]:
        return [self.generate(request) for request in req.requests]

    @torch.no_grad()
    def generate(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        sampling_params = req.sampling_params
        prompts, negative_prompt = self._extract_prompts([req.prompt])
        if negative_prompt:
            logger.warning(
                "Ideogram 4 CFG uses an unconditional branch rather than a negative prompt; "
                "negative_prompt is ignored."
            )
        num_images = max(1, int(sampling_params.num_outputs_per_prompt or 1))
        prompts = prompts * num_images

        height, width = normalize_min_aligned_size(
            sampling_params.height or self.default_sample_size * self.vae_scale_factor,
            sampling_params.width or self.default_sample_size * self.vae_scale_factor,
            self.vae_scale_factor * PATCH_SIZE,
        )
        num_steps, gw_per_step, schedule = self._resolve_sampling_plan(sampling_params, (height, width))

        inputs = self._build_inputs(prompts, height, width)
        llm_features = self._encode_text(inputs["token_ids"], inputs["text_position_ids"], inputs["indicator"])

        device = self._execution_device
        generator = sampling_params.generator
        if generator is None:
            generator = torch.Generator(device=device)
            if sampling_params.seed is not None:
                generator.manual_seed(sampling_params.seed)

        if sampling_params.latents is not None:
            latents = sampling_params.latents.to(device=device, dtype=torch.float32)
        else:
            latents = torch.randn(
                len(prompts),
                inputs["num_image_tokens"],
                self.transformer.config.in_channels,
                dtype=torch.float32,
                device=device,
                generator=generator,
            )

        sigmas = (
            torch.as_tensor(sampling_params.sigmas, dtype=torch.float32, device=device)
            if sampling_params.sigmas
            else schedule(make_step_intervals(num_steps).to(device))
        )

        z = self.diffuse(
            latents,
            llm_features,
            inputs["position_ids"],
            inputs["segment_ids"],
            inputs["indicator"],
            inputs["max_text_tokens"],
            num_steps,
            gw_per_step,
            sigmas,
            self._negative_model(sampling_params),
        )

        if sampling_params.output_type == "latent":
            return DiffusionOutput(
                output=latents,
                stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
            )
        return self._decode(z, inputs["grid_h"], inputs["grid_w"])
