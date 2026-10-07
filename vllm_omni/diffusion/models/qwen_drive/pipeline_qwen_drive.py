# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen-Drive planning-expert pipeline (diffusion stage that consumes the VLM stage's KV cache).

``DiffusionEngine.step() -> pipeline.forward(req)``. The upstream AR stage sends the post-rotary K/V of the
VLM's full-attention layers (``req.past_key_values``) and the mRoPE position of the last prompt token
(``req.kv_metadata["rope_anchor"]``). Per-scene numeric inputs arrive in ``sampling_params.extra_args``.
Output: ``DiffusionOutput(output={"trajectories": ndarray[num_samples, points, 3]})``.

The pipeline self-loads one planner (``planner_subfolder`` in the deploy ``model_config``) from the model
directory, so no ``trust_remote_code`` is needed: ``config.json`` is read as plain JSON.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.qwen_drive.planning_expert import (
    PlannerSettings,
    PlanningExpert,
    plan_trajectories,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = init_logger(__name__)

DEFAULT_PLANNER_SUBFOLDER = "planner-rl"
_WEIGHT_PREFIX = "planning_expert."


def _post_process(x):
    return x


def get_qwen_drive_post_process_func(od_config: OmniDiffusionConfig):
    del od_config
    return _post_process


def _as_tensor(value: Any, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(value), dtype=dtype, device=device)


class QwenDrivePlannerPipeline(nn.Module):
    """KV-conditioned flow-matching trajectory planner (registered as ``QwenDrivePlannerPipeline``)."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.prefix = prefix
        model_config = dict(od_config.model_config or {})
        self.planner_subfolder = str(model_config.get("planner_subfolder", DEFAULT_PLANNER_SUBFOLDER))

        self.model_dir = self._resolve_model_dir(od_config.model, self.planner_subfolder)
        with open(os.path.join(self.model_dir, "config.json")) as f:
            self.settings = PlannerSettings.from_hf_config(json.load(f))

        self._device = self._resolve_device()
        self._dtype = self._resolve_dtype(od_config)
        self.expert = PlanningExpert(self.settings)
        self._load_planner()
        self.expert.to(device=self._device, dtype=self._dtype).eval()

    @staticmethod
    def _resolve_model_dir(model: str | None, planner_subfolder: str) -> str:
        if not model:
            raise ValueError("QwenDrivePlannerPipeline needs od_config.model")
        if os.path.isdir(model):
            return model
        from vllm_omni.transformers_utils.repo_utils import hf_api

        return hf_api().snapshot_download(repo_id=model, allow_patterns=["config.json", f"{planner_subfolder}/*"])

    @staticmethod
    def _resolve_dtype(od_config: OmniDiffusionConfig) -> torch.dtype:
        dt = od_config.dtype
        if isinstance(dt, torch.dtype):
            return dt
        return getattr(torch, str(dt).split(".")[-1], torch.bfloat16)

    @staticmethod
    def _resolve_device() -> torch.device:
        from vllm_omni.diffusion.distributed.utils import get_local_device

        try:
            return get_local_device()
        except Exception:  # noqa: BLE001
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_planner(self) -> None:
        from safetensors.torch import load_file

        path = os.path.join(self.model_dir, self.planner_subfolder, "model.safetensors")
        logger.info("QwenDrivePlannerPipeline: loading planner weights from %s", path)
        weights = {k.removeprefix(_WEIGHT_PREFIX): v for k, v in load_file(path).items()}
        self.expert.load_state_dict(weights, strict=True)

    def load_weights(self, weights=()):  # noqa: D401
        """No-op for the diffusion loader: the planner self-loads in ``__init__`` (no ``weights_sources``)."""
        for _ in weights:
            pass
        return None

    # ------------------------------------------------------------------
    @staticmethod
    def _rope_anchor(req: OmniDiffusionRequest, extra_args: dict[str, Any]) -> int:
        meta = getattr(req, "kv_metadata", None) or {}
        anchor = meta.get("rope_anchor", extra_args.get("rope_anchor"))
        if anchor is None:
            raise ValueError("missing rope_anchor (kv_metadata['rope_anchor'] or extra_args['rope_anchor'])")
        return int(np.asarray(anchor).reshape(-1)[-1])

    def _scene_cache(self, req: OmniDiffusionRequest) -> list[tuple[torch.Tensor, torch.Tensor]]:
        kv = getattr(req, "past_key_values", None)
        if kv is None:
            raise ValueError("missing past_key_values: the VLM stage did not send its KV cache")
        # Entries of layers the sender skipped (non-attention) are None; the rest are in layer order.
        pairs = [(k, v) for k, v in zip(kv.key_cache, kv.value_cache, strict=True) if k is not None]
        expected = self.settings.expert.num_kv_sources
        if len(pairs) != expected:
            raise ValueError(f"expected KV of {expected} full-attention layers, got {len(pairs)}")
        return [
            (k.to(self._device, self._dtype).unsqueeze(0), v.to(self._device, self._dtype).unsqueeze(0))
            for k, v in pairs
        ]

    def _ego_status(self, extra_args: dict[str, Any]) -> torch.Tensor:
        if extra_args.get("ego_status") is not None:
            return _as_tensor(extra_args["ego_status"], torch.float32, self._device).reshape(1, -1)
        parts = [extra_args["ego_velocity"], extra_args["ego_acceleration"], extra_args["driving_command"]]
        return _as_tensor(np.concatenate([np.asarray(p, dtype=np.float32).reshape(-1) for p in parts]), torch.float32, self._device).reshape(1, -1)

    @torch.inference_mode()
    def forward(self, req: OmniDiffusionRequest, **kwargs) -> DiffusionOutput:
        extra_args = getattr(req.sampling_params, "extra_args", None) or {}

        if getattr(req, "past_key_values", None) is None and "history" not in extra_args:
            # Engine warmup / dummy run: no scene, return zeros.
            return DiffusionOutput(
                output={"trajectories": np.zeros((1, self.settings.num_future_points, 3), dtype=np.float32)}
            )

        try:
            scene_cache = self._scene_cache(req)
            anchor = torch.full((3, 1), self._rope_anchor(req, extra_args), dtype=torch.long, device=self._device)
            trajectories = plan_trajectories(
                self.expert,
                self.settings,
                scene_cache,
                anchor,
                history=_as_tensor(extra_args["history"], torch.float64, self._device).unsqueeze(0),
                history_velocity=_as_tensor(extra_args["history_velocity"], torch.float32, self._device).unsqueeze(0),
                history_acceleration=_as_tensor(
                    extra_args["history_acceleration"], torch.float32, self._device
                ).unsqueeze(0),
                nav_command=torch.tensor([int(extra_args["nav_command"])], dtype=torch.long, device=self._device),
                ego_status=self._ego_status(extra_args),
                num_samples=int(extra_args.get("num_samples", 1)),
                num_steps=int(extra_args.get("num_steps") or self.settings.num_inference_steps),
                seed=int(extra_args.get("seed") if extra_args.get("seed") is not None else self.settings.noise_seed),
            )
        except (KeyError, ValueError) as e:
            return DiffusionOutput(error=f"QwenDrivePlannerPipeline: {e}")

        output: dict[str, Any] = {"trajectories": trajectories.float().cpu().numpy()}
        if extra_args.get("reasoning") is not None:
            output["reasoning"] = extra_args["reasoning"]
        return DiffusionOutput(output=output)
