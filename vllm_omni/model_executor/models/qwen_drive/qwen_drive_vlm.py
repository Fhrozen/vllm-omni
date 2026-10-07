# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen-Drive VLM stage: the Qwen3.5 vision-language model of a Qwen-Drive checkpoint.

The checkpoint stores the VLM under the ``vlm.`` prefix next to the planning expert (served by the diffusion
stage), so the checkpoint's nested ``vlm_config`` is flattened for vLLM (see ``OmniEngineArgs``) and the prefix is
stripped here. The stage can also hand the mRoPE position of the last prompt token to the planner stage, which
continues the VLM's rotary positions for its waypoint tokens.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import torch
from vllm.logger import init_logger
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration

logger = init_logger(__name__)

_WEIGHT_PREFIX = "vlm."
_MAX_TRACKED_REQUESTS = 256


class QwenDriveVLMForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    def __init__(self, *, vllm_config, prefix: str = "model"):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # req_id -> mRoPE position (shape [3]) of the last token scheduled for that request.
        self._last_mrope_position: dict[str, torch.Tensor] = {}

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Planner/perception tensors live in other files; anything without the `vlm.` prefix is not ours.
        return super().load_weights(
            (name[len(_WEIGHT_PREFIX) :], weight) for name, weight in weights if name.startswith(_WEIGHT_PREFIX)
        )

    def _clear_warmup_state(self) -> None:
        self._last_mrope_position.clear()

    def prepare_runner_inputs(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        req_ids: Sequence[str],
        num_computed_tokens: Sequence[int],
        num_scheduled_tokens: Sequence[int],
        input_ids_buffer: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Remember the mRoPE position of each request's last scheduled token (the planner's anchor)."""
        if positions is not None and positions.ndim == 2:
            ends = np.cumsum(np.asarray(num_scheduled_tokens)) - 1
            for req_id, end in zip(req_ids, ends.tolist(), strict=False):
                self._last_mrope_position[req_id] = positions[:, end].clone()
            while len(self._last_mrope_position) > _MAX_TRACKED_REQUESTS:
                self._last_mrope_position.pop(next(iter(self._last_mrope_position)))
        return input_ids, positions

    def get_kv_transfer_metadata(
        self,
        req_id: str,
        *,
        num_computed_tokens: int | None = None,
    ) -> dict[str, Any] | None:
        position = self._last_mrope_position.pop(req_id, None)
        if position is None:
            return None
        # All three mRoPE sections agree on a text token, which is always the last prompt token.
        return {"rope_anchor": int(position[-1].item())}
