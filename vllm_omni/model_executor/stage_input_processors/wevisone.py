# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage input processor for WeVisOne-VL: VLM (stage 0) -> pixel decoder (stage 1).

Stage 0 carries the pixel-decoder contract in the ``segmentation`` payload group
(``vllm_omni/data_entry_keys.py``).  This builder turns one request's accumulated
stage-0 output into the per-request payload the decoder stage reads:

``features``  pre-merger ViT feature maps, ``[layers, C, H, W]``
``grid_hw``   pre-merge ``(h, w)`` of that image
``t_seg``     ``<OBJ>`` query rows, ``[num_queries, llm_dim]``
``boxes``     normalized ``xyxy`` boxes, ``[N, 4]``, ordered to match the rows
``canvas``    decoder CNN canvas, ``[1, 3, S, S]`` float RGB in ``[0, 1]``
``mask_hw``   original image size the masks are resampled back to

Boxes are reconstructed from the *generated* token ids because the pooling output
has no text: the request's decoded response is not available here.  Scanning only
generated tokens matters -- a ``visual_prompt`` request can carry ``<box>`` spans
in its prompt, and the reference parses boxes from the response alone.
"""

import logging
from collections.abc import Mapping
from typing import Any

import torch

from vllm_omni.data_entry_keys import OmniPayload
from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.model_executor.models.wevisone.constants import (
    SEG_TOKEN_ID_FALLBACK,
    parse_seg_boxes_from_ids,
)

logger = logging.getLogger(__name__)

# Keys that carry a complete result rather than a per-step delta.  ``features``,
# ``grid_hw``, ``canvas`` and ``mask_hw`` are emitted once at prefill; ``boxes`` is
# rebuilt here from the whole response.  ``segmentation.t_seg`` keeps CONCAT_DIM0
# semantics so query rows accumulate across decode steps.
_FULL_PAYLOAD_REPLACE_KEYS: frozenset[str] = frozenset(
    {
        "segmentation.features",
        "segmentation.grid_hw",
        "segmentation.canvas",
        "segmentation.mask_hw",
        "segmentation.boxes",
    }
)


def _ensure_list(values: Any) -> list[Any]:
    return list(values) if isinstance(values, (list, tuple)) else []


def _tensor_or_none(value: Any) -> torch.Tensor | None:
    return value if isinstance(value, torch.Tensor) else None


def _seg_field(pooling_output: Mapping[str, Any], name: str) -> Any:
    """Read one ``segmentation`` field from a flat or nested pooling output."""
    nested = pooling_output.get("segmentation")
    if isinstance(nested, Mapping):
        return nested.get(name)
    return pooling_output.get(f"segmentation.{name}")


def vlm2pixel_decoder_full_payload(
    transfer_manager: Any,
    pooling_output: Mapping[str, Any],
    request: OmniEngineCoreRequest,
) -> OmniPayload | None:
    """Pack one request's complete stage-0 output for the pixel-decoder stage."""
    del transfer_manager
    rid = getattr(request, "request_id", None)
    if not isinstance(pooling_output, Mapping):
        logger.warning(
            "vlm2pixel_decoder_full_payload: pooling_output not a dict (type=%s) for req=%s; "
            "consumer wait gate may hang.",
            type(pooling_output).__name__,
            rid,
        )
        return None

    features = _tensor_or_none(_seg_field(pooling_output, "features"))
    grid_hw = _tensor_or_none(_seg_field(pooling_output, "grid_hw"))
    t_seg = _tensor_or_none(_seg_field(pooling_output, "t_seg"))
    canvas = _tensor_or_none(_seg_field(pooling_output, "canvas"))
    mask_hw = _tensor_or_none(_seg_field(pooling_output, "mask_hw"))

    prompt_token_ids = _ensure_list(getattr(request, "prompt_token_ids", []) or [])
    all_token_ids = _ensure_list(getattr(request, "all_token_ids", None) or [])
    if not all_token_ids:
        all_token_ids = prompt_token_ids + _ensure_list(getattr(request, "output_token_ids", []) or [])
    generated_token_ids = all_token_ids[len(prompt_token_ids):]

    boxes = parse_seg_boxes_from_ids(generated_token_ids)
    if not boxes:
        # Text-only tasks (ocr, count_*, freeform answers) produce no <OBJ> and no
        # boxes; the decoder stage returns None for them and never touches the
        # sidecar. Withholding the payload keeps those requests text-only.
        if t_seg is not None and t_seg.numel() > 0:
            logger.warning(
                "vlm2pixel_decoder_full_payload: req=%s has %s <OBJ> query rows but no parsed "
                "boxes; masks cannot be decoded, shipping text-only.",
                rid,
                int(t_seg.shape[0]),
            )
        return None

    seg_token_id = SEG_TOKEN_ID_FALLBACK

    # Row i of t_seg is the last-decoder-layer output for the token fed at position
    # i, so a <OBJ> at position p contributes row p. Rows for tokens that were
    # predicted but never fed back in are absent, which is why the guard below is
    # a check rather than an assumption.
    rows = [i for i, token_id in enumerate(all_token_ids) if token_id == seg_token_id]
    available = int(t_seg.shape[0]) if t_seg is not None else 0
    usable = [row for row in rows if row < available]
    if len(usable) != len(boxes):
        logger.warning(
            "vlm2pixel_decoder_full_payload: req=%s has %s <OBJ> positions (%s usable rows) "
            "but %s parsed boxes; dropping the request's masks. t_seg rows=%s ids_len=%s",
            rid,
            len(rows),
            len(usable),
            len(boxes),
            available,
            len(all_token_ids),
        )
        return None
    if not usable:
        return None

    query_rows = t_seg[usable]  # type: ignore[index]

    payload: OmniPayload = {
        "ids": {"all": list(all_token_ids), "prompt": list(prompt_token_ids)},
        "segmentation": {
            "features": features.detach().cpu() if features is not None else None,
            "grid_hw": grid_hw.detach().cpu() if grid_hw is not None else None,
            "t_seg": query_rows.detach().cpu().contiguous(),
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "canvas": canvas.detach().cpu() if canvas is not None else None,
            "mask_hw": mask_hw.detach().cpu() if mask_hw is not None else None,
        },
        "meta": {"finished": torch.tensor(True, dtype=torch.bool)},
    }
    return payload


__all__ = ["vlm2pixel_decoder_full_payload", "_FULL_PAYLOAD_REPLACE_KEYS"]
