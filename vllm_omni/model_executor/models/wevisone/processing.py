# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Per-image decoder canvas carried alongside the Qwen3-VL vision inputs.

Ported from the Tencent WeVisOne-VL reference implementation (Apache-2.0).

The pixel decoder always sees the *original* image resized bicubically to
``cnn_image_size`` — never Qwen's smart-resized patches, whose aspect ratio is
reshuffled by patch merging. ``vllm`` hands the model only the processed vision
inputs, so the canvas is produced here as an extra multimodal field:

    ``wevisone_canvas`` — ``[N, 3, S, S]`` float RGB in [0, 1], one canvas per image
    ``wevisone_hw``     — ``[N, 2]`` original (H, W) the masks are resampled back to

Both are plain tensors (``keep_on_cpu=True``) because they are read by the stage
input processor rather than consumed by the vision tower; ``Qwen3VLForConditional-
Generation._parse_and_validate_multimodal_inputs`` ignores unknown kwargs.
"""

from collections.abc import Mapping

import numpy as np
import torch
from PIL import Image
from transformers import BatchFeature
from vllm.logger import init_logger
from vllm.model_executor.models.qwen3_vl import Qwen3VLMultiModalProcessor
from vllm.multimodal.inputs import MultiModalFieldConfig

from .config import Qwen3VLSegConfig

logger = init_logger(__name__)

CANVAS_KEY = "wevisone_canvas"
HW_KEY = "wevisone_hw"

#: Default canvas side, matching ``Qwen3VLSegConfig.cnn_image_size``.
CANVAS_SIZE = Qwen3VLSegConfig().cnn_image_size


def pil_to_float_rgb(image: Image.Image) -> torch.Tensor:
    """``[3, H, W]`` float RGB in [0, 1], exactly as the reference feeds the CNN."""
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def build_canvas(image: Image.Image, size: int = CANVAS_SIZE) -> torch.Tensor:
    """Bicubic square canvas for the CNN stem, from the un-resized image."""
    return pil_to_float_rgb(image.resize((size, size), resample=Image.Resampling.BICUBIC))


def _as_pil_image(item: object) -> Image.Image:
    if isinstance(item, Image.Image):
        return item
    raise ValueError(
        "WeVisOne masks require PIL image inputs: the pixel decoder needs the "
        f"original pixels, got {type(item).__name__}."
    )


class WeVisOneMultiModalProcessor(Qwen3VLMultiModalProcessor):
    """Qwen3-VL processor + the decoder canvas for every image item.

    The canvas is computed from the raw PIL image before any smart-resize, so
    stage 1 can rebuild the pixel decoder's input and return masks at the
    original resolution.
    """

    def _postprocess_hf_mm_data(
        self,
        mm_inputs: Mapping[str, object],
        hf_processor_mm_kwargs: Mapping[str, object],
        processed_data: BatchFeature,
    ) -> BatchFeature:
        """Add the canvas fields to whatever the HF processor produced.

        ``Qwen3VLMultiModalProcessor`` overrides ``_apply_hf_processor_main`` and
        calls the HF processor directly rather than through ``_call_hf_processor``,
        so overriding the latter never fires on that path. ``_postprocess_hf_mm_data``
        is the one hook both the base and the Qwen3-VL paths run through, so the
        canvas is attached there.

        vLLM assembles the inputs as ``{f"{modality}s": items}`` (parse.py), so the
        pictures arrive under ``images``; the singular modality name is accepted for
        callers that pass it directly.
        """
        images = mm_inputs.get("images") or mm_inputs.get("image") or []
        if not images:
            return processed_data

        size = self.canvas_size()
        pil_images = [_as_pil_image(item) for item in images]
        processed_data[CANVAS_KEY] = torch.stack([build_canvas(image, size) for image in pil_images])
        processed_data[HW_KEY] = torch.tensor([(image.height, image.width) for image in pil_images], dtype=torch.int64)
        return processed_data

    def canvas_size(self) -> int:
        """Canvas side for this model, honouring an overridden seg config."""
        seg_config = getattr(self.info.get_hf_config(), "seg_config", None)
        if isinstance(seg_config, dict):
            return int(seg_config.get("cnn_image_size", CANVAS_SIZE))
        return int(getattr(seg_config, "cnn_image_size", CANVAS_SIZE))

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: dict,
    ) -> dict:
        fields = dict(super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs))
        if CANVAS_KEY in hf_inputs:
            # One canvas per image item, kept on CPU for cross-stage transfer.
            fields[CANVAS_KEY] = MultiModalFieldConfig.batched("image", keep_on_cpu=True)
            fields[HW_KEY] = MultiModalFieldConfig.batched("image", keep_on_cpu=True)
        return fields
