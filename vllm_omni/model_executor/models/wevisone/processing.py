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

import numpy as np
import torch
from PIL import Image
from transformers import BatchFeature

from vllm.model_executor.models.qwen3_vl import Qwen3VLMultiModalProcessor
from vllm.multimodal.inputs import MultiModalFieldConfig

from .config import Qwen3VLSegConfig

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

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: dict,
        mm_kwargs: dict,
        tok_kwargs: dict,
    ) -> BatchFeature:
        hf_inputs = super()._call_hf_processor(prompt, mm_data, mm_kwargs, tok_kwargs)

        images = mm_data.get("images") or []
        if not images:
            return hf_inputs

        canvases: list[torch.Tensor] = []
        heights: list[int] = []
        widths: list[int] = []
        for item in images:
            image = _as_pil_image(item)
            canvases.append(build_canvas(image, self.canvas_size()))
            heights.append(image.height)
            widths.append(image.width)

        hf_inputs[CANVAS_KEY] = torch.stack(canvases)
        hf_inputs[HW_KEY] = torch.tensor(
            list(zip(heights, widths, strict=True)), dtype=torch.int64
        )
        return hf_inputs

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
