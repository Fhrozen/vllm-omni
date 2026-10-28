# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Image-content choices for WeVisOne-VL mask output.

A pixel-decoder stage ships one stacked ``[N, 1, H, W]`` tensor per request, so
``_create_image_choice`` emits one PNG per leading slice and labels each part with
``mask_index``. Mask ``i`` pairs with box ``i``, and the boxes live in the text
choice content (``metrics`` carries model metadata only on the diffusion serving
path), so ``mask_index`` is the only link between a mask and its box on the wire.
"""

from __future__ import annotations

import base64
from io import BytesIO
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _image_choice(image_data, *, stage_durations=None, peak_memory_mb=None):
    omni_outputs = SimpleNamespace(
        request_output=None,
        stage_durations=stage_durations or {},
        peak_memory_mb=peak_memory_mb,
        images=[],
        outputs=[SimpleNamespace(multimodal_output={"image": image_data})],
    )
    choices = OmniOpenAIServingChat._create_image_choice(  # type: ignore[misc]
        None,
        omni_outputs=omni_outputs,
        role="assistant",
        request=SimpleNamespace(return_token_ids=False),
    )
    assert len(choices) == 1
    content = choices[0].message.content
    assert isinstance(content, list)
    return content


def _decoded_png(part):
    from PIL import Image

    url = part["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    raw = base64.b64decode(url.split(",", 1)[1])
    return Image.open(BytesIO(raw))


def test_stacked_masks_become_one_png_per_leading_slice():
    # One [1, H, W] mask per leading slice, as the pixel-decoder stage emits them.
    masks = torch.tensor(
        [
            [[[0.0, 1.0], [0.2, 0.0]]],
            [[[0.4, 0.4], [0.6, 1.0]]],
            [[[1.0, 0.0], [0.0, 0.0]]],
        ],
        dtype=torch.float32,
    )
    assert masks.shape == (3, 1, 2, 2)

    content = _image_choice(masks)

    assert len(content) == 3
    assert [part["mask_index"] for part in content] == [0, 1, 2]
    for part in content:
        assert part["type"] == "image_url"
        image = _decoded_png(part)
        # Mask logits are probabilities in [0, 1], so the PNG is grayscale and
        # scaled by 255 rather than re-normalised.
        assert image.mode == "L"
        assert image.size == (2, 2)

    first = _decoded_png(content[0])
    assert first.getpixel((0, 0)) == 0
    assert first.getpixel((1, 0)) == 255
    assert first.getpixel((0, 1)) == 51


def test_mask_order_matches_box_order_of_the_text_choice():
    # Three query rows in response order: the part ordinals must stay in the
    # order the decoder produced them, which is the order of the boxes.
    masks = torch.arange(3 * 1 * 2 * 2, dtype=torch.float32).reshape(3, 1, 2, 2) / 12.0
    content = _image_choice(masks)
    assert [part["mask_index"] for part in content] == list(range(3))
    pixels = [_decoded_png(part).getpixel((0, 0)) for part in content]
    assert pixels == [int(value * 255) for value in (0.0, 4.0 / 12.0, 8.0 / 12.0)]


def test_single_mask_keeps_its_own_rank_and_gets_no_mask_index():
    # A single mask is an ordinary image, not a stack: pairing is unambiguous and
    # clients do not need mask_index for it.
    content = _image_choice(torch.full((1, 1, 2, 2), 0.5, dtype=torch.float32))
    assert len(content) == 1
    assert "mask_index" not in content[0]
    assert _decoded_png(content[0]).mode == "L"


def test_empty_payload_still_produces_a_choice_with_a_fallback_message():
    # Text-only requests reach the decoder stage with a zero-numel tensor: the
    # request still needs a choice, and no mask is invented for it.
    content = _image_choice(torch.empty(0))
    assert content == [{"type": "text", "text": "Image generation completed but no images were produced."}]


def test_zero_numel_slots_are_skipped_without_inventing_masks():
    # Every slot of a degenerate stack is empty, so no PNG is emitted and the
    # choice falls back to the text placeholder instead of a blank mask.
    all_empty = torch.zeros((3, 1, 0, 2), dtype=torch.float32)
    assert _image_choice(all_empty) == [
        {"type": "text", "text": "Image generation completed but no images were produced."}
    ]

    # Slice ordinals are recorded only for the slices that survive the skip, so a
    # skipped slot never shifts the mask i <-> box i pairing of the parts that do
    # get emitted.
    masks = torch.zeros((2, 1, 2, 2), dtype=torch.float32)
    assert [part["mask_index"] for part in _image_choice(masks)] == [0, 1]


def test_stage_metrics_are_carried_on_every_mask_part():
    stage_durations = {"stage_0": 1.2, "stage_1": 0.3}
    content = _image_choice(
        torch.zeros((2, 1, 2, 2), dtype=torch.float32), stage_durations=stage_durations, peak_memory_mb=12.5
    )
    assert all(part["stage_durations"] == stage_durations for part in content)
    assert all(part["peak_memory_mb"] == 12.5 for part in content)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
