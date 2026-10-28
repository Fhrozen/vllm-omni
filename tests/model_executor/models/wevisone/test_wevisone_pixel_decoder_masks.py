# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Mask post-processing contract of the WeVisOne-VL pixel decoder.

Mirrors the reference ``pixel_decoder._masks_from_logits`` in
``extra_repos/WeVisOne-VL/wevisone_vl/pixel_decoder.py``: the decoder returns
logits, sigmoid is applied *after* the bilinear resample back to the original
image size, and the result is always ``[num_masks, 1, H, W]`` float32 on CPU.
Mask order equals box order, so mask ``i`` pairs with box ``i``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.wevisone import pixel_decoder

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_masks_are_sigmoid_resampled_to_the_original_image_size():
    logits = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]], dtype=torch.float32)

    masks = pixel_decoder._masks_from_logits(logits, height=4, width=6)

    assert masks.shape == (1, 1, 4, 6)
    assert masks.dtype == torch.float32
    assert masks.is_contiguous()
    assert masks.device.type == "cpu"
    assert masks.min() >= 0.0 and masks.max() <= 1.0

    # Bilinear resampling keeps the corners at the sigmoid of the corner logits.
    corners = torch.sigmoid(logits[0, 0])
    assert torch.allclose(masks[0, 0, 0, 0], corners[0, 0])
    assert torch.allclose(masks[0, 0, -1, -1], corners[-1, -1])


def test_constant_logits_resample_to_a_constant_half():
    # Sigmoid(0) == 0.5 and interpolation of a constant plane is exact, so the
    # whole mask is 0.5 — a deterministic check that no normalisation sneaks in.
    masks = pixel_decoder._masks_from_logits(torch.zeros(2, 1, 3, 3), height=7, width=5)
    assert masks.shape == (2, 1, 7, 5)
    assert torch.allclose(masks, torch.full_like(masks, 0.5))


def test_mask_order_matches_box_order():
    first = torch.full((1, 1, 2, 2), -4.0)
    second = torch.full((1, 1, 2, 2), 4.0)
    masks = pixel_decoder._masks_from_logits(torch.cat([first, second]), height=2, width=2)

    assert masks[0, 0].max() < 0.1  # box 0 -> mask 0
    assert masks[1, 0].min() > 0.9  # box 1 -> mask 1


def test_single_mask_logits_gain_the_batch_dimension():
    masks = pixel_decoder._masks_from_logits(torch.zeros(2, 3), height=2, width=2)
    assert masks.shape == (1, 1, 2, 2)


def test_decoder_logit_layouts_are_accepted():
    # ``[num_masks, h, w]`` is what ``WeVisOneMaskDecoder.forward`` actually returns.
    for logits, expected in [
        (torch.zeros(2, 3, 4), (2, 1, 2, 2)),
        (torch.zeros(2, 1, 3, 4), (2, 1, 2, 2)),
        (torch.zeros(3, 4), (1, 1, 2, 2)),
    ]:
        assert pixel_decoder._masks_from_logits(logits, height=2, width=2).shape == expected


def test_unexpected_logit_shapes_are_rejected():
    with pytest.raises(ValueError, match="unexpected mask logit shape"):
        pixel_decoder._masks_from_logits(torch.zeros(1, 2, 2, 2), height=2, width=2)

    with pytest.raises(ValueError, match="unexpected mask logit shape"):
        pixel_decoder._masks_from_logits(torch.zeros(4), height=2, width=2)


def test_segmentation_view_accepts_nested_and_flat_payloads():
    nested = {"segmentation": {"t_seg": torch.zeros(1), "boxes": torch.zeros(1, 4)}}
    assert set(pixel_decoder._segmentation_view(nested)) == {"t_seg", "boxes"}

    flat = {
        "segmentation.t_seg": torch.zeros(1),
        "segmentation.boxes": torch.zeros(1, 4),
        "ids.all": [1, 2],
    }
    view = pixel_decoder._segmentation_view(flat)
    assert set(view) == {"t_seg", "boxes"}


def _bare_decoder():
    """A decoder instance with the lazy sidecar load already satisfied.

    ``_decode_request`` loads the sidecar as soon as a request carries query
    rows, so validation tests must not trip that path.
    """
    decoder = pixel_decoder.WeVisOnePixelDecoder.__new__(pixel_decoder.WeVisOnePixelDecoder)
    decoder._loaded = True
    return decoder


def _payload():
    return {
        # Cross-stage payloads carry stacked feature layers, not one [1,C,H,W] map.
        "segmentation.features": torch.zeros(2, 3, 2, 2),
        "segmentation.grid_hw": torch.tensor([[2, 2]], dtype=torch.int32),
        "segmentation.t_seg": torch.zeros(1, 3),
        "segmentation.boxes": torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
        "segmentation.canvas": torch.zeros(1, 3, 4, 4),
        "segmentation.mask_hw": torch.tensor([[4, 4]], dtype=torch.int32),
    }


def test_decode_request_needs_features_boxes_and_canvas():
    decoder = _bare_decoder()

    incomplete = _payload()
    incomplete.pop("segmentation.features")
    with pytest.raises(ValueError, match="no pre-merger feature maps"):
        decoder._decode_request(incomplete)

    incomplete = _payload()
    incomplete.pop("segmentation.boxes")
    with pytest.raises(ValueError, match="no boxes parsed from the response"):
        decoder._decode_request(incomplete)

    incomplete = _payload()
    incomplete.pop("segmentation.canvas")
    with pytest.raises(ValueError, match="no decoder canvas"):
        decoder._decode_request(incomplete)


def test_hw_pair_accepts_plain_and_batched_shapes():
    # The processor emits grid/mask sizes as batched("image") fields, so a
    # single-image request arrives as [[h, w]]; both layouts must read as (h, w).
    assert pixel_decoder._hw_pair(torch.tensor([4, 4], dtype=torch.int32)) == (4, 4)
    assert pixel_decoder._hw_pair(torch.tensor([[4, 4]], dtype=torch.int32)) == (4, 4)
    assert pixel_decoder._hw_pair(torch.tensor([[2, 2]], dtype=torch.int32)) == (2, 2)


def test_decode_request_rejects_mismatched_shapes():
    decoder = _bare_decoder()

    mismatched = _payload()
    mismatched["segmentation.boxes"] = torch.zeros((2, 4))
    with pytest.raises(ValueError, match=r"#OBJ=1 does not match #boxes=2"):
        decoder._decode_request(mismatched)

    bad_grid = _payload()
    bad_grid["segmentation.grid_hw"] = torch.tensor([[2, 2, 2]], dtype=torch.int32)
    with pytest.raises(ValueError, match=r"grid payload must be \(h, w\)"):
        decoder._decode_request(bad_grid)

    bad_size = _payload()
    bad_size["segmentation.mask_hw"] = torch.tensor([[0, 4]], dtype=torch.int32)
    with pytest.raises(ValueError, match=r"positive \(H, W\)"):
        decoder._decode_request(bad_size)

    bad_features = _payload()
    bad_features["segmentation.features"] = torch.zeros(3, 2, 2)
    with pytest.raises(ValueError, match=r"\[layers, C, H, W\]"):
        decoder._decode_request(bad_features)


def test_payload_tensors_reach_the_decoder_on_the_stage_device():
    """Canvas and feature maps are kept on CPU, so the forward call must move them.

    ``keep_on_cpu=True`` keeps the cross-stage payload serialisable, which leaves
    the decoder weights on the stage device and the tensors on CPU — that mismatch
    fails inside the CNN. A meta device stands in for the GPU here so the move is
    observable without a CUDA test.
    """
    decoder = _bare_decoder()
    stage_device = torch.device("meta")
    decoder.vllm_config = SimpleNamespace(device_config=SimpleNamespace(device=stage_device))
    decoder.seg_config = SimpleNamespace(cnn_image_size=4)

    seen: dict[str, object] = {}

    def fake_decoder(image, features, t_seg, boxes, grid):
        seen["devices"] = {
            "image": image.device,
            "features": {feature.device for feature in features},
            "t_seg": t_seg.device,
            "boxes": boxes.device,
        }
        seen["grid"] = grid
        return {"mask_logits": torch.zeros(1, 1, 2, 2)}

    decoder.decoder = fake_decoder
    masks = decoder._decode_request(_payload())

    assert seen["devices"] == {
        "image": stage_device,
        "features": {stage_device},
        "t_seg": stage_device,
        "boxes": stage_device,
    }
    assert seen["grid"] == (2, 2)
    # Masks are returned on CPU, which is what the client receives.
    assert masks.device.type == "cpu"


def test_text_only_requests_stop_before_the_sidecar_load(tmp_path):
    # Text-only requests reach stage 1 with an empty payload: that is handled by
    # returning a zero-numel tensor, not by loading the sidecar.
    decoder = _bare_decoder()
    assert pixel_decoder._segmentation_view({}) == {}

    no_sidecar = decoder._decode_request({"segmentation.t_seg": torch.empty(0)})
    assert no_sidecar is None
    # A local checkpoint without the sidecar fails only when masks are asked for.
    with pytest.raises(FileNotFoundError, match="Text-only tasks do not need it"):
        pixel_decoder._resolve_sidecar(str(tmp_path), None, None)
