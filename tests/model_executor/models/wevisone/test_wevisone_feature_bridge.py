# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""L1 feature-bridge contract for WeVisOne-VL (CPU, no weights).

Qwen flattens patches in ``[block_y, block_x, intra_y, intra_x]`` order so each
group of ``merge_size**2`` adjacent tokens feeds its merger. The decoder needs
ordinary row-major 2-D maps, so ``premerge_tokens_to_2d`` must undo that shuffle
exactly — a plain ``reshape`` scrambles every ``m x m`` neighbourhood.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.wevisone import feature_bridge

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _shuffle_to_tokens(plane: torch.Tensor, merge_size: int) -> torch.Tensor:
    """Flatten a row-major map the way Qwen's processor orders patches: block-major."""
    h, w, c = plane.shape
    m = merge_size
    blocks = plane.reshape(h // m, m, w // m, m, c).permute(0, 2, 1, 3, 4)
    return blocks.reshape(h * w, c)


def test_premerge_tokens_to_2d_undoes_the_patch_shuffle():
    plane = torch.arange(6 * 8 * 3, dtype=torch.float32).reshape(6, 8, 3)
    tokens = _shuffle_to_tokens(plane, merge_size=2)

    # Without the inverse permutation the round trip would not close.
    scrambled = tokens.reshape(6, 8, 3).permute(2, 0, 1).unsqueeze(0)
    restored = feature_bridge.premerge_tokens_to_2d(tokens, (6, 8), 2)
    assert torch.equal(restored, plane.permute(2, 0, 1).unsqueeze(0))
    assert not torch.equal(restored, scrambled)
    assert tuple(restored.shape) == (1, 3, 6, 8)


def test_premerge_tokens_to_2d_keeps_a_single_neighbourhood_together():
    """Each merge block must land in its own m x m window of the restored map."""
    tokens = torch.arange(4 * 4, dtype=torch.float32).reshape(16, 1)
    plane = feature_bridge.premerge_tokens_to_2d(tokens, (4, 4), 2)[0, 0]
    # Token order is block-major: block (0,0) covers values 0..3 at rows 0-1, cols 0-1,
    # then block (0,1) covers 4..7 at rows 0-1, cols 2-3.
    assert plane.tolist() == [[0.0, 1.0, 4.0, 5.0], [2.0, 3.0, 6.0, 7.0], [8.0, 9.0, 12.0, 13.0], [10.0, 11.0, 14.0, 15.0]]


def test_premerge_tokens_to_2d_rejects_shape_mismatch():
    tokens = torch.zeros(16, 3)
    with pytest.raises(ValueError, match="not divisible by merge size"):
        feature_bridge.premerge_tokens_to_2d(tokens, (5, 4), 2)
    # Grid divisible by the merge size but with the wrong number of tokens.
    with pytest.raises(ValueError, match="do not match grid"):
        feature_bridge.premerge_tokens_to_2d(tokens, (6, 4), 2)
    with pytest.raises(ValueError, match="do not match grid"):
        feature_bridge.premerge_tokens_to_2d(tokens.reshape(4, 4, 3), (4, 4), 2)


def test_split_concat_tokens_splits_per_image():
    tokens = torch.arange(10, dtype=torch.float32).reshape(10, 1)
    parts = feature_bridge.split_concat_tokens(tokens, [4, 6])
    assert len(parts) == 2
    assert torch.equal(parts[0], tokens[:4])
    assert torch.equal(parts[1], tokens[4:])
    assert feature_bridge.split_concat_tokens(None, [4]) == []

    with pytest.raises(ValueError, match="token count mismatch"):
        feature_bridge.split_concat_tokens(tokens, [3, 6])


def test_decoder_canvas_is_the_anisotropic_original_resize():
    canvas = torch.zeros(1, 3, 1024, 1024)
    assert torch.equal(feature_bridge.assert_cnn_rgb(canvas, 1024), canvas)
    assert tuple(feature_bridge.assert_cnn_rgb(canvas[0], 1024).shape) == (1, 3, 1024, 1024)

    # Smart-resize (and a non-RGB layout) must not reach the decoder.
    with pytest.raises(ValueError, match="cnn_image_size"):
        feature_bridge.assert_cnn_rgb(torch.zeros(1, 3, 672, 672), 1024, sample_idx=0)
    with pytest.raises(ValueError, match=r"\[1,3,H,W\]"):
        feature_bridge.assert_cnn_rgb(torch.zeros(1, 1, 1024, 1024), 1024, sample_idx=1)


def test_premerge_grid_rejects_video():
    assert feature_bridge.premerge_grid_hw(torch.tensor([[1, 16, 32], [1, 8, 8]])) == [(16, 32), (8, 8)]
    with pytest.raises(ValueError, match="video not supported"):
        feature_bridge.premerge_grid_hw(torch.tensor([[4, 8, 8]]))


def test_extract_t_seg_selects_obj_rows_in_order():
    hidden = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    ids = torch.tensor([[1, 152678, 2, 2, 152678], [152678, 1, 2, 2, 2]])
    rows = feature_bridge.extract_t_seg(hidden, ids, 152678)
    assert len(rows) == 2
    assert torch.equal(rows[0], hidden[0, [1, 4]])
    assert torch.equal(rows[1], hidden[1, [0]])


def _fake_visual():
    return SimpleNamespace(
        # deepstack_visual_indexes [5, 11, 17] plus the final block 23, whose
        # capture point is the final merger rather than a DeepStack merger.
        deepstack_visual_indexes=[5, 11, 17],
        deepstack_merger_list=[nn.Identity() for _ in range(3)],
        blocks=list(range(24)),
        merger=nn.Identity(),
    )


def test_capture_points_are_the_merger_inputs():
    visual = _fake_visual()
    capture = feature_bridge.VisualFeatureCapture(SimpleNamespace(visual=visual), (5, 11, 17, 23))
    assert tuple(capture.feature_layers) == (5, 11, 17, 23)
    assert capture._get_visual() is visual

    available = set(visual.deepstack_visual_indexes) | {len(visual.blocks) - 1}
    assert set(capture.feature_layers).issubset(available)

    # A layer with no merger input cannot be captured, and the error says so.
    capture.feature_layers = (5, 11, 17, 22)
    with pytest.raises(RuntimeError, match="cannot capture pre-merge ViT layers"):
        with capture:
            pass


def test_capture_records_the_state_before_each_merger():
    visual = _fake_visual()
    capture = feature_bridge.VisualFeatureCapture(SimpleNamespace(visual=visual), (5, 11, 17, 23))
    layer_11 = torch.full((4, 3), 7.0)
    block_23 = torch.full((2, 3), 9.0)

    with capture:
        visual.deepstack_merger_list[1](layer_11)
        visual.merger(block_23)

    # Hooks are removed on exit, so a later merger input is not captured.
    visual.deepstack_merger_list[0](torch.zeros(1, 3))
    assert sorted(capture.premerge_features) == [11, 23]
    assert torch.equal(capture.premerge_features[11], layer_11)
    assert torch.equal(capture.premerge_features[23], block_23)
