# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Capture Qwen3-VL features before every spatial merger.

Ported from the Tencent WeVisOne-VL reference implementation (Apache-2.0).

Qwen orders patches as ``[block_y, block_x, intra_y, intra_x]`` so each group of
``merge_size**2`` adjacent tokens can be consumed by its merger.  The helpers in
this file capture the inputs of the three DeepStack mergers and the final merger,
then undo that shuffle before exposing ordinary row-major 2-D feature maps.

The final LM layer is captured by ``wevisone_vl.py`` instead of a hook: vLLM
executes the decoder stack under ``torch.compile``, where module hooks do not
fire, so the capture must happen inside the compiled forward.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def premerge_grid_hw(image_grid_thw: torch.Tensor) -> list[tuple[int, int]]:
    """Per-image patch grid before Qwen's spatial merger."""
    grids = []
    for t, h, w in image_grid_thw.tolist():
        if int(t) != 1:
            raise ValueError(f"video not supported by segmentation decoder, got t={t}")
        grids.append((int(h), int(w)))
    return grids


def as_token_rows(tokens: torch.Tensor) -> torch.Tensor:
    """Return a merger input as ``[N, C]`` patch rows.

    vLLM's vision blocks hand the mergers the attention-layout tensor ``[N, 1, C]``,
    while the reference implementation sees ``[N, C]``. The patch order is the same,
    so only the singleton sequence dim is folded away.
    """
    if tokens.dim() == 3:
        return tokens.reshape(-1, int(tokens.shape[-1]))
    return tokens


def split_concat_tokens(tokens: torch.Tensor, counts: Sequence[int]) -> list[torch.Tensor]:
    """Split a concatenated [sum N_i, C] tensor into per-image [N_i, C]."""
    if tokens is None:
        return []
    if sum(counts) != tokens.shape[0]:
        raise ValueError(f"token count mismatch: sum(counts)={sum(counts)} vs tokens={tokens.shape[0]}")
    return list(torch.split(tokens, list(counts), dim=0))


def assert_cnn_rgb(
    image: torch.Tensor,
    cnn_image_size: int,
    sample_idx: int = 0,
) -> torch.Tensor:
    """Require decoder CNN RGB ``[1,3,S,S]`` / ``[3,S,S]`` (anisotropic orig resize).

    This canvas is **not** the Qwen smart-resize used by the vision encoder.
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.dim() != 4 or image.shape[1] != 3:
        raise ValueError(f"sample {sample_idx}: decoder image must be [1,3,H,W] or [3,H,W], got {tuple(image.shape)}")
    got = (int(image.shape[-2]), int(image.shape[-1]))
    expect = (int(cnn_image_size), int(cnn_image_size))
    if got != expect:
        raise ValueError(
            f"sample {sample_idx}: decoder CNN image HW {got} != {expect}. "
            "Resize the original photo to cnn_image_size x cnn_image_size "
            "(not Qwen smart-resize)."
        )
    return image


def premerge_tokens_to_2d(
    feature_tokens: torch.Tensor,
    grid_hw: tuple[int, int],
    spatial_merge_size: int,
) -> torch.Tensor:
    """Undo Qwen's merge-local patch shuffle and return one ``[C,H,W]`` map.

    The processor flattens patches in ``[H/m,W/m,m,m]`` order.  A direct
    ``reshape(H,W)`` silently scrambles every ``m x m`` neighbourhood.

    No batch dim is added here: a captured map belongs to one image, and the
    cross-stage payload stacks the layers of that image into ``[layers,C,H,W]``.
    The decoder re-adds its batch dim when it calls the reference-style decoder.
    """
    h, w = (int(grid_hw[0]), int(grid_hw[1]))
    m = int(spatial_merge_size)
    if h % m or w % m:
        raise ValueError(f"pre-merge grid {(h, w)} is not divisible by merge size {m}")
    if feature_tokens.dim() != 2 or int(feature_tokens.shape[0]) != h * w:
        raise ValueError(f"pre-merge tokens {tuple(feature_tokens.shape)} do not match grid {h}x{w}")
    c = int(feature_tokens.shape[-1])
    x = feature_tokens.reshape(h // m, w // m, m, m, c)
    x = x.permute(0, 2, 1, 3, 4).contiguous().reshape(h, w, c)
    return x.permute(2, 0, 1).contiguous()


def extract_t_seg(
    last_hidden_state: torch.Tensor,
    input_ids: torch.Tensor,
    seg_token_id: int,
) -> list[torch.Tensor]:
    """Per-sample LM hidden states at ``<OBJ>`` positions."""
    result = []
    for b in range(input_ids.shape[0]):
        pos = (input_ids[b] == seg_token_id).nonzero(as_tuple=False).flatten()
        result.append(last_hidden_state[b, pos])
    return result


class VisualFeatureCapture:
    """Capture layers 5/11/17/23 exactly at their merger inputs."""

    def __init__(self, qwen, feature_layers: Sequence[int] = (5, 11, 17, 23)):
        self.qwen = qwen
        self.feature_layers = tuple(int(x) for x in feature_layers)
        self.premerge_features = {}
        self._handles = []

    def _capture_premerge(self, layer_idx: int):
        def hook(_module, inputs):
            if not inputs:
                raise RuntimeError(f"empty merger input for ViT layer {layer_idx}")
            self.premerge_features[int(layer_idx)] = as_token_rows(inputs[0])

        return hook

    def __enter__(self):
        visual = self._get_visual()
        self.premerge_features.clear()
        deep_indexes = tuple(int(x) for x in visual.deepstack_visual_indexes)
        mergers = visual.deepstack_merger_list
        final_idx = len(visual.blocks) - 1
        available = set(deep_indexes) | {final_idx}
        missing = [x for x in self.feature_layers if x not in available]
        if missing:
            raise RuntimeError(
                f"cannot capture pre-merge ViT layers {missing}; merger inputs exist at {sorted(available)}"
            )
        for idx, merger in zip(deep_indexes, mergers):
            if idx in self.feature_layers:
                self._handles.append(merger.register_forward_pre_hook(self._capture_premerge(idx)))
        if final_idx in self.feature_layers:
            self._handles.append(visual.merger.register_forward_pre_hook(self._capture_premerge(final_idx)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        return False

    def _get_visual(self):
        root = self.qwen
        base = getattr(root, "model", root)
        visual = getattr(base, "visual", None) or getattr(root, "visual", None)
        if visual is None:
            raise RuntimeError("cannot locate qwen visual module for feature capture")
        return visual


def per_sample_premerge_feature_maps(
    premerge_features: dict[int, torch.Tensor],
    feature_layers: Sequence[int],
    image_grid_thw: torch.Tensor,
    spatial_merge_size: int,
) -> list[torch.Tensor]:
    """Split concatenated raw ViT tokens into one ``[layers,C,H,W]`` tensor per image.

    Cross-stage payloads carry a single tensor per request, so the captured
    layers are stacked here; the decoder reads them back as that shape.
    """
    grids = premerge_grid_hw(image_grid_thw)
    counts = [h * w for h, w in grids]
    ordered = []
    for idx in feature_layers:
        feat = premerge_features.get(int(idx))
        if feat is None:
            raise RuntimeError(f"ViT layer {idx} was not captured before its merger")
        ordered.append(split_concat_tokens(feat, counts))

    return [
        torch.stack([premerge_tokens_to_2d(level[sample_idx], grid, spatial_merge_size) for level in ordered], dim=0)
        for sample_idx, grid in enumerate(grids)
    ]
