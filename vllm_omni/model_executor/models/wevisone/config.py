# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Final decoder configuration for the WeVisOne-VL v1 checkpoint.

Ported from the Tencent WeVisOne-VL reference implementation (Apache-2.0).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

ARCHITECTURE_VERSION = "wevisone_vl_v1"


@dataclass
class Qwen3VLSegConfig:
    """Only fields that instantiate the released 9.6M-parameter decoder."""

    architecture_version: str = ARCHITECTURE_VERSION
    vision_feature_layers: tuple[int, ...] = (5, 11, 17, 23)

    hidden_dim: int = 192
    mask_feature_dim: int = 64
    num_fourier_freqs: int = 32

    msda_encoder_layers: int = 3
    msda_num_levels: int = 4
    msda_num_points: int = 4

    transformer_layers: int = 6
    transformer_heads: int = 8
    transformer_ffn_dim: int = 768
    transformer_dropout: float = 0.0

    group_norm_groups: int = 32
    stem_mid_dim: int = 128
    alpha_cascade_dim: int = 48
    alpha_cascade_blocks: int = 2
    cnn_image_size: int = 1024

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> Qwen3VLSegConfig:
        if not values:
            return cls()
        known = cls.__dataclass_fields__  # type: ignore[attr-defined]
        cleaned = {key: value for key, value in values.items() if key in known}
        # The sidecar stores the capture layers as a JSON list while the default
        # is a tuple; without this the shipped cfg reads as drift and rebuilds
        # an identical module at startup.
        layers = cleaned.get("vision_feature_layers")
        if isinstance(layers, list):
            cleaned["vision_feature_layers"] = tuple(layers)
        return cls(**cleaned)


SEG_SPECIAL_TOKENS = ("<OBJ>",)
