# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Architecture description of the Ideogram 4 DiT.

The released checkpoint ships a plain ``transformer/config.json`` whose keys are
read here as data, so no remote-code model definition is imported. ``emb_dim``
is not stored in the checkpoint and is derived from ``num_heads * head_dim``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

__all__ = ["Ideogram4Config", "Ideogram4TextConfig"]


@dataclass(frozen=True)
class Ideogram4Config:
    emb_dim: int = 4608
    num_layers: int = 34
    num_heads: int = 18
    head_dim: int = 256
    intermediate_size: int = 12288
    # The reference source calls this ``adanln_dim``; the checkpoint key is
    # ``adaln_dim``. Keep the checkpoint spelling and derive the reference
    # value from it.
    adaln_dim: int = 512
    in_channels: int = 128
    llm_features_dim: int = 53248
    norm_eps: float = 1e-5
    rope_theta: float = 5_000_000.0
    mrope_section: tuple[int, ...] = (24, 20, 20)

    @classmethod
    def from_hf_config(cls, config: Mapping[str, object]) -> "Ideogram4Config":
        """Build the config from the ``transformer/config.json`` mapping."""
        num_heads = int(config["num_attention_heads"])
        head_dim = int(config["attention_head_dim"])
        return cls(
            emb_dim=num_heads * head_dim,
            num_layers=int(config["num_layers"]),
            num_heads=num_heads,
            head_dim=head_dim,
            intermediate_size=int(config["intermediate_size"]),
            adaln_dim=int(config["adaln_dim"]),
            in_channels=int(config["in_channels"]),
            llm_features_dim=int(config["llm_features_dim"]),
            norm_eps=float(config.get("norm_eps", 1e-5)),
            rope_theta=float(config.get("rope_theta", 5_000_000.0)),
            mrope_section=tuple(int(v) for v in config.get("mrope_section", (24, 20, 20))),
        )


@dataclass(frozen=True)
class Ideogram4TextConfig:
    """Text-side Qwen3VL architecture as stored in ``text_encoder/config.json``.

    The vision tower is not ported: Ideogram 4 conditions the DiT on language
    model activations only, so the checkpoint's ``visual.*`` tensors are skipped
    by the weight source.
    """

    vocab_size: int = 151936
    hidden_size: int = 4096
    num_layers: int = 36
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 12288
    norm_eps: float = 1e-6
    rope_theta: float = 5_000_000.0
    mrope_section: tuple[int, ...] = (24, 20, 20)

    @classmethod
    def from_hf_config(cls, config: Mapping[str, object]) -> "Ideogram4TextConfig":
        """Build the config from the ``text_encoder/config.json`` text sub-mapping."""
        rope = config.get("rope_parameters") or config.get("rope_scaling") or {}
        rope = rope if isinstance(rope, Mapping) else {}
        return cls(
            vocab_size=int(config["vocab_size"]),
            hidden_size=int(config["hidden_size"]),
            num_layers=int(config["num_hidden_layers"]),
            num_heads=int(config["num_attention_heads"]),
            num_kv_heads=int(config.get("num_key_value_heads", config["num_attention_heads"])),
            head_dim=int(config.get("head_dim", int(config["hidden_size"]) // int(config["num_attention_heads"]))),
            intermediate_size=int(config["intermediate_size"]),
            norm_eps=float(config.get("rms_norm_eps", 1e-6)),
            rope_theta=float(rope.get("rope_theta", config.get("rope_theta", 5_000_000.0))),
            mrope_section=tuple(int(v) for v in rope.get("mrope_section", (24, 20, 20))),
        )
