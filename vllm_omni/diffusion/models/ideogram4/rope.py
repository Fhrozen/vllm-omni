# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""RoPE primitives shared by the Ideogram 4 transformer and text encoder.

Both sides use the same interleaved multi-axis RoPE (``mrope_section``
``(24, 20, 20)``, ``rope_theta`` 5e6). Text-only inputs carry the same position
on all three axes, so the interleaved result reduces to plain RoPE; sharing one
implementation keeps the frequency math in a single place.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "Ideogram4MRoPE",
    "apply_rotary_pos_emb",
    "rotate_half",
    "sinusoidal_embedding",
]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to a ``[B, L, heads, head_dim]`` tensor with ``[B, L, head_dim]`` cos/sin."""
    return x * cos.unsqueeze(2) + rotate_half(x) * sin.unsqueeze(2)


def sinusoidal_embedding(t: torch.Tensor, dim: int, scale: float = 1e4) -> torch.Tensor:
    """Sinusoidal time embedding, matching the reference's ``log(scale)/(half-1)`` frequencies."""
    t = t.to(torch.float32)
    half = dim // 2
    freq = math.log(scale) / (half - 1)
    freq = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device) * -freq)
    emb = t.unsqueeze(-1) * freq
    emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class Ideogram4MRoPE(nn.Module):
    """Interleaved multi-axis RoPE over ``(t, h, w)`` position ids."""

    def __init__(self, head_dim: int, base: float, mrope_section: tuple[int, ...]) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.head_dim = head_dim
        self.mrope_section = tuple(mrope_section)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.dim() != 3 or position_ids.shape[-1] != 3:
            raise ValueError(f"position_ids must be (B, L, 3), got {tuple(position_ids.shape)}")
        pos = position_ids.permute(2, 0, 1).float()
        inv_freq = self.inv_freq[None, None, :, None].float().expand(3, pos.shape[1], -1, 1)
        freqs = (inv_freq @ pos.unsqueeze(2)).transpose(2, 3)  # (3, B, L, head_dim // 2)

        interleaved = freqs[0].clone()
        for axis, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[axis] * 3
            idx = torch.arange(offset, length, 3, device=freqs.device)
            interleaved[..., idx] = freqs[axis][..., idx]
        emb = torch.cat((interleaved, interleaved), dim=-1)
        return emb.cos(), emb.sin()
