# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Communication-free shard-layout helpers for the Ideogram 4 transformer.

Under Ulysses sequence parallelism each rank holds a contiguous slice of the
padded sequence, but after the all-to-all the attention op sees the
*rank-concatenated* global sequence. Ideogram 4 attention is block-diagonal
(text tokens attend to text tokens, image tokens to image tokens), so the mask
has to be built for the reordered global sequence. Every rank can derive that
layout from the shard group alone, without any communication.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from vllm_omni.diffusion.forward_context import get_sp_shard_original_seq_len
from vllm_omni.diffusion.models.ideogram4.constants import SEQUENCE_PADDING_INDICATOR


@dataclass(frozen=True, slots=True)
class ShardLayout:
    """How one globally padded sequence maps onto the SP ranks."""

    original_seq_len: int
    world_size: int
    rank: int

    @classmethod
    def resolve(
        cls,
        shard_group: str,
        *,
        local_seq_len: int,
        original_seq_len: int | None = None,
    ) -> "ShardLayout":
        """Read a boundary's layout from the ForwardContext.

        Falls back to an unsharded layout (world_size=1) when SP is off or the
        tensor was never split, in which case ``local_seq_len`` is the whole
        sequence.

        ``original_seq_len`` overrides the recorded length. Callers that know
        their own pre-split length must pass it: the negative CFG branch is
        image-only, so the length recorded by the positive branch's split does
        not describe it.
        """
        from vllm_omni.diffusion.distributed.parallel_state import (
            get_sequence_parallel_rank,
            get_sequence_parallel_world_size,
        )

        if original_seq_len is None:
            original_seq_len = get_sp_shard_original_seq_len(shard_group)
        if original_seq_len is None:
            return cls(original_seq_len=local_seq_len, world_size=1, rank=0)
        return cls(
            original_seq_len=original_seq_len,
            world_size=get_sequence_parallel_world_size(),
            rank=get_sequence_parallel_rank(),
        )

    @property
    def padded_seq_len(self) -> int:
        return math.ceil(self.original_seq_len / self.world_size) * self.world_size

    @property
    def local_seq_len(self) -> int:
        return self.padded_seq_len // self.world_size

    @property
    def padding_size(self) -> int:
        return self.padded_seq_len - self.original_seq_len

    def bounds(self, rank: int) -> tuple[int, int]:
        """Global position range held by ``rank`` in the padded sequence."""
        start = rank * self.local_seq_len
        return start, start + self.local_seq_len


def _pad_segment_ids(segment_ids: torch.Tensor, padded_seq_len: int) -> torch.Tensor:
    if segment_ids.shape[1] == padded_seq_len:
        return segment_ids
    pad = segment_ids.new_full(
        (segment_ids.shape[0], padded_seq_len - segment_ids.shape[1]),
        SEQUENCE_PADDING_INDICATOR,
    )
    return torch.cat((segment_ids, pad), dim=1)


def rank_concat_segment_mask_or_none(
    segment_ids: torch.Tensor,
    shard_group: str,
    *,
    like: torch.Tensor,
    required: bool = False,
    original_seq_len: int | None = None,
) -> torch.Tensor | None:
    """Block-diagonal mask over the sequence as attention sees it.

    ``segment_ids`` is the unsplit ``(B, L)`` tensor produced before the shard
    boundary. The returned mask is ``(B, 1, world_size * local_seq_len,
    world_size * local_seq_len)`` in rank-concatenated order, matching the
    tensor the attention op receives after the all-to-all. Returns ``None`` for
    an all-valid mask (single sample, no padding) unless the caller insists on
    one, so the unmasked fast path stays available.

    ``original_seq_len`` defaults to ``segment_ids.shape[1]``, the length the
    shard boundary saw, and is only needed when ``segment_ids`` was already
    sliced down to that length.
    """
    layout = ShardLayout.resolve(
        shard_group,
        local_seq_len=like.shape[1],
        original_seq_len=segment_ids.shape[1] if original_seq_len is None else original_seq_len,
    )
    capacity = layout.local_seq_len
    if segment_ids.shape[1] > layout.padded_seq_len:
        raise ValueError(
            f"segment_ids length {segment_ids.shape[1]} exceeds the padded shard length "
            f"{layout.padded_seq_len} for shard group {shard_group!r}"
        )

    padded = _pad_segment_ids(segment_ids, layout.padded_seq_len)
    if layout.world_size == 1:
        ordered = padded
    else:
        ordered = torch.cat(
            [padded[:, start:end] for start, end in (layout.bounds(r) for r in range(layout.world_size))],
            dim=1,
        )

    if (ordered >= 0).all():
        return None if not required else ordered.new_ones(
            (ordered.shape[0], 1, ordered.shape[1], ordered.shape[1]), dtype=torch.bool
        )

    mask = (ordered.unsqueeze(2) == ordered.unsqueeze(1)).unsqueeze(1)
    return mask.to(device=like.device)
