# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Sequence-layout constants for the Ideogram 4 diffusion stage.

Values are ported from the reference implementation of ``ideogram-ai/ideogram-4``
and must stay byte-identical to it; the reference package is never imported.
"""

from __future__ import annotations

# ``segment_ids`` marker for tokens that are padding rather than real content.
# Padding rows only attend to padding rows, and their outputs are discarded by
# the pipeline, so this value only matters for the block-diagonal mask.
SEQUENCE_PADDING_INDICATOR = -1

# ``indicator`` markers used by the transformer to tell the two halves of the
# packed sequence apart. ``0`` is padding, and ``1`` is unused (reserved).
OUTPUT_IMAGE_INDICATOR = 2
LLM_TOKEN_INDICATOR = 3

# RoPE position offset applied to image tokens so text and image positions
# never overlap in the rotary embedding.
IMAGE_POSITION_OFFSET = 65536

# Text-encoder layers whose hidden states are concatenated into the conditioning
# tensor consumed by the DiT (``llm_features_dim = hidden_size * len(...)``).
QWEN3_VL_ACTIVATION_LAYERS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)

# Latent patch factor and VAE spatial downsample factor. A latent cell covers
# ``PATCH_SIZE * PATCH_SIZE`` channels-worth of the packed latent, and the image
# grid step is their product times the VAE scale factor.
PATCH_SIZE = 2
AE_SCALE_FACTOR = 8

# Longest prompt the tokenizer output may have before the pipeline rejects it.
MAX_TEXT_TOKENS = 2048

__all__ = [
    "AE_SCALE_FACTOR",
    "IMAGE_POSITION_OFFSET",
    "LLM_TOKEN_INDICATOR",
    "MAX_TEXT_TOKENS",
    "OUTPUT_IMAGE_INDICATOR",
    "PATCH_SIZE",
    "QWEN3_VL_ACTIVATION_LAYERS",
    "SEQUENCE_PADDING_INDICATOR",
]
