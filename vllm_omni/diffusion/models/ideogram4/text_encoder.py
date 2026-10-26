# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Native Qwen3VL language model used as the Ideogram 4 text encoder.

The checkpoint stores ``language_model.*`` tensors with HF Qwen3VL naming
(``q_proj``/``k_proj``/``v_proj``/``o_proj``, per-head ``q_norm``/``k_norm``,
SwiGLU ``gate_proj``/``up_proj``/``down_proj``), so the module tree below keeps
those names and the loader can stream safetensors straight into it. The vision
tower is not ported: Ideogram 4 conditions on text only, and the weight source
skips ``visual.*``.

Causality is encoded inside the attention mask rather than through
``Attention(causal=True)``. The attention layer cannot combine causal masking
with an explicit mask, so the text encoder builds one combined causal-and-padding
mask and always calls attention with ``causal=False``. Sequence parallelism is
skipped here: the encoder runs once per request before the DiT, and the SP plan
only covers the transformer.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.layers.norm import RMSNorm
from vllm_omni.diffusion.models.ideogram4.config import Ideogram4TextConfig
from vllm_omni.diffusion.models.ideogram4.constants import QWEN3_VL_ACTIVATION_LAYERS
from vllm_omni.diffusion.models.ideogram4.fp8_linear import Fp8Linear, load_streamed_weights
from vllm_omni.diffusion.models.ideogram4.rope import Ideogram4MRoPE, apply_rotary_pos_emb

logger = init_logger(__name__)

__all__ = [
    "Ideogram4Qwen3VLAttention",
    "Ideogram4Qwen3VLMLP",
    "Ideogram4Qwen3VLTextEncoder",
    "Ideogram4Qwen3VLTextModel",
]


def build_text_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Combine causality and padding into one ``(B, 1, L, L)`` bool mask.

    ``attention_mask`` is the ``(B, L)`` padding mask of attendable positions,
    which for Ideogram 4 means the text tokens: image tokens sit after the text
    and are never visible to the encoder. Rows for padding positions would be all
    ``False`` and turn softmax into NaN, so such rows are allowed to attend to
    themselves; their garbage outputs are zeroed by the caller. Keeping this in
    one place means no backend ever sees ``causal=True`` together with a mask.
    """
    if attention_mask.dim() != 2:
        raise ValueError(f"attention_mask must be (B, L), got {tuple(attention_mask.shape)}")
    batch_size, seq_len = attention_mask.shape
    device = attention_mask.device
    allowed_key = attention_mask.to(torch.bool)
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    mask = causal & allowed_key[:, None, :]
    padding_query = ~allowed_key
    eye = torch.eye(seq_len, dtype=torch.bool, device=device)
    mask = mask | (padding_query[:, None, :] & eye[None, :, None])
    return mask.unsqueeze(1)


class Ideogram4Qwen3VLAttention(nn.Module):
    """Grouped-query self attention with per-head RMSNorm and interleaved RoPE."""

    def __init__(self, config: Ideogram4TextConfig, compute_dtype: torch.dtype, prefix: str = "") -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5

        self.q_proj = Fp8Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False, compute_dtype=compute_dtype)
        self.k_proj = Fp8Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False, compute_dtype=compute_dtype)
        self.v_proj = Fp8Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False, compute_dtype=compute_dtype)
        self.o_proj = Fp8Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False, compute_dtype=compute_dtype)
        self.q_norm = RMSNorm(self.head_dim, eps=config.norm_eps, dtype=compute_dtype)
        self.k_norm = RMSNorm(self.head_dim, eps=config.norm_eps, dtype=compute_dtype)

        # ``causal=False`` is deliberate: the caller supplies a mask that already
        # encodes causality, and the two cannot be combined.
        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            softmax_scale=self.scaling,
            causal=False,
            num_kv_heads=self.num_kv_heads,
            prefix=prefix,
            skip_sequence_parallel=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        cos, sin = position_embeddings

        query = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        value = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        query = self.q_norm(query)
        key = self.k_norm(key)
        query = apply_rotary_pos_emb(query, cos, sin)
        key = apply_rotary_pos_emb(key, cos, sin)

        attn_output = self.attn(query, key, value, attn_metadata)
        return self.o_proj(attn_output.flatten(2, 3).type_as(hidden_states))


class Ideogram4Qwen3VLMLP(nn.Module):
    """SwiGLU MLP matching the checkpoint's ``gate_proj``/``up_proj``/``down_proj``."""

    def __init__(self, config: Ideogram4TextConfig, compute_dtype: torch.dtype) -> None:
        super().__init__()
        self.gate_proj = Fp8Linear(config.hidden_size, config.intermediate_size, bias=False, compute_dtype=compute_dtype)
        self.up_proj = Fp8Linear(config.hidden_size, config.intermediate_size, bias=False, compute_dtype=compute_dtype)
        self.down_proj = Fp8Linear(config.intermediate_size, config.hidden_size, bias=False, compute_dtype=compute_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Ideogram4Qwen3VLDecoderLayer(nn.Module):
    """Pre-norm decoder block; activations are captured *after* each block."""

    def __init__(
        self,
        config: Ideogram4TextConfig,
        compute_dtype: torch.dtype,
        layer_index: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.norm_eps, dtype=compute_dtype)
        self.self_attn = Ideogram4Qwen3VLAttention(config, compute_dtype, prefix=f"{prefix}{layer_index}.")
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.norm_eps, dtype=compute_dtype)
        self.mlp = Ideogram4Qwen3VLMLP(config, compute_dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states), attn_metadata, position_embeddings
        )
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


class Ideogram4Qwen3VLTextModel(nn.Module):
    """Language tower that emits the per-layer activations Ideogram 4 conditions on."""

    def __init__(
        self,
        config: Ideogram4TextConfig,
        compute_dtype: torch.dtype,
        activation_layers: tuple[int, ...] = QWEN3_VL_ACTIVATION_LAYERS,
        prefix: str = "language_model",
    ) -> None:
        super().__init__()
        self.config = config
        self.compute_dtype = compute_dtype
        self.activation_layers = tuple(activation_layers)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=compute_dtype)
        self.layers = nn.ModuleList(
            [
                Ideogram4Qwen3VLDecoderLayer(config, compute_dtype, i, prefix=f"{prefix}.layers.")
                for i in range(config.num_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps, dtype=compute_dtype)
        # Text-only inputs carry the same position on all three axes, so the
        # interleaved multi-axis RoPE reduces to plain RoPE here.
        self.rotary_emb = Ideogram4MRoPE(config.head_dim, config.rope_theta, config.mrope_section)

    def forward(
        self,
        token_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Return captured activations in ``self.activation_layers`` order.

        The final norm is deliberately not applied: the reference conditions on
        raw per-layer activations, and the caller zeroes padding positions.
        """
        hidden_states = self.embed_tokens(token_ids)
        cos, sin = self.rotary_emb(position_ids)
        cos = cos.to(hidden_states.dtype)
        sin = sin.to(hidden_states.dtype)

        mask = build_text_attention_mask(attention_mask)
        attn_metadata = AttentionMetadata(
            attn_mask=mask,
            extra={"attention_mask_mode": "arbitrary"},
        )

        tap_set = set(self.activation_layers)
        captures: list[torch.Tensor] = []
        for layer_index, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, attn_metadata, (cos, sin))
            if layer_index in tap_set:
                captures.append(hidden_states)
        return captures

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return load_streamed_weights(self, weights, self.compute_dtype)


class Ideogram4Qwen3VLTextEncoder(nn.Module):
    """Container that keeps the checkpoint's ``language_model`` prefix resolvable."""

    def __init__(
        self,
        config: Ideogram4TextConfig,
        compute_dtype: torch.dtype,
        activation_layers: tuple[int, ...] = QWEN3_VL_ACTIVATION_LAYERS,
    ) -> None:
        super().__init__()
        self.config = config
        self.compute_dtype = compute_dtype
        self.language_model = Ideogram4Qwen3VLTextModel(
            config, compute_dtype, activation_layers, prefix="language_model"
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> list[torch.Tensor]:
        return self.language_model(token_ids, position_ids, attention_mask)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return self.language_model.load_weights(weights)
