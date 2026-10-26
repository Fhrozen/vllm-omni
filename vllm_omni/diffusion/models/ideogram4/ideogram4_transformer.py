# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Native vLLM-Omni port of the Ideogram 4 flow-matching transformer.

The released checkpoint is a plain Diffusers-style folder, but the pipeline and
model classes are not real Diffusers classes, so this port is self-contained and
never imports the reference package. Semantics follow the reference exactly:
per-row FP8 weights dequantized inside the forward pass, interleaved MRoPE,
block-diagonal segment masking, SwiGLU MLPs, and AdaLN modulation with
``tanh`` gates and ``1 + scale`` shifts.

Sequence parallelism is declared with ``_sp_plan``: the ``sequence_prepare``
boundary splits the packed sequence, and ``final_layer`` gathers it back.
``segment_ids`` and the AdaLN conditioning stay replicated, so the block-diagonal
mask is rebuilt for the rank-concatenated sequence that Ulysses produces after
its all-to-all (see ``sp_layout``).
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.sp_plan import (
    SequenceParallelInput,
    SequenceParallelOutput,
)
from vllm_omni.diffusion.layers.norm import LayerNorm, RMSNorm
from vllm_omni.diffusion.models.ideogram4.config import Ideogram4Config
from vllm_omni.diffusion.models.ideogram4.constants import (
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
)
from vllm_omni.diffusion.models.ideogram4.fp8_linear import Fp8Linear, load_streamed_weights
from vllm_omni.diffusion.models.ideogram4.rope import (
    Ideogram4MRoPE,
    apply_rotary_pos_emb,
    sinusoidal_embedding,
)
from vllm_omni.diffusion.models.ideogram4.sp_layout import rank_concat_segment_mask_or_none

logger = init_logger(__name__)

# Name used by ``_sp_plan`` and by the mask helper; both must agree.
SHARD_GROUP = "sequence"

__all__ = [
    "Ideogram4Attention",
    "Ideogram4EmbedScalar",
    "Ideogram4FinalLayer",
    "Ideogram4MLP",
    "Ideogram4SequencePrepare",
    "Ideogram4Transformer2DModel",
    "Ideogram4TransformerBlock",
]


class Ideogram4Attention(nn.Module):
    """Block-diagonal segment-masked attention through the vLLM-Omni attention layer."""

    def __init__(
        self,
        config: Ideogram4Config,
        compute_dtype: torch.dtype,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.compute_dtype = compute_dtype
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.emb_dim

        self.qkv = Fp8Linear(self.hidden_size, 3 * self.hidden_size, bias=False, compute_dtype=compute_dtype)
        self.norm_q = RMSNorm(self.head_dim, eps=config.norm_eps)
        self.norm_k = RMSNorm(self.head_dim, eps=config.norm_eps)
        self.o = Fp8Linear(self.hidden_size, self.hidden_size, bias=False, compute_dtype=compute_dtype)

        # The shared Attention layer owns backend selection and parallel dispatch;
        # FP8 here is weight-only through Fp8Linear, so no quant_config is passed.
        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            softmax_scale=self.head_dim**-0.5,
            causal=False,
            prefix=prefix,
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim).unbind(2)
        query = self.norm_q(qkv[0])
        key = self.norm_k(qkv[1])
        value = qkv[2]

        query = apply_rotary_pos_emb(query, cos, sin)
        key = apply_rotary_pos_emb(key, cos, sin)

        attn_output = self.attn(query, key, value, attn_metadata)
        return self.o(attn_output.flatten(2, 3).type_as(x))


class Ideogram4MLP(nn.Module):
    """SwiGLU MLP matching the checkpoint's ``w1``/``w2``/``w3`` naming."""

    def __init__(self, config: Ideogram4Config, compute_dtype: torch.dtype) -> None:
        super().__init__()
        self.w1 = Fp8Linear(config.emb_dim, config.intermediate_size, bias=False, compute_dtype=compute_dtype)
        self.w2 = Fp8Linear(config.intermediate_size, config.emb_dim, bias=False, compute_dtype=compute_dtype)
        self.w3 = Fp8Linear(config.emb_dim, config.intermediate_size, bias=False, compute_dtype=compute_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Ideogram4TransformerBlock(nn.Module):
    """Pre-norm transformer block with AdaLN modulation."""

    def __init__(
        self,
        config: Ideogram4Config,
        compute_dtype: torch.dtype,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.attention = Ideogram4Attention(config, compute_dtype, f"{prefix}attention.")
        self.feed_forward = Ideogram4MLP(config, compute_dtype)

        self.attention_norm1 = RMSNorm(config.emb_dim, eps=config.norm_eps)
        self.ffn_norm1 = RMSNorm(config.emb_dim, eps=config.norm_eps)
        self.attention_norm2 = RMSNorm(config.emb_dim, eps=config.norm_eps)
        self.ffn_norm2 = RMSNorm(config.emb_dim, eps=config.norm_eps)

        self.adaln_modulation = Fp8Linear(
            config.adaln_dim, 4 * config.emb_dim, bias=True, compute_dtype=compute_dtype
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        adaln_input: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        mod = self.adaln_modulation(adaln_input)
        scale_msa, gate_msa, scale_mlp, gate_mlp = mod.chunk(4, dim=-1)
        gate_msa = torch.tanh(gate_msa)
        gate_mlp = torch.tanh(gate_mlp)
        scale_msa = 1.0 + scale_msa
        scale_mlp = 1.0 + scale_mlp

        attn_out = self.attention(self.attention_norm1(x) * scale_msa, cos, sin, attn_metadata)
        x = x + gate_msa * self.attention_norm2(attn_out)
        x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
        return x


class Ideogram4EmbedScalar(nn.Module):
    """Sinusoidal scalar embedding followed by a small MLP."""

    def __init__(self, dim: int, compute_dtype: torch.dtype, input_range: tuple[float, float] = (0.0, 1.0)) -> None:
        super().__init__()
        self.dim = dim
        self.compute_dtype = compute_dtype
        self.range_min, self.range_max = input_range
        if self.range_max <= self.range_min:
            raise ValueError("input_range max must exceed min")
        self.mlp_in = Fp8Linear(dim, dim, bias=True, compute_dtype=compute_dtype)
        self.mlp_out = Fp8Linear(dim, dim, bias=True, compute_dtype=compute_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        scaled = 1e4 * (x - self.range_min) / (self.range_max - self.range_min)
        emb = sinusoidal_embedding(scaled, self.dim).to(self.compute_dtype)
        emb = F.silu(self.mlp_in(emb))
        return self.mlp_out(emb)


class Ideogram4FinalLayer(nn.Module):
    """Final AdaLN-modulated projection back to latent channels."""

    def __init__(self, config: Ideogram4Config, compute_dtype: torch.dtype) -> None:
        super().__init__()
        self.norm_final = LayerNorm(config.emb_dim, eps=1e-6, elementwise_affine=False)
        self.linear = Fp8Linear(config.emb_dim, config.in_channels, bias=True, compute_dtype=compute_dtype)
        self.adaln_modulation = Fp8Linear(config.adaln_dim, config.emb_dim, bias=True, compute_dtype=compute_dtype)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        scale = 1.0 + self.adaln_modulation(F.silu(c))
        return self.linear(self.norm_final(x) * scale)


class Ideogram4SequencePrepare(nn.Module):
    """Boundary where the packed sequence is sharded.

    Outputs 0 and 1 (hidden states and position ids) are split by ``_sp_plan``;
    outputs 2 and 3 (segment ids and AdaLN conditioning) stay replicated, the
    first because the block-diagonal mask needs the whole sequence, the second
    because it is per-sample rather than per-token.
    """

    def __init__(self, config: Ideogram4Config, compute_dtype: torch.dtype) -> None:
        super().__init__()
        self.config = config
        self.compute_dtype = compute_dtype
        self.input_proj = Fp8Linear(config.in_channels, config.emb_dim, bias=True, compute_dtype=compute_dtype)
        self.llm_cond_norm = RMSNorm(config.llm_features_dim, eps=1e-6)
        self.llm_cond_proj = Fp8Linear(
            config.llm_features_dim, config.emb_dim, bias=True, compute_dtype=compute_dtype
        )
        self.t_embedding = Ideogram4EmbedScalar(config.emb_dim, compute_dtype)
        self.adaln_proj = Fp8Linear(config.emb_dim, config.adaln_dim, bias=True, compute_dtype=compute_dtype)
        self.embed_image_indicator = nn.Embedding(2, config.emb_dim)

    def forward(
        self,
        llm_features: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        position_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        indicator: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        llm_features = llm_features.to(self.compute_dtype)
        x = x.to(self.compute_dtype)
        t = t.to(self.compute_dtype)

        indicator = indicator.to(torch.long)
        llm_token_mask = (indicator == LLM_TOKEN_INDICATOR).to(x.dtype).unsqueeze(-1)
        output_image_mask = (indicator == OUTPUT_IMAGE_INDICATOR).to(x.dtype).unsqueeze(-1)

        llm_features = llm_features * llm_token_mask
        x = self.input_proj(x * output_image_mask) * output_image_mask

        t_cond = self.t_embedding(t)
        if t.dim() == 1:
            t_cond = t_cond.unsqueeze(1)
        adaln_input = F.silu(self.adaln_proj(t_cond))

        llm_features = self.llm_cond_proj(self.llm_cond_norm(llm_features)) * llm_token_mask

        hidden_states = x + llm_features
        hidden_states = hidden_states + self.embed_image_indicator(
            (indicator == OUTPUT_IMAGE_INDICATOR).to(torch.long)
        )

        return hidden_states, position_ids, segment_ids, adaln_input


class Ideogram4Transformer2DModel(nn.Module):
    """Ideogram 4 DiT with sequence-parallel and layerwise-offload hooks."""

    _repeated_blocks = ["Ideogram4TransformerBlock"]
    _layerwise_offload_blocks_attrs = ["layers"]

    _sp_plan = {
        "sequence_prepare": {
            0: SequenceParallelInput(
                split_dim=1, expected_dims=3, split_output=True, auto_pad=True, shard_group=SHARD_GROUP
            ),
            1: SequenceParallelInput(
                split_dim=1, expected_dims=3, split_output=True, auto_pad=True, shard_group=SHARD_GROUP
            ),
            # 2 (segment_ids) and 3 (adaln_input) stay replicated.
        },
        "final_layer": SequenceParallelOutput(gather_dim=1, expected_dims=3, shard_group=SHARD_GROUP),
    }

    def __init__(
        self,
        *,
        config: Ideogram4Config,
        od_config: OmniDiffusionConfig | None = None,
        prefix: str = "transformer",
        compute_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        # ``config`` is resolved by the pipeline from
        # ``od_config.tf_model_config.params``: TransformerConfig is a dataclass,
        # not the raw HF mapping, so the model cannot derive the architecture
        # from it directly.
        self.od_config = od_config
        self.prefix = prefix
        self.config = config
        self.compute_dtype = compute_dtype or torch.bfloat16
        self.parallel_config = od_config.parallel_config if od_config is not None else None

        self.sequence_prepare = Ideogram4SequencePrepare(config, self.compute_dtype)
        self.rotary_emb = Ideogram4MRoPE(config.head_dim, config.rope_theta, config.mrope_section)
        self.layers = nn.ModuleList(
            [
                Ideogram4TransformerBlock(config, self.compute_dtype, prefix=f"{prefix}.layers.{i}.")
                for i in range(config.num_layers)
            ]
        )
        self.final_layer = Ideogram4FinalLayer(config, self.compute_dtype)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return load_streamed_weights(self, weights, self.compute_dtype)

    def forward(
        self,
        llm_features: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        position_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        indicator: torch.Tensor,
    ) -> torch.Tensor:
        """Velocity prediction over the packed ``[text | image]`` sequence.

        Only positions with ``indicator == OUTPUT_IMAGE_INDICATOR`` are
        meaningful; the caller slices them off after the gather.
        """
        hidden_states, position_ids, segment_ids, adaln_input = self.sequence_prepare(
            llm_features, x, t, position_ids, segment_ids, indicator
        )

        cos, sin = self.rotary_emb(position_ids)
        cos = cos.to(hidden_states.dtype)
        sin = sin.to(hidden_states.dtype)

        # The mask describes the rank-concatenated sequence attention sees after
        # the all-to-all, so it is built from the unsplit segment ids. It is
        # skipped entirely when nothing is padded, keeping the fast path.
        attn_mask = rank_concat_segment_mask_or_none(segment_ids, SHARD_GROUP, like=hidden_states)
        attn_metadata = (
            AttentionMetadata(attn_mask=attn_mask, extra={"attention_mask_mode": "arbitrary"})
            if attn_mask is not None
            else None
        )

        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin, adaln_input, attn_metadata)

        out = self.final_layer(hidden_states, adaln_input)
        return out.to(torch.float32)
