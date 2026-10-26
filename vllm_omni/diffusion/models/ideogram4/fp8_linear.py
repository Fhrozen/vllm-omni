# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Weight-only FP8 linears used by the Ideogram 4 release.

Every ``nn.Linear`` in the released checkpoint is stored as FP8 E4M3 with one
scale per output row, and the reference dequantizes the weight inside the
forward pass while keeping activations in the compute dtype. This module keeps
that behaviour on purpose: it does not call ``torch._scaled_mm``, so it needs no
compute-capability gate and runs unchanged on consumer Blackwell (sm_120),
Ada, Hopper and ROCm. Weights stay FP8 in memory, which is what makes the two
9.3 GB transformers fit alongside the text encoder.

The helpers also accept a non-quantized checkpoint: a layer whose weight arrives
in a floating dtype is kept dense, so a future bf16 release loads through the
same code path.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "FP8_E4M3_MAX",
    "FP8_SCALE_SUFFIX",
    "FP8_TEXT_ENCODER_CONFIG_FLAG",
    "FP8_WEIGHT_DTYPE",
    "Fp8Linear",
    "is_fp8_state_dict",
    "load_fp8_state_dict",
    "load_streamed_weights",
    "quantize_weight_to_fp8",
    "swap_linears_to_fp8",
]

FP8_E4M3_MAX = 448.0
FP8_WEIGHT_DTYPE = torch.float8_e4m3fn
FP8_SCALE_SUFFIX = ".weight_scale"
# Non-standard key the released text-encoder config uses to flag its FP8 weights.
FP8_TEXT_ENCODER_CONFIG_FLAG = "ideogram_fp8_weight_only"


def quantize_weight_to_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-wise symmetric quantization, returning ``(weight_fp8, scale_fp32)``."""
    weight = weight.to(torch.float32)
    amax = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = amax / FP8_E4M3_MAX
    quantized = (weight / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(FP8_WEIGHT_DTYPE)
    return quantized, scale.squeeze(1).to(torch.float32)


class Fp8Linear(nn.Module):
    """``nn.Linear`` over an FP8 weight, dequantized per forward pass.

    ``weight`` and ``bias`` are buffers rather than parameters so that a
    quantized weight is never cast to the compute dtype by a blanket
    ``model.to(dtype)``. ``weight_scale`` stays float32.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        self.register_buffer("weight", torch.empty(out_features, in_features, dtype=FP8_WEIGHT_DTYPE))
        self.register_buffer("weight_scale", None)
        self.register_buffer("bias", None if not bias else torch.empty(out_features, dtype=compute_dtype))

    def load_param(self, name: str, tensor: torch.Tensor) -> bool:
        """Assign one checkpoint entry; returns whether ``name`` belongs to this layer.

        Existing storage is filled in place so the layer keeps its device; meta or
        empty placeholders are replaced by the incoming tensor.
        """
        if name == "weight":
            value = tensor if tensor.dtype == FP8_WEIGHT_DTYPE else tensor.to(self.compute_dtype)
        elif name == "weight_scale":
            value = tensor.to(torch.float32)
        elif name == "bias":
            value = tensor.to(self.compute_dtype)
        else:
            return False
        storage = getattr(self, name)
        if storage is None or storage.is_meta:
            setattr(self, name, value)
        else:
            storage.copy_(value.to(storage.device))
        return True

    @property
    def quantized(self) -> bool:
        return self.weight.dtype == FP8_WEIGHT_DTYPE

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        if weight.dtype == FP8_WEIGHT_DTYPE:
            scale = self.weight_scale.to(x.dtype)
            weight = weight.to(x.dtype) * scale.unsqueeze(1)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


def is_fp8_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
    """True when the state dict carries FP8 weights or per-row FP8 scales."""
    return any(key.endswith(FP8_SCALE_SUFFIX) for key in state_dict) or any(
        tensor.dtype == FP8_WEIGHT_DTYPE for tensor in state_dict.values()
    )


def swap_linears_to_fp8(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
    compute_dtype: torch.dtype = torch.bfloat16,
    prefix: str = "",
) -> int:
    """Replace quantized ``nn.Linear`` submodules with :class:`Fp8Linear`.

    Only layers that have a matching ``*.weight_scale`` entry are swapped, so a
    mixed quantized/dense checkpoint keeps its dense layers as ``nn.Linear``.
    Returns the number of swapped layers.
    """
    swapped = 0
    for name, child in list(module.named_children()):
        if not isinstance(child, nn.Linear):
            continue
        full_name = f"{prefix}{name}"
        if f"{full_name}{FP8_SCALE_SUFFIX}" not in state_dict:
            continue
        replacement = Fp8Linear(
            child.in_features,
            child.out_features,
            bias=child.bias is not None,
            compute_dtype=compute_dtype,
        )
        replacement.load_param("weight", state_dict[full_name])
        replacement.load_param("weight_scale", state_dict[f"{full_name}{FP8_SCALE_SUFFIX}"])
        if child.bias is not None:
            replacement.load_param("bias", state_dict[f"{full_name}.bias"])
        setattr(module, name, replacement)
        swapped += 1
    for name, child in module.named_children():
        if isinstance(child, nn.Module) and not isinstance(child, Fp8Linear):
            swapped += swap_linears_to_fp8(child, state_dict, compute_dtype, f"{prefix}{name}.")
    return swapped


def load_fp8_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    assign: bool = False,
    strict: bool = True,
) -> None:
    """Load a state dict that may mix FP8 weights, FP32 scales and dense tensors.

    FP8 tensors move to ``device`` without a dtype cast, ``*.weight_scale``
    entries stay float32, other floating tensors are cast to ``dtype``, and
    everything else is moved as-is.
    """
    prepared = {}
    for key, tensor in state_dict.items():
        if tensor.dtype == FP8_WEIGHT_DTYPE:
            prepared[key] = tensor.to(device)
        elif key.endswith(FP8_SCALE_SUFFIX):
            prepared[key] = tensor.to(torch.float32).to(device)
        elif tensor.is_floating_point():
            prepared[key] = tensor.to(dtype).to(device)
        else:
            prepared[key] = tensor.to(device)
    model.load_state_dict(prepared, strict=strict, assign=assign)
    model.to(device)


def load_streamed_weights(
    model: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    compute_dtype: torch.dtype,
    *,
    skip_unexpected: bool = True,
    dtype_for: Callable[[str], torch.dtype] | None = None,
) -> set[str]:
    """Apply checkpoint entries one at a time without materializing a state dict.

    Entries owned by an :class:`Fp8Linear` go through :meth:`Fp8Linear.load_param`
    so FP8 weights and their row scales keep their dtypes. Every other tensor is
    cast to ``compute_dtype`` (FP8 tensors are left alone) and copied into the
    matching parameter or buffer; meta placeholders are replaced instead of copied
    into. Tensors arriving from the safetensors iterator are on CPU, so each
    assignment targets the storage's own device. Returns the set of checkpoint
    names that were applied, which the loader uses for its strict check.

    ``dtype_for`` optionally overrides the cast dtype per checkpoint name, which
    keeps a force-upcast VAE in float32 while the rest of the model runs in the
    pipeline compute dtype.
    """
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    loaded: set[str] = set()
    for name, tensor in weights:
        target = params.get(name)
        if target is None:
            target = buffers.get(name)
        if target is None:
            if skip_unexpected:
                continue
            raise ValueError(f"Unexpected checkpoint key for {type(model).__name__}: {name}")
        module_path, _, attribute = name.rpartition(".")
        owner = model.get_submodule(module_path) if module_path else model
        dtype = dtype_for(name) if dtype_for is not None else compute_dtype
        value = tensor if tensor.dtype == FP8_WEIGHT_DTYPE else tensor.to(dtype)
        if isinstance(owner, Fp8Linear):
            if not owner.load_param(attribute, value):
                if skip_unexpected:
                    continue
                raise ValueError(f"Unexpected FP8 checkpoint key: {name}")
            loaded.add(name)
            continue
        if target.is_meta:
            target.data = value
        else:
            target.data.copy_(value.to(target.device))
        loaded.add(name)
    return loaded
