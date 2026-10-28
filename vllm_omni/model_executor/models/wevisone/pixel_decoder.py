# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage-1 pixel decoder for WeVisOne-VL.

Ported from the Tencent WeVisOne-VL reference implementation (Apache-2.0).

This stage owns the 9.6M-parameter mask decoder. It reads the per-request
``segmentation`` payload that stage 0 hands over (pre-merger ViT feature maps,
the pre-merge grid, the ``<OBJ>`` query rows, the boxes parsed from the
response, the decoder canvas and the original image size), runs the decoder in
float32, and emits one stacked mask tensor ``[num_masks, 1, H, W]`` per request
under ``image``. Masks are resampled back to the ORIGINAL image size with
bilinear interpolation, which is what the reference ``_masks_from_logits`` does.

Two deliberate properties:

* Mask order equals box order, so mask ``i`` pairs with box ``i``. Nothing is
  transported to identify the pairing; serving derives ``mask_index`` from
  enumeration.
* The sidecar ``mask_decoder.pt`` is loaded lazily. Text-only tasks (which
  generate no ``<OBJ>``) run without it, and the only failure is when a mask
  is actually requested.

Boxes are not reported here: ``metrics`` carries model metadata only on the
diffusion serving path, so the ``<box>`` spans in stage 0's text response are
the box contract for this pipeline.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .config import Qwen3VLSegConfig
from .constants import SEG_TOKEN_ID_FALLBACK
from .decoder import build_mask_decoder
from .feature_bridge import assert_cnn_rgb

logger = init_logger(__name__)

__all__ = ["WeVisOnePixelDecoder"]

SIDECAR_FILENAME = "mask_decoder.pt"

# Sidecar values the checkpoint is shipped with; a mismatch means the payload
# and the weights were built for different checkpoints.
_EXPECTED_SIDECAR_SEG_TOKEN_ID = SEG_TOKEN_ID_FALLBACK
_EXPECTED_SIDECAR_SPATIAL_MERGE_SIZE = 2


def _resolve_sidecar(model_path: str, revision: str | None, download_dir: str | None) -> str:
    """Locate ``mask_decoder.pt``, mirroring the reference ``_resolve_sidecar``.

    A local checkpoint directory must contain the sidecar; a repository id is
    fetched from the Hugging Face hub.
    """
    local = os.path.join(model_path, SIDECAR_FILENAME)
    if os.path.isdir(model_path):
        if not os.path.isfile(local):
            raise FileNotFoundError(
                f"WeVisOne masks need {SIDECAR_FILENAME}; it is not in {model_path}. Text-only tasks do not need it."
            )
        return local
    from vllm_omni.transformers_utils.repo_utils import hf_api

    return hf_api().hf_hub_download(
        repo_id=model_path,
        filename=SIDECAR_FILENAME,
        revision=revision,
        cache_dir=download_dir,
    )


def _segmentation_view(info: Mapping[str, Any]) -> dict[str, Any]:
    """Return the ``segmentation`` fields of one request, nested or flat.

    The wire keeps nested groups as dicts, but some runner paths flatten them to
    ``segmentation.<field>`` before the stage sees the payload.
    """
    nested = info.get("segmentation")
    if isinstance(nested, Mapping):
        return dict(nested)
    flat = {
        key.split(".", 1)[1]: value for key, value in info.items() if key.startswith("segmentation.") and "." in key
    }
    return flat


def _hw_pair(value: Tensor) -> tuple[int, int]:
    """Read an ``(h, w)`` pair from a plain or batched tensor.

    The multimodal processor emits ``grid_hw``/``mask_hw`` as ``batched("image")``
    fields, so a single-image request arrives as ``[[h, w]]`` and unpacking the
    nested row directly would fail.
    """
    values = value.reshape(-1).tolist()
    return int(values[0]), int(values[1])


def _masks_from_logits(logits: Tensor, height: int, width: int) -> Tensor:
    """Sigmoid and bilinear-resize mask logits back to the original size.

    Shapes follow the reference exactly: ``[num_masks, h, w]`` (what the decoder
    returns), ``[num_masks, 1, h, w]`` and ``[h, w]`` are the accepted layouts.
    Returns ``[num_masks, 1, H, W]`` float32 in ``[0, 1]``, which is what a client
    receives as one PNG per mask.
    """
    if logits.ndim == 4 and logits.shape[1] == 1:
        rows = logits[:, 0]
    elif logits.ndim == 3:
        rows = logits
    elif logits.ndim == 2:
        rows = logits[None]
    else:
        raise ValueError(f"unexpected mask logit shape: {tuple(logits.shape)}")
    probabilities = F.interpolate(
        rows.float().sigmoid()[None],
        size=(int(height), int(width)),
        mode="bilinear",
        align_corners=False,
    )[0]
    return probabilities.unsqueeze(1).contiguous()


class WeVisOnePixelDecoder(nn.Module):
    """The WeVisOne mask decoder, driven by the generation runner."""

    # Consumes the image-derived part of the cross-stage payload; the value is
    # what serving reports as this stage's input modality.
    input_modalities = "image"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        del prefix
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model

        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True

        model_config = vllm_config.model_config
        hf_config = getattr(model_config, "hf_config", None)
        text_config = getattr(hf_config, "text_config", None) or hf_config
        vision_config = getattr(hf_config, "vision_config", None)

        # The sidecar carries the decoder cfg but not the backbone dimensions,
        # so those come from the checkpoint config.
        self.seg_config = Qwen3VLSegConfig()
        self.llm_dim = int(getattr(text_config, "hidden_size", 2560))
        visual_hidden = int(getattr(vision_config, "hidden_size", 1024))
        self.visual_in_dims = [visual_hidden] * len(self.seg_config.vision_feature_layers)

        # Skeleton only. vLLM's memory profiler walks the module tree at
        # startup, before any weight loader has run.
        self.decoder = build_mask_decoder(self.seg_config, self.llm_dim, self.visual_in_dims)
        self._loaded = False

    # ------------------------------------------------------------- vLLM hooks

    def embed_input_ids(self, input_ids: Tensor, **_: Any) -> Tensor:
        """Stable dummy embedding: this stage never reads token embeddings."""
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: Tensor | OmniOutput, sampling_metadata: Any = None) -> None:
        del hidden_states, sampling_metadata
        return None

    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> set[str]:
        """Ignore the VLM safetensors and load the decoder sidecar.

        The stage shares the checkpoint directory with stage 0, so the iterator
        yields Qwen3-VL weights that this module has no parameters for. It is
        drained anyway: the loader streams from a file handle the caller closes
        only once the iterator is exhausted, so abandoning it wedges startup.
        """
        try:
            for _ in weights:
                pass
        except RuntimeError as exc:
            if "Cannot find any model weights" not in str(exc):
                raise
            logger.debug("WeVisOne pixel-decoder stage has no root-level weights, as expected")
        return self._load_sidecar()

    def _ensure_loaded(self) -> None:
        """Load on first use when the weight loader skipped ``load_weights``.

        ``load_format=dummy`` (used by warmup and memory profiling) never calls
        ``load_weights``, which would leave the decoder holding its random
        skeleton and make the dummy run produce noise masks.
        """
        if not self._loaded:
            self._load_sidecar()

    def _load_sidecar(self) -> set[str]:
        """Read ``mask_decoder.pt`` and check it against the module skeleton.

        Loaded with ``strict=True``: a partial load would produce plausible masks
        that are wrong, which is worse than a startup failure.
        """
        path = _resolve_sidecar(
            self.model_path,
            getattr(self.vllm_config.model_config, "revision", None),
            getattr(self.vllm_config.load_config, "download_dir", None),
        )
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, Mapping):
            raise ValueError(f"{SIDECAR_FILENAME} is not a checkpoint mapping")
        state_dict = checkpoint.get("state_dict")
        if not isinstance(state_dict, Mapping):
            raise ValueError(f"{SIDECAR_FILENAME} has no state_dict")

        cfg = Qwen3VLSegConfig.from_dict(dict(checkpoint.get("cfg") or {}))
        if cfg.to_dict() != self.seg_config.to_dict():
            logger.info("WeVisOne decoder cfg from sidecar differs from defaults, rebuilding")
            self.seg_config = cfg
            self.decoder = build_mask_decoder(cfg, self.llm_dim, self.visual_in_dims)

        sidecar_seg_id = int(checkpoint.get("seg_token_id", _EXPECTED_SIDECAR_SEG_TOKEN_ID))
        if sidecar_seg_id != _EXPECTED_SIDECAR_SEG_TOKEN_ID:
            raise ValueError(
                f"sidecar seg_token_id={sidecar_seg_id} does not match the "
                f"<OBJ> id {_EXPECTED_SIDECAR_SEG_TOKEN_ID} used to select query rows"
            )
        sidecar_merge = int(checkpoint.get("spatial_merge_size", _EXPECTED_SIDECAR_SPATIAL_MERGE_SIZE))
        if sidecar_merge != _EXPECTED_SIDECAR_SPATIAL_MERGE_SIZE:
            raise ValueError(
                f"sidecar spatial_merge_size={sidecar_merge} does not match "
                f"{_EXPECTED_SIDECAR_SPATIAL_MERGE_SIZE} used to undo the pre-merge shuffle"
            )

        self.decoder.load_state_dict({str(name): tensor for name, tensor in state_dict.items()}, strict=True)
        self.decoder.to(device=self.vllm_config.device_config.device, dtype=torch.float32).eval()
        self._loaded = True

        loaded = {f"decoder.{name}" for name, _ in self.decoder.named_parameters()}
        logger.info(
            "WeVisOne pixel decoder loaded from %s device=%s parameters=%d",
            path,
            self.vllm_config.device_config.device,
            self.decoder.num_parameters(),
        )
        return loaded

    # ------------------------------------------------------------ decoding

    def _decode_request(self, info: Mapping[str, Any]) -> Tensor | None:
        """Decode one request's payload into ``[num_masks, 1, H, W]``, or ``None``.

        ``None`` means the request asked for no masks, which is normal for
        text-only tasks; they run without the sidecar at all. ``forward`` turns
        it into an empty tensor so the request still has a per-request payload.
        """
        seg = _segmentation_view(info)
        boxes = seg.get("boxes")
        if not isinstance(boxes, Tensor) or boxes.numel() == 0:
            return None

        t_seg = seg.get("t_seg")
        if not isinstance(t_seg, Tensor) or t_seg.numel() == 0:
            return None

        self._ensure_loaded()

        features = seg.get("features")
        grid_hw = seg.get("grid_hw")
        canvas = seg.get("canvas")
        mask_hw = seg.get("mask_hw")
        if not isinstance(features, Tensor) or features.numel() == 0:
            raise ValueError(
                "a request with <OBJ> queries carries no pre-merger feature maps, so its masks cannot be decoded"
            )
        if features.ndim != 4:
            raise ValueError(f"feature payload must be [layers, C, H, W], got {tuple(features.shape)}")
        if not isinstance(canvas, Tensor):
            raise ValueError("a request with <OBJ> queries carries no decoder canvas")

        if boxes.shape[0] != t_seg.shape[0]:
            raise ValueError(f"#OBJ={t_seg.shape[0]} does not match #boxes={boxes.shape[0]}")
        if not isinstance(grid_hw, Tensor) or grid_hw.numel() != 2:
            got = tuple(grid_hw.shape) if isinstance(grid_hw, Tensor) else grid_hw
            raise ValueError(f"grid payload must be (h, w), got {got}")
        height, width = _hw_pair(mask_hw) if isinstance(mask_hw, Tensor) else (0, 0)
        if height <= 0 or width <= 0:
            raise ValueError(f"mask payload must carry a positive (H, W), got {(height, width)}")

        # The canvas and feature maps are registered with ``keep_on_cpu=True`` so the
        # cross-stage payload stays serialisable; the decoder weights live on the
        # stage device, so every tensor is moved before the forward call.
        device = self.vllm_config.device_config.device
        visual_features_2d = [feature.unsqueeze(0).to(device) for feature in features]
        grid = _hw_pair(grid_hw)
        outputs = self.decoder(
            assert_cnn_rgb(canvas.to(device), self.seg_config.cnn_image_size),
            visual_features_2d,
            t_seg.to(device),
            boxes.to(device),
            grid,
        )
        masks = _masks_from_logits(outputs["mask_logits"], height, width)
        return masks.to("cpu").contiguous()

    def forward(
        self,
        input_ids: Tensor | None = None,
        positions: Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: Tensor | None = None,
        runtime_additional_information: Sequence[Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        """Decode every request in the step and return one mask tensor per request.

        Raises:
            ValueError: If a payload that needs masks is malformed. Failing the
                step is deliberate: a request whose masks cannot be decoded has
                nothing to return, and reporting an error is better than leaving
                the caller with a text-only response for a mask task.
        """
        del input_ids, positions, intermediate_tensors, inputs_embeds

        infos = runtime_additional_information
        if infos is None:
            infos = kwargs.get("model_intermediate_buffer")
        infos = infos or []

        # vLLM profiles generation stages once with synthetic token inputs,
        # before any cross-stage payload exists.
        if not infos:
            return OmniOutput(
                text_hidden_states=torch.zeros(1, device=self.vllm_config.device_config.device),
                multimodal_outputs=None,
            )

        masks: list[Tensor] = []
        for info in infos:
            request_masks = self._decode_request(info if isinstance(info, Mapping) else {})
            if request_masks is None:
                # Per-request values are copied through ``split_mapping_outputs``,
                # which only accepts tensors, so a mask-less request still needs
                # one. The empty tensor contributes no content part, leaving
                # that request with its text choice alone.
                request_masks = torch.empty(0)
            masks.append(request_masks)

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"image": masks},
        )
