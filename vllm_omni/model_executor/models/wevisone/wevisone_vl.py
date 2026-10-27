# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage 0 of WeVisOne-VL: the Qwen3-VL backbone plus feature capture.

Ported from the Tencent WeVisOne-VL reference implementation (Apache-2.0).

Stage 0 is an ordinary Qwen3-VL that additionally produces the two tensors the
pixel decoder needs, exposed as the ``segmentation`` group of the cross-stage
payload:

``segmentation.features``
    Batch-order list, one ``[num_capture_layers, C, H, W]`` map per request.
    The maps are the ViT outputs taken *before* each spatial merger.
``segmentation.grid_hw``
    Batch-order list, one ``[2]`` pre-merge ``(h, w)`` per request.
``segmentation.t_seg``
    Full-batch ``[num_tokens, llm_dim]`` rows of the last decoder layer output.
    ``to_payload_element`` slices this per request; the stage input processor
    reduces the accumulated rows to the ``<OBJ>`` queries.
``segmentation.canvas`` / ``segmentation.mask_hw``
    Batch-order lists coming from ``WeVisOneMultiModalProcessor``: the original
    image bicubically resized to the decoder canvas, and the original ``(H, W)``
    the masks are resampled back to.

Both the list-shaped groups and the full-batch tensor are only valid in the raw
model output, before the payload is partitioned per request.

Two constraints follow from the reference implementation and are enforced here:

* Images are captured inside ``embed_multimodal``, which is where the vision
  tower runs. The vision tower is not ``support_torch_compile``-decorated, so
  forward pre-hooks still fire there; the decoder stack is compiled, so the last
  layer capture happens inside the compiled forward instead of a hook.
* One image per request. A per-request payload entry can only hold one grid, so
  a request carrying several images cannot be mapped to ``features``.
"""

from __future__ import annotations

from itertools import islice

import torch
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.distributed.parallel_state import PPMissingLayer
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
    Qwen3VLForConditionalGeneration,
    Qwen3LLMForCausalLM,
    Qwen3LLMModel,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from vllm_omni.data_entry_keys import OmniPayload
from vllm_omni.model_executor.models.output_templates import OmniOutput

from .config import Qwen3VLSegConfig
from .processing import CANVAS_KEY, HW_KEY, WeVisOneMultiModalProcessor
from .feature_bridge import (
    VisualFeatureCapture,
    per_sample_premerge_feature_maps,
    premerge_grid_hw,
)


_DYNAMIC_ARG_DIMS = {
    "input_ids": 0,
    "positions": -1,
    "intermediate_tensors": 0,
    "inputs_embeds": 0,
    # the same shape as input_embeds
    "deepstack_input_embeds": 0,
}


@support_torch_compile(dynamic_arg_dims=_DYNAMIC_ARG_DIMS)
class WeVisOneQwen3LLMModel(Qwen3LLMModel):
    """Qwen3 decoder stack that also returns the last layer output.

    The reference hooks ``layers[-1]`` and reads the value that layer produced,
    which is *before* the final norm.  vLLM defers the residual addition until
    the next RMSNorm, so the logical last-layer state has to be reconstructed
    right after the last layer call.
    """

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        # args for deepstack
        deepstack_input_embeds: IntermediateTensors | None = None,
        return_last_layer_output: bool = False,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer_idx, layer in islice(
            enumerate(self.layers), self.start_layer, self.end_layer
        ):
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )

            if deepstack_input_embeds is not None and layer_idx in range(
                0, len(deepstack_input_embeds)
            ):
                hidden_states = (
                    hidden_states
                    + deepstack_input_embeds[f"deepstack_input_embeds_{layer_idx}"]
                )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        # Reconstruct the logical last-layer output before the final norm.
        last_layer_output = (
            hidden_states.clone() if residual is None else hidden_states + residual
        )
        hidden_states, _ = self.norm(hidden_states, residual)

        if return_last_layer_output:
            return hidden_states, last_layer_output.view(-1, last_layer_output.shape[-1])
        return hidden_states, None


class WeVisOneQwen3LLMForCausalLM(Qwen3LLMForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Qwen3LLMForCausalLM, self).__init__()
        config = vllm_config.model_config.hf_config.text_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config
        self.model = WeVisOneQwen3LLMModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix="lm_head",
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )


@MULTIMODAL_REGISTRY.register_processor(
    WeVisOneMultiModalProcessor,
    info=Qwen3VLProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class WeVisOneVLForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """Qwen3-VL whose forward emits the pixel-decoder inputs as ``segmentation``."""

    have_multimodal_outputs = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # Replaced wholesale so weight names stay `language_model.model.*`.
        self.language_model = WeVisOneQwen3LLMForCausalLM(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "language_model")
        )

        self.seg_config = Qwen3VLSegConfig()
        self.spatial_merge_size = int(self.config.vision_config.spatial_merge_size)
        self._pending_features: list[torch.Tensor] = []
        self._pending_grid_hw: list[torch.Tensor] = []
        self._pending_canvas: list[torch.Tensor] = []
        self._pending_mask_hw: list[torch.Tensor] = []

    def embed_multimodal(self, **kwargs: object):
        """Run the vision tower while capturing the pre-merger feature maps."""
        self._pending_features = []
        self._pending_grid_hw = []
        self._pending_canvas = []
        self._pending_mask_hw = []

        grid_thw = kwargs.get("image_grid_thw")
        if grid_thw is None or self.visual is None:
            return super().embed_multimodal(**kwargs)

        with VisualFeatureCapture(self, self.seg_config.vision_feature_layers) as capture:
            embeddings = super().embed_multimodal(**kwargs)

        per_sample = per_sample_premerge_feature_maps(
            capture.premerge_features,
            self.seg_config.vision_feature_layers,
            grid_thw,
            self.spatial_merge_size,
        )
        self._pending_features = [torch.stack(maps, dim=0) for maps in per_sample]
        self._pending_grid_hw = [
            torch.tensor([h, w], dtype=torch.int64) for h, w in premerge_grid_hw(grid_thw)
        ]
        # Extra fields produced by WeVisOneMultiModalProcessor, already in image
        # order and kept on CPU so the stage input processor can forward them.
        canvases = kwargs.get(CANVAS_KEY)
        hw = kwargs.get(HW_KEY)
        if canvases is not None and hw is not None:
            self._pending_canvas = [
                canvas.unsqueeze(0).contiguous() for canvas in canvases
            ]
            self._pending_mask_hw = [row for row in hw]
        return embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if intermediate_tensors is not None:
            inputs_embeds = None

        if (
            self.use_deepstack
            and inputs_embeds is not None
            and get_pp_group().is_first_rank
        ):
            deepstack_input_embeds = self._get_deepstack_input_embeds(
                inputs_embeds.size(0)
            )
        else:
            deepstack_input_embeds = None

        hidden_states, last_layer_output = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            deepstack_input_embeds=deepstack_input_embeds,
            return_last_layer_output=True,
        )

        if inputs_embeds is not None and get_pp_group().is_first_rank:
            self._clear_deepstack_input_embeds(inputs_embeds.size(0))

        return hidden_states, last_layer_output

    def make_omni_output(
        self, model_outputs: torch.Tensor | OmniOutput, **kwargs: object
    ) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs

        if isinstance(model_outputs, tuple):
            text_hidden_states, last_layer_output = model_outputs
        else:
            text_hidden_states, last_layer_output = model_outputs, None

        text_hidden_states = text_hidden_states.reshape(-1, text_hidden_states.shape[-1])
        segmentation: OmniPayload = {}

        if last_layer_output is not None:
            # Full-batch rows: the payload builder slices these per request.
            segmentation["t_seg"] = last_layer_output.view(-1, last_layer_output.shape[-1])

        features = self._pending_features
        if features:
            spans = kwargs.get("request_token_spans")
            if spans is not None and len(features) != len(spans):
                raise ValueError(
                    f"captured {len(features)} image feature maps for {len(spans)} "
                    "requests; WeVisOne supports one image per request"
                )
            segmentation["features"] = features
            segmentation["grid_hw"] = self._pending_grid_hw
            if self._pending_canvas:
                segmentation["canvas"] = self._pending_canvas
                segmentation["mask_hw"] = self._pending_mask_hw
            # Consume the capture: a decode step has no vision output.
            self._pending_features = []
            self._pending_grid_hw = []
            self._pending_canvas = []
            self._pending_mask_hw = []

        return OmniOutput(
            text_hidden_states=text_hidden_states,
            multimodal_outputs={"segmentation": segmentation} if segmentation else None,
        )
