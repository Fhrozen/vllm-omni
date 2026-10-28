# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""WeVisOne-VL pipeline topology: Qwen3-VL VLM (stage 0) -> pixel decoder (stage 1).

Stage 0 runs the Qwen3-VL backbone and captures the two feature sets the mask
decoder needs (``feature_bridge``): pre-merger ViT block outputs and the last
decoder-layer output at ``<OBJ>`` rows before the final norm. Its pooling output
is the ``segmentation`` modality, which the stage input processor turns into a
per-request payload; boxes are reconstructed from generated token ids, so
``skip_special_tokens`` must stay off and ``stop_token_ids`` must match the
checkpoint's ``eos_token_id``.

Stage 1 holds the 9.6M-parameter pixel decoder and emits one stacked
``[num_masks, 1, H, W]`` probability tensor per request as the client-visible
``image`` output. Text-only tasks (detection, count, OCR) produce no masks and
pass through stage 1 unchanged, and the sidecar weights are loaded lazily, so
those tasks never touch ``mask_decoder.pt``.

The pixel decoder is numerically sensitive to its input contract — the canvas is
the original image resized bicubically to ``cnn_image_size`` with no smart-resize
and no mean/std normalization — so stage 1 runs eagerly.
"""

from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig

_PROC = "vllm_omni.model_executor.stage_input_processors.wevisone"

# ``<|im_end|>`` in the shipped tokenizer; mirrored in the deploy yaml so the
# sampling stop list and the box parser agree.
IM_END_TOKEN_ID = 151645

WEVISONE_VL_PIPELINE = PipelineConfig(
    model_type="wevisone_vl",
    model_arch="WeVisOneVLForConditionalGeneration",
    hf_architectures=("Qwen3VLForConditionalGeneration",),
    default_deploy_config_name="wevisone_vl.yaml",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="wevisone_vl",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            # Both stages are client-visible: stage 0 carries the box spans, stage 1
            # carries the masks. Mask i answers box i, so dropping either half of the
            # response would leave the pairing unverifiable.
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="text+segmentation",
            model_arch="WeVisOneVLForConditionalGeneration",
            custom_process_next_stage_input_func=f"{_PROC}.vlm2pixel_decoder_full_payload",
            # Boxes come from token ids, so special tokens must survive decoding.
            sampling_constraints={
                "detokenize": True,
                "skip_special_tokens": False,
                "stop_token_ids": [IM_END_TOKEN_ID],
            },
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="pixel_decoder",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            requires_full_payload_input=True,
            final_output=True,
            final_output_type="image",
            engine_output_type="image",
            model_arch="WeVisOnePixelDecoder",
        ),
    ),
)

# Text-only deployment: the VLM alone, no sidecar weights and no second stage.
# Selected explicitly (the checkpoint resolves to ``wevisone_vl``), and mask
# requests are rejected because nothing can decode them.
WEVISONE_VL_TEXT_ONLY_PIPELINE = PipelineConfig(
    model_type="wevisone_vl_text_only",
    model_arch="WeVisOneVLForConditionalGeneration",
    hf_architectures=("Qwen3VLForConditionalGeneration",),
    default_deploy_config_name="wevisone_vl_text_only.yaml",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="wevisone_vl",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="text",
            model_arch="WeVisOneVLForConditionalGeneration",
            sampling_constraints={"skip_special_tokens": False},
        ),
    ),
)

__all__ = ["WEVISONE_VL_PIPELINE", "WEVISONE_VL_TEXT_ONLY_PIPELINE"]
