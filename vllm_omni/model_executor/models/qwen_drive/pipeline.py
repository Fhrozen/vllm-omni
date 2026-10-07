# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen-Drive pipeline topology (frozen).

  Stage 0: VLM (AR)         — reasoning / VQA text; `modalities=["text"]` stops here.
  Stage 1: VLM prefill (AR) — re-reads the (closed-turn) prompt and ships the K/V of the full-attention layers.
  Stage 2: planning expert  — flow-matching trajectory conditioned on that K/V; `modalities=["trajectory"]`.

Stage 1 exists because vLLM never computes the KV of the last sampled token, while the planner reads a cache that
ends with ``<|im_end|>\\n``.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.qwen_drive"
_VLM_ARCH = "QwenDriveVLMForConditionalGeneration"

QWEN_DRIVE_PIPELINE = PipelineConfig(
    model_type="qwen_drive",
    default_deploy_config_name="qwen_drive.yaml",
    model_arch=_VLM_ARCH,
    hf_architectures=("QwenDriveForPlanning",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="vlm",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            model_arch=_VLM_ARCH,
            engine_output_type="text",
            sampling_constraints={"detokenize": True},
        ),
        StagePipelineConfig(
            stage_id=1,
            # The orchestrator forwards the request's processed image features only to "thinker" stages.
            model_stage="thinker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(0,),
            final_output=False,
            model_arch=_VLM_ARCH,
            engine_output_type="text",
            custom_process_input_func=f"{_PROC}.vlm_to_prefill",
            omni_kv_config={
                "need_send_cache": True,
                "kv_transfer_criteria": {"type": "prefill_finished"},
                # Full-attention layers of the Qwen3.5 backbone (every 4th layer); the others hold GDN state.
                "kv_layer_indices": [3, 7, 11, 15, 19, 23, 27, 31],
            },
            sampling_constraints={"detokenize": False},
        ),
        StagePipelineConfig(
            stage_id=2,
            model_stage="planner",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(1,),
            final_output=True,
            final_output_type="trajectory",
            model_arch="QwenDrivePlannerPipeline",
            custom_process_input_func=f"{_PROC}.prefill_to_planner",
            omni_kv_config={"need_recv_cache": True},
        ),
    ),
)
