# E7: Pipeline, deploy and stage bridges

Depends on: E5, E6.

## Inputs
- Templates: `vllm_omni/model_executor/models/bagel/pipeline.py`, `vllm_omni/model_executor/stage_input_processors/bagel.py`, `vllm_omni/deploy/bagel.yaml`, `vllm_omni/deploy/pi0.yaml`, `vllm_omni/config/pipeline_registry.py`, `vllm_omni/config/stage_config.py`
- Outputs of E5, E6; spike verdicts S1/S3

## Outputs
- `vllm_omni/model_executor/models/qwen_drive/pipeline.py` (`QWEN_DRIVE_PLAN_PIPELINE`, `QWEN_DRIVE_VQA_PIPELINE`): stage 0 `LLM_AR` (text), stage 1 `LLM_AR` (`omni_kv_config` need_send_cache, `prefill_finished`, `kv_layer_indices`), stage 2 `DIFFUSION` (`need_recv_cache`, `final_output_type="trajectory"`)
- `vllm_omni/model_executor/stage_input_processors/qwen_drive.py`: bridge 0->1 (prompt ids + reasoning + `<|im_end|>\n`; direct mode passes the prompt unchanged, stage 0 `max_tokens=1`), bridge 1->2 (extra_args + anchor)
- `vllm_omni/deploy/qwen_drive_plan.yaml`, `qwen_drive_vqa.yaml` (`trust_remote_code: true`, `shared_memory_connector`, devices per D6, small `max_model_len`, `planner_subfolder`)
- Entries in `vllm_omni/config/pipeline_registry.py`

## Verify
Offline end-to-end (docker, 2 GPUs) on demo scenes: direct and reasoning trajectories vs golden (ADE vs golden < 0.1 m; exact match not required because of different attention kernels).

## Exit criteria
Offline run passes; Architecture.md updated; Progress.md updated.
