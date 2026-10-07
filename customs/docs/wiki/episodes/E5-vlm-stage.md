# E5: Stage 0/1 VLM in vllm-omni

Depends on: E3, E4 (S4, S5).

## Goal
vllm-omni serves the VLM part of `QwenDriveForPlanning` as an AR stage that (a) generates text and (b) can ship the post-rotary K/V of the 8 full-attention layers plus the rope anchor to the next stage.

## Inputs
- `extra_repos/Qwen-Drive-1.0-4B-hf` (E3), golden K/V and text (E2), spike verdicts (E4)
- Templates: `vllm_omni/model_executor/models/bagel/`, `vllm_omni/model_executor/models/gepard/gepard_talker.py` (imports vllm Qwen3_5), `vllm_omni/model_executor/models/registry.py`
- `vllm_omni/distributed/omni_connectors/kv_transfer_manager.py`, `.../utils/kv_utils.py`, `vllm_omni/worker/gpu_ar_model_runner.py`, `vllm_omni/worker_v2/omni_ar_model_runner.py`

## Outputs
- `vllm_omni/model_executor/models/qwen_drive/{__init__.py,qwen_drive_vlm.py}`: class `QwenDriveForPlanning` wrapping vllm `Qwen3_5ForConditionalGeneration` built from `hf_config.vlm_config`; weight mapper strips `vlm.`, skips `planning_expert.*`/`perception.*`; exposes the rope anchor through `custom_metadata`
- Registry entry in `vllm_omni/model_executor/models/registry.py`
- KV layer filter: `omni_kv_config["kv_layer_indices"]` honored in `kv_transfer_manager.py` (and plumbed through the AR runners)
- Unit tests for the filter (CPU) under `tests/distributed/`

## Verify (docker, GPU)
- Greedy VQA text equals golden `vqa_<i>.txt`.
- K/V of full-attention layers from a vLLM prefill vs golden: cosine > 0.99 per layer (bf16).
- Anchor equals golden anchor.

## Exit criteria
Parity checks pass, wiki Architecture.md updated, Progress.md updated.
