# E6: Planning expert diffusion pipeline

Depends on: E3 (code to port is in the exported repo). Parallel with E5.

## Goal
A vllm-omni diffusion stage that runs the Planning Expert from received VLM K/V and returns trajectories.

## Inputs
- `extra_repos/Qwen-Drive-1.0/src/qwen_drive/planning_expert.py`, `trajectory.py`, `configuration_qwen_drive.py`
- Golden (E2): K/V, anchor, trajectories, per-sample seeds
- Templates: `vllm_omni/diffusion/models/pi0/pipeline_pi0.py`, `vllm_omni/deploy/pi0.yaml`, `vllm_omni/diffusion/models/pi0_pipeline_config.py`, `vllm_omni/diffusion/models/lingbot_video/pipeline_lingbot_video.py` (subfolder selection via `od_config.model_config`), `vllm_omni/diffusion/registry.py`, `vllm_omni/diffusion/request.py`

## Outputs
- `vllm_omni/diffusion/models/qwen_drive/{__init__.py,planning_expert.py,pipeline_qwen_drive.py}`
- Registry entry `QwenDrivePlannerPipeline` in `vllm_omni/diffusion/registry.py`
- Tests: `tests/diffusion/models/qwen_drive/test_planning_expert.py` (tiny random config on CPU: shapes, determinism, per-sample seed `seed+k` batch invariance) and a golden-based GPU test

## Rules
- Verbatim port of numerics (bf16 rotary table, Fourier table in module dtype, RMSNorm in fp32, 10 Euler steps, `min_one_minus_t=0.1`).
- Weights: `<model>/<planner_subfolder>/model.safetensors`, strip `planning_expert.`, strict load. `planner_subfolder` from `od_config.model_config` (default `planner-rl`).
- Request data from `sampling_params.extra_args`: `history`, `history_velocity`, `history_acceleration`, `ego_status`, `nav_command`, `num_samples`, `seed`, `num_steps`. Scene K/V and anchor from the received KV payload.
- Output: `DiffusionOutput(output={"trajectories": ndarray[n,50,3]})`.

## Verify
Given golden K/V + anchor, trajectories match golden (max abs error < 1e-2 m, bf16).

## Exit criteria
Tests pass in docker, Progress.md updated.
