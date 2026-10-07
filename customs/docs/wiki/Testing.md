# Testing

All commands run on the host and execute inside docker (`customs/qwen_drive/docker_run.sh`, repo mounted at `/workspace`, venv `/workspace/.venv`). Prerequisites: `setup_env.sh` (E1), golden data (E2: `run_ref_baseline.sh`), HF export (E3: `export_hf.sh`). Default deploy is the FP8 one for 16 GB GPUs.

| What | Command | Notes |
|------|---------|-------|
| Unit + parity (GPU 0) | `customs/qwen_drive/run_unit_tests.sh` | `tests/diffusion/models/qwen_drive/test_planning_expert.py`: 5 CPU tests, 12 golden parity (atol 1e-3), 2 pipeline-contract. Markers: `core_model`/`cpu` for units, `advanced_model`/`local_model`/`diffusion` for golden tests (skipped without checkpoint/golden/CUDA) |
| Offline 3-stage run | `customs/qwen_drive/run_e2e_offline.sh --num-scenes 3 --modes direct,reasoning` | prints ADE vs golden; add `--deploy-config /workspace/vllm_omni/deploy/qwen_drive.yaml` for BF16 |
| Online e2e (both GPUs) | `customs/qwen_drive/run_e2e_test.sh` | `tests/e2e/online_serving/test_qwen_drive.py`: VQA, direct x3 scenes, reasoning x3 scenes (11 tests incl. concurrency, client abort, VQA streaming and planner-sft, ~5 min). The script passes `--run-level=advanced_model`; with the default `core_model` level the harness patches the deploy yaml to `load_format: dummy` (random weights!) |
| BF16 on 2x16 GB | `customs/qwen_drive/run_e2e_offline.sh --deploy-config /workspace/customs/qwen_drive/deploy_bf16_16gb.yaml --num-scenes 3` | one sequence, 5000-token context |
| Noise floor of the reference | `QD_GPUS='"device=0"' customs/qwen_drive/docker_run.sh "python customs/qwen_drive/spikes/noise_floor.py"` | eager vs sdpa ADE 0.02-0.06 m |
| BF16 accuracy (big GPU) | `QD_DEPLOY=/workspace/vllm_omni/deploy/qwen_drive.yaml QD_MAX_ADE=0.02 customs/qwen_drive/run_e2e_test.sh` | expect ADE vs golden at the kernel-noise level (0.02-0.1 m), not zero |
| Planner variant | `QD_PLANNER=sft` (and set `planner_subfolder: planner-sft` in the deploy yaml) | golden files exist for both |

Env knobs of the e2e test: `QD_MODEL_DIR`, `QD_DEPLOY`, `QD_GOLDEN_DIR`, `QD_SCENES`, `QD_IMAGE_ROOT`, `QD_PLANNER`, `QD_MAX_ADE` (default 0.15 m, FP8).

Lint: `ruff` is installed into the venv on demand (`uv pip install ruff`); changed files pass `ruff check` and `ruff format --check`.

Pass criteria seen so far (2x RTX 5060 Ti): unit 19/19; offline 6/6 FP8 and 6/6 BF16; online 11/11 FP8; ADE vs the HF golden 0.026-0.112 m, same order as the reference's own eager-vs-sdpa noise (0.02-0.06 m).
