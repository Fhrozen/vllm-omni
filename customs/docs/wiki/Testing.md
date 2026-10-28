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

Pass criteria seen so far (measured on the earlier 2x RTX 5060 Ti host, not re-run on the current 2x RTX 3060 machine): unit 19/19; offline 6/6 FP8 and 6/6 BF16; online 11/11 FP8; ADE vs the HF golden 0.026-0.112 m, same order as the reference's own eager-vs-sdpa noise (0.02-0.06 m).

## Ideogram 4

All commands run inside the same container (`customs/qwen_drive/docker_run.sh "<cmd>"`). Nothing has been executed yet — the container/image is not running, so only `py_compile` and YAML parsing have been verified on the host.

| What | Command | Notes |
|------|---------|-------|
| L1 unit + reference parity | `pytest tests/diffusion/models/ideogram4/test_ideogram4.py -v -m "core_model and cpu"` | 39 tests: config mapping, latent norm, schedule direction, presets, fp8 roundtrip, `Fp8Linear` buffers/dequant, sequence layout, text mask (padding-query self-attention), SP plan, `extra_args`, post-process range, streaming metadata. Auto-collected by CI (`test-ready.yml`), no per-model wiring |
| L2 serving contract | `pytest tests/e2e/online_serving/test_ideogram4.py -v -m "core_model and diffusion"` | JSON-only response, image part, `stream: true` -> 400. `local_model`: gated checkpoint, skipped without `IG_MODEL_DIR` or `VLLM_CI_HF_TOKEN` |
| L4 weekly expansion | `pytest tests/e2e/online_serving/test_ideogram4_expansion.py -v -m "slow and diffusion and H100 and cards_1" --run-level "full_model"` | clone vs keep, per-request override, `num_steps`/`guidance_schedule` length mismatch -> 400, 2-GPU USP. Wired in `test-weekly.yml` |
| HTTP 400 contract | `pytest tests/dfx/reliability/invalid_param_test/test_invalid_diffusion_chat.py -v -m "slow and H100 and cards_1"` | Lives in `invalid_param_test/` (chat route), not `e2e/online_serving/`, per the repo convention. Wired through `reliability_invalid_param_h100` / `_l4` |

Golden parity procedure (once a checkpoint and the reference snapshot are available):
1. Run the reference `Ideogram4Pipeline` on a fixed prompt/seed at 1024x1024 and save the decoded image.
2. Run the port through `examples/offline_inference/ideogram4.py` with the same seed, `dit_copies=keep`, no SP, no offload.
3. Compare pixel arrays with `atol 1e-3` on the `[-1, 1]` tensor before post-processing; report clone-vs-keep as a measured deviation, not as parity.
4. Record peak memory for `clone` and `keep` (the `keep` case is the one that needs layerwise offload on a 16 GB card).

Lint: changed files must pass `ruff check` and `ruff format --check` inside the venv.

## WeVisOne-VL

Runs through `customs/wevisone/docker_run.sh` (same image and shared `.venv` as
`customs/qwen_drive/docker_run.sh`, plus `--entrypoint bash` for images whose `ENTRYPOINT` is not a shell,
e.g. `vllm/vllm-openai`). The wrapper exports `PYTHONPATH=/workspace` (`QD_PYTHONPATH` to override) because
the shared venv has no editable install of `vllm_omni`.

| What | Command | Notes |
|------|---------|-------|
| L1 CPU unit tests | `bash customs/wevisone/run_tests.sh` | **Executed on this machine**: 311 passed / 8 deselected in the scoped set (the WeVisOne model tests, the stage-input-processor tests, the mask-choice, modality, stage-bridge, model_extras and helper assertion files). `tests/model_executor/models/wevisone` plus `test_wevisone.py` alone measured 127 passed / 4 deselected. Container deps installed on demand: `aenum omegaconf prettytable regex diffusers pytest_asyncio pytest_xdist pyyaml pytest_mock av soundfile opencc-python-reimplemented filelock psutil requests`; `opencc` is imported by `tests/helpers/media.py`, so without it `tests/helpers/tests/test_assertions.py` fails to import. Knobs: `WVS_MARKERS`, `WVS_GPUS`, `WVS_IMAGE`; extra args go to pytest |
| Golden parity | `WVS_MARKERS='local_model and cpu' bash customs/wevisone/run_tests.sh tests/model_executor/models/wevisone/test_wevisone_golden_parity.py` | 4 passed against `customs/wevisone/golden/matting/`. `--run-level` does not accept `local_model`; select with `-m` only |
| Reference oracle | `bash customs/wevisone/run_golden.sh` | writes `golden/<task>/result.json` + `masks/mask_NNN.npy`; masks differ run to run (~5e-4), so comparisons use tolerances (box 0.001, mean mask 0.02) |
| L2 online serving (2 GPUs) | `bash customs/wevisone/run_e2e_test.sh` | the test starts its own server; `WVS_GPUS` defaults to `device=0,1` because `wevisone_vl.yaml` puts stage 0 on device 0 and the decoder on device 1. **Executed on this machine**: 2 passed in 184.54 s (matting 109.74 s, server startup 98.036 s) with `WEVISONE_MODEL_PATH=/workspace/extra_repos/WeVisOne-VL-4B` |
| L4 expansion (2 GPUs) | `WVS_LEVEL=full_model bash customs/wevisone/run_e2e_test.sh tests/e2e/online_serving/test_wevisone_vl_expansion.py` | `WVS_LEVEL` sets both `-m` and `--run-level`; needs weights for `tencent/WeVisOne-VL-4B` or `WEVISONE_MODEL_PATH`. **Executed on this machine**: 4 passed in 121.28 s on 2x RTX 3060 with `WEVISONE_MODEL_PATH=/workspace/extra_repos/WeVisOne-VL-4B` (server startup ~80 s). The multi-object case answered with four `<OBJ>` queries inside one `<box>` span and returned four masks; the box-less case completed in ~1.7 s with no mask parts |

### Why the WeVisOne e2e is `advanced_model` only

`tests/helpers/stage_config.py` patches every stage to `load_format: dummy` for `core_model` runs, so a
`core_model` e2e serves random weights and the AR stage emits random tokens — there is no box span to
pair a mask with. The WeVisOne e2e therefore carries only `advanced_model`, and the `core_model` Buildkite
step for it was removed from `.buildkite/cuda/test-ready.yml`; `test-merge.yml` keeps the
`advanced_model and cuda` step. Quote from the test file: "Real weights are required: a core_model run
patches every stage to dummy weights, so the AR stage would emit random tokens and no box span to pair a
mask with."

Gates measured clean: `check_spdx_header.py`, `check_test_marks.py`, `check_buildkite.py`,
`check_forbidden_imports.py`.

Lint with the pre-commit pin (`ruff==0.14.10`, installed on demand in the shared venv): `ruff check` is
clean on every WeVisOne file, and `ruff format` reformatted 11 of them (typing modernisation `UP006`/
`UP035`/`UP045`/`UP037`, import sorting `I001`, two unused imports, one dead local, two long lines in
`customs/wevisone/`). Tests re-run after the fixes: still 224 passed / 8 deselected and 4 golden-parity
passed. Note that `ruff` 0.13 formats differently from the pinned 0.14.10, so use the pinned version.
