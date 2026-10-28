# Environment

## Hardware (host nelson-lab02)
- 2x NVIDIA GeForce RTX 3060, 12288 MiB each, compute capability 8.6 (Ampere, sm_86), driver 580.95.05.
  Verified with `nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv`.
- Implications: a 4B bf16 VLM is ~9.2 GB of weights, so one stage fits per card with
  `gpu_memory_utilization` ~0.85-0.90 and a small `max_model_len`; keep `max_num_seqs` at 1 for the
  tightest memory budget. `RTX3060` is registered as a CUDA SKU marker in `pyproject.toml`, but
  `.buildkite/common/ci_mirror_hardwares.yml` has no consumer-GPU pool, so `RTX3060` tests are
  local-only and CI still selects them under the H100/B200 markers.
- Earlier measurements (qwen_drive, ideogram) were taken on host nelson-lab03 with 2x RTX 5060 Ti
  16 GB (sm_120); the pass criteria quoted for those models in [Testing](Testing.md) come from that
  machine and have not been re-measured here.

## Docker
- Image: `fhrozen/vllm-omni:builder-cuda13-u24` (from `docker/Dockerfile.dev`; uv, HF_HOME=/workspace/.hf_cache, UV_CACHE_DIR=/workspace/.uv_cache).
- Repo mounted at `/workspace`; venv at `/workspace/.venv` (python lives in `/workspace/.uv-bin`, only valid inside docker).
- Do not use the host `.venv/bin/python`.
- Fallback when the builder image cannot be pulled (Docker Hub 401, as on the current host): the local
  `vllm/vllm-openai:latest` works with the shared `.venv` through the mount. It has
  `ENTRYPOINT ["vllm","serve"]`, so `--entrypoint bash` is required — that is what
  `customs/wevisone/docker_run.sh` does. In that image `pip install -e ".[dev]"` fails and
  `pyproject.toml[dev]` does not install the dev extras, so the pytest plugins
  (`pytest-asyncio`, `pytest-cov`, `pytest-mock`, `pytest-shard`, `pytest-xdist`, `pytest-subtests`)
  and `aenum`/`omegaconf` have to be installed individually; `python` is absent, use `python3`, and set
  `PYTHONPATH=/workspace`.

## Scripts (`customs/qwen_drive/`)
- `docker_run.sh "<cmd>"`: run a command in docker with the venv active. Env: `QD_GPUS` (default `all`), `QD_IMAGE`, `QD_EXTRA_DOCKER_ARGS`.
- `setup_env.sh`: venv + vllm 0.31.0 + `pip install -e .[dev]` + reference `qwen_drive` (`--no-deps`).
- `env_check.sh`: prints versions, GPUs and vllm Qwen3.5 support.

## Gotchas
- The container runs as root: files created under `/workspace` are root-owned on the host. Use `trap 'chown -R <uid>:<gid> <paths>' EXIT` in wrappers (see `run_ref_baseline.sh`) or `docker_run.sh "chown ..."`.
- Pin one GPU with `QD_GPUS='"device=0"'` (note the inner quotes).

## Installed versions (E1)
- vllm 0.31.0, transformers 5.14.1, torch 2.13.0+cu132 (CUDA 13.2), python 3.12.
- Missing: flash_attn, fla (flash-linear-attention), causal_conv1d. vLLM has its own GDN kernels; only the HF reference path is affected.

## Paths
- Reference code: `extra_repos/Qwen-Drive-1.0` (src/qwen_drive, scripts/demo.py, data/demo/{planning_scenes.jsonl,frames/}).
- Original checkpoint: `extra_repos/Qwen-Drive-1.0-4B` (cloned from HF `Qwen/Qwen-Drive-1.0-4B`).
- Export target: `extra_repos/Qwen-Drive-1.0-4B-hf` (E3).
