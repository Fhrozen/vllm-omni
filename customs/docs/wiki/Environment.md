# Environment

## Hardware (host nelson-lab03)
- 2x NVIDIA GeForce RTX 5060 Ti, 16311 MiB each, compute capability 12.0 (Blackwell, sm_120).
- Implications: VLM bf16 weights = 8.5 GB (`model.safetensors`), each planner = 2.0 GB. Tight memory; keep `max_model_len` small and `gpu_memory_utilization` ~0.85.

## Docker
- Image: `fhrozen/vllm-omni:builder-cuda13-u24` (from `docker/Dockerfile.dev`; uv, HF_HOME=/workspace/.hf_cache, UV_CACHE_DIR=/workspace/.uv_cache).
- Repo mounted at `/workspace`; venv at `/workspace/.venv` (python lives in `/workspace/.uv-bin`, only valid inside docker).
- Do not use the host `.venv/bin/python`.

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
