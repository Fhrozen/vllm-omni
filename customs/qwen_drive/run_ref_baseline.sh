#!/usr/bin/env bash
# E2: generate golden outputs twice (determinism check). GPU 0 only (VLM + expert fit in 16 GB).
# Usage: customs/qwen_drive/run_ref_baseline.sh [extra ref_golden.py args]

set -e

SCRIPT_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
OWNER="$(id -u):$(id -g)"

# The container runs as root; hand the outputs back to the host user even on failure.
QD_GPUS='"device=0"' "${SCRIPT_DIR}/docker_run.sh" "\
trap 'chown -R ${OWNER} customs/docs/golden extra_repos/golden_kv extra_repos/golden_rerun extra_repos/golden_kv_rerun 2>/dev/null' EXIT; \
rm -rf customs/docs/golden extra_repos/golden_kv extra_repos/golden_rerun extra_repos/golden_kv_rerun; \
python customs/qwen_drive/ref_golden.py $* && \
python customs/qwen_drive/ref_golden.py --out /workspace/extra_repos/golden_rerun --kv-out /workspace/extra_repos/golden_kv_rerun $* && \
python customs/qwen_drive/ref_golden.py --compare /workspace/customs/docs/golden /workspace/extra_repos/golden_rerun"
