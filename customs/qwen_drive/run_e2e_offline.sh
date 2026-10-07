#!/usr/bin/env bash
# E5/E7: offline run of the 3-stage pipeline vs the golden trajectories (needs both GPUs). Extra args go to e2e_offline.py.
# Usage: customs/qwen_drive/run_e2e_offline.sh [--deploy-config /workspace/vllm_omni/deploy/qwen_drive.yaml] [--modes direct]

set -e

SCRIPT_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")

"${SCRIPT_DIR}/docker_run.sh" "python customs/qwen_drive/e2e_offline.py $*"
