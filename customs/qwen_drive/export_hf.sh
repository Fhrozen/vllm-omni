#!/usr/bin/env bash
# E3: export the HF repo (host, stdlib only) then verify it with remote code in docker (GPU 0).
# Usage: customs/qwen_drive/export_hf.sh [--copy]

set -e

SCRIPT_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")

python3 "${SCRIPT_DIR}/export_hf.py" "$@"
QD_GPUS='"device=0"' "${SCRIPT_DIR}/docker_run.sh" "python customs/qwen_drive/verify_hf_export.py"
