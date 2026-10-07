#!/usr/bin/env bash
# Serve Qwen-Drive with vllm-omni inside docker (host network, port 8091).
# Usage: customs/qwen_drive/serve.sh [deploy-config]   (default: the FP8 deploy for 16 GB GPUs)

set -e

SCRIPT_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
DEPLOY=${1:-/workspace/vllm_omni/deploy/qwen_drive_fp8.yaml}
MODEL=${QD_MODEL:-/workspace/extra_repos/Qwen-Drive-1.0-4B-hf}

"${SCRIPT_DIR}/docker_run.sh" "vllm serve ${MODEL} --omni --deploy-config ${DEPLOY} \
    --chat-template ${MODEL}/chat_template_drive.jinja --trust-remote-code --port 8091 \
    --stage-init-timeout 900"
