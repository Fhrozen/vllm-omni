#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Serve WeVisOne-VL with vllm-omni inside docker (host network, port 8091).
# Usage: customs/wevisone/serve.sh [deploy-config]

# Two stages: the Qwen3-VL thinker on GPU 0 and the pixel decoder on GPU 1.
# On a single GPU, run the text-only pipeline instead (--pipeline wevisone_vl_text_only),
# or edit the deploy yaml so both stages use the same device -- masks need the
# decoder stage either way.
set -e

SCRIPT_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
DEPLOY=${1:-/workspace/vllm_omni/deploy/wevisone_vl.yaml}
MODEL=${WVS_MODEL:-/workspace/extra_repos/WeVisOne-VL-4B}

"${SCRIPT_DIR}/../qwen_drive/docker_run.sh" "vllm serve ${MODEL} --omni --deploy-config ${DEPLOY} \
    --port 8091 --stage-init-timeout 900"
