#!/usr/bin/env bash
# Serve Ideogram 4 with vllm-omni inside docker (host network, port 8091).
#
# Usage: docs/ideogram-4/serve.sh [deploy-config]
#   default: docs/ideogram-4/deploy_ideogram4_fp8.yaml
#
# Startup knobs go through --custom-pipeline-args (there is no --extra-args CLI flag):
#   --custom-pipeline-args '{"dit_copies": "clone"}'   # one shared DiT (default, fits 16 GB)
#   --custom-pipeline-args '{"dit_copies": "keep"}'   # two instances, reference parity
# The same keys also work as `extra_args` on a request, per request.
#
# Layerwise offload for a single 16 GB GPU:
#   --diffusion-offload-config '{"mode":"layer","components":["dit"]}'

set -e

REPO=${REPO:-/workspace}
DEPLOY=${1:-${REPO}/docs/ideogram-4/deploy_ideogram4_fp8.yaml}
MODEL=${IG4_MODEL:-ideogram-ai/ideogram-4-fp8}
DIT_COPIES=${DIT_COPIES:-clone}

"${REPO}/customs/qwen_drive/docker_run.sh" "vllm serve ${MODEL} --omni \
    --deploy-config ${DEPLOY} \
    --custom-pipeline-args '{\"dit_copies\": \"${DIT_COPIES}\"}' \
    --port 8091 --stage-init-timeout 900"
