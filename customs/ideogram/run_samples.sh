#!/usr/bin/env bash
# Generate Ideogram 4 samples from a JSON list inside the container.
# Usage: customs/ideogram/run_samples.sh [json] [out-dir] [extra args passed to the script]
#
# The spec and the output directory are shared with the host, so images written
# under /workspace/... appear in the repo checkout.
set -e

SCRIPT_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
WORKDIR=$(realpath "${SCRIPT_DIR}/../..")
JSON=${1:-/workspace/customs/ideogram/samples.json}
OUT=${2:-/workspace/customs/ideogram/output}
shift 2 2>/dev/null || true

"${WORKDIR}/customs/qwen_drive/docker_run.sh" \
    "python customs/ideogram/generate_samples.py --json ${JSON} --out-dir ${OUT} $*"
