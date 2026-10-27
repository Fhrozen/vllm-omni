#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Run samples.json against a running server; start it first with customs/wevisone/serve.sh.
# Usage: customs/wevisone/run_samples.sh [samples.json] [out-dir] [extra args...]

set -e

SCRIPT_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
JSON=${1:-/workspace/customs/wevisone/samples.json}
OUT=${2:-/workspace/customs/wevisone/output}
shift $(( $# < 2 ? $# : 2 ))

"${SCRIPT_DIR}/../qwen_drive/docker_run.sh" \
    "python customs/wevisone/generate_samples.py --json ${JSON} --out-dir ${OUT} $*"

# Review the overlays afterwards:
#   python customs/wevisone/visualize.py --result ${OUT}/matting
