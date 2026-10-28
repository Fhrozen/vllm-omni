#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Generate WeVisOne-VL golden fixtures with the reference implementation.
# Usage: customs/wevisone/run_golden.sh [extra args passed to ref_golden.py]

set -e

SCRIPT_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
WORKDIR=$(realpath "${SCRIPT_DIR}/../..")

cd "$WORKDIR"
"${SCRIPT_DIR}/docker_run.sh" \
    "python customs/wevisone/ref_golden.py --decode-masks $*"
