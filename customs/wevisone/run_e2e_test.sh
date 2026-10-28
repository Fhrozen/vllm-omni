#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Run a WeVisOne-VL online-serving e2e test in docker (the test starts its own server).
# Extra args go to pytest.
# Usage: customs/wevisone/run_e2e_test.sh [test file] [pytest args]
# Env:   WVS_LEVEL (pytest -m / --run-level, default "advanced_model"),
#        WVS_MODEL (model path inside the container), WVS_GPUS (docker --gpus value),
#        WVS_IMAGE (image tag).
#
# Defaults run the L2 baseline with the advanced_model gate, mirroring
# customs/qwen_drive/run_e2e_test.sh. The full pipeline needs two GPUs because stage 0
# (Qwen3-VL) and stage 1 (the pixel decoder) each take one device in
# vllm_omni/deploy/wevisone_vl.yaml, so WVS_GPUS defaults to device=0,1; the text-only
# case in the same file needs one.
#
# Examples:
#   bash customs/wevisone/run_e2e_test.sh
#   bash customs/wevisone/run_e2e_test.sh tests/e2e/online_serving/test_wevisone_vl_expansion.py
#   WVS_LEVEL=full_model bash customs/wevisone/run_e2e_test.sh tests/e2e/online_serving/test_wevisone_vl_expansion.py
#   bash customs/wevisone/run_e2e_test.sh -k mask_index

set -e

SCRIPT_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")

FILE=${1:-tests/e2e/online_serving/test_wevisone_vl.py}
shift 2>/dev/null || true

LEVEL=${WVS_LEVEL:-advanced_model}
MODEL=${WVS_MODEL:-/workspace/extra_repos/WeVisOne-VL-4B}
export WVS_GPUS=${WVS_GPUS:-device=0,1}

"${SCRIPT_DIR}/docker_run.sh" \
    "WEVISONE_MODEL_PATH=${MODEL} python -m pytest -v -s -p no:cacheprovider --run-level=${LEVEL} ${FILE} -m '${LEVEL} and cuda' $*"
