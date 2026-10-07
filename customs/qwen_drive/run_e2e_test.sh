#!/usr/bin/env bash
# Run the Qwen-Drive online e2e test (starts its own server; needs both GPUs). Extra args go to pytest.
# Usage: customs/qwen_drive/run_e2e_test.sh [-k direct] ;  BF16: QD_DEPLOY=/workspace/vllm_omni/deploy/qwen_drive.yaml ...

set -e

SCRIPT_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
EXTRA_ENV=""
for var in QD_DEPLOY QD_PLANNER QD_MAX_ADE; do
    [ -n "${!var}" ] && EXTRA_ENV="${EXTRA_ENV} ${var}=${!var}"
done

"${SCRIPT_DIR}/docker_run.sh" "${EXTRA_ENV} python -m pytest -v -s -p no:cacheprovider --run-level=advanced_model tests/e2e/online_serving/test_qwen_drive.py $*"
