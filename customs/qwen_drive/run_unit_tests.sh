#!/usr/bin/env bash
# Run the Qwen-Drive unit/parity tests in docker (GPU 0). Extra args go to pytest.
# Usage: customs/qwen_drive/run_unit_tests.sh [pytest args]

set -e

SCRIPT_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")

QD_GPUS='"device=0"' "${SCRIPT_DIR}/docker_run.sh" "python -m pytest -v -p no:cacheprovider \
    tests/diffusion/models/qwen_drive $*"
