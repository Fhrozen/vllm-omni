#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Run the WeVisOne-VL CPU unit tests in docker (single GPU pin). Extra args go to pytest.
# Usage: customs/wevisone/run_tests.sh [pytest args]
# Env: WVS_MARKERS (pytest -m expression, default "core_model and cpu"),
#      WVS_GPUS (docker --gpus value, default "device=0"), WVS_IMAGE (image tag).
# Runs through customs/wevisone/docker_run.sh, which uses the shared .venv created by
# customs/qwen_drive/setup_env.sh. When the builder image is not available locally, point
# WVS_IMAGE at an image that already carries vllm + vllm_omni, e.g.
#   WVS_IMAGE=vllm/vllm-openai:latest customs/wevisone/run_tests.sh
#
# Golden parity is marked local_model and needs the checkpoint plus the committed
# fixtures, so it is deselected by default. Run it by switching the marker expression
# (the other files then contribute nothing, so the run reduces to the parity file):
#   WVS_MARKERS='local_model and cpu' customs/wevisone/run_tests.sh
# (--run-level only accepts core_model/advanced_model/full_model; it selects the
#  assertion strictness of the runtime helpers, not marker-based collection.)

set -e

SCRIPT_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")

MARKERS=${WVS_MARKERS:-core_model and cpu}
FILES="tests/model_executor/models/wevisone \
    tests/model_executor/stage_input_processors/test_wevisone.py \
    tests/engine/test_data_entry_keys.py \
    tests/engine/test_output_modality.py \
    tests/entrypoints/openai_api/test_modality_validation.py \
    tests/entrypoints/openai_api/test_serving_chat_mask_choices.py \
    tests/model_extras/test_model_extras.py"

WVS_GPUS=${WVS_GPUS:-device=0} "${SCRIPT_DIR}/docker_run.sh" \
    "python -m pytest -v -p no:cacheprovider -m '${MARKERS}' ${FILES} $*"
