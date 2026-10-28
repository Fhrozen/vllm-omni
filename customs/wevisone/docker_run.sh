#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Run a command inside the container with the shared venv, like customs/qwen_drive/docker_run.sh.
# Usage: customs/wevisone/docker_run.sh "<command>"
# Env:   WVS_IMAGE      image tag (defaults to QD_IMAGE, then the builder image)
#        WVS_GPUS       docker --gpus value, default "device=0". A comma-separated
#                       device list must reach docker inside quotes, otherwise docker
#                       splits it into separate specs and reads the second part as a
#                       device count ("cannot set both Count and DeviceIDs").
#        WVS_ENTRYPOINT default "bash"; needed because vllm/vllm-openai has
#                       ENTRYPOINT ["vllm","serve"], which would otherwise swallow the command.
#        QD_PYTHONPATH  PYTHONPATH inside the container, default "/workspace", so the
#                       mounted source is importable without an editable install.
# The shared .venv is created once with customs/qwen_drive/setup_env.sh and is visible
# through the workspace mount, so the same environment is used by every customs/ script.
#
# Example with the local image:
#   WVS_IMAGE=vllm/vllm-openai:latest customs/wevisone/docker_run.sh \
#       'python -m pytest -q -m "core_model and cpu" tests/model_executor/stage_input_processors/test_wevisone.py'

set -e

SCRIPT_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
WORKDIR=$(realpath "${SCRIPT_DIR}/../..")

IMAGE=${WVS_IMAGE:-${QD_IMAGE:-fhrozen/vllm-omni:builder-cuda13-u24}}
if ! docker image inspect "${IMAGE}" > /dev/null 2>&1; then
    echo "Image '${IMAGE}' is not available locally." >&2
    echo "Build it with customs/qwen_drive/build_image.sh, or point WVS_IMAGE at an image" >&2
    echo "that already contains vllm + vllm_omni (e.g. vllm/vllm-openai:latest)." >&2
    exit 1
fi

# shellcheck disable=SC2086
docker run \
    --gpus "\"${WVS_GPUS:-device=0}\"" \
    --rm \
    --ipc=host \
    --network host \
    ${QD_EXTRA_DOCKER_ARGS} \
    --entrypoint "${WVS_ENTRYPOINT:-bash}" \
    --env "PYTHONPATH=${QD_PYTHONPATH:-/workspace}" \
    -v "${WORKDIR}":/workspace \
    "${IMAGE}" \
    -c "cd /workspace && . /workspace/.venv/bin/activate && $1"
