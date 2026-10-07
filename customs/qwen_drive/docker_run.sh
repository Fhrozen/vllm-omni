#!/usr/bin/env bash
# Run a command inside the builder docker image with the repo mounted at /workspace and the uv venv active.
# Usage: customs/qwen_drive/docker_run.sh "<command>"
# Env: QD_IMAGE (image tag), QD_GPUS (docker --gpus value, default "all"), QD_EXTRA_DOCKER_ARGS.

set -e

if [ $# -lt 1 ]; then
    echo "usage: $0 \"<command>\"" >&2
    exit 1
fi

SCRIPT_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
WORKDIR=$(realpath "${SCRIPT_DIR}/../..")
docker_tag=${QD_IMAGE:-fhrozen/vllm-omni:builder-cuda13-u24}
gpus=${QD_GPUS:-all}

if [ ! "$(docker images -q "$docker_tag")" ]; then
    echo "Docker image does not exist, building it..."
    (cd "$WORKDIR" && docker build -f docker/Dockerfile.dev -t "$docker_tag" .)
fi

# shellcheck disable=SC2086
docker run --gpus "$gpus" --rm --ipc=host ${QD_EXTRA_DOCKER_ARGS} \
    -v "$WORKDIR":/workspace "$docker_tag" \
    bash -c "cd /workspace && . /workspace/.venv/bin/activate && $1"
