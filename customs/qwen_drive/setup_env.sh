#!/usr/bin/env bash
# E1: create /workspace/.venv (if missing) and install vllm, vllm-omni (editable) and the Qwen-Drive reference package.
# The reference package is only used for parity/golden generation, never by vllm-omni itself.

set -e

SCRIPT_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
WORKDIR=$(realpath "${SCRIPT_DIR}/../..")
docker_tag=${QD_IMAGE:-fhrozen/vllm-omni:builder-cuda13-u24}

if [ ! "$(docker images -q "$docker_tag")" ]; then
    (cd "$WORKDIR" && docker build -f docker/Dockerfile.dev -t "$docker_tag" .)
fi

docker run --gpus all --rm -v "$WORKDIR":/workspace --ipc=host "$docker_tag" bash -c "cd /workspace; \
    [ -x .venv/bin/python ] || uv venv --python 3.12 --seed; \
    source .venv/bin/activate; \
    python -c 'import vllm' 2>/dev/null || uv pip install vllm==0.31.0 --torch-backend=auto \
        --extra-index-url https://wheels.vllm.ai/db9527a46873454610df6dbedf79a36d6bf1a7f6; \
    uv pip install -e '.[dev]'; \
    uv pip install --no-deps -e extra_repos/Qwen-Drive-1.0; \
    uv pip list"
