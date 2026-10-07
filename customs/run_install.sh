#!/usr/bin/env bash

set -e

THISPATH=$(realpath "$(pwd)")
WORKDIR=$(realpath "${THISPATH}/..")

docker_tag=fhrozen/vllm-omni:builder-cuda13-u24

if [ ! "$(docker images -q "$docker_tag")" ]; then
    echo "Docker image does not exist, building it..."
    cd "$WORKDIR"
    docker build -f docker/Dockerfile.dev -t "$docker_tag" .
    cd "$THISPATH"
fi

docker run --gpus all --rm -v "$WORKDIR":/workspace "$docker_tag" bash -c "uv venv --python 3.12 --seed; \
    source .venv/bin/activate; \
    uv pip install vllm==0.31.0 --torch-backend=auto \
        --extra-index-url https://wheels.vllm.ai/db9527a46873454610df6dbedf79a36d6bf1a7f6; \
    uv pip install -e .; uv pip list"
