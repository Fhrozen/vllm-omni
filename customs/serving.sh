#!/usr/bin/env bash

set -e

. .venv/bin/activate

vllm serve Tongyi-MAI/Z-Image-Turbo \
  --omni \
  --port 8080 \
  --stage-overrides '{"0": {"tensor_parallel_size": 2, "devices": "0,1", "quantization": "fp8"}}'
