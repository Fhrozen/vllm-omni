#!/usr/bin/env bash
# E1: print the facts the plan depends on (versions, GPUs, vllm Qwen3.5 VLM support).

set -e

SCRIPT_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")

"${SCRIPT_DIR}/docker_run.sh" '
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv
python - <<EOF
import torch, transformers, vllm
print("vllm", vllm.__version__, "transformers", transformers.__version__, "torch", torch.__version__, torch.version.cuda)
import vllm.model_executor.models.qwen3_5 as m
print("qwen3_5 classes:", [n for n in dir(m) if n.startswith("Qwen3_5")])
from vllm.model_executor.models.registry import ModelRegistry
print("registry has Qwen3_5ForConditionalGeneration:", "Qwen3_5ForConditionalGeneration" in ModelRegistry.get_supported_archs())
for mod in ("flash_attn", "fla", "causal_conv1d", "qwen_drive"):
    try:
        __import__(mod); print(mod, "OK")
    except Exception as e:
        print(mod, "MISSING", type(e).__name__)
EOF
'
