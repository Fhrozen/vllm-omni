#!/usr/bin/env bash
#
# Install vllm-omni into the base image's Python environment.
#
# The base image (vllm/vllm-openai) already ships vllm, torch and the CUDA
# runtime, so this script only installs vllm-omni itself (non-editable) into
# the system interpreter. It is invoked from docker/Dockerfile.custom with the
# working directory set to the copied source tree.
#
set -euo pipefail

# Resolve the base image's system interpreter (the one that already has vllm).
PY="$(python3 -c 'import sys; print(sys.executable)')"
echo "Installing vllm-omni into: ${PY}"

# Non-editable install so the resulting image is self-contained: no source
# directory is required at runtime. vllm + torch + CUDA come from the base image.
uv pip install --python "${PY}" --no-cache-dir .

# Sanity checks: fail the build if the install did not produce a working stack.
# Note: importing vllm_omni registers the omni archs into vllm's model registry
# via the `vllm.general_plugins` entry point, so a successful import is the
# meaningful check (the registry class itself is vllm's private `_ModelRegistry`).
"${PY}" - <<'PYCHECK'
import vllm
import vllm_omni
from vllm_omni.model_executor.models.registry import _OMNI_MODELS

print(f"vllm {vllm.__version__} | vllm_omni {vllm_omni.__version__}")
print(f"omni archs defined: {len(_OMNI_MODELS)}")
PYCHECK

echo "vllm-omni install complete."
