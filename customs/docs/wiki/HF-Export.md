# HF export (E3)

Output: `extra_repos/Qwen-Drive-1.0-4B-hf/` (git-ignored). Built by `customs/qwen_drive/export_hf.py` (stdlib only; hard-links weights, `--copy` to duplicate).
Verified by `customs/qwen_drive/verify_hf_export.py` (docker, GPU 0): remote-code load with `qwen_drive` blocked reproduces the E2 golden bit-for-bit (VQA text, reasoning text, trajectories for both planners, direct and reasoning).

## Layout
- Weights: `model.safetensors` (VLM, keys `vlm.model.*`), `planner-sft/`, `planner-rl/` (keys `planning_expert.*`). No `perception/`.
- Tokenizer/templates/configs copied unchanged; `config.json` gets `auto_map` for `AutoConfig`, `AutoModel`, `AutoModelForImageTextToText`.
- Code (relative imports only): `configuration_qwen_drive.py`, `modeling_qwen_drive.py`, `planning_expert.py`, `scene.py`, `trajectory.py`.
- `README.md`: original model card plus a self-contained-export note.

## Differences from the original code
Only `modeling_qwen_drive.py` changes (patches asserted to apply exactly once in `export_hf.py`):
- `_resolve_planner(planner, base)`: `planner` may be an existing directory, a sub-folder of the local model dir (`"planner-rl"`), or a sub-folder of a hub repo (`hf_hub_download`).
- `from_pretrained(..., planner=...)` uses it.

## Usage
```python
from transformers import AutoModel
model = AutoModel.from_pretrained("extra_repos/Qwen-Drive-1.0-4B-hf", trust_remote_code=True,
                                  planner="planner-rl", dtype="bfloat16", attn_implementation="sdpa").cuda()
```
Scene loading (`read_scene_file`) lives in the original repo (`qwen_drive.benchmarks`) and is not shipped; build `DrivingScene` objects directly from `scene.py` or use the vllm-omni client (E8).

## Notes
- For vllm-omni, `trust_remote_code` is needed only to resolve `QwenDriveConfig` (nested `vlm_config`); the model class comes from vllm-omni's own registry (E5).
- Upload to the hub requires real files: run with `--copy` (hard links are fine locally).
