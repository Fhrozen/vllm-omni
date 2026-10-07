# E3: Single HF repo with remote code

Depends on: E2 (parity check). Parallel with E4.

## Goal
One directory that works with `AutoModel.from_pretrained(..., trust_remote_code=True)` and `vllm serve --trust-remote-code` without the `Qwen-Drive-1.0` package. Perception excluded.

## Inputs
- Code: `extra_repos/Qwen-Drive-1.0/src/qwen_drive/{configuration_qwen_drive,modeling_qwen_drive,planning_expert,scene,trajectory}.py`
- Checkpoint: `extra_repos/Qwen-Drive-1.0-4B` (all except `perception/`, `.git`, `.DS_Store`, `.ms_upload_cache`)
- Golden: `customs/docs/golden/` from E2

## Outputs
- `customs/qwen_drive/export_hf.py`, `export_hf.sh`
- `extra_repos/Qwen-Drive-1.0-4B-hf/` containing: weights (symlink or copy, flag), tokenizer/chat template/preprocessor/generation configs, `planner-sft/`, `planner-rl/`, flat code files with relative imports only, updated `config.json` with `auto_map` (`AutoConfig`, `AutoModel`, `AutoModelForImageTextToText`), README model card note
- `customs/docs/wiki/HF-Export.md`: layout, how loading works, differences from the original code

## Changes to the copied code
- Remove imports of the `qwen_drive` package namespace (`from .x import y` only between shipped files). `scene.py` must not import dropped modules.
- `load_planner` accepts a subfolder name relative to the repo (`planner="planner-rl"`), resolved with `snapshot_download`/`name_or_path` for hub ids.
- `flash_attn` import stays lazy; fallback SDPA (already the case).
- Keep behavior identical otherwise.

## Verify (in a venv/env WITHOUT `qwen_drive` installed; e.g. `uv pip uninstall qwen-drive` in a scratch venv or `PYTHONPATH` check)
- `AutoConfig.from_pretrained(path, trust_remote_code=True)` and `AutoModel.from_pretrained(path, trust_remote_code=True, planner="planner-rl")`.
- Output matches E2 golden (direct, reasoning, VQA).
- `python -c "import qwen_drive"` fails in that env.

## Exit criteria
Parity with golden, wiki page written, Progress.md updated.
