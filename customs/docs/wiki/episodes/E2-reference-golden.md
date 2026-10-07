# E2: Reference golden outputs

Depends on: E1. Status: see Progress.md.

## Goal
Produce reproducible reference outputs from the original `qwen_drive` package, used as the truth for every later parity test.

## Inputs
- `extra_repos/Qwen-Drive-1.0/scripts/demo.py` (usage pattern), `src/qwen_drive/*` (reference API)
- `extra_repos/Qwen-Drive-1.0/data/demo/planning_scenes.jsonl`, `data/demo/frames/` (sample scenes + images)
- `extra_repos/Qwen-Drive-1.0-4B` (VLM + `planner-sft`, `planner-rl`)

## Outputs
- `customs/qwen_drive/ref_golden.py` and `customs/qwen_drive/run_ref_baseline.sh` (docker wrapper)
- `customs/docs/golden/` (small files only): per scene `scene_<i>.json` (inputs summary), `traj_{sft,rl}_{direct,reasoning}.npy` ([6,50,3]), `reasoning_<i>.txt`, `vqa_<i>.txt`, `prompt_ids_<i>_{direct,reasoning}.npy`, `image_grid_thw_<i>.npy`, `anchor_<i>.json`
- Large artifacts (K/V of the 8 full-attention layers, post-rotary) go to `extra_repos/golden_kv/` (not committed)
- Findings.md entry: safetensors key names (prefix `vlm.`?), `driving_command` length, token counts per scene, timings, GPU memory.

## Steps
1. Write `ref_golden.py`: load model with `QwenDriveForPlanning.from_pretrained(path, planner=..., dtype=bfloat16, attn_implementation="sdpa").cuda()` (single 16 GB GPU is enough), load scenes as `demo.py` does.
2. For N=3 scenes: direct planning (`num_samples=6, seed=42`) for both planners; reasoning planning for planner-rl and planner-sft; VQA with default params. Save prompt ids, grid, anchor (`_rope_positions(...)[:, :, -1]`), K/V via `_scene_cache`.
3. Dump safetensors header keys of `model.safetensors` and a planner file (prefixes, dtypes) to Findings.md.
4. Run twice and assert identical outputs (determinism).

## Exit criteria
Golden files exist, second run is bit-identical, Progress.md updated.
