# Decisions

## User decisions (fixed)
- Scope: VQA + direct planning + reasoning planning. No perception head.
- Planners: `planner-sft` and `planner-rl` both in the exported HF repo; selected per server via deploy yaml `model_config.planner_subfolder` or `--stage-overrides`.
- HF export excludes `perception/`. Must work with `--trust-remote-code`, without the `Qwen-Drive-1.0` package.
- Hardware: 2 GPUs (see Environment: 2x RTX 5060 Ti 16 GB). Tests run in docker with uv (`. /workspace/.venv/bin/activate`).
- API: `/v1/chat/completions`, images in messages, numeric inputs in `extra_body`, trajectory in a custom response field.
- Docs only in `customs/docs`; main docs/recipes untouched.

## Design decisions (revisable, log changes below)
- D1: Qwen-Drive config is NOT re-implemented in vllm-omni; loaded through `trust_remote_code` (BAGEL precedent). Stage 0 never imports `qwen_drive`.
- D2: Pipeline = Stage 0 AR (reasoning/VQA text) -> Stage 1 AR prefill-only on closed-turn prompt (sends KV of 8 full-attn layers) -> Stage 2 diffusion expert (trajectory). Reason: vLLM does not compute KV of the last sampled token, but the reference appends `<|im_end|>\n` before planning.
- D3: Two deploy yamls: `qwen_drive_vqa.yaml` (1 stage) and `qwen_drive_plan.yaml` (3 stages). Revisit after spike S1.
- D4: KV layer filter (`kv_layer_indices` in `omni_kv_config`) added to the KV transfer manager; mRoPE anchor via `custom_metadata` (spike S3).
- D5: HF export path: `extra_repos/Qwen-Drive-1.0-4B-hf`.
- D6 (hardware-driven): each AR stage loads a 9 GB VLM, so stage 0 and stage 1 cannot share a 16 GB GPU. Place stage 0 on GPU 0, stage 1 on GPU 1, expert (2 GB) on GPU 0 or 1. Revisit if a single-AR-stage design (model runner appends the closed-turn tokens) proves feasible in S4.

## Change log
- (none yet)
