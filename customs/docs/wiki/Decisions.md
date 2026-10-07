# Decisions

## User decisions (fixed)
- Scope: VQA + direct planning + reasoning planning. No perception head.
- Planners: `planner-sft` and `planner-rl` both in the exported HF repo; selected per server via deploy yaml `model_config.planner_subfolder` or `--stage-overrides`.
- HF export excludes `perception/`. Must work with `--trust-remote-code`, without the `Qwen-Drive-1.0` package.
- Hardware: 2 GPUs (see Environment: 2x RTX 5060 Ti 16 GB). Tests run in docker with uv (`. /workspace/.venv/bin/activate`).
- API: `/v1/chat/completions`, images in messages, numeric inputs in `extra_body`, trajectory in a custom response field.
- Docs only in `customs/docs`; main docs/recipes untouched.

## Design decisions (revisable, log changes below)
- D1: Qwen-Drive config is NOT re-implemented in vllm-omni and stage 0 never imports `qwen_drive`. The nested `vlm_config` is unwrapped into a flat Qwen3.5 config for the VLM stages (callable `hf_overrides` or a flattened `hf_config_path`, E5 picks one); the diffusion stage reads `config.json` as plain JSON (Pi0 style), so vllm-omni works with or without `--trust-remote-code` (the flag is only needed for HF `AutoModel` use).
- D2: Pipeline = Stage 0 AR (reasoning/VQA text) -> Stage 1 AR prefill-only on closed-turn prompt (sends KV of 8 full-attn layers) -> Stage 2 diffusion expert (trajectory). Reason: vLLM does not compute KV of the last sampled token, but the reference appends `<|im_end|>\n` before planning.
- D3: ONE deploy yaml `qwen_drive.yaml` with 3 stages; routing by request `modalities` (`["text"]` = VQA, stops after stage 0; `["trajectory"]` = planning). Stage 0 `final_output_type=text`, stage 2 `final_output_type=trajectory`. (Changed from two yamls after spike S1.)
- D3b: Prompt is rendered by a dedicated chat template (+ `chat_template_kwargs` mode) and images are pre-resized by the client; no server-side prompt hook (spike S2: joined-text tokenization == reference ids). Direct mode: client sends `max_tokens=1` for stage 0.
- D4: KV layer filter (`kv_layer_indices` in `omni_kv_config`) added to the KV transfer manager; the layer->kv_caches index map must come from layer names, not identity (verify in E5). mRoPE anchor = position of the last token of the stage-1 prompt, delivered through `get_kv_transfer_metadata` -> `kv_metadata` (E5 chooses runner-state vs closed-form).
- D5: HF export path: `extra_repos/Qwen-Drive-1.0-4B-hf`.
- D6 (hardware-driven): each AR stage loads a 9 GB VLM, so stage 0 and stage 1 cannot share a 16 GB GPU. Place stage 0 on GPU 0, stage 1 on GPU 1, expert (2 GB) on GPU 0 or 1. Revisit if a single-AR-stage design (model runner appends the closed-turn tokens) proves feasible in S4.

## Change log
- 2026-10-07 (E5/E7): user asked to use fp8 for tests so the model fits 16 GB GPUs; BF16 evaluation will happen later on a larger-VRAM machine. Added `qwen_drive_fp8.yaml` (online `fp8_per_tensor` on the two VLM stages; planner stays BF16) used by all docker tests; `qwen_drive.yaml` is the BF16 deploy. Stage 1 is named `thinker` so the orchestrator forwards the processed image features (D2 unchanged otherwise). Prefix caching is disabled in both yamls (hybrid model).
- 2026-10-07 (E4): D1 refined, D3 changed to a single yaml, D3b added, D4 refined. D2/D6 confirmed (a single AR stage cannot append `<|im_end|>\n` to the KV; vLLM runs the VLM on one 16 GB GPU with 2.4 GiB of KV).
