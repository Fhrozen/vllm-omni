# Findings (append-only)

## Reference model (read in planning phase)
- `QwenDriveForPlanning(PreTrainedModel)` = `vlm` (`AutoModelForImageTextToText` from `vlm_config`, qwen3_5) + `planning_expert`. `base_model_prefix="vlm"`. File: `extra_repos/Qwen-Drive-1.0/src/qwen_drive/modeling_qwen_drive.py`.
- Checkpoint layout: `model.safetensors` (VLM only; expert reported missing on purpose), `planner-{sft,rl}/model.safetensors` (keys optionally prefixed `planning_expert.`, stripped by `load_planner`, strict load), `perception/` (ignored).
- VLM text: hidden 2560, 32 layers, 16 heads, 4 KV heads, head_dim 256, hybrid linear/full attention (`layer_types`), 8 full-attention layers, partial_rotary 0.25, mrope_section [11,11,10], vocab 248320. Vision: depth 24, hidden 1024, patch 16, merge 2, temporal patch 2, out 2560.
- Expert: 32 layers, hidden 1024, 16 heads/4 KV heads/head_dim 256, mlp 3584, `layers_per_kv=4` (expert layer i attends over VLM cache index i//4 = i-th full-attention layer group), adaLN conditioning (time + nav + ego), joint attention over [scene K/V ; waypoint K/V], non-causal.
- Sampling: 10 Euler steps, x-prediction (endpoint) -> velocity `(endpoint - x)/max(1-t, 0.1)`, noise `randn` per sample from `torch.Generator(seed + k)`, default seed 42. Trajectory scale `[165, 25, 1.5703125]`.
- Numerical quirks that must be reproduced: Fourier frequency table and rotary inv_freq are computed in the module dtype (bf16 rounding); positions cast to that dtype.
- Prompt is built from token ids by `QwenDriveProcessor.build_input_ids` (scene.py), NOT through the chat template: segments tokenized separately; per view `<FRONT VIEW>`, `<FRONT LEFT VIEW>`, `<FRONT RIGHT VIEW>`, then `frame: i` + vision tokens for each of 4 frames; instruction text; direct mode closes an empty assistant turn, reasoning mode appends `REASONING_REQUEST` and leaves the assistant header open.
- Images: per-frame `target_size` resize, then snap to multiple of 32 within budget (current 921600 px, history 174080 px), normalized mean=std=0.5, patches built manually (`encode_images`).
- Reasoning mode: greedy `generate` (min 10, max 256 tokens, stop at `<|im_end|>`/eos), then `<|im_end|>` + `\n` forwarded through the VLM to extend the cache; anchor = last prompt rope position + len(closed_turn).
- Planner input: history `[16,3]`, history_velocity/acceleration `[16,2]`, `ego_status` = concat(ego_velocity 2, ego_acceleration 2, driving_command) with `ego_status_dim=8` so `driving_command` should have 4 entries (confirm on demo data in E2), `nav_command` int 0..2.

## vllm-omni
- KV transfer (`vllm_omni/distributed/omni_connectors/kv_transfer_manager.py`): sends at request finish, all layers, no layer filter; `custom_metadata` supported (`vllm_omni/worker/gpu_ar_model_runner.py` ~L857-886).
- Pi0 (`vllm_omni/diffusion/models/pi0/pipeline_pi0.py`): self-loading diffusion pipeline, `final_output_type="action"`, request data in `sampling_params.extra_args`.
- Subfolder selection for diffusion weights: `lingbot_video` reads `od_config.model_config[...]`.
- Diffusion registry: `vllm_omni/diffusion/registry.py` (`_DIFFUSION_MODELS`), AR registry: `vllm_omni/model_executor/models/registry.py`, pipelines: `vllm_omni/config/pipeline_registry.py`.

## Golden data (E2, 2026-10-07)
- Demo file has 4 scenes; golden covers scenes 0-2. Image root = `extra_repos/Qwen-Drive-1.0/data/demo` (paths like `frames/scene0_0.jpg`). Scene loader: `qwen_drive.benchmarks.read_scene_file(path, image_root=, image_archive=None, num_history_points=16, limit=)` -> samples with `.scene .token .future_trajectory`.
- `driving_command` has 4 entries, `ego_status` length 8, history `[16,3]`. nav_command: scene0=0 (straight), scene1=1 (left), scene2=2 (right).
- Prompt length (direct) = 3385 / 3383 / 3383 tokens; rope anchor direct = 522 / 520 / 520 (all 3 mRoPE sections equal); reasoning anchor = 555 / 554 / 551 (anchor includes `<|im_end|>\n`). Hence mRoPE positions are far below token count (images compress positions).
- K/V payload per scene (8 layers, bf16, 4 KV heads, head_dim 256): ~3.4k tokens -> ~105 MB direct; `extra_repos/golden_kv/scene_<i>_{direct,reasoning}.pt` = `{kv: [(k,v)]*8 of [1,S,4,256], anchor: [3,1]}`.
- Safetensors keys: VLM file has 723 keys all prefixed `vlm.model.` (e.g. `vlm.model.language_model.layers.0.linear_attn.A_log`), bf16, no `lm_head` key listed under another prefix (check tied embeddings in E5). Planner files: 358 keys, all `planning_expert.*`, bf16 (320 under `planning_expert.layers`), identical key sets for sft and rl.
- Reasoning output examples: scene1 -> "Turn left at the clear intersection and accelerate to the target speed."; reasoning pass generated ids saved in `generated_ids_<i>.npy`.
- Reference runs on one 16 GB GPU in bf16 with SDPA (no flash_attn/fla/causal_conv1d, torch fallback for GDN). Deterministic across runs.
- Gotcha: docker runs as root, so files written to `/workspace` are root-owned; scripts must `chown` outputs back (see `run_ref_baseline.sh`).

## Environment (E1, 2026-10-07)
- vllm 0.31.0 already provides `Qwen3_5ForConditionalGeneration` (`vllm.model_executor.models.qwen3_5`); transformers 5.14.1 (matches Qwen-Drive pin >=5.14,<5.15).
- Host has 2x RTX 5060 Ti 16 GB (sm_120).

## HF export (E3, 2026-10-07)
- `import qwen_drive` registers `qwen_drive` in `AutoConfig`/`AutoModel` (`__init__.py`); a registered class wins over `auto_map`. Verification therefore deletes the registration (see `verify_hf_export.py`). In vllm-omni never import the reference package.
- Loading via remote code prints a non-fatal `Do you wish to run the custom code? [y/N]` prompt and a "model of type `qwen_drive` to instantiate a model of type ``" warning when the nested `vlm_config` is built (stdin not a tty -> default continues). Pass `trust_remote_code=True` everywhere to avoid blocking in interactive shells; re-check this prompt does not hang the vllm engine process (E5/E7).
- Remote module name is `transformers_modules.Qwen_hyphen_Drive_...` (HF dynamic module cache under `HF_HOME`).
