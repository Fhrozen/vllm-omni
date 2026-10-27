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

## Spikes (E4, 2026-10-07)
Scripts: `customs/qwen_drive/spikes/{s5_vllm_vlm.py,qd_probe.py,s2_tokenize_roundtrip.py}`. Run with `QD_GPUS='"device=0"' customs/qwen_drive/docker_run.sh "PYTHONPATH=customs/qwen_drive/spikes python customs/qwen_drive/spikes/s5_vllm_vlm.py"`.

### S5 / S7 / S6: plain vLLM serves the VLM (VERDICT: works)
- vLLM 0.31.0 loads the exported repo's VLM on ONE 16 GB RTX 5060 Ti: weights+non-torch 9.04 GiB, peak activation 1.7 GiB, KV 2.42 GiB (= 64,428 tokens) at `gpu_memory_utilization=0.85`, `max_model_len=8192`, `enforce_eager=True`, `limit_mm_per_prompt={"image": 12}`. Init takes ~42 s; first request ~32 s (triton warmup), next ~3.4 s for 128 tokens.
- Needed pieces: (1) config unwrap: `hf_overrides` as a CALLABLE returns vLLM's own `vllm.transformers_utils.configs.qwen3_5.Qwen3_5Config(**vlm_config.to_dict())` with `architectures=["<wrapper>"]` (vLLM's mm processor type-checks vLLM's class, not transformers'); vLLM first calls the callable with a dummy config (must return it unchanged when `vlm_config` is absent). (2) wrapper subclass of `Qwen3_5ForConditionalGeneration` whose `load_weights` keeps only `vlm.*` names and strips the 4-char prefix (parent mapper then maps `model.language_model.`/`model.visual.`). The mm processor registration is inherited by the subclass (no re-registration needed). (3) `video_preprocessor_config.json` in the repo dir (added to `export_hf.py`): transformers resolves the video processor from the dir's `model_type` (`qwen_drive`), which is unknown.
- Alternative to the callable (serialisable across stage processes): flatten `vlm_config` into a temp dir `config.json` and set `hf_config_path` (precedent: `OmniEngineArgs._patch_empty_hf_config` in `vllm_omni/engine/arg_utils.py`). Decide in E5.
- Prefix caching is ENABLED by default for this hybrid model in vLLM 0.31 (`enable_prefix_caching=True`, 4 KV cache groups, lcm block size 272). Effectiveness (second identical request) not measured yet -> E10. KV cache layout log: `Using LBNHC KV cache layout` -> check against `normalize_layer_kv` in E5.
- Greedy VQA vs golden: output starts with `</think>` (model closes an empty think block); the reference strips it (`_strip_thinking`). Scene 1 matches the golden after stripping; scene 0 diverges at the first answer token (near-tie or numeric/preprocessing difference) -> decide with the K/V parity test in E5, not with text equality.

### S2: no prompt hook needed (VERDICT: chat template is enough)
- Tokenizing the joined text reproduces the reference `build_input_ids` ids exactly (6/6 golden prompts, direct + reasoning). So the prompt can be rendered as TEXT: `<|im_start|>user\n` + per view [`<FRONT VIEW>`, then per frame `frame: i` + `<|vision_start|><|image_pad|><|vision_end|>`] + instruction + `<|im_end|>\n<|im_start|>assistant\n` (+ `<|im_end|>\n` for direct mode; reasoning request appended to the instruction for reasoning mode). No `<think>` and no system turn.
- Design: ship a dedicated Jinja template (`chat_template_kwargs` selects mode) and send images in message order; images must be pre-resized client-side (per-frame `target_size`, then snap to multiples of 32, PIL bicubic) so vLLM's resize is a no-op. Per-request `max_tokens` applies to stage 0 (direct mode: `max_tokens=1`; reasoning: 256, greedy).
- Stage inputs: extra_body -> `sampling_params.extra_args` per stage exists (`apply_declared_extra_args` in `serving_chat.py` ~L3333); declare the qwen_drive keys.

### S1: routing (VERDICT: single 3-stage deploy, route by `modalities`)
- `modalities` selects the final stage (`_compute_final_stage_id` in `vllm_omni/entrypoints/omni_base.py` ~L458, `get_final_stage_id_for_e2e` in `vllm_omni/entrypoints/utils.py` ~L149). BAGEL precedent (stage 0 text final, stage 1 image final). So: stage 0 `final_output=True, final_output_type="text"`, stage 2 `final_output_type="trajectory"`; VQA request: `modalities=["text"]` (stops after stage 0); planning: `modalities=["trajectory"]`. D3 changed accordingly (one deploy yaml).
- Caveat: subagent notes, verify in E7 that stage 1 (non-final) is skipped for text-only and that reasoning text can be returned alongside the trajectory (plan: bridge copies reasoning text to stage 2 and stage 2 returns it in its output dict).

### S8: response (VERDICT: needs a code change)
- `serving_chat.py` only has branches for text/audio/image final outputs (non-streaming ~L4240-4320). Add a `trajectory` branch and a response field; extend `protocol/chat_completion.py`.

### S3 / S4: KV transfer (VERDICT: works with a layer filter + anchor metadata; verify empirically in E5)
- Sender: `OmniKVTransferManager.handle_finished_requests_kv_transfer` (`kv_transfer_manager.py` ~L977) iterates `self.kv_caches` (list), `normalize_layer_kv` (`utils/kv_utils.py` ~L400-501) returns None for non-attention tensors (skipped, `None` entries stay in `layer_blocks`). Receiver diffusion stage sees `req.past_key_values` (`key_cache[i]`/`value_cache[i]` as `[seq, heads, dim]`) and `req.kv_metadata`.
- UNVERIFIED (subagent claims, must be tested in E5): that `kv_caches` is ordered by decoder layer index, that mamba/GDN state tensors are skipped cleanly, and that block ids from the scheduler address the attention groups correctly in a hybrid model (4 KV cache groups, lcm block 272). Build the layer->list-index map from layer names rather than assuming identity. Golden K/V (`extra_repos/golden_kv/`) is the ground truth.
- Hook: `model.get_kv_transfer_metadata(req_id, num_computed_tokens)` (called in `gpu_ar_model_runner.py` ~L857-886; only BAGEL implements it) -> merged into `custom_metadata` -> receiver `req.kv_metadata`. Anchor rule (verified against golden): anchor = mRoPE position of the LAST token of the stage-1 prompt (all 3 sections equal). Candidates for computing it: (a) per-request mrope positions kept by the runner (`requests[req_id]`) passed to the hook, (b) closed form `sum(text tokens) + sum(max(h',w') per image) - 1` with merged grids from stage-0 mm features. Test against golden anchors 522/520/520 (direct), 555/554/551 (reasoning).
- `kv_transfer_criteria` `prefill_finished`: seq_len = prompt tokens only. A single AR stage cannot include `<|im_end|>\n` after generation (last sampled token has no KV) -> D2 stage 1 re-prefill stays.

### Hardware note
- Stage 0 (9 GB) and stage 1 (9 GB) need different GPUs (D6). The expert (2 GB) fits next to either.

## E5/E7 implementation findings (2026-10-07)
- End-to-end offline (fp8, 2x16 GB): `customs/qwen_drive/run_e2e_offline.sh --num-scenes 3 --modes direct,reasoning` -> trajectories `[6,50,3]` with ADE vs the BF16 reference golden 0.027-0.055 m (6/6 cases). Received K/V of the 8 full-attention layers vs golden: cosine mean 0.88-0.98 per layer (fp8 weights), anchor exactly equal to golden (522). seq_len 3385.
- Placeholders: stage 0 must receive ONE `<|image_pad|>` per image (vLLM expands it from the image grid). Sending pre-expanded ids + images makes vLLM-Omni expand again (6427 tokens). Stage 0 reports its prompt expanded; stage 1 must keep those expanded ids and receive the already-processed image features, which the orchestrator forwards only to a stage whose `model_stage == "thinker"` (`vllm_omni/engine/orchestrator.py` ~L2648). Hence stage 1 is named `thinker` in `pipeline.py`. Raw `multi_modal_data` on a bridged `OmniTokensPrompt` is NOT processed (no placeholders get expanded).
- Hybrid KV transfer needed a fix: `omni_ar_scheduler._mark_request_for_kv_transfer` used `block_ids_tuple[0]`, which is a GDN/Mamba group (1 block) in Qwen3.5; now uses the first `AttentionSpec` group (`_kv_transfer_group_index`). `kv_caches` list index == decoder layer index (verified with golden), so `kv_layer_indices` works as designed. `cache_config.block_size` is 272 for this model.
- Prefix caching: vllm-omni raises `OmniPrefixCacheUnmatchError` for hybrid models ("requires a single full-attention kv-cache group; found 4") -> `enable_prefix_caching: false` in the deploy yamls. S6 verdict: not usable; stage 1 re-prefills the prompt (acceptable, ~3.4k tokens).
- The mRoPE anchor comes from `QwenDriveVLMForConditionalGeneration.prepare_runner_inputs` (records `positions[:, last scheduled token]` per request) + `get_kv_transfer_metadata` -> `kv_metadata["rope_anchor"]`. Only the v1 AR runner (default) calls these hooks.
- Diffusion payload key: a non-media output must use the key `"trajectory"` (or `"actions"`); other keys are treated as image payloads. The diffusion KV manager attaches received KV to `req` AND `req.sampling_params` (`past_key_values`, `kv_metadata`); the pipeline reads either. Output type `"trajectory"` is declared in `vllm_omni/diffusion/model_metadata.py`.
- The Omni script needs an `if __name__ == "__main__":` guard (stages spawn processes).
- fp8: `quantization: fp8_per_tensor` per stage in the yaml works (the name `fp8` is deprecated); quantizes 250 layers incl. the vision tower and GDN projections; each VLM stage takes 5.47 GiB (vs 9.04 GiB bf16). Stage init ~65 s, first request ~30 s.
- A debug aid used for the K/V parity check was removed from the pipeline; re-add by dumping `req.past_key_values` and compare with `customs/qwen_drive/spikes/compare_kv.py`.

## E8/E9 findings (2026-10-07)
- Online path works end to end (non-streaming). The diffusion stage must get its own prompt: without a `custom_process_input_func` (`prefill_to_planner`, empty prompt) the orchestrator forwards the processed original prompt, whose `MultiModalKwargsItems` cannot be serialized to the diffusion stage.
- Planner inputs must be sent as `extra_args` (canonical request field routed to diffusion stages). Declaring flat fields through `vllm_omni/model_extras` did NOT route them for this multi-stage pipeline (tested, then removed).
- The frontend multimodal processor cache must be off (`mm_processor_cache_gb: 0`): it sends hashes only for repeated images and stage 1 then fails with "Multi-modal cache miss" (or reuses wrong features).
- `min_tokens` in the stage default sampling params breaks `max_tokens=1` requests; pass `min_tokens` per request.
- Test harness gotcha: tests run at `--run-level=core_model` by default, which rewrites every deploy yaml stage to `load_format: dummy` (random weights: outputs of `\n` tokens, wrong trajectories). Use `--run-level=advanced_model` (done in `run_e2e_test.sh`).
- Stage 1 re-prefill costs about 0.65 s per request (3.4k tokens, fp8) on top of 0.63 s stage 0 and 1.0 s planner (6 samples): ~2.6 s direct request when warm.

## E10 findings (2026-10-08)
- BF16 fits the 2x16 GB host with `customs/qwen_drive/deploy_bf16_16gb.yaml` (stage 0 + planner on GPU 0 at util 0.72, stage 1 on GPU 1, `max_num_seqs: 1`, `max_model_len: 5000`). Offline ADE vs the HF/sdpa golden: scene0 0.112/0.103 m (direct/reasoning), scene1 0.026/0.039, scene2 0.026/0.027.
- Noise floor: the reference itself, switching only `attn_implementation` sdpa -> eager, differs from the golden by ADE 0.023 / 0.028 / 0.058 m (max |d| up to 0.24). So FP8 (0.028-0.055 m) and BF16 (0.026-0.112 m) deviations from the golden are at kernel-noise level; an ADE bound of 0.15 m is the right test threshold, and exact parity is not expected from vLLM's different attention/GDN kernels.
- Online robustness (FP8): 5 concurrent mixed requests all correct; a client disconnect mid-request does not break later requests; VQA streaming works; planner-sft serves correctly via `model_config.planner_subfolder`. `trajectory` is returned for non-streaming requests only.

## OpenAI request example (2026-10-22)
- `customs/qwen_drive/openai_request_example.py`: documented client for `/v1/chat/completions`; modes vqa/direct/reasoning; `--print-payload` (truncated) and `--payload-out FILE` (full JSON for `curl -d @file`). Mirrors `client_example.py` (same `smart_resize`/`prepare_image`/`history`), self-contained, no `qwen_drive` import.
- Verified payload structure in docker (scene 0): 12 image parts (base64 PNG data URLs) + 16 text parts (3 view labels + 12 `frame: i` + instruction); direct: `modalities ["trajectory"]`, `max_tokens 1`, `extra_args` history 16x3 / vel+acc 16x2 / `driving_command [0,1,0,0]` / `nav_command 0`; reasoning: `["text","trajectory"]`, `max_tokens 256`, `min_tokens 10`; vqa: images + question only, no `extra_args`.
- Docs: `customs/docs/wiki/OpenAI-Request.md` (request contract, extra_args table, image preprocessing, curl workflow, mode-difference table); linked from Home.md.

## E11 findings — WeVisOne-VL (2026-10-08)
- Checkpoint `extra_repos/WeVisOne-VL-4B` reports `architectures=["Qwen3VLForConditionalGeneration"]`,
  `model_type="qwen3_vl"`, no `auto_map`, so `StageConfigFactory._match_pipeline_key` resolves
  `tencent/WeVisOne-VL-4B` to the `wevisone_vl` pipeline key by checkpoint name; the model_extras family
  rule therefore resolves from the checkpoint name, not the HF config. No HF export needed.
- Token ids: `<OBJ>` 152678 (added, `special: false`), `<box>` 152672, `</box>` 152673, coords
  `<0>..<1000>` from 151669, eos `<|im_end|>` 151645, `image_token` 151655, vocab 152680;
  `coord_token_id_base(152678) = 151669`.
- Box parsers agree with the golden fixture: text and token-id parsers both give
  `[[0.009, 0.065, 0.753, 1.0]]`. A box is recorded only when followed by `<OBJ>`; `</box>` clears the
  in-box state, so a prompt box closed before any `<OBJ>` is not a query row. Only generated tokens may
  be scanned (prompt ids must be excluded).
- `mask_decoder.pt`: 4 keys, `state_dict` 296 entries / 9,609,764 elements (no prefix), `cfg`
  (`wevisone_vl_v1`, feature layers as a JSON **list** [5,11,17,23], hidden 192, mask feature 64, 32
  Fourier freqs, MSDA 3/4/4, transformer 6/8/768, GN 32, stem 128, alpha cascade 48x2, cnn 1024),
  `seg_token_id` 152678, `spatial_merge_size` 2. `Qwen3VLSegConfig.from_dict` normalizes list -> tuple so
  the strict path does not log a spurious rebuild.
- Golden matting fixture (`customs/wevisone/golden/matting/`, seed 1234, `seabreeze.jpg` 1400x933): box
  `[[0.009, 0.065, 0.753, 1.0]]`, response
  `<ref>the person</ref><box><9><65><753><1000><OBJ></box>`, mask `(933, 1400)` float32 in [0,1],
  min 4.9e-9 max 0.9999964 mean 0.2726. Generated by the reference oracle on GPU 0.
  Reproducibility measured with three runs: text and boxes are identical, masks differ by up to
  5.1e-4, so the test compares masks with tolerances (boxes 0.001, mean mask 0.02) rather than exact
  equality.
- Canvas rule verified from the reference engine: bicubic resize of the ORIGINAL PIL image to
  `cnn_image_size`, float32/255 -> plain float RGB [0,1], no mean/std (0.5 mean/std is the vision
  processor's), never Qwen smart-resize. Delivered as extra `batched("image", keep_on_cpu=True)` fields
  `wevisone_canvas` / `wevisone_hw` by the custom mm processor.
- Mask post-processing (`_masks_from_logits`) is verbatim parity; constant-zero logits -> all 0.5 is the
  deterministic test anchor. Interpolating to the mask's own size is an identity, so the golden mask is
  reproduced through the inverse sigmoid with max error 5.96e-8.
- Feature-bridge shuffle inverse is block-major:
  `reshape(h//m, m, w//m, m, c).permute(0,2,1,3,4)`, and the merge-size divisibility check runs before
  the shape check in `premerge_tokens_to_2d`.
- `segmentation` is a non-drainable modality (`NON_DRAINABLE_MODALITIES`), so its keys survive
  `drain_delta_payload` for the decoder stage; `segmentation.t_seg` must stay `CONCAT_DIM0` across decode
  steps, so it is excluded from `_FULL_PAYLOAD_REPLACE_KEYS`. Stage 1 needs an fp32 forward at S=1024; the
  alpha cascade is the memory hotspot (`expandable_segments:True`).
- Serving: masks are one stacked `[N,1,H,W]` tensor per request under `image` (dict-valued payload keys
  are rejected by `split_mapping_outputs`); `_create_image_choice` skips zero-numel slices, scales [0,1]
  floats to uint8 by truncation (0.5 -> 127), and sets `mask_index` only for stacked multi-mask answers,
  with ordinals recorded after the skip. Boxes stay in the text choice content because `metrics` is only
  populated on the diffusion serving path (`WEVISONE_EXTRA_OUTPUT_PARAMS = frozenset()`).
- Host limitation (not yet resolved): the docker image is unavailable (Docker Hub 401) and the host
  interpreter has no `vllm`, so L1 pytest cannot run here. Only the files that do not import `vllm` were
  executed (33 passed). Hook parity and mask PSNR vs the reference remain unmeasured.
