# E11: WeVisOne-VL (pixel decoder + mask responses)

Depends on: E1 (environment). Use the `vllm-omni-test` skill for markers and CI wiring rules, and the
`add-diffusion-model`/`review-pr` skills for the stage-1 runner conventions.

Model page: [WeVisOne-VL](../WeVisOne-VL.md) (goal, decisions, findings, testing table).

## Inputs
- Reference package and checkpoint: `extra_repos/WeVisOne-VL` (`wevisone_vl/{engine,prompts,decoder,feature_bridge}.py`),
  `extra_repos/WeVisOne-VL-4B` (`config.json`, `tokenizer.json`, `mask_decoder.pt`), demo image
  `extra_repos/WeVisOne-VL/assets/demo/seabreeze.jpg`.
- Precedents: `vllm_omni/model_executor/models/aura_omni/qwen3_vl.py` (processing-info subclass +
  `register_processor`), `qwen3_omni_moe_thinker.py` (layer dict capture), `nemotron_voicechat_thinker.py`
  (`postprocess` + `make_omni_output`), `fish_speech/dac_decoder.py` (lazy sidecar load),
  `stage_input_processors/thinker2talker.py`, `tests/helpers/serving_chat.py`.

## Outputs
- `vllm_omni/model_executor/models/wevisone/`: `constants.py`, `config.py`, `decoder.py`, `msda.py`,
  `feature_bridge.py`, `wevisone_vl.py`, `processing.py`, `pipeline.py`.
- `vllm_omni/data_entry_keys.py`: `Segmentation` TypedDict + `SegmentationStruct` + `_NESTED_KEYS`;
  `vllm_omni/outputs/output_modality.py`: non-drainable `SEGMENTATION` modality.
- `vllm_omni/model_executor/stage_input_processors/wevisone.py` (`vlm2pixel_decoder_full_payload`).
- Registries: `_OMNI_MODELS` (both archs), `OMNI_PIPELINES` (`wevisone_vl`, `wevisone_vl_text_only`);
  `vllm_omni/deploy/wevisone_vl.yaml`, `wevisone_vl_text_only.yaml`.
- `vllm_omni/model_extras/wevisone.py` + `_EXTRA_SPECS` + checkpoint-name family rule; stacked-mask
  support in `_create_image_choice` (`mask_index` ordinals).
- `customs/wevisone/`: README, `openai_request_example.py`, `samples.json` (11 tasks),
  `generate_samples.py`, `visualize.py`, `serve.sh`, `run_samples.sh`, `run_golden.sh`, `run_tests.sh`,
  `run_e2e_test.sh`, `docker_run.sh`, `ref_golden.py`, `golden/matting/`.
- Tests: `tests/model_executor/models/wevisone/` (parsers, feature bridge, decoder masks,
  configuration+pipeline, golden parity), `tests/model_executor/stage_input_processors/test_wevisone.py`,
  `tests/engine/test_data_entry_keys.py`, `tests/engine/test_output_modality.py`,
  `tests/entrypoints/openai_api/test_serving_chat_mask_choices.py`, `tests/model_extras/test_model_extras.py`;
  L2 `tests/e2e/online_serving/test_wevisone_vl.py`, L4 `test_wevisone_vl_expansion.py`, and the shared
  `tests/helpers/assertions.py::assert_chat_mask_choices`.
- Buildkite: `x-wevisone-vl` anchor + `omni_wevisone_vl_function`/`omni_wevisone_vl_expansion` in
  `.buildkite/common/ci_source_file_dependencies.yml`, L2 steps in `test-ready.yml` and `test-merge.yml`,
  L4 key added to the nightly 2-GPU step.

## What was done
- Golden oracle ran on this host in an offline venv (`/tmp/wevis-venv`, GPU 0): seed 1234, fixed image, matting
  prompt → `customs/wevisone/golden/matting/{result.json, masks/mask_000.npy, mask_000.png}`.
  Measured across three runs: response and boxes are identical, mask tensors differ by up to 5.1e-4
  (GPU kernels are not bit-reproducible) — parity tests must use tolerances, never exact mask equality.
- Both box parsers (text and token-id) reproduce the golden box exactly; the feature-bridge shuffle
  inverse is verbatim parity with the reference, including block-major patch order.
- CPU unit tests: 33 passed for the files that do not import `vllm` in the offline host venv. `py_compile`,
  SPDX, 120-column, and `check_test_marks.py` all clean.
- Docker run on the 2-GPU host (`vllm/vllm-openai:latest`, shared `.venv`, `--entrypoint bash`, GPU 0):
  **224 passed / 8 deselected** for the L1 CPU set, **4 passed** for golden parity, all four pre-commit
  checks exit 0. Three defects were found and fixed there: the stage-processor test fixture built the bare
  group dict instead of the nested `pooling_output` view, the serving-chat mask fixture used a `(3,2,1,2)`
  shape instead of the real `[N,1,H,W]`, and the stage processor's mismatch handling was changed to the
  reference `min(len(boxes), t_seg rows)` positional pairing.
- New docker wrappers: `customs/wevisone/docker_run.sh` (mirrors `customs/qwen_drive/docker_run.sh`, adds the
  `--entrypoint bash` override and exports `PYTHONPATH=/workspace`) and `customs/wevisone/run_tests.sh`
  (`WVS_MARKERS`, `WVS_GPUS`, `WVS_IMAGE`); `serve.sh`, `run_samples.sh`, `run_golden.sh` now delegate to the
  local wrapper. All `bash -n` clean and executable; `run_tests.sh` verified by an actual run.
- Lint with the pinned `ruff==0.14.10`: check clean across the WeVisOne files, 11 files reformatted (typing
  modernisation, import sorting, 2 unused imports, 1 dead local, 2 long lines in `customs/wevisone/`), suite
  re-ran green afterwards.
- e2e coverage: `tests/e2e/online_serving/test_wevisone_vl.py` — mask/box pairing through
  `assert_chat_mask_choices` on the full pipeline, and a text-only case that asserts the decoder sidecar is
  never loaded; `test_wevisone_vl_expansion.py` — one mask per box, original-image resolution
  (`213x149`, not divisible by merge factor 28), and per-request canvas on one server. Reusable checks added
  to `tests/helpers/assertions.py`; per-case `parametrize` ids, no branching on `case_id`.
- L2 executed on this machine (2x RTX 3060, `vllm/vllm-openai:latest`, `WVS_GPUS="device=0,1"`,
  `WEVISONE_MODEL_PATH=/workspace/extra_repos/WeVisOne-VL-4B`): **2 passed in 184.54 s**, matting case
  109.74 s, server startup 98.036 s, stages `[0: 0.69 s, 1: 0.64 s]`, HTTP 200, mask 0 gray range 0-254.
  The text-only case passed too, so the sidecar-free path is confirmed. Five defects were found by running
  it and each is now covered by an L1 test:
  1. the processor hook had to move to `_postprocess_hf_mm_data`, because
     `Qwen3VLMultiModalProcessor._apply_hf_processor_main` calls `info.ctx.call_hf_processor` directly and
     never reaches `_call_hf_processor`;
  2. `ProcessorBatchItems.get_processor_data()` keys image data as `images`, not `image`;
  3. `keep_on_cpu=True` payload fields arrive on CPU while the decoder weights are on the stage device, so
     `_decode_request` must move them to `vllm_config.device_config.device`;
  4. `_masks_from_logits` must accept the ndim-3 `[num_queries, h, w]` layout the reference accepts;
  5. with two final-output stages the response emits one choice per final output type, so
     `assert_chat_mask_choices` collects parts across all choices and stage 0 declares
     `final_output_type="text"`; a single-mask payload legitimately has no `mask_index`;
  6. boxes are counted as `<OBJ>`-terminated coordinate groups rather than `<box>` spans, because one span
     can answer several queries (`count_box_queries` in `tests/helpers/assertions.py`);
  7. a box-less answer must still reach the consumer stage: the processor now ships an empty mask
     selection instead of `None`, and `_decode_request` returns `None` for an empty box list before loading
     the sidecar.
- L4 expansion **executed on this machine**: 4 passed in 121.28 s on 2x RTX 3060. The first case had to ask
  for mattes (`segmentation`) instead of `referring`, because a box is only a mask query when `<OBJ>` follows
  its coordinates; it returned four masks for the four queries in one span. The added box-less case completes
  in ~1.7 s with no mask parts — before the fix it hung until the 600 s connector input-wait timeout.
  L2 re-run after the `_decode_request` reorder: still 2 passed (186.68 s).
- Buildkite wiring landed: registry anchor `x-wevisone-vl` + keys `omni_wevisone_vl_function` /
  `omni_wevisone_vl_expansion`, L2 steps in `test-ready.yml` and `test-merge.yml`, the L4 key added to the
  nightly 2-GPU step. `check_buildkite.py` exits 0 on the four files, and marker collection in docker selects
  the intended cases (`1/2`, `2`, `4`). dfx was skipped: no WeVisOne error path is verifiable at the HTTP
  layer as implemented (Findings.md).
- `customs/wevisone/run_e2e_test.sh` added (`WVS_LEVEL`, `WVS_MODEL`, `WVS_GPUS` default `device=0,1`),
  executable, `bash -n` clean, and used for the L2 run above.
- Buildkite: the `core_model` L1 step was removed from `test-ready.yml` because `core_model` runs patch
  every stage to `load_format: dummy` (`tests/helpers/stage_config.py`), which cannot produce a box span;
  the e2e file carries `advanced_model` only, and `RTX3060` was registered as a CUDA SKU marker in
  `pyproject.toml` for local runs (no consumer-GPU pool exists in `ci_mirror_hardwares.yml`).

## Exit criteria
Unit tests pass in docker on the 2-GPU host, hook-parity deltas are recorded in Findings.md, golden
parity (boxes 0.001, mean mask 0.02) passes for `run_golden.sh`, and the manual API check confirms the
mask/box pairing. Progress.md updated.
**Partly met** — unit tests and golden parity pass in docker on the 2-GPU host, the L2 serving tests pass
on this machine (mask/box pairing confirmed over HTTP, text-only path confirmed sidecar-free), the L4
expansion passes on this machine, and all pre-commit gates exit 0. Hook-parity deltas and mask PSNR vs the
reference are still outstanding.

## Open items
- L4 expansion: closed, executed green on this machine (`WVS_LEVEL=full_model bash
  customs/wevisone/run_e2e_test.sh tests/e2e/online_serving/test_wevisone_vl_expansion.py`). L2 is executed
  and green on this machine.
- dfx invalid-param cases: closed as not writable as implemented (no WeVisOne error path is verifiable at
  the HTTP layer); reopens only if server-side task validation is added.
- Hook parity measurement and mask PSNR vs the reference (blocks all tolerance claims).
- Text-only pipeline smoke run without `mask_decoder.pt` — closed, the L2 text-only case passes and asserts
  the sidecar is never loaded.
