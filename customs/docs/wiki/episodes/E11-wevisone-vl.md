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
  `generate_samples.py`, `visualize.py`, `serve.sh`, `run_samples.sh`, `run_golden.sh`,
  `ref_golden.py`, `golden/matting/`.
- Tests: `tests/model_executor/models/wevisone/` (parsers, feature bridge, decoder masks,
  configuration+pipeline, golden parity), `tests/model_executor/stage_input_processors/test_wevisone.py`,
  `tests/engine/test_data_entry_keys.py`, `tests/engine/test_output_modality.py`,
  `tests/entrypoints/openai_api/test_serving_chat_mask_choices.py`, `tests/model_extras/test_model_extras.py`.

## What was done
- Golden oracle ran on this host in an offline venv (`/tmp/wevis-venv`, GPU 0): seed 1234, fixed image, matting
  prompt → `customs/wevisone/golden/matting/{result.json, masks/mask_000.npy, mask_000.png}`.
  Measured across three runs: response and boxes are identical, mask tensors differ by up to 5.1e-4
  (GPU kernels are not bit-reproducible) — parity tests must use tolerances, never exact mask equality.
- Both box parsers (text and token-id) reproduce the golden box exactly; the feature-bridge shuffle
  inverse is verbatim parity with the reference, including block-major patch order.
- CPU unit tests: 33 passed for the files that do not import `vllm`. `py_compile`, SPDX, 120-column, and
  `check_test_marks.py` all clean.

## Exit criteria
Unit tests pass in docker on the 2-GPU host, hook-parity deltas are recorded in Findings.md, golden
parity (boxes 0.001, mean mask 0.02) passes for `run_golden.sh`, and the manual API check confirms the
mask/box pairing. Progress.md updated. **Not met yet** — no vllm runtime is available on this host.

## Open items
- L2 `tests/e2e/online_serving/test_wevisone_vl.py`, L4 expansion, Buildkite `omni_wevisone_vl_function`
  wiring, dfx invalid-param cases (unknown task / missing sidecar).
- Hook parity measurement and mask PSNR vs the reference (blocks all tolerance claims).
- Text-only pipeline smoke run without `mask_decoder.pt`.
