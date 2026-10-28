# Progress

Update this file at the end of every episode. Status: `todo` | `in-progress` | `done` | `blocked`.

| ID | Status | Notes / outputs |
|----|--------|-----------------|
| E0 | done | wiki + `customs/qwen_drive/{docker_run,setup_env,env_check}.sh`; the other scripts were added by their episodes (all present) |
| E1 | done | vllm 0.31.0, transformers 5.14.1, torch 2.13.0+cu132; `Qwen3_5ForConditionalGeneration` registered in vllm; `qwen_drive` reference installed (--no-deps). flash_attn/fla/causal_conv1d NOT installed (HF reference falls back to torch GDN, slow). |
| E2 | done | `customs/qwen_drive/{ref_golden.py,run_ref_baseline.sh}`; golden in `customs/docs/golden/` (3 scenes, 6 samples, seed 42, sdpa bf16), K/V in `extra_repos/golden_kv/` (638 MB, git-ignored); two runs bit-identical |
| E3 | done | `customs/qwen_drive/{export_hf.py,export_hf.sh,verify_hf_export.py}`; `extra_repos/Qwen-Drive-1.0-4B-hf`; remote-code load matches golden bit-for-bit; see HF-Export.md |
| E4 | done | spike scripts in `customs/qwen_drive/spikes/`; verdicts in Findings.md ("Spikes (E4)"); `video_preprocessor_config.json` added to the export; Decisions D1/D3/D3b/D4 updated. Open items carried to E5: KV list-index mapping on hybrid caches, anchor source, flatten-config mechanism; E8: `trajectory` response branch; E10: prefix-cache effectiveness |
| E5 | done | `vllm_omni/model_executor/models/qwen_drive/qwen_drive_vlm.py` (wrapper: strips `vlm.`, anchor hooks), registry entry, `OmniEngineArgs._flatten_nested_hf_config` (`arg_utils.py`), `kv_layer_indices` filter (`kv_transfer_manager.py`), attention-group block ids (`core/sched/omni_ar_scheduler.py`). Verified through the offline pipeline run (E7) |
| E6 | done | `vllm_omni/diffusion/models/qwen_drive/{__init__,planning_expert,pipeline_qwen_drive}.py`; registry entry `QwenDrivePlannerPipeline` (+ no-cache-accel, post-process) in `vllm_omni/diffusion/registry.py`; tests `tests/diffusion/models/qwen_drive/test_planning_expert.py` (19 pass: 5 CPU units, 12 golden parity at atol 1e-3, 2 pipeline-contract); run with `customs/qwen_drive/run_unit_tests.sh`. Golden now also stores `planner_inputs_<i>.npz` |
| E7 | done | `model_executor/models/qwen_drive/pipeline.py`, `model_executor/stage_input_processors/qwen_drive.py` (`vlm_to_prefill`), `config/pipeline_registry.py`, `deploy/qwen_drive.yaml` (bf16) and `deploy/qwen_drive_fp8.yaml`; `customs/qwen_drive/{e2e_offline.py,run_e2e_offline.sh}`: 6/6 (3 scenes x direct/reasoning) ADE vs golden 0.03-0.055 m with fp8. BF16 yaml not run yet (needs ~24 GB total) |
| E8 | done | `customs/qwen_drive/{client_example.py,chat_template_drive.jinja,serve.sh}`; `trajectory` response field (`protocol/chat_completion.py`) + branch in `serving_chat.py`; stage 1->2 bridge `prefill_to_planner`; export ships the chat template. Verified live: direct/reasoning/VQA over HTTP, ADE vs golden identical to offline (0.045/0.055 m scene 1) |
| E9 | done | `tests/e2e/online_serving/test_qwen_drive.py` (7 tests pass, FP8), `customs/qwen_drive/{run_e2e_test.sh,run_unit_tests.sh}`; Testing.md, Usage.md written |
| E10 | done | ruff clean; online suite grew to 11 tests (concurrency x5, client abort, VQA streaming, planner-sft) all passing; BF16 measured with `customs/qwen_drive/deploy_bf16_16gb.yaml` (fits 2x16 GB: 1 seq, 5000 ctx); noise floor measured (`spikes/noise_floor.py`); Architecture/Usage/Testing/Risks refreshed |
| E11 | done | WeVisOne-VL: code-complete for phases A-F. `vllm_omni/model_executor/models/wevisone/` (constants, config, decoder, msda, feature_bridge, wevisone_vl, processing, pipeline), `Segmentation` payload group + non-drainable `SEGMENTATION` modality, `stage_input_processors/wevisone.py`, both registries, `deploy/wevisone_vl.yaml` + `wevisone_vl_text_only.yaml`, `model_extras/wevisone.py`, stacked-mask `mask_index` in `_create_image_choice`; `customs/wevisone/` (README, client, samples.json 11 tasks, generator, visualize, serve/run scripts, ref_golden + golden fixture); tests for parsers/bridge/decoder masks/pipeline/stage bridge/payload schema/serving choice/model_extras; `customs/wevisone/{docker_run.sh,run_tests.sh,run_e2e_test.sh}` docker wrappers. Verified in docker: L1 CPU **311 passed / 8 deselected** in the scoped set, golden parity **4 passed**, gates exit 0, `ruff==0.14.10` check + format clean. **L2 e2e executed on 2x RTX 3060: 2 passed in 186.68 s** (matting mask decoded, HTTP 200). **L4 expansion executed on the same machine: 4 passed in 121.28 s.** Two defects found by running L4 are fixed and covered by L1 tests: boxes must be counted as `<OBJ>`-terminated coordinate groups (`count_box_queries`), and a box-less answer must still ship a payload with an empty mask selection instead of `None` (it otherwise parked in `WAITING_FOR_INPUT` until the 600 s connector timeout). **Hook parity executed: 4 passed** — CPU dual-run exact, GPU dual-run within the fp32 noise floor, mask PSNR 77.7 dB between towers |

## Commands run (reproducible)
- E1: `customs/qwen_drive/setup_env.sh` then `customs/qwen_drive/env_check.sh`.
- E2: `customs/qwen_drive/run_ref_baseline.sh --num-scenes 3` (runs GPU 0 only, twice, then compares).
- E3: `customs/qwen_drive/export_hf.sh` (export on host, verify in docker GPU 0).
- E4: see Findings.md "Spikes (E4)" for the exact commands.
- E5/E7: `customs/qwen_drive/run_e2e_offline.sh --num-scenes 3 --modes direct,reasoning`.
- E8: `customs/qwen_drive/serve.sh` then `customs/qwen_drive/docker_run.sh "python customs/qwen_drive/client_example.py --wait 900 --mode direct --scene 1 --golden customs/docs/golden"`.
- E9: `customs/qwen_drive/run_e2e_test.sh`.
- E6: `customs/qwen_drive/run_unit_tests.sh` (GPU 0; add `-k pipeline_forward` for the pipeline-contract test).
- E11: `python customs/wevisone/ref_golden.py --decode-masks --tasks matting` (offline venv, GPU 0,
  seed 1234) produced `customs/wevisone/golden/matting/`; four runs (the committed fixture plus three
  fresh) agree on text/boxes and differ in masks by up to 5.1e-4; `bash customs/wevisone/run_golden.sh` compares
  it; CPU tests run in the offline venv for the files
  that do not import `vllm`:
  `pytest -q -c /dev/null --confcutdir=tests/model_executor/models/wevisone -p no:cacheprovider -m 'not gpu and not cuda' tests/model_executor/models/wevisone/test_wevisone_parsers.py tests/model_executor/models/wevisone/test_wevisone_feature_bridge.py`.
  Gates: `python tools/pre_commit/check_test_marks.py` and `check_spdx_header.py` (both exit 0).
  Full L1 run must happen in docker — the host interpreter has no `vllm`.
- E11 docker (image `vllm/vllm-openai:latest`, shared `.venv`, `--entrypoint bash`, GPU 0):
  `WVS_IMAGE=vllm/vllm-openai:latest bash customs/wevisone/run_tests.sh` → **224 passed, 8 deselected**;
  `WVS_MARKERS='local_model and cpu' bash customs/wevisone/run_tests.sh tests/model_executor/models/wevisone/test_wevisone_golden_parity.py`
  → **4 passed**; all four `tools/pre_commit` checks exit 0.
- E11 e2e on 2 GPUs (`WVS_GPUS="device=0,1"`, weights from `extra_repos/WeVisOne-VL-4B`):
  `WVS_IMAGE=vllm/vllm-openai:latest WVS_GPUS="device=0,1" bash customs/wevisone/run_e2e_test.sh`
  → **2 passed in 184.54 s** (matting 109.74 s, startup 98.036 s). Write the log to a file and grep it
  rather than piping through `tail`, which hides the failure lines.
  Two test bugs and one implementation deviation were fixed to get there: the stage-processor fixture now builds the
  nested (`{"segmentation": {...}}`) view of `pooling_output` instead of the bare group dict, the serving-chat mask
  fixture uses the real `[N, 1, H, W]` shape, and box↔query pairing follows the reference `min(len(boxes), t_seg rows)`
  positional rule instead of inventing pairings.
  Lint with the pre-commit pin (`ruff==0.14.10`, installed in the shared venv): `ruff check` is clean on every
  WeVisOne file; 11 files needed formatting (typing modernisation, import sorting, 2 unused imports, 1 dead local,
  2 long lines under `customs/wevisone/`), and the suite re-ran green afterwards (224 passed / 8 deselected).
  `customs/wevisone/docker_run.sh` now exports `PYTHONPATH=/workspace` so the `customs/` scripts can import
  `vllm_omni` from the mount.
- E11 hook parity on GPU 0 (`vllm/vllm-openai:latest`, `python -m pytest` inside the shared venv):
  `customs/wevisone/parity_capture.py --side reference --device cpu --out customs/wevisone/golden/hooks/cpu`
  then the same with `--side port` (the port's capture logic runs against the reference tower on CPU),
  followed by `bash customs/wevisone/run_tests.sh tests/model_executor/models/wevisone/test_wevisone_hook_parity.py`
  → **4 passed**. The harness writes `manifest.json` plus `layer_NNN.npy`, `t_seg.npy`, `canvas.npy` and
  `mask.npy` per side; both sides must be separate processes, and `--dtype float32 --tf32 off` is what keeps
  the noise floor near 1e-5 relative (a 12 GiB card needs `--dtype bfloat16` for a full-model run).
  Throwaway probes used while narrowing the cause were deleted; the surviving harness is `parity_capture.py`.
- E11 e2e + CI wiring (authored, collection-verified, gate-verified — execution needs GPUs):
  `tests/e2e/online_serving/test_wevisone_vl.py` (2 cases) and
  `test_wevisone_vl_expansion.py` (3 cases) with the shared helper
  `tests/helpers/assertions.py::assert_chat_mask_choices`; collection measured in docker
  (`1/2` ready, `2` merge, `3` nightly); registry key `omni_wevisone_vl_function` +
  `omni_wevisone_vl_expansion`; `customs/wevisone/run_e2e_test.sh` (`WVS_LEVEL`, `WVS_MODEL`,
  `WVS_GPUS` default `device=0,1`) is executable and `bash -n` clean but has never been run.
  Gates re-run after the last YAML edit: `check_buildkite.py` (with the four files passed explicitly),
  `check_test_marks.py`, `check_spdx_header.py`, `check_forbidden_imports.py` — all exit 0; ruff 0.14.10
  check + format clean on the changed test files.

- E11 L4 expansion on 2 GPUs (`WVS_LEVEL=full_model`, `WVS_GPUS="device=0,1"`):
  `WVS_IMAGE=vllm/vllm-openai:latest WVS_GPUS="device=0,1" WVS_LEVEL=full_model \
    bash customs/wevisone/run_e2e_test.sh tests/e2e/online_serving/test_wevisone_vl_expansion.py`
  → **4 passed in 121.28 s**. Case 1 asks for mattes (`segmentation`, "each colored square") because only
  matte-producing tasks emit `<OBJ>`; it answered four queries inside one `<box>` span and returned four
  masks. Case 4 (`count_box`, no `<OBJ>`) completed in ~1.7 s with no mask parts — before the fix it hung
  until the 600 s connector input-wait timeout. Container deps for the suite: `aenum omegaconf prettytable
  regex diffusers pytest_asyncio pytest_xdist pyyaml pytest_mock av soundfile opencc-python-reimplemented`
  (plus `filelock psutil requests` for the runtime helpers); `opencc` is needed by `tests/helpers/media.py`,
  without it `tests/helpers/tests/test_assertions.py` fails 24 tests.

## Next
E11 is the active episode. Remaining, in order:
1. ~~Run the CPU tests in docker~~ done: 224 passed / 8 deselected via `bash customs/wevisone/run_tests.sh` (see Commands run).
2. ~~Lint the changed files~~ done with `ruff==0.14.10`: check clean, 11 files formatted, suite re-run green.
3. Hook-parity gate on GPU: capture hooks fire under `enforce_eager` on stage 0, captured ViT layers 5/11/17/23 and the last LM layer match the reference values.
4. Golden mask parity + PSNR on GPU (tolerances: box 0.001, mean mask 0.02 — masks are not bit-reproducible).
5. ~~L2 e2e test file~~ authored and **executed green on 2x RTX 3060** (2 passed, mask choice + `mask_index`
   pairing, text-only no-sidecar case) via `bash customs/wevisone/run_e2e_test.sh`; the L4 expansion likewise
   (**4 passed in 121.28 s**, including the box-less case that previously hung).
6. dfx invalid-param cases: **closed as not writable as implemented** — `extra_body` names are merged
   without validation and task validation is client-side, so no status code can be claimed (Findings.md).
   Reopens if server-side task validation is added.
7. ~~Buildkite wiring~~ done: `omni_wevisone_vl_function` / `omni_wevisone_vl_expansion` in
   `.buildkite/common/ci_source_file_dependencies.yml`, L2 steps in `test-ready.yml` and `test-merge.yml`,
   L4 key added to the nightly 2-GPU step; `check_buildkite.py` exits 0 and the marker expressions collect
   the intended cases.

Optional follow-ups (E1-E10): streaming of the `trajectory` field, throughput benchmarks with larger `max_num_seqs` on the big-VRAM machine (`vllm_omni/deploy/qwen_drive.yaml`).
