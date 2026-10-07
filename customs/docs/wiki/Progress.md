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

## Commands run (reproducible)
- E1: `customs/qwen_drive/setup_env.sh` then `customs/qwen_drive/env_check.sh`.
- E2: `customs/qwen_drive/run_ref_baseline.sh --num-scenes 3` (runs GPU 0 only, twice, then compares).
- E3: `customs/qwen_drive/export_hf.sh` (export on host, verify in docker GPU 0).
- E4: see Findings.md "Spikes (E4)" for the exact commands.
- E5/E7: `customs/qwen_drive/run_e2e_offline.sh --num-scenes 3 --modes direct,reasoning`.
- E8: `customs/qwen_drive/serve.sh` then `customs/qwen_drive/docker_run.sh "python customs/qwen_drive/client_example.py --wait 900 --mode direct --scene 1 --golden customs/docs/golden"`.
- E9: `customs/qwen_drive/run_e2e_test.sh`.
- E6: `customs/qwen_drive/run_unit_tests.sh` (GPU 0; add `-k pipeline_forward` for the pipeline-contract test).

## Next
All episodes done. Optional follow-ups: streaming of the `trajectory` field, throughput benchmarks with larger `max_num_seqs` on the big-VRAM machine (`vllm_omni/deploy/qwen_drive.yaml`), CI wiring (Buildkite) if wanted.
