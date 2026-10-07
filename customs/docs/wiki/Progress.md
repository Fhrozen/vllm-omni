# Progress

Update this file at the end of every episode. Status: `todo` | `in-progress` | `done` | `blocked`.

| ID | Status | Notes / outputs |
|----|--------|-----------------|
| E0 | done | wiki + `customs/qwen_drive/{docker_run,setup_env,env_check}.sh`; `run_ref_baseline/run_unit_tests/run_e2e_test/export_hf` stubs still to be created by their episodes |
| E1 | done | vllm 0.31.0, transformers 5.14.1, torch 2.13.0+cu132; `Qwen3_5ForConditionalGeneration` registered in vllm; `qwen_drive` reference installed (--no-deps). flash_attn/fla/causal_conv1d NOT installed (HF reference falls back to torch GDN, slow). |
| E2 | done | `customs/qwen_drive/{ref_golden.py,run_ref_baseline.sh}`; golden in `customs/docs/golden/` (3 scenes, 6 samples, seed 42, sdpa bf16), K/V in `extra_repos/golden_kv/` (638 MB, git-ignored); two runs bit-identical |
| E3 | done | `customs/qwen_drive/{export_hf.py,export_hf.sh,verify_hf_export.py}`; `extra_repos/Qwen-Drive-1.0-4B-hf`; remote-code load matches golden bit-for-bit; see HF-Export.md |
| E4 | done | spike scripts in `customs/qwen_drive/spikes/`; verdicts in Findings.md ("Spikes (E4)"); `video_preprocessor_config.json` added to the export; Decisions D1/D3/D3b/D4 updated. Open items carried to E5: KV list-index mapping on hybrid caches, anchor source, flatten-config mechanism; E8: `trajectory` response branch; E10: prefix-cache effectiveness |
| E5 | todo | |
| E6 | done | `vllm_omni/diffusion/models/qwen_drive/{__init__,planning_expert,pipeline_qwen_drive}.py`; registry entry `QwenDrivePlannerPipeline` (+ no-cache-accel, post-process) in `vllm_omni/diffusion/registry.py`; tests `tests/diffusion/models/qwen_drive/test_planning_expert.py` (19 pass: 5 CPU units, 12 golden parity at atol 1e-3, 2 pipeline-contract); run with `customs/qwen_drive/run_unit_tests.sh`. Golden now also stores `planner_inputs_<i>.npz` |
| E7 | todo | |
| E8 | todo | |
| E9 | todo | |
| E10 | todo | |

## Commands run (reproducible)
- E1: `customs/qwen_drive/setup_env.sh` then `customs/qwen_drive/env_check.sh`.
- E2: `customs/qwen_drive/run_ref_baseline.sh --num-scenes 3` (runs GPU 0 only, twice, then compares).
- E3: `customs/qwen_drive/export_hf.sh` (export on host, verify in docker GPU 0).
- E4: see Findings.md "Spikes (E4)" for the exact commands.
- E6: `customs/qwen_drive/run_unit_tests.sh` (GPU 0; add `-k pipeline_forward` for the pipeline-contract test).

## Next
E5 (VLM stage + KV layer filter) is next; then E7 (pipeline/deploy wiring), E8, E9, E10. The expert pipeline contract E7 must satisfy: `req.past_key_values.{key_cache,value_cache}` (lists, `None` for skipped layers, `[seq, 4, 256]` each, 8 non-None), `req.kv_metadata["rope_anchor"]` (int), `sampling_params.extra_args` keys `history [16,3]`, `history_velocity [16,2]`, `history_acceleration [16,2]`, `ego_status [8]` (or `ego_velocity`+`ego_acceleration`+`driving_command`), `nav_command`, optional `num_samples`, `seed`, `num_steps`, `reasoning`; deploy `model_config.planner_subfolder` (default `planner-rl`), `model_class_name: QwenDrivePlannerPipeline`.
