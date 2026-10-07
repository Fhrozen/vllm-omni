# Progress

Update this file at the end of every episode. Status: `todo` | `in-progress` | `done` | `blocked`.

| ID | Status | Notes / outputs |
|----|--------|-----------------|
| E0 | done | wiki + `customs/qwen_drive/{docker_run,setup_env,env_check}.sh`; `run_ref_baseline/run_unit_tests/run_e2e_test/export_hf` stubs still to be created by their episodes |
| E1 | done | vllm 0.31.0, transformers 5.14.1, torch 2.13.0+cu132; `Qwen3_5ForConditionalGeneration` registered in vllm; `qwen_drive` reference installed (--no-deps). flash_attn/fla/causal_conv1d NOT installed (HF reference falls back to torch GDN, slow). |
| E2 | done | `customs/qwen_drive/{ref_golden.py,run_ref_baseline.sh}`; golden in `customs/docs/golden/` (3 scenes, 6 samples, seed 42, sdpa bf16), K/V in `extra_repos/golden_kv/` (638 MB, git-ignored); two runs bit-identical |
| E3 | done | `customs/qwen_drive/{export_hf.py,export_hf.sh,verify_hf_export.py}`; `extra_repos/Qwen-Drive-1.0-4B-hf`; remote-code load matches golden bit-for-bit; see HF-Export.md |
| E4 | todo | |
| E5 | todo | |
| E6 | todo | |
| E7 | todo | |
| E8 | todo | |
| E9 | todo | |
| E10 | todo | |

## Commands run (reproducible)
- E1: `customs/qwen_drive/setup_env.sh` then `customs/qwen_drive/env_check.sh`.
- E2: `customs/qwen_drive/run_ref_baseline.sh --num-scenes 3` (runs GPU 0 only, twice, then compares).
- E3: `customs/qwen_drive/export_hf.sh` (export on host, verify in docker GPU 0).

## Next
E4 (spikes) is next, then E5 (VLM stage) and E6 (expert pipeline) in parallel. Unit/golden inputs for E5/E6 are ready (E2), the HF dir for serving is ready (E3).
