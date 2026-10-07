# E4: Spikes

Depends on: E1. Output of this episode = answers in Findings.md (+ Decisions.md changes). Prototype code goes in `customs/qwen_drive/spikes/` only; do not touch `vllm_omni/` yet.

| ID | Question | How | Default if inconclusive |
|----|----------|-----|-------------------------|
| S1 | Per-request routing of 1-stage (VQA) vs 3-stage (plan) | read `vllm_omni/entrypoints/openai/serving_chat.py`, `vllm_omni/config/stage_config.py`, `modalities` handling | two deploy yamls (D3) |
| S2 | Where to hook an exact prompt-token builder (images + prompt_token_ids) in chat serving and the input processor | `serving_chat.py`, `vllm_omni/inputs/`, `vllm_omni/engine/*input*` | offline API (`prompt_token_ids` + `multi_modal_data`) first, HTTP later |
| S3 | How to provide the rope anchor to stage 2 | `gpu_ar_model_runner.py` ~L857-886 `custom_metadata`; or recompute from `input_ids`+`image_grid_thw` | `custom_metadata` |
| S4 | KV extraction with hybrid caches: does `kv_caches` include GDN layers; what does `normalize_layer_kv` (`distributed/omni_connectors/utils/kv_utils.py` ~L400-501) return; can one AR stage append `<|im_end|>\n` instead of a second stage | run a tiny vllm-omni AR stage on Qwen3.5 in docker and inspect | add `kv_layer_indices` filter (D4); keep D2 |
| S5 | How to build vllm `Qwen3_5ForConditionalGeneration` from `hf_config.vlm_config` and map `vlm.` weight prefix | read `vllm/model_executor/models/qwen3_5.py` in `/workspace/.venv/lib/python3.12/site-packages`, `hf_to_vllm_mapper` | wrapper subclass with patched vllm_config + mapper |
| S6 | Does vllm 0.31 support prefix caching for hybrid Qwen3.5 (to skip the re-prefill in stage 1) | start `vllm serve` with `--enable-prefix-caching` on the 4B VLM | accept double prefill |
| S7 | Can vllm run Qwen3.5-4B VLM on 16 GB sm_120 with max_model_len ~8192 | smoke run with `vllm serve` using the HF-exported dir (needs E3) or the original dir with a `hf_overrides` for arch | lower context / enforce_eager |

## Exit criteria
Each spike has a verdict and a link to evidence in Findings.md. Update Decisions.md (D2/D3/D4/D6) if a verdict changes the design.
