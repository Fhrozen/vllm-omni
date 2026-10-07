# Risks

| # | Risk | Mitigation / owner episode |
|---|------|----------------------------|
| R1 | 16 GB GPUs: VLM (8.5 GB) + vision activations + KV; two AR stages cannot share one GPU | D6; small `max_model_len`, `gpu_memory_utilization` ~0.85; consider single-AR-stage design (E4) |
| R2 | KV extraction on hybrid (GDN + full-attn) caches may mis-handle non-attention layers | S4; add `kv_layer_indices` filter; verify K/V against HF reference (E5) |
| R3 | vLLM never computes KV of the last sampled token; reference needs `<|im_end|>\n` in cache | D2 (prefill-only stage 1) |
| R4 | Chat template/tokenization differs from `build_input_ids` | S2; exact prompt-token builder (E8) |
| R5 | bf16 rotary/Fourier rounding differences change trajectories | port quirks verbatim; golden tolerance (E6) |
| R6 | Prefix caching on hybrid models may be unsupported -> double prefill (~4.3k tokens) | S6; acceptable cost |
| R7 | Blackwell sm_120 kernels (FA, GDN) availability in vllm 0.31 | verify in E5; fall back to SDPA/triton |
| R8 | HF dynamic-module loading of sibling files and nested `AutoConfig` sub_configs | E3 test in a venv without `qwen_drive` |
| R9 | Cross-GPU KV transfer via shared memory (CPU) latency, ~140 MB per 4.3k-token scene | measure in E10 |
| R10 | Streaming responses do not carry the `trajectory` field | open by design; non-streaming only (Usage.md); VQA streaming tested |
| R11 | Concurrency/abort behavior | closed: 5 concurrent requests and a client abort tested (E10) |
| R12 | BF16 numbers | closed: measured on 2x16 GB with deploy_bf16_16gb.yaml; deviation is kernel-noise level |
