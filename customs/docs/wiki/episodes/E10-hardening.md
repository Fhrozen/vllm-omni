# E10: Hardening and final docs

Depends on: E9.

## Tasks
- `num_samples>1` batching, abort handling, concurrent requests.
- Memory tuning on 16 GB GPUs (`gpu_memory_utilization`, `max_model_len`), measure KV payload and cross-GPU connector latency (R9).
- Re-evaluate D2/D6 (single AR stage?) and prefix caching (S6) with measurements.
- Final docs in `customs/docs/wiki/`: Architecture.md, Usage.md (serve + client commands), Testing.md, Risks.md, Progress.md.

## Exit criteria
All wiki pages current; Progress.md all `done`.
