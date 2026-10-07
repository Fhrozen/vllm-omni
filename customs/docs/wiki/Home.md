# Qwen-Drive in vllm-omni: wiki home

Goal: serve Qwen-Drive-1.0-4B (VQA, direct planning, reasoning planning) with vllm-omni, export one self-contained
HF repo usable with `--trust-remote-code` (no `Qwen-Drive-1.0` package), and test it in docker.

## Read this first (any agent resuming work)
1. [Progress](Progress.md): which episode is done / next, exact commands that were run.
2. [Decisions](Decisions.md): user decisions and design choices. Do not change silently; log changes there.
3. [Findings](Findings.md): append-only facts discovered while working.
4. Your episode page in [episodes/](episodes/).
5. [Environment](Environment.md): docker/uv commands and hardware.
6. [Risks](Risks.md), [Architecture](Architecture.md).

## Rules
- Documentation lives only in `customs/docs`. Never edit `docs/`, `recipes/`, `mkdocs.yml`.
- Run anything GPU/python-related in docker: `customs/qwen_drive/docker_run.sh "<cmd>"` (activates `/workspace/.venv`).
- `extra_repos/` is git-ignored; it holds the reference code (`Qwen-Drive-1.0`) and checkpoint (`Qwen-Drive-1.0-4B`).
- vllm-omni code must never import `qwen_drive` (the reference package). It is only for golden/parity scripts.
- At the end of every episode: update Progress.md, append to Findings.md, commit nothing unless asked.

## Episode index
| ID | Title | Depends on |
|----|-------|------------|
| E0 | Wiki + docker helpers | - |
| E1 | Environment | E0 |
| E2 | Reference golden outputs | E1 |
| E3 | Single HF repo with remote code | E2 |
| E4 | Spikes (unknowns) | E1 |
| E5 | Stage 0/1 VLM in vllm-omni | E3, E4 |
| E6 | Planning expert diffusion pipeline | E3 |
| E7 | Pipeline, deploy, bridges | E5, E6 |
| E8 | Online API | E7 |
| E9 | Tests in docker | E6-E8 |
| E10 | Hardening + final docs | E9 |
