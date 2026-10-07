# E9: Tests (docker)

Depends on: E6-E8 (unit tests can start after E6). Use the `vllm-omni-test` skill (`.claude/skills/vllm-omni-test/SKILL.md`) for markers and CI wiring rules.

## Inputs
- `tests/e2e/online_serving/test_bagel.py`, `tests/helpers/runtime.py` (`OmniServer`, `OmniServerParams`), `tests/helpers/mark.py`, `customs/qwen_drive/docker_run.sh`
- Golden files (`customs/docs/golden/`), HF export dir `extra_repos/Qwen-Drive-1.0-4B-hf` (container path `/workspace/extra_repos/Qwen-Drive-1.0-4B-hf`)

## Outputs
- Unit: `tests/diffusion/models/qwen_drive/test_planning_expert.py`, `tests/model_executor/stage_input_processors/test_qwen_drive_processor.py`, KV filter test
- E2E: `tests/e2e/online_serving/test_qwen_drive.py` (VQA text; direct planning shape/finite/ADE vs golden; reasoning returns text + trajectory; both planners)
- `customs/qwen_drive/run_unit_tests.sh`, `run_e2e_test.sh` (docker wrappers: `docker_run.sh "pytest -v ..."`)
- `customs/docs/wiki/Testing.md`: commands, markers, prerequisites, CI-like invocation

## Exit criteria
All unit and e2e tests pass in docker on the 2-GPU host. Progress.md updated.
