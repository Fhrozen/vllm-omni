# E8: Online API

Depends on: E7.

## Goal
`vllm serve <hf-dir> --omni --trust-remote-code --deploy-config qwen_drive_plan.yaml` answers a chat request containing the 12 driving images and returns text + trajectories.

## Inputs
- `vllm_omni/entrypoints/openai/serving_chat.py`, `.../protocol/chat_completion.py`, `.../serving_video.py` (action output extraction for Pi0)
- `extra_repos/Qwen-Drive-1.0/src/qwen_drive/scene.py` (`QwenDriveProcessor`, `DrivingScene`), `data/demo/planning_scenes.jsonl`
- S2 verdict (hook location)

## Outputs
- Exact prompt/token builder ported from `QwenDriveProcessor.build_input_ids` + image resizing (`smart_resize`, per-view budgets) inside vllm-omni (no `qwen_drive` import), attached at the hook found in S2
- Request schema `extra_body.qwen_drive`: `mode` (`vqa|direct_planning|reasoning_planning`), `history`, `history_velocity`, `history_acceleration`, `ego_velocity`, `ego_acceleration`, `driving_command`, `nav_command`, `instruction_text?`, `num_samples`, `seed`, `num_steps`
- Response: `choices[0].message.content` (reasoning/VQA text) + `trajectories` (`[n,50,3]`) field
- `customs/qwen_drive/client_example.py` (reads a demo scene, base64 images, sends request)

## Verify
Prompt token ids from the server builder equal reference `build_input_ids` on demo scenes (unit test, CPU). Live request returns finite trajectories.

## Exit criteria
Client example works in docker against a running server; Progress.md updated.
