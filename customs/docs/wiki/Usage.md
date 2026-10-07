# Usage

1. Export the repo: `customs/qwen_drive/export_hf.sh` -> `extra_repos/Qwen-Drive-1.0-4B-hf` (also ships `chat_template_drive.jinja`).
2. Serve: `customs/qwen_drive/serve.sh [deploy]` (docker, host network, port 8091). Equivalent command:
   `vllm serve <hf-dir> --omni --deploy-config vllm_omni/deploy/qwen_drive_fp8.yaml --chat-template <hf-dir>/chat_template_drive.jinja --trust-remote-code`
   - `qwen_drive_fp8.yaml`: FP8 VLM stages for 16 GB GPUs (GPU0: stage 0 + planner, GPU1: stage 1). `qwen_drive.yaml`: BF16 (~9 GB per VLM stage; set `devices` per stage; the planner needs ~2.5 GB).
   - Planner: `model_config.planner_subfolder` of stage 2 (`planner-rl` default, `planner-sft`).
3. Request (see `customs/qwen_drive/client_example.py`, modes `vqa|direct|reasoning`):
   - `messages`: one user message with text and `image_url` parts in reading order (view label, `frame: i`, image, ..., instruction). Images must be pre-resized like the reference (the client does it).
   - `chat_template_kwargs: {"qwen_drive_mode": "direct"|"reasoning"}` (omit for VQA).
   - `modalities`: `["text"]` (VQA), `["trajectory"]` (direct), `["text","trajectory"]` (reasoning: text + trajectory).
   - `max_tokens`: 1 for direct (the stage-0 text is unused), 256 + `min_tokens: 10` for reasoning, `temperature: 0`.
   - `extra_args` (routed to the planner stage): `history [16,3]`, `history_velocity [16,2]`, `history_acceleration [16,2]`, `ego_velocity`, `ego_acceleration`, `driving_command [4]` (or `ego_status [8]`), `nav_command` (0 straight, 1 left, 2 right), optional `num_samples`, `num_steps`, `noise_seed`. Flat top-level fields are NOT routed; use `extra_args`.
   - Response: `choices[0].message.content` (text) and `trajectory` `[num_samples, 50, 3]` (x, y, heading; metres/radians, ego frame, 10 Hz).
4. Limits: non-streaming only for the `trajectory` field; prefix caching is off (hybrid model); the multimodal processor cache is off (stage 1 reuses stage 0's processed images).
