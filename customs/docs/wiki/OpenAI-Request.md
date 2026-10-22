# Submitting requests to Qwen-Drive over the OpenAI API

How to build and send a request to a Qwen-Drive-1.0-4B server launched with vllm-omni.
The endpoint is the standard OpenAI-compatible `POST /v1/chat/completions`; Qwen-Drive adds
two extensions: planner inputs in `extra_args` and a top-level `trajectory` response field.
A ready-to-run client: `customs/qwen_drive/openai_request_example.py`.

## 1. Start the server

```bash
customs/qwen_drive/serve.sh            # docker, host network, port 8091
# equivalent:
vllm serve /workspace/extra_repos/Qwen-Drive-1.0-4B-hf --omni \
    --deploy-config vllm_omni/deploy/qwen_drive_fp8.yaml \
    --chat-template /workspace/extra_repos/Qwen-Drive-1.0-4B-hf/chat_template_drive.jinja \
    --trust-remote-code --port 8091 --stage-init-timeout 900
```

Health check: `GET /health` returns 200 when ready (first start takes several minutes).

## 2. Request schema

One user message. Content parts are in **reading order**: view label, `frame: i`, image, ...,
instruction. There are 12 images: 3 views (FRONT, FRONT LEFT, FRONT RIGHT) x 4 frames
(t-1.5s, t-1.0s, t-0.5s, t=0s). The last text part is the driving instruction (history poses
+ navigation command), taken verbatim from the scene record's `messages[0].content`.

| Field | Value | Notes |
|---|---|---|
| `model` | HF repo path/name | e.g. `/workspace/extra_repos/Qwen-Drive-1.0-4B-hf` |
| `temperature` | `0.0` | greedy |
| `messages` | one user message | text + `image_url` parts as above |
| `chat_template_kwargs` | `{"qwen_drive_mode": "direct"\|"reasoning"}` | omit for VQA |
| `modalities` | `["trajectory"]` / `["text","trajectory"]` / `["text"]` | direct / reasoning / VQA |
| `max_tokens` | `1` (direct) / `256` (reasoning, VQA) | direct stage-0 text is unused |
| `min_tokens` | `10` (reasoning only) | guarantees a rationale sentence |
| `extra_args` | planner inputs, see below | **flat top-level fields are NOT routed** |
| `stream` | must be omitted/false | `trajectory` is non-streaming only |

### `extra_args` (routed to the diffusion planner stage)

| Key | Shape | Units / values |
|---|---|---|
| `history` | `[16, 3]` | (x, y, heading), metres/radians, ego frame, 10 Hz, oldest first, last row = (0,0,0) |
| `history_velocity` | `[16, 2]` | m/s |
| `history_acceleration` | `[16, 2]` | m/s^2 |
| `ego_velocity` | `[2]` | m/s |
| `ego_acceleration` | `[2]` | m/s^2 |
| `driving_command` | `[4]` | one-hot: left, straight, right, unknown |
| `nav_command` | int | 0 straight, 1 left, 2 right |
| `num_samples` | int, optional | number of trajectories to sample (default 6 in the example) |
| `num_steps` | int, optional | flow-matching Euler steps (default 10) |
| `noise_seed` | int, optional | seed for reproducible sampling |

From a scene record (`planning_scenes.jsonl`): prefer the `*_1p5s_10hz` history keys
(`hist_traj_1p5s_10hz`, `hist_vel_1p5s_10hz`, `hist_acc_1p5s_10hz`), falling back to `*_10hz`,
and pad to 16 rows by repeating the oldest row. `ego_velocity`/`ego_acceleration`/
`driving_command` come from `trajectory.ego_status`; `nav_command` from `trajectory.nav_command`.

### Image preprocessing (client-side, mandatory)

The server does not resize. Each frame must be sent exactly as the reference preprocessing
produces it, or the model sees different pixels than it was trained on:

1. Resize to the per-frame target size from the scene record: history frames `384x416`,
   current frame (last of each view) `720x799` (PIL bicubic).
2. `smart_resize` to a multiple of 32 within the pixel budget: 921600 px for current frames,
   174080 px for history frames.
3. Encode as PNG and send as a `data:image/png;base64,...` URL in an `image_url` part.

`openai_request_example.py::prepare_image` implements this.

## 3. Response

```json
{
  "choices": [{"message": {"role": "assistant", "content": "<text or empty>"}}],
  "trajectory": [[[x, y, heading], ... 50 points] ... num_samples samples]
}
```

- `choices[0].message.content`: the VLM text (empty for direct; the one-sentence rationale for
  reasoning; the answer for VQA).
- `trajectory` (top-level, not in `choices`): `[num_samples, 50, 3]` — x, y, heading in
  metres/radians, ego frame of the current timestamp, 10 Hz, 5 s horizon. `null`/absent for VQA.

## 4. Worked example (scene 0 of the demo data)

Demo data: `extra_repos/Qwen-Drive-1.0/data/demo/planning_scenes.jsonl` (4 scenes) with images
in `data/demo/frames/` (`scene0_0.jpg` ... `scene0_11.jpg`). Scene 0: `nav_command` 0
(straight), `driving_command` `[0, 1, 0, 0]`.

Generate the payload and send it with curl:

```bash
customs/qwen_drive/docker_run.sh "python customs/qwen_drive/openai_request_example.py \
    --mode direct --scene 0 --payload-out /tmp/qd_payload.json"
curl -s http://localhost:8091/v1/chat/completions \
    -H 'Content-Type: application/json' -d @/tmp/qd_payload.json | python -m json.tool
```

Or run the full client (builds, sends, prints text + trajectory endpoint):

```bash
customs/qwen_drive/docker_run.sh "python customs/qwen_drive/openai_request_example.py --mode direct --scene 0"
```

Truncated payload shape (direct mode; image URLs elided):

```json
{
  "model": "/workspace/extra_repos/Qwen-Drive-1.0-4B-hf",
  "temperature": 0.0,
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "<FRONT VIEW>"},
    {"type": "text", "text": "frame: 0"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
    {"type": "text", "text": "frame: 1"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
    {"type": "text", "text": "frame: 2"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
    {"type": "text", "text": "frame: 3"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
    {"type": "text", "text": "<FRONT LEFT VIEW>"},
    "... 8 more frame/image parts ...",
    {"type": "text", "text": "The input images are organized by camera view. ... 2. Active navigation command: [GO STRAIGHT]"}
  ]}],
  "chat_template_kwargs": {"qwen_drive_mode": "direct"},
  "modalities": ["trajectory"],
  "max_tokens": 1,
  "extra_args": {
    "history": [[0.0, 0.0, 0.0], "... 16 rows ..."],
    "history_velocity": [[0.0, 0.0], "... 16 rows ..."],
    "history_acceleration": [[0.0, 0.0], "... 16 rows ..."],
    "ego_velocity": [0.0, -0.0],
    "ego_acceleration": [0.0, 0.0],
    "driving_command": [0, 1, 0, 0],
    "nav_command": 0,
    "num_samples": 6
  }
}
```

## 5. Mode differences

| | VQA | Direct planning | Reasoning planning |
|---|---|---|---|
| `modalities` | `["text"]` | `["trajectory"]` | `["text", "trajectory"]` |
| `chat_template_kwargs` | omit | `{"qwen_drive_mode": "direct"}` | `{"qwen_drive_mode": "reasoning"}` |
| `max_tokens` / `min_tokens` | 256 | 1 | 256 / 10 |
| `extra_args` | not needed | required | required |
| Last text part | free question (images only, no instruction) | scene instruction | scene instruction + fixed reasoning suffix |
| Response | text | trajectory | rationale text + trajectory |

## 6. Limits

- `trajectory` is **non-streaming only** (streaming returns text only).
- Prefix caching is off (hybrid model); the multimodal processor cache is off (stage 1 reuses
  stage 0's processed images).
- See [Usage](Usage.md) for deploy configs and [Risks](Risks.md) R10 for the streaming limitation.
