"""Submit a Qwen-Drive demo scene to a vllm-omni server through the OpenAI-compatible API.

This is the documented, copy-pasteable example for requesting Qwen-Drive-1.0-4B served by
vllm-omni (see customs/docs/wiki/OpenAI-Request.md). It builds a /v1/chat/completions
payload from one demo scene in planning_scenes.jsonl and posts it to the server.

    # Direct planning (trajectory only):
    python customs/qwen_drive/openai_request_example.py --mode direct --scene 0

    # Reasoning planning (one-sentence rationale + trajectory):
    python customs/qwen_drive/openai_request_example.py --mode reasoning --scene 0

    # VQA (text only, free question):
    python customs/qwen_drive/openai_request_example.py --mode vqa --question "What is ahead?"

    # Print the JSON payload (image data URLs truncated) or write it to a file for curl:
    python customs/qwen_drive/openai_request_example.py --mode direct --scene 0 --print-payload
    python customs/qwen_drive/openai_request_example.py --mode direct --scene 0 --payload-out payload.json
    curl -s http://localhost:8091/v1/chat/completions -H 'Content-Type: application/json' -d @payload.json

Minimal payload skeleton (direct planning):

    {
        "model": "<hf-repo>",
        "temperature": 0.0,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "<FRONT VIEW>"},
            {"type": "text", "text": "frame: 0"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
            ... 12 images total (3 views x 4 frames, pre-resized client-side) ...
            {"type": "text", "text": "<driving instruction with history poses + nav command>"}
        ]}],
        "chat_template_kwargs": {"qwen_drive_mode": "direct"},
        "modalities": ["trajectory"],
        "max_tokens": 1,
        "extra_args": {
            "history": [[x, y, heading] x 16],
            "history_velocity": [[vx, vy] x 16],
            "history_acceleration": [[ax, ay] x 16],
            "ego_velocity": [vx, vy],
            "ego_acceleration": [ax, ay],
            "driving_command": [0, 1, 0, 0],
            "nav_command": 0,
            "num_samples": 6
        }
    }

Response: choices[0].message.content (text) plus a top-level "trajectory" field
[num_samples, 50, 3] (x, y, heading; metres/radians, ego frame, 10 Hz, 5 s horizon).
The trajectory field is only present in non-streaming responses.

Needs only requests, numpy and pillow. Images are resized exactly like the reference
preprocessing so the server sees the same pixels. Server: customs/qwen_drive/serve.sh.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

REASONING_REQUEST = "\n\nGive a one-sentence brief reasoning of the ego's future driving decision ONLY."
FACTOR = 32  # patch size 16 x spatial merge 2
CURRENT_PIXELS, HISTORY_PIXELS = 921600, 174080


def smart_resize(height: int, width: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """Round to multiples of FACTOR and scale into [min_pixels, max_pixels], like the reference."""
    h, w = round(height / FACTOR) * FACTOR, round(width / FACTOR) * FACTOR
    if h * w > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h, w = math.floor(height / beta / FACTOR) * FACTOR, math.floor(width / beta / FACTOR) * FACTOR
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h, w = math.ceil(height * beta / FACTOR) * FACTOR, math.ceil(width * beta / FACTOR) * FACTOR
    return h, w


def prepare_image(path: Path, target_size: tuple[int, int] | None, budget: int) -> str:
    """Resize a frame to the reference target size, then smart_resize within the pixel budget.

    Returns a base64 PNG data URL for the image_url part.
    """
    image = Image.open(path).convert("RGB")
    max_pixels = budget
    if target_size is not None:
        image = image.resize(target_size, Image.BICUBIC)
        max_pixels = 12800 * FACTOR**2
    height, width = smart_resize(image.height, image.width, 4 * FACTOR**2, max_pixels)
    image = image.resize((width, height), Image.BICUBIC)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def history(trajectory: dict, key: str, count: int = 16) -> list:
    """Last `count` history rows for hist_traj/hist_vel/hist_acc, preferring the 1.5 s 10 Hz variant."""
    values = trajectory.get(f"{key}_1p5s_10hz") or trajectory[f"{key}_10hz"]
    array = np.asarray(values, dtype=np.float32)
    if array.shape[0] < count:
        array = np.concatenate([np.repeat(array[:1], count - array.shape[0], axis=0), array], axis=0)
    return array[-count:].tolist()


def build_request(record: dict, image_root: Path, mode: str, question: str, model: str) -> dict:
    """Build the /v1/chat/completions body for one demo scene record.

    modes: vqa (text only), direct (trajectory only), reasoning (text + trajectory).
    """
    content = record["messages"][0]["content"]
    images = [item for item in content if "image" in item]
    per_view = len(images) // 3
    parts, image_index = [], 0
    for item in content:
        if "text" in item:
            parts.append({"type": "text", "text": item["text"]})
            continue
        is_current = (image_index % per_view) == per_view - 1
        size = (int(item["resized_width"]), int(item["resized_height"])) if item.get("resized_width") else None
        url = prepare_image(image_root / item["image"], size, CURRENT_PIXELS if is_current else HISTORY_PIXELS)
        parts.append({"type": "image_url", "image_url": {"url": url}})
        image_index += 1

    body: dict = {"model": model, "temperature": 0.0}
    if mode == "vqa":
        # The VQA prompt is the images followed by the question (no driving instruction).
        parts = [p for p in parts if p["type"] == "image_url"] + [{"type": "text", "text": question}]
        body.update(messages=[{"role": "user", "content": parts}], modalities=["text"], max_tokens=256)
        return body

    if mode == "reasoning":
        parts[-1] = {"type": "text", "text": parts[-1]["text"] + REASONING_REQUEST}
    trajectory = record["trajectory"]
    ego = trajectory["ego_status"]
    body.update(
        messages=[{"role": "user", "content": parts}],
        chat_template_kwargs={"qwen_drive_mode": "direct" if mode == "direct" else "reasoning"},
        modalities=["trajectory"] if mode == "direct" else ["text", "trajectory"],
        max_tokens=1 if mode == "direct" else 256,
        # Planner inputs travel as extra_args, which the server routes to the diffusion (planner) stage.
        # Flat top-level fields are NOT routed.
        extra_args=dict(
            history=history(trajectory, "hist_traj"),
            history_velocity=history(trajectory, "hist_vel"),
            history_acceleration=history(trajectory, "hist_acc"),
            ego_velocity=ego["ego_velocity"],
            ego_acceleration=ego["ego_acceleration"],
            driving_command=ego["driving_command"],
            nav_command=int(trajectory["nav_command"]),
            num_samples=6,
        ),
    )
    if mode == "reasoning":
        body["min_tokens"] = 10
    return body


def truncate_payload(body: dict) -> dict:
    """Copy of the payload with image data URLs shortened, for readable printing."""
    shown = json.loads(json.dumps(body))
    for part in shown["messages"][0]["content"]:
        if part.get("type") == "image_url":
            url = part["image_url"]["url"]
            part["image_url"]["url"] = url[:40] + f"... ({len(url)} chars)"
    return shown


def wait_ready(url: str, timeout: float) -> None:
    deadline = time.time() + timeout
    while True:
        try:
            if requests.get(f"{url}/health", timeout=5).status_code == 200:
                return
        except requests.RequestException:
            pass
        if time.time() > deadline:
            raise TimeoutError(f"server at {url} not ready after {timeout:.0f}s")
        time.sleep(5)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--url", default="http://localhost:8091")
    p.add_argument("--model", default="/workspace/extra_repos/Qwen-Drive-1.0-4B-hf")
    p.add_argument("--scenes", default="/workspace/extra_repos/Qwen-Drive-1.0/data/demo/planning_scenes.jsonl")
    p.add_argument("--image-root", type=Path, default=Path("/workspace/extra_repos/Qwen-Drive-1.0/data/demo"))
    p.add_argument("--scene", type=int, default=0)
    p.add_argument("--mode", choices=["vqa", "direct", "reasoning"], default="direct")
    p.add_argument("--question", default="Describe the traffic scene and the safest action.")
    p.add_argument("--print-payload", action="store_true", help="print the payload (images truncated) and exit")
    p.add_argument("--payload-out", type=Path, default=None, help="write the full payload JSON to this file and exit")
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--wait", type=float, default=0, help="seconds to wait for the server to become healthy")
    args = p.parse_args()

    with open(args.scenes) as f:
        record = json.loads(f.readlines()[args.scene])
    body = build_request(record, args.image_root, args.mode, args.question, args.model)

    if args.print_payload:
        print(json.dumps(truncate_payload(body), indent=2))
        return
    if args.payload_out is not None:
        args.payload_out.write_text(json.dumps(body))
        print(f"wrote payload to {args.payload_out} ({args.payload_out.stat().st_size / 1e6:.1f} MB)")
        print(f"curl -s {args.url}/v1/chat/completions -H 'Content-Type: application/json' -d @{args.payload_out}")
        return

    wait_ready(args.url, args.wait)
    reply = requests.post(f"{args.url}/v1/chat/completions", json=body, timeout=args.timeout)
    reply.raise_for_status()
    data = reply.json()

    for choice in data.get("choices", []):
        print("text:", (choice["message"].get("content") or "").strip()[:400])
    trajectory = data.get("trajectory")
    if trajectory is not None:
        array = np.asarray(trajectory)
        print("trajectory:", array.shape, "endpoint of sample 0:", np.round(array[0, -1], 3).tolist())
    else:
        print("trajectory: None (VQA mode or streaming)")


if __name__ == "__main__":
    main()
