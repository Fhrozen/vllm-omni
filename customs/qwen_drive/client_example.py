"""Send a Qwen-Drive demo scene to a vllm-omni server through /v1/chat/completions.

    python customs/qwen_drive/client_example.py --mode direct --scene 0
    modes: vqa | direct | reasoning

Needs only requests, numpy and pillow; the images are resized exactly like the reference preprocessing so the server
sees the same pixels. Server: customs/qwen_drive/serve.sh.
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
    h, w = round(height / FACTOR) * FACTOR, round(width / FACTOR) * FACTOR
    if h * w > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h, w = math.floor(height / beta / FACTOR) * FACTOR, math.floor(width / beta / FACTOR) * FACTOR
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h, w = math.ceil(height * beta / FACTOR) * FACTOR, math.ceil(width * beta / FACTOR) * FACTOR
    return h, w


def prepare_image(path: Path, target_size: tuple[int, int] | None, budget: int) -> str:
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
    values = trajectory.get(f"{key}_1p5s_10hz") or trajectory[f"{key}_10hz"]
    array = np.asarray(values, dtype=np.float32)
    if array.shape[0] < count:
        array = np.concatenate([np.repeat(array[:1], count - array.shape[0], axis=0), array], axis=0)
    return array[-count:].tolist()


def build_request(record: dict, image_root: Path, mode: str, question: str, model: str) -> dict:
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
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8091")
    p.add_argument("--model", default="/workspace/extra_repos/Qwen-Drive-1.0-4B-hf")
    p.add_argument("--scenes", default="/workspace/extra_repos/Qwen-Drive-1.0/data/demo/planning_scenes.jsonl")
    p.add_argument("--image-root", type=Path, default=Path("/workspace/extra_repos/Qwen-Drive-1.0/data/demo"))
    p.add_argument("--scene", type=int, default=0)
    p.add_argument("--mode", choices=["vqa", "direct", "reasoning"], default="direct")
    p.add_argument("--question", default="Describe the traffic scene and the safest action.")
    p.add_argument("--golden", type=Path, default=None, help="directory with the E2 golden files for a comparison")
    p.add_argument("--planner", default="rl")
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--wait", type=float, default=0, help="seconds to wait for the server to become healthy")
    args = p.parse_args()

    wait_ready(args.url, args.wait)
    with open(args.scenes) as f:
        record = json.loads(f.readlines()[args.scene])
    body = build_request(record, args.image_root, args.mode, args.question, args.model)
    reply = requests.post(f"{args.url}/v1/chat/completions", json=body, timeout=args.timeout)
    reply.raise_for_status()
    data = reply.json()

    for choice in data.get("choices", []):
        print("text:", (choice["message"].get("content") or "").strip()[:400])
    trajectory = data.get("trajectory")
    if trajectory is not None:
        array = np.asarray(trajectory)
        print("trajectory:", array.shape, "endpoint of sample 0:", np.round(array[0, -1], 3).tolist())
        if args.golden is not None and args.mode != "vqa":
            want = np.load(args.golden / f"traj_{args.planner}_{args.mode}_{args.scene}.npy")
            ade = np.linalg.norm(array[..., :2] - want[..., :2], axis=-1).mean()
            print(f"ADE vs golden: {ade:.4f} m")


if __name__ == "__main__":
    main()
