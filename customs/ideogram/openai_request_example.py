"""Submit an Ideogram 4 caption to a vllm-omni server through the OpenAI-compatible API.

Copy-pasteable example for `Ideogram4Pipeline` (see customs/ideogram/README.md and
customs/docs/wiki/Ideogram-4.md). Ideogram 4 is text-to-image only: one caption, no
reference images, no negative prompt, and the response is non-streaming only.

    # Generate and save the image:
    python customs/ideogram/openai_request_example.py "a red fox sitting on a park bench"

    # Print the JSON payload, or write it to a file for curl:
    python customs/ideogram/openai_request_example.py "a red fox" --print-payload
    python customs/ideogram/openai_request_example.py "a red fox" --payload-out payload.json
    curl -s http://localhost:8091/v1/chat/completions -H 'Content-Type: application/json' -d @payload.json

Payload skeleton:

    {
        "model": "ideogram-ai/ideogram-4-fp8",
        "messages": [{"role": "user", "content": "a red fox sitting on a park bench"}],
        "extra_args": {
            "preset": "V4_QUALITY_48",
            "height": 1024,
            "width": 1024,
            "seed": 42
        }
    }

`stream` must be omitted or false: the pipeline produces one image per request and cannot
answer a streaming request with incremental chunks.

Response: `choices[0].message.content` is a list of image parts, one per generated image:

    [{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."},
      "stage_durations": {...}, "peak_memory_mb": 0.0}]

Needs only requests. Server: customs/ideogram/serve.sh.
"""

from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path

import requests

DEFAULT_MODEL = "ideogram-ai/ideogram-4-fp8"
DEFAULT_URL = "http://localhost:8091/v1/chat/completions"


def build_payload(args) -> dict:
    extra_args = {
        "preset": args.preset,
        "height": args.height,
        "width": args.width,
        "seed": args.seed,
    }
    if args.num_steps is not None:
        extra_args["num_inference_steps"] = args.num_steps
    if args.guidance_schedule:
        extra_args["guidance_schedule"] = args.guidance_schedule
    if args.mu is not None:
        extra_args["mu"] = args.mu
    if args.std is not None:
        extra_args["std"] = args.std

    return {
        "model": args.model,
        "messages": [{"role": "user", "content": args.caption}],
        "extra_args": extra_args,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("caption", help="The caption to generate from.")
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--preset", default="V4_QUALITY_48", choices=("V4_QUALITY_48", "V4_DEFAULT_20", "V4_TURBO_12"))
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-steps", type=int, help="Override the preset's step count.")
    p.add_argument("--guidance-schedule", type=float, nargs="+", help="Per-step guidance weight, loop order.")
    p.add_argument("--mu", type=float, help="Logit-normal mean; defaults to the resolution-aware value.")
    p.add_argument("--std", type=float, help="Logit-normal std; defaults to the resolution-aware value.")
    p.add_argument("--out-dir", type=Path, default=Path("/tmp/ideogram4"))
    p.add_argument("--payload-out", type=Path, help="Write the JSON payload here instead of sending it.")
    p.add_argument("--print-payload", action="store_true")
    args = p.parse_args()

    payload = build_payload(args)
    if args.print_payload:
        print(json.dumps(payload, indent=2))
    if args.payload_out:
        args.payload_out.write_text(json.dumps(payload, indent=2))
        print(f"Payload written to {args.payload_out}")
        return

    started = time.time()
    response = requests.post(args.url, json=payload, timeout=600)
    response.raise_for_status()
    body = response.json()
    if body.get("error"):
        raise SystemExit(f"Server error: {json.dumps(body['error'])}")

    content = body["choices"][0]["message"]["content"]
    parts = content if isinstance(content, list) else [{"type": "text", "text": content}]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for i, part in enumerate(parts):
        if part.get("type") != "image_url":
            print(f"Text part: {part.get('text', '')}")
            continue
        image = base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
        path = args.out_dir / f"{body.get('id', 'request')}_{i}.png"
        path.write_bytes(image)
        print(f"Wrote {path} ({len(image)} bytes)")

    usage = body.get("usage") or {}
    if usage.get("time"):
        print(f"server generation time: {usage['time']:.2f}s")
    print(f"round trip: {time.time() - started:.2f}s")


if __name__ == "__main__":
    main()
