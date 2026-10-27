#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Submit one WeVisOne-VL request to a vllm-omni server through the OpenAI-compatible API.

Copy-pasteable example for the WeVisOne-VL pipeline (see
customs/docs/wiki/WeVisOne-VL.md). Run it against a server started with
``customs/wevisone/serve.sh``. It needs ``requests`` plus the repo package for
the prompt family, so run it from the repo checkout or inside the container.

    # Mask task: text answer plus one PNG part per mask.
    python customs/wevisone/openai_request_example.py --task matting \
        --query "the person" --image extra_repos/WeVisOne-VL/assets/demo/seabreeze.jpg

    # Text-only task: the response carries boxes but no image parts.
    python customs/wevisone/openai_request_example.py --task detection \
        --query "person\nchair" --image extra_repos/WeVisOne-VL/assets/demo/seabreeze.jpg

    # Print the JSON payload, or write it for curl:
    python customs/wevisone/openai_request_example.py --task ocr --print-payload
    python customs/wevisone/openai_request_example.py --task ocr --payload-out payload.json
    curl -s http://localhost:8091/v1/chat/completions -H 'Content-Type: application/json' -d @payload.json

The task instruction is built with ``build_prompt``, which is the family the
checkpoint was trained on; ``--prompt`` sends a one-off instruction instead.
Reference boxes are pixel-space ``[x1, y1, x2, y2]`` and only ``visual_prompt``
reads them.

Payload skeleton:

    {
        "model": "tencent/WeVisOne-VL-4B",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
                {"type": "text", "text": "Locate the instances ... : the person."}
            ]}
        ]
    }

Response: one text choice with the raw structured answer (``<box>`` spans are
kept because the stage must not strip special tokens), plus one image choice
whose parts carry ``mask_index`` -- mask i is the answer to box i.

    choices[0].message.content -> "<ref>the person</ref><box><9><65><753><1000><OBJ></box>"
    choices[1].message.content -> [{"type": "image_url", ..., "mask_index": 0}]

Box order matches mask order, so a client pairs them by enumerating both.
"""

from __future__ import annotations

import argparse
import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from vllm_omni.model_executor.models.wevisone.constants import (
    SYSTEM_PROMPT,
    TASKS,
    build_prompt,
    parse_seg_boxes,
)

DEFAULT_MODEL = "tencent/WeVisOne-VL-4B"
DEFAULT_URL = "http://localhost:8091/v1/chat/completions"
DEFAULT_IMAGE = Path(__file__).resolve().parents[2] / "extra_repos" / "WeVisOne-VL" / "assets" / "demo" / "seabreeze.jpg"


@dataclass
class RequestSpec:
    """One request, so the JSON sample file and the CLI build the same payload."""

    model: str = DEFAULT_MODEL
    image: Path = DEFAULT_IMAGE
    task: str = "freeform"
    query: str = ""
    prompt: str | None = None
    reference_boxes: list[list[int]] | None = None
    image_size: tuple[int, int] | None = None
    max_tokens: int | None = None


def image_part(image: Path) -> dict[str, Any]:
    """Reference the image the way the API accepts it: data URL for local files."""
    if str(image).startswith(("http://", "https://")):
        return {"type": "image_url", "image_url": {"url": str(image)}}
    data = base64.b64encode(Path(image).read_bytes()).decode("utf-8")
    suffix = Path(image).suffix.lstrip(".").lower() or "png"
    return {"type": "image_url", "image_url": {"url": f"data:image/{suffix};base64,{data}"}}


def instruction(spec: RequestSpec) -> str:
    return spec.prompt or build_prompt(
        spec.task,
        spec.query,
        reference_boxes=spec.reference_boxes,
        image_size=spec.image_size or (0, 0),
    )


def build_payload(spec: RequestSpec) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [image_part(spec.image), {"type": "text", "text": instruction(spec)}],
        },
    ]
    payload: dict[str, Any] = {"model": spec.model, "messages": messages}
    if spec.max_tokens:
        payload["max_tokens"] = spec.max_tokens
    return payload


def parse_response(data: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Split a chat response into the text choice and the mask parts.

    Text and masks arrive as separate choices, so collect both instead of
    assuming the first choice holds everything.
    """
    text = ""
    masks: list[dict[str, Any]] = []
    for choice in data.get("choices", []):
        content = choice.get("message", {}).get("content")
        if isinstance(content, str):
            text += content
        elif isinstance(content, list):
            masks.extend(part for part in content if part.get("type") == "image_url")
    return text, masks


def write_result(out_dir: Path, spec: RequestSpec, text: str, masks: list[dict[str, Any]]) -> None:
    """Write ``result.json`` and ``masks/`` in the shape the golden fixtures use."""
    masks_dir = out_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    mask_files = []
    for index, part in enumerate(masks):
        ordinal = int(part.get("mask_index", index))
        data = base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
        name = f"mask_{ordinal:03d}.png"
        (masks_dir / name).write_bytes(data)
        mask_files.append(f"masks/{name}")

    boxes = parse_seg_boxes(text)
    record: dict[str, Any] = {
        "task": spec.task,
        "prompt": instruction(spec),
        "response": text,
        "boxes": boxes,
        # Mask i is the answer to box i, so the mapping is positional; writing
        # it out keeps the pairing explicit for whatever reads the result.
        "mask_to_box": [
            {"mask_index": index, "box": boxes[index]} for index in range(min(len(masks), len(boxes)))
        ],
        "mask_files": mask_files,
        "image": str(spec.image),
    }
    (out_dir / "result.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--image", default=DEFAULT_IMAGE, help="local file path or http(s) URL")
    p.add_argument("--task", default="freeform", choices=TASKS)
    p.add_argument("--query", default="", help="labels (newline separated) or a phrase")
    p.add_argument("--prompt", help="one-off instruction; overrides task + query")
    p.add_argument("--reference-box", type=int, nargs=4, action="append", metavar="X1 Y1 X2 Y2")
    p.add_argument("--image-size", type=int, nargs=2, help="W H used by visual_prompt box encoding")
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--out-dir", help="write result.json and masks/ here; omitted means no files")
    p.add_argument("--print-payload", action="store_true")
    p.add_argument("--payload-out", help="write the payload JSON for curl")
    cli_args = p.parse_args()

    spec = RequestSpec(
        model=cli_args.model,
        image=Path(cli_args.image),
        task=cli_args.task,
        query=cli_args.query,
        prompt=cli_args.prompt,
        reference_boxes=cli_args.reference_box,
        image_size=cli_args.image_size,
        max_tokens=cli_args.max_tokens,
    )
    if not str(spec.image).startswith(("http://", "https://")) and not spec.image.is_file():
        raise SystemExit(f"image not found: {spec.image}")

    payload = build_payload(spec)
    if cli_args.print_payload:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    if cli_args.payload_out:
        Path(cli_args.payload_out).write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    response = requests.post(cli_args.url, json=payload, timeout=600)
    response.raise_for_status()
    text, masks = parse_response(response.json())

    boxes = parse_seg_boxes(text)
    print(f"task:    {spec.task}")
    print(f"response: {text}")
    print(f"boxes:    {boxes}")
    print(f"masks:    {len(masks)}")
    for index, part in enumerate(masks):
        url = part.get("image_url", {}).get("url", "")
        print(f"  mask {part.get('mask_index', index)}: {len(base64.b64decode(url.split(',', 1)[1]))} bytes")

    if cli_args.out_dir:
        write_result(Path(cli_args.out_dir), spec, text, masks)
        print(f"wrote {Path(cli_args.out_dir) / 'result.json'}")


if __name__ == "__main__":
    main()
