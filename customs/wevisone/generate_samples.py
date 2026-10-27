# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Run every sample in ``samples.json`` against a running WeVisOne-VL server.

Start the server with ``customs/wevisone/serve.sh``, then:

    python customs/wevisone/generate_samples.py

    # A subset, a different server, or a different output directory:
    python customs/wevisone/generate_samples.py --only segmentation matting
    python customs/wevisone/generate_samples.py --url http://localhost:8091/v1/chat/completions
    python customs/wevisone/generate_samples.py --out-dir /tmp/wevisone

Each sample writes ``<out-dir>/<id>/result.json`` plus ``masks/mask_NNN.png``, in
the same shape the golden fixtures use, so ``--golden`` compares the served
answer against the reference run recorded by ``customs/wevisone/ref_golden.py``.

    python customs/wevisone/generate_samples.py --only matting --golden customs/wevisone/golden

Comparison checks the raw response text, the parsed boxes, and each mask against
the reference mask file. Text-only samples have no masks, so only the text and
boxes are compared for them.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image

from openai_request_example import DEFAULT_URL, RequestSpec, build_payload, parse_response, write_result
from vllm_omni.model_executor.models.wevisone.constants import parse_seg_boxes

SAMPLES_FILE = Path(__file__).resolve().parent / "samples.json"
REPO_ROOT = Path(__file__).resolve().parents[2]
FIXED_IMAGE = REPO_ROOT / "extra_repos" / "WeVisOne-VL" / "assets" / "demo" / "seabreeze.jpg"

BOX_TOLERANCE = 0.001
MASK_TOLERANCE = 0.02


def load_samples(path: Path) -> list[dict[str, Any]]:
    samples = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(samples, list):
        raise SystemExit(f"{path} must contain a JSON array of samples")
    return samples


def served_mask(part: dict[str, Any]) -> np.ndarray:
    """Decode one served mask part into a grayscale array for comparison."""
    url = part["image_url"]["url"]
    raw = base64.b64decode(url.split(",", 1)[1]) if url.startswith("data:") else requests.get(url, timeout=60).content
    return np.asarray(Image.open(io.BytesIO(raw)).convert("L"), dtype=np.float32) / 255.0


def compare(entry: dict[str, Any], text: str, masks: list[dict[str, Any]], golden_dir: Path) -> list[str]:
    """Return differences between a served result and its golden fixture."""
    fixture = golden_dir / entry["id"] / "result.json"
    if not fixture.is_file():
        return [f"no golden fixture for {entry['id']}"]
    reference = json.loads(fixture.read_text(encoding="utf-8"))
    problems: list[str] = []

    if entry.get("task") != reference.get("task"):
        problems.append(f"task {entry.get('task')} != fixture {reference.get('task')}")
    if text != reference.get("response"):
        problems.append(f"response {text!r} != fixture {reference.get('response')!r}")

    boxes = parse_seg_boxes(text)
    for index, (served, expected) in enumerate(zip(boxes, reference.get("boxes", []))):
        if any(abs(s - e) > BOX_TOLERANCE for s, e in zip(served, expected)):
            problems.append(f"box {index} {served} != fixture {expected}")

    if len(masks) != len(reference.get("mask_files", [])):
        problems.append(f"{len(masks)} masks != fixture {len(reference.get('mask_files', []))}")

    for index, part in enumerate(masks):
        name = reference.get("mask_files", [])[index]
        path = golden_dir / entry["id"] / name
        if not path.is_file():
            problems.append(f"missing fixture mask {name}")
            continue
        expected = np.load(path)
        served = served_mask(part)
        if served.shape != expected.shape:
            problems.append(f"mask {index} shape {served.shape} != fixture {expected.shape}")
        elif np.abs(served - expected).mean() > MASK_TOLERANCE:
            problems.append(f"mask {index} mean diff {np.abs(served - expected).mean():.4f}")
    return problems


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--json", default=str(SAMPLES_FILE))
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--model", default="tencent/WeVisOne-VL-4B")
    p.add_argument("--image", default=str(FIXED_IMAGE))
    p.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "output"))
    p.add_argument("--golden", default=str(Path(__file__).resolve().parent / "golden"))
    p.add_argument("--only", nargs="+", help="sample ids to run")
    p.add_argument("--no-compare", action="store_true")
    args = p.parse_args()

    samples = load_samples(Path(args.json))
    if args.only:
        samples = [entry for entry in samples if entry.get("id") in args.only]
    if not samples:
        raise SystemExit("no samples selected")

    out_dir = Path(args.out_dir)
    golden_dir = Path(args.golden)
    failures = 0
    for entry in samples:
        spec = RequestSpec(
            model=args.model,
            image=Path(entry.get("image") or args.image),
            task=entry["task"],
            query=entry.get("query", ""),
            prompt=entry.get("prompt"),
            reference_boxes=[entry["reference_box"]] if entry.get("reference_box") else None,
        )
        response = requests.post(args.url, json=build_payload(spec), timeout=600)
        response.raise_for_status()
        text, masks = parse_response(response.json())
        sample_dir = out_dir / entry["id"]
        write_result(sample_dir, spec, text, masks)
        print(f"{entry['id']}: {len(masks)} masks, response {text!r}")
        if args.no_compare or not golden_dir.is_dir():
            continue
        problems = compare(entry, text, masks, golden_dir)
        if problems:
            failures += 1
            print(f"  differs from golden: {'; '.join(problems)}")
        else:
            print("  matches golden")

    if failures:
        raise SystemExit(f"{failures} sample(s) differ from the golden fixtures")


if __name__ == "__main__":
    main()
