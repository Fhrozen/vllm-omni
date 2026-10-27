#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Golden oracle for WeVisOne-VL parity tests.

Runs the released reference implementation and writes reference fixtures
that the ported pipeline is compared against. This is the only place where the
reference package (``extra_repos/WeVisOne-VL/wevisone_vl``) is imported;
``vllm_omni`` code must never depend on it.

Run inside docker (the reference package and its dependencies only exist there):

    customs/qwen_drive/docker_run.sh "python customs/wevisone/ref_golden.py"

Outputs, per task, under ``--out`` (default ``customs/wevisone/golden``):
  * ``result.json``  - prompt, response, normalized xyxy boxes, mask list
  * ``masks/mask_NNN.npy`` - float32 mask at the original image size
  * ``masks/mask_NNN.png`` - uint8 grayscale copy for human review

Greedy decoding (``temperature=0``) plus a fixed seed and fixed image makes the
text and box outputs reproducible. GPU kernels are not bit-reproducible: four
runs on this host differed in the mask tensors by up to 5.1e-4, so parity checks
compare float masks with a tolerance instead of exact equality.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = REPO_ROOT / "extra_repos" / "WeVisOne-VL"
if not (REFERENCE_ROOT / "wevisone_vl").is_dir():
    raise SystemExit(f"reference package not found: {REFERENCE_ROOT}")
sys.path.insert(0, str(REFERENCE_ROOT))

from wevisone_vl.engine import DEFAULT_MODEL_ID, parse_seg_boxes  # noqa: E402
from wevisone_vl.prompts import TASKS, build_prompt  # noqa: E402

FIXED_IMAGE = REFERENCE_ROOT / "assets" / "demo" / "seabreeze.jpg"
SEED = 1234

# Shared input set: the request harnesses read the same samples.json, so the
# golden fixtures and the served requests exercise identical prompts.
SAMPLES_FILE = Path(__file__).resolve().parent / "samples.json"
SAMPLES = json.loads(SAMPLES_FILE.read_text(encoding="utf-8"))


def _write_mask_pair(masks_dir: Path, index: int, mask: np.ndarray) -> str:
    masks_dir.mkdir(parents=True, exist_ok=True)
    array = np.asarray(mask, dtype=np.float32)
    np.save(masks_dir / f"mask_{index:03d}.npy", array)
    Image.fromarray(np.uint8(np.clip(array, 0.0, 1.0) * 255), mode="L").save(
        masks_dir / f"mask_{index:03d}.png"
    )
    return f"masks/mask_{index:03d}.npy"


def main() -> int:
    parser = argparse.ArgumentParser(description="Write WeVisOne-VL golden fixtures.")
    parser.add_argument("--model", default=str(REPO_ROOT / "extra_repos" / "WeVisOne-VL-4B"))
    parser.add_argument("--image", default=str(FIXED_IMAGE))
    parser.add_argument("--out", default=str(REPO_ROOT / "customs" / "wevisone" / "golden"))
    parser.add_argument("--tasks", default=",".join(TASKS), help="comma-separated task subset")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--decode-masks", action="store_true", help="also run the pixel decoder")
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    tasks = [task.strip().lower() for task in args.tasks.split(",") if task.strip()]
    unknown = [task for task in tasks if task not in TASKS]
    if unknown:
        raise SystemExit(f"unknown task(s): {', '.join(unknown)}")

    random.seed(args.seed)
    try:
        import torch

        torch.manual_seed(args.seed)
    except ImportError:  # pragma: no cover - the oracle runs in docker with torch
        torch = None  # type: ignore[assignment]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    engine: Any = None
    if args.decode_masks:
        from wevisone_vl.engine import WeVisOneVL

        engine = WeVisOneVL(model_name_or_path=args.model, device=args.device)

    index: dict[str, list[str]] = {}
    for entry in SAMPLES:
        task = entry["task"]
        if task not in tasks:
            continue
        query = entry.get("query", "")
        reference_boxes = entry.get("reference_box")
        reference_boxes = [reference_boxes] if reference_boxes else None
        prompt = build_prompt(
            task,
            query,
            reference_boxes=reference_boxes,
            image_size=(image.width, image.height),
        )
        if engine is not None:
            result = engine.predict(
                image,
                task=task,
                query=query,
                reference_boxes=reference_boxes,
            )
            response, boxes, masks = result.response, result.boxes, result.masks
            assert result.prompt == prompt, f"prompt drift for task {task}"
        else:
            raise SystemExit(
                "golden fixtures need --decode-masks so masks, boxes and responses "
                "are recorded from the same reference run"
            )

        task_dir = out_dir / task
        masks_dir = task_dir / "masks"
        mask_files = [_write_mask_pair(masks_dir, i, m) for i, m in enumerate(masks)]
        record = {
            "task": task,
            "prompt": prompt,
            "response": response,
            "boxes": boxes,
            "parse_seg_boxes": parse_seg_boxes(response),
            "image_size": [image.width, image.height],
            "seed": args.seed,
            "mask_files": mask_files,
        }
        (task_dir / "result.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        index[task] = record["mask_files"]
        print(f"{task}: {len(boxes)} boxes, {len(masks)} masks")

    (out_dir / "index.json").write_text(
        json.dumps(
            {
                "image": str(Path(args.image)),
                "model": args.model,
                "seed": args.seed,
                "tasks": index,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
