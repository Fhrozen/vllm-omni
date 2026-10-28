# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Overlay WeVisOne-VL masks and boxes on the source image for visual review.

Reads a result directory written by either example harness -- the served samples
(``masks/mask_NNN.png``) or the golden fixtures (``masks/mask_NNN.npy``) -- and
writes ``overlay_NNN.png`` next to it.

    python customs/wevisone/visualize.py --result customs/wevisone/output/matting
    python customs/wevisone/visualize.py --result customs/wevisone/golden/matting

Each mask is tinted where it is above 0.5 and its box is outlined, so the overlay
shows whether the pairing the API delivers holds: mask i answers box i. Served
results carry that pairing explicitly as ``mask_to_box``; golden records list the
boxes in the same order as the mask files, so the positional fallback is used there.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXED_IMAGE = REPO_ROOT / "extra_repos" / "WeVisOne-VL" / "assets" / "demo" / "seabreeze.jpg"

COLORS = ((76, 175, 80), (244, 67, 54), (63, 81, 181), (255, 193, 7), (156, 39, 176))


def load_mask(path: Path) -> np.ndarray:
    """Read a mask written as numpy or PNG into a single-channel float array."""
    if path.suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float32)
    array = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    return array


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--result", required=True, help="directory holding result.json and masks/")
    p.add_argument("--image", help="source image; defaults to the record or the fixed demo image")
    p.add_argument("--out-dir", help="where to write overlays; defaults to the result directory")
    args = p.parse_args()

    result_dir = Path(args.result)
    record = json.loads((result_dir / "result.json").read_text(encoding="utf-8"))
    image_path = Path(args.image or record.get("image") or FIXED_IMAGE)
    if not image_path.is_file():
        raise SystemExit(f"source image not found: {image_path}")

    base = Image.open(image_path).convert("RGB")
    pairing = record.get("mask_to_box") or [
        {"mask_index": index, "box": box} for index, box in enumerate(record.get("boxes", []))
    ]
    out_dir = Path(args.out_dir) if args.out_dir else result_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for index, entry in enumerate(pairing):
        mask_path = result_dir / "masks" / f"mask_{entry['mask_index']:03d}"
        candidate = next(
            (p for p in (mask_path.with_suffix(".npy"), mask_path.with_suffix(".png")) if p.is_file()),
            None,
        )
        if candidate is None:
            print(f"mask {entry['mask_index']}: no file under {result_dir / 'masks'}")
            continue
        mask = load_mask(candidate)
        if mask.shape[:2] != (base.height, base.width):
            mask = np.asarray(Image.fromarray(np.uint8(mask * 255), mode="L").resize(base.size), dtype=np.float32)

        color = COLORS[index % len(COLORS)]
        overlay = base.copy()
        tint = np.asarray(overlay, dtype=np.float32)
        selected = mask >= 0.5
        tint[selected] = (tint[selected] * 0.55 + np.asarray(color, dtype=np.float32) * 0.45).clip(0, 255)
        overlay = Image.fromarray(np.uint8(tint), mode="RGB")

        draw = ImageDraw.Draw(overlay)
        box = entry["box"]
        x1, y1, x2, y2 = (value * base.width for value in box)
        if x2 - x1 < 4 or y2 - y1 < 4:
            # Point targets are boxes with equal corners, so draw a marker.
            draw.ellipse([x1 - 6, y1 - 6, x1 + 6, y1 + 6], outline=color, width=3)
        else:
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        name = f"overlay_{entry['mask_index']:03d}.png"
        overlay.save(out_dir / name)
        print(f"wrote {out_dir / name}")


if __name__ == "__main__":
    main()
