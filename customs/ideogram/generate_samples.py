# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Generate Ideogram 4 images from a JSON list of image details.

The input is a JSON array; each element describes one image. Every sampling key
uses the same name the OpenAI API takes in `extra_body`, so one entry works for
both the offline `Omni` path and a server request.

    # Generate every sample in the default file:
    python customs/ideogram/generate_samples.py

    # A different spec, output directory, or subset:
    python customs/ideogram/generate_samples.py --json customs/ideogram/samples.json --out-dir /tmp/ideogram4
    python customs/ideogram/generate_samples.py --only fox portrait --out-dir /tmp/ideogram4

    # Print the payloads the same entries would produce for the API (no GPU):
    python customs/ideogram/generate_samples.py --dry-run
    python customs/ideogram/generate_samples.py --dry-run --only fox > fox.json

Entry schema (all keys optional except `caption`):

    {
        "id": "fox",                      # label used for output names and logs
        "caption": "a red fox ...",       # the prompt
        "output": "fox.png",              # file name inside --out-dir
        "notes": "optional free text",

        "height": 1024, "width": 1024,   # must be multiples of 16 (AE_SCALE_FACTOR * PATCH_SIZE)
        "seed": 42,
        "num_inference_steps": 48,        # or the alias "num_steps"; preset value used when absent
        "num_outputs_per_prompt": 1,
        "sigmas": null,                   # explicit schedule, diffusers order (index 0 = noisiest)
        "output_type": "pil",

        "preset": "V4_QUALITY_48",        # V4_QUALITY_48 | V4_DEFAULT_20 | V4_TURBO_12
        "guidance_schedule": [3.0, ...],  # per-step weight in LOOP order, length must match the step count
        "mu": 0.0, "std": 1.5,           # logit-normal schedule; resolution-aware value used when absent
        "dit_copies": "keep"              # only honoured when the server loaded both DiT copies
    }

`guidance_schedule` is in loop-index order: index 0 is the last (final polish)
step, index `num_steps - 1` is the first. `dit_copies` is a startup knob; a
per-entry value only takes effect when the pipeline was started with `keep`, so
the startup default is passed once through `--dit-copies`.

Server equivalent: `customs/ideogram/serve.sh` then
`python customs/ideogram/openai_request_example.py "<caption>"`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vllm_omni.diffusion.models.ideogram4.constants import AE_SCALE_FACTOR, PATCH_SIZE
from vllm_omni.diffusion.utils.param_utils import apply_declared_extra_args
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_extras.ideogram4 import IDEOGRAM4_EXTRA_BODY_PARAMS

SCRIPT_DIR = Path(__file__).resolve().parent

# Keys the script consumes itself; everything else is a sampling key.
SCRIPT_ONLY_KEYS = frozenset({"id", "caption", "output", "notes"})
SAMPLING_KEYS = frozenset(
    {
        "height",
        "width",
        "seed",
        "num_inference_steps",
        "num_steps",
        "num_outputs_per_prompt",
        "sigmas",
        "output_type",
    }
)
KNOB_KEYS = IDEOGRAM4_EXTRA_BODY_PARAMS
KNOWN_KEYS = SCRIPT_ONLY_KEYS | SAMPLING_KEYS | KNOB_KEYS
ALIGNMENT = AE_SCALE_FACTOR * PATCH_SIZE


def _load_samples(path: Path) -> list[dict]:
    raw = sys.stdin.read() if str(path) == "-" else path.read_text()
    entries = json.loads(raw)
    if not isinstance(entries, list):
        raise SystemExit(f"{path} must contain a JSON list of image details")

    samples = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise SystemExit(f"entry {index} is not a JSON object")
        unknown = sorted(set(entry) - KNOWN_KEYS)
        if unknown:
            raise SystemExit(f"entry {index} has unknown key(s) {unknown}; known keys are {sorted(KNOWN_KEYS)}")
        if not str(entry.get("caption", "")).strip():
            raise SystemExit(f"entry {index} has an empty caption")

        height = int(entry.get("height") or 1024)
        width = int(entry.get("width") or 1024)
        if height % ALIGNMENT or width % ALIGNMENT:
            raise SystemExit(
                f"entry {index} ({entry.get('id', index)}): height/width must be multiples of "
                f"{ALIGNMENT}, got {height}x{width}"
            )

        steps = entry.get("num_steps") or entry.get("num_inference_steps")
        schedule = entry.get("guidance_schedule")
        if schedule and steps and len(schedule) != int(steps):
            raise SystemExit(
                f"entry {index} ({entry.get('id', index)}): guidance_schedule has {len(schedule)} "
                f"entries but the step count is {steps}"
            )
        samples.append(entry)
    return samples


def _extra_body(entry: dict) -> dict:
    """The request body for one entry: sampling keys only, in the API's shape."""
    return {key: entry[key] for key in sorted(KNOWN_KEYS - SCRIPT_ONLY_KEYS) if key in entry}


def _payload(entry: dict, model: str) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": entry["caption"]}],
        "extra_body": _extra_body(entry),
    }


def _sampling_params(entry: dict, body: dict) -> OmniDiffusionSamplingParams:
    params = OmniDiffusionSamplingParams(
        height=body.get("height"),
        width=body.get("width"),
        seed=body.get("seed"),
        num_outputs_per_prompt=body.get("num_outputs_per_prompt", 1),
        sigmas=body.get("sigmas"),
        output_type=body.get("output_type") or "pil",
    )
    steps = body.get("num_steps") or body.get("num_inference_steps")
    if steps is not None:
        params.num_inference_steps = int(steps)
    apply_declared_extra_args(params, KNOB_KEYS, body)
    return params


def _output_path(out_dir: Path, entry: dict, index: int) -> Path:
    name = entry.get("output") or f"{entry.get('id') or 'sample'}.png"
    path = out_dir / name
    if index > 0:
        path = out_dir / f"{path.stem}_{index}{path.suffix}"
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--json",
        type=Path,
        default=SCRIPT_DIR / "samples.json",
        help="JSON list of image details, or - for stdin",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/ideogram4_samples"))
    parser.add_argument("--only", nargs="+", help="Generate only these entry ids")
    parser.add_argument("--model", default="ideogram-ai/ideogram-4-fp8")
    parser.add_argument("--deploy-config", default="customs/ideogram/deploy_ideogram4_fp8.yaml")
    parser.add_argument(
        "--dit-copies",
        choices=("keep", "clone"),
        help="Startup default for entries that do not set it",
    )
    parser.add_argument("--ulysses-degree", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Print request payloads instead of generating")
    args = parser.parse_args()

    samples = _load_samples(args.json)
    if args.only:
        samples = [entry for entry in samples if entry.get("id") in args.only]
        if not samples:
            raise SystemExit(f"--only matched nothing: {args.only}")

    if args.dry_run:
        for entry in samples:
            print(json.dumps(_payload(entry, args.model), indent=2))
        return

    omni_kwargs = {"model": args.model, "deploy_config": args.deploy_config}
    if args.dit_copies:
        omni_kwargs["custom_pipeline_args"] = {"dit_copies": args.dit_copies}
    if args.ulysses_degree > 1:
        omni_kwargs["ulysses_degree"] = args.ulysses_degree

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    omni = Omni(**omni_kwargs)

    rows = []
    for entry in samples:
        body = _extra_body(entry)
        params = _sampling_params(entry, body)
        outputs = omni.generate({"prompt": entry["caption"]}, sampling_params_list=[params])
        images = [image for output in outputs for image in (getattr(output, "images", None) or [])]
        if not images:
            print(f"{entry.get('id')}: no image in the response", file=sys.stderr)
            continue
        for i, image in enumerate(images):
            path = _output_path(out_dir, entry, i)
            image.save(path)
            rows.append(
                {
                    "id": entry.get("id") or f"sample_{i}",
                    "size": f"{body.get('height') or 1024}x{body.get('width') or 1024}",
                    "steps": params.num_inference_steps or (body.get("preset") or "V4_QUALITY_48"),
                    "path": str(path),
                }
            )

    print()
    for row in rows:
        print(f"{row['id']:<12} {row['size']:>9}  steps={row['steps']:<14} {row['path']}")
    print(f"\n{len(rows)} image(s) in {out_dir}")


if __name__ == "__main__":
    main()
