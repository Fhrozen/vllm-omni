"""Offline end-to-end run of Ideogram 4 through vllm-omni (no server needed).

    python docs/ideogram-4/e2e_offline.py "a red fox sitting on a park bench"
    python docs/ideogram-4/e2e_offline.py "a red fox" --preset V4_TURBO_12 --output /tmp/fox.png
    python docs/ideogram-4/e2e_offline.py "a red fox" --dit-copies clone --ulysses-degree 2

Single-stage pipeline: Qwen3VL text encoder -> 34-layer FP8 DiT -> Flux2-style VAE.
`--dit-copies clone` shares the transformer weights between the conditional and
unconditional branches (one copy instead of two, the default); `keep` loads the
separately shipped `unconditional_transformer` for reference parity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_extras.ideogram4 import IDEOGRAM4_EXTRA_BODY_PARAMS


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("caption", help="The caption to generate from.")
    p.add_argument("--model", default="ideogram-ai/ideogram-4-fp8")
    p.add_argument("--deploy-config", default="docs/ideogram-4/deploy_ideogram4_fp8.yaml")
    p.add_argument("--preset", default="V4_QUALITY_48", choices=("V4_QUALITY_48", "V4_DEFAULT_20", "V4_TURBO_12"))
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-steps", type=int, help="Override the preset's step count.")
    p.add_argument("--guidance-schedule", type=float, nargs="+", help="Per-step guidance weight, loop order.")
    p.add_argument("--mu", type=float, help="Logit-normal mean; defaults to the resolution-aware value.")
    p.add_argument("--std", type=float, help="Logit-normal std; defaults to the resolution-aware value.")
    p.add_argument("--dit-copies", default="clone", choices=("keep", "clone"))
    p.add_argument("--ulysses-degree", type=int, default=1)
    p.add_argument("--output", type=Path, default=Path("/tmp/ideogram4_output.png"))
    args = p.parse_args()

    omni_kwargs = {"model": args.model, "deploy_config": args.deploy_config}
    if args.ulysses_degree > 1:
        omni_kwargs["ulysses_degree"] = args.ulysses_degree

    omni = Omni(**omni_kwargs)

    extra_args = {"preset": args.preset, "dit_copies": args.dit_copies}
    if args.guidance_schedule:
        extra_args["guidance_schedule"] = args.guidance_schedule
    if args.mu is not None:
        extra_args["mu"] = args.mu
    if args.std is not None:
        extra_args["std"] = args.std

    params = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        seed=args.seed,
        extra_args={key: value for key, value in extra_args.items() if key in IDEOGRAM4_EXTRA_BODY_PARAMS},
    )
    # `num_steps` is not a declared extra arg, so the step count has to go through
    # the sampling param the pipeline actually reads.
    if args.num_steps is not None:
        params.num_inference_steps = args.num_steps

    outputs = omni.generate({"prompt": args.caption}, sampling_params_list=[params])
    images = next((getattr(o, "images", None) for o in outputs if getattr(o, "images", None)), None)
    if not images:
        raise SystemExit(f"No images in {outputs}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for i, image in enumerate(images):
        path = args.output if i == 0 else args.output.parent / f"{args.output.stem}_{i}{args.output.suffix}"
        image.save(path)
        print(f"Saved {path}")

    for output in outputs:
        if getattr(output, "stage_durations", None):
            print(json.dumps(output.stage_durations, indent=2))


if __name__ == "__main__":
    main()
