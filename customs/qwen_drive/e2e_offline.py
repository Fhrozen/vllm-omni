"""E5/E7: offline end-to-end run of the 3-stage Qwen-Drive pipeline against the E2 golden trajectories.

Run via customs/qwen_drive/run_e2e_offline.sh. Uses the reference package only to read demo scenes/frames.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

p = argparse.ArgumentParser()
p.add_argument("--model", default="/workspace/extra_repos/Qwen-Drive-1.0-4B-hf")
p.add_argument("--deploy-config", default="/workspace/vllm_omni/deploy/qwen_drive_fp8.yaml")
p.add_argument("--golden", type=Path, default=Path("/workspace/customs/docs/golden"))
p.add_argument("--scenes", default="/workspace/extra_repos/Qwen-Drive-1.0/data/demo/planning_scenes.jsonl")
p.add_argument("--image-root", default="/workspace/extra_repos/Qwen-Drive-1.0/data/demo")
p.add_argument("--num-scenes", type=int, default=1)
p.add_argument("--modes", default="direct,reasoning")
p.add_argument("--planner", default="rl")
p.add_argument("--num-samples", type=int, default=6)
p.add_argument("--stage-init-timeout", type=int, default=900)
args = p.parse_args()


def collapse_placeholders(ids, pad_id):
    """One placeholder per image: the serving layer expands it from the image grid."""
    return [t for i, t in enumerate(ids) if not (t == pad_id and i and ids[i - 1] == pad_id)]


def resized_image(proc, frame, budget):
    from qwen_drive.scene import smart_resize

    image = frame.load()
    max_pixels = budget
    if frame.target_size is not None:
        width, height = frame.target_size
        image = TF.resize(image, [height, width], interpolation=InterpolationMode.BICUBIC)
        max_pixels = proc.grid_pixel_limit
    width, height = image.size
    gh, gw = smart_resize(height, width, proc.factor, proc.min_pixels, max_pixels)
    return TF.resize(image, [gh, gw], interpolation=InterpolationMode.BICUBIC)


def main() -> None:
    from qwen_drive.benchmarks import read_scene_file
    from qwen_drive.configuration_qwen_drive import QwenDriveConfig
    from qwen_drive.scene import QwenDriveProcessor
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    config = QwenDriveConfig.from_pretrained(args.model)
    proc = QwenDriveProcessor(AutoTokenizer.from_pretrained(args.model), config)
    samples = list(
        read_scene_file(
            args.scenes, image_root=args.image_root, image_archive=None, num_history_points=16, limit=args.num_scenes
        )
    )
    omni = Omni(model=args.model, deploy_config=args.deploy_config, stage_init_timeout=args.stage_init_timeout)
    results = {}
    try:
        for i, sample in enumerate(samples):
            scene = sample.scene
            per_view = scene.num_camera_frames
            images = [
                resized_image(proc, f, config.current_image_pixels if (idx % per_view) == per_view - 1 else config.history_image_pixels)
                for idx, f in enumerate(scene.frames_in_order())
            ]
            planner_inputs = np.load(args.golden / f"planner_inputs_{i}.npz")
            for mode in args.modes.split(","):
                ids = np.load(args.golden / f"prompt_ids_{i}_{mode}.npy")[0].tolist()
                ids = collapse_placeholders(ids, config.vlm_config.image_token_id)
                stage0 = (
                    SamplingParams(temperature=0.0, max_tokens=1)
                    if mode == "direct"
                    else SamplingParams(temperature=0.0, max_tokens=256, min_tokens=10)
                )
                params = [
                    stage0,
                    SamplingParams(temperature=0.0, max_tokens=1),
                    OmniDiffusionSamplingParams(
                        extra_args={
                            "history": planner_inputs["history"][0],
                            "history_velocity": planner_inputs["history_velocity"][0],
                            "history_acceleration": planner_inputs["history_acceleration"][0],
                            "ego_status": planner_inputs["ego_status"][0],
                            "nav_command": int(planner_inputs["nav_command"][0]),
                            "num_samples": args.num_samples,
                        }
                    ),
                ]
                prompt = {
                    "prompt_token_ids": ids,
                    "multi_modal_data": {"image": images},
                    "modalities": ["trajectory"],
                }
                outputs = omni.generate(prompt, params, use_tqdm=False)
                final = [o for o in outputs if "trajectory" in (getattr(o, "multimodal_output", None) or {})]
                assert final, "no trajectory output; got " + str(
                    [(o.stage_id, o.final_output_type, list((getattr(o, "multimodal_output", None) or {}).keys())) for o in outputs]
                )
                traj = np.asarray(final[0].multimodal_output["trajectory"])
                want = np.load(args.golden / f"traj_{args.planner}_{mode}_{i}.npy")
                diff = np.abs(traj - want)
                ade = np.linalg.norm(traj[..., :2] - want[..., :2], axis=-1).mean()
                results[f"{i}_{mode}"] = {"max_abs": float(diff.max()), "ade_vs_golden_m": float(ade)}
                print(f"scene {i} {mode}: shape {traj.shape} max|d| {diff.max():.4f} ADE-vs-golden {ade:.4f} m", flush=True)
    finally:
        omni.close()
    print("RESULT", json.dumps(results))


if __name__ == "__main__":
    main()
