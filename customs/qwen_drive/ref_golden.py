"""E2: golden outputs from the reference `qwen_drive` package (parity truth for vllm-omni).

Run via customs/qwen_drive/run_ref_baseline.sh (docker). Not used by vllm_omni itself.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import torch
from qwen_drive import QwenDriveForPlanning
from qwen_drive.benchmarks import read_scene_file
from safetensors import safe_open

PLANNERS = ("planner-sft", "planner-rl")


def key_summary(path: Path) -> dict:
    with safe_open(str(path), framework="pt") as f:
        keys = list(f.keys())
        dtype = str(f.get_tensor(keys[0]).dtype)
    prefixes = collections.Counter(".".join(k.split(".")[:2]) for k in keys)
    return {"num_keys": len(keys), "first_dtype": dtype, "prefix_counts": dict(prefixes.most_common(12)), "sample": keys[:3]}


def to_device(inputs: dict, device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}


def save_cache(path: Path, cache, anchor) -> None:
    torch.save({"kv": [(k.cpu(), v.cpu()) for k, v in cache], "anchor": anchor.cpu()}, path)


def generate(args) -> None:
    model_dir = Path(args.model)
    out = Path(args.out)
    kv_out = Path(args.kv_out)
    out.mkdir(parents=True, exist_ok=True)
    kv_out.mkdir(parents=True, exist_ok=True)

    summary = {"vlm": key_summary(model_dir / "model.safetensors")}
    for name in PLANNERS:
        summary[name] = key_summary(model_dir / name / "model.safetensors")
    (out / "safetensors_keys.json").write_text(json.dumps(summary, indent=2))

    model = QwenDriveForPlanning.from_pretrained(model_dir, dtype=torch.bfloat16, attn_implementation="sdpa")
    model = model.to("cuda").eval()
    samples = list(
        read_scene_file(
            args.scenes,
            image_root=args.image_root,
            image_archive=None,
            num_history_points=model.config.num_history_points,
            limit=args.num_scenes,
        )
    )
    processor = model.processor

    # Capture the raw generated ids of the reasoning pass.
    captured = {}
    original_generate = model.vlm.generate

    def capture_generate(*a, **kw):
        result = original_generate(*a, **kw)
        captured["sequences"] = (result.sequences if hasattr(result, "sequences") else result).cpu()
        return result

    model.vlm.generate = capture_generate

    for i, sample in enumerate(samples):
        scene = sample.scene
        info = {
            "token": sample.token,
            "nav_command": int(scene.nav_command),
            "driving_command_len": len(scene.driving_command),
            "ego_status_len": int(scene.ego_status.shape[0]),
            "history_shape": list(scene.history.shape),
        }

        vqa = model.generate_text(scene.frames_in_order(), args.question, max_new_tokens=args.vqa_max_new_tokens)
        (out / f"vqa_{i}.txt").write_text(vqa.text)

        direct_inputs = to_device(processor(scene, with_reasoning=False, device="cpu"), model.device)
        np.save(out / f"prompt_ids_{i}_direct.npy", direct_inputs["input_ids"].cpu().numpy())
        np.save(out / f"image_grid_thw_{i}.npy", direct_inputs["image_grid_thw"].cpu().numpy())
        planner_keys = ("history", "history_velocity", "history_acceleration", "ego_status", "nav_command")
        np.savez(out / f"planner_inputs_{i}.npz", **{k: direct_inputs[k].cpu().numpy() for k in planner_keys})
        cache, anchor = model._prefill(direct_inputs)
        save_cache(kv_out / f"scene_{i}_direct.pt", cache, anchor)
        info["seq_len_direct"] = int(direct_inputs["input_ids"].shape[1])
        info["anchor_direct"] = anchor[:, 0].tolist()

        reason_inputs = to_device(processor(scene, with_reasoning=True, device="cpu"), model.device)
        np.save(out / f"prompt_ids_{i}_reasoning.npy", reason_inputs["input_ids"].cpu().numpy())
        r_cache, r_anchor, reasoning = model._prefill_with_reasoning(reason_inputs, model.config.max_reasoning_tokens)
        (out / f"reasoning_{i}.txt").write_text(reasoning)
        np.save(out / f"generated_ids_{i}.npy", captured["sequences"][0, reason_inputs["input_ids"].shape[1] :].numpy())
        save_cache(kv_out / f"scene_{i}_reasoning.pt", r_cache, r_anchor)
        info["anchor_reasoning"] = r_anchor[:, 0].tolist()

        for name in PLANNERS:
            model.load_planner(model_dir / name)
            tag = name.split("-")[1]
            for mode, c, a, inp in (("direct", cache, anchor, direct_inputs), ("reasoning", r_cache, r_anchor, reason_inputs)):
                traj = model._plan_from_cache(
                    c, a, inp, num_samples=args.num_samples, num_steps=model.config.num_inference_steps, seed=model.config.noise_seed
                )
                np.save(out / f"traj_{tag}_{mode}_{i}.npy", traj)
        (out / f"scene_{i}.json").write_text(json.dumps(info, indent=2))
        print(f"scene {i} done: {info}", flush=True)


def compare(a: Path, b: Path) -> None:
    bad = []
    for fa in sorted(a.glob("*")):
        fb = b / fa.name
        if fa.suffix == ".npy":
            same = np.array_equal(np.load(fa), np.load(fb))
        else:
            same = fa.read_bytes() == fb.read_bytes()
        if not same:
            bad.append(fa.name)
    print("IDENTICAL" if not bad else f"DIFFERENT: {bad}")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/workspace/extra_repos/Qwen-Drive-1.0-4B")
    p.add_argument("--scenes", default="/workspace/extra_repos/Qwen-Drive-1.0/data/demo/planning_scenes.jsonl")
    p.add_argument("--image-root", default="/workspace/extra_repos/Qwen-Drive-1.0/data/demo")
    p.add_argument("--out", default="/workspace/customs/docs/golden")
    p.add_argument("--kv-out", default="/workspace/extra_repos/golden_kv")
    p.add_argument("--num-scenes", type=int, default=3)
    p.add_argument("--num-samples", type=int, default=6)
    p.add_argument("--question", default="Describe the traffic scene and the safest action.")
    p.add_argument("--vqa-max-new-tokens", type=int, default=128)
    p.add_argument("--compare", nargs=2, type=Path)
    args = p.parse_args()
    if args.compare:
        compare(*args.compare)
    else:
        generate(args)
