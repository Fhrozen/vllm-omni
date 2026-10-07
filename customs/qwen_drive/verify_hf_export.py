"""E3: load the exported HF repo via remote code (qwen_drive blocked) and compare with the E2 golden.

Run via customs/qwen_drive/verify_hf_export.sh (docker).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# The reference package is only used to read demo scenes; it is blocked before the model loads.
from qwen_drive.benchmarks import read_scene_file

p = argparse.ArgumentParser()
p.add_argument("--model", default="/workspace/extra_repos/Qwen-Drive-1.0-4B-hf")
p.add_argument("--golden", type=Path, default=Path("/workspace/customs/docs/golden"))
p.add_argument("--scenes", default="/workspace/extra_repos/Qwen-Drive-1.0/data/demo/planning_scenes.jsonl")
p.add_argument("--image-root", default="/workspace/extra_repos/Qwen-Drive-1.0/data/demo")
p.add_argument("--num-scenes", type=int, default=3)
p.add_argument("--question", default="Describe the traffic scene and the safest action.")
args = p.parse_args()

samples = list(read_scene_file(args.scenes, image_root=args.image_root, image_archive=None, num_history_points=16, limit=args.num_scenes))

for name in [m for m in sys.modules if m == "qwen_drive" or m.startswith("qwen_drive.")]:
    del sys.modules[name]
sys.modules["qwen_drive"] = None  # any later `import qwen_drive` raises ImportError

# Importing the reference package registered qwen_drive with the auto classes; undo it so the repo's auto_map is used.
from transformers.models.auto.configuration_auto import CONFIG_MAPPING  # noqa: E402
from transformers.models.auto.modeling_auto import MODEL_MAPPING  # noqa: E402

CONFIG_MAPPING._extra_content.pop("qwen_drive", None)
for key in [k for k in MODEL_MAPPING._extra_content if getattr(k, "model_type", None) == "qwen_drive"]:
    MODEL_MAPPING._extra_content.pop(key)

from transformers import AutoModel  # noqa: E402

model = AutoModel.from_pretrained(
    args.model, trust_remote_code=True, planner="planner-sft", dtype=torch.bfloat16, attn_implementation="sdpa"
).cuda().eval()
print("loaded", type(model).__module__, flush=True)
assert type(model).__module__.startswith("transformers_modules"), type(model).__module__

processor = model.processor
CameraFrame = sys.modules[type(processor).__module__].CameraFrame
failures = []


def check(name: str, got, want, atol: float = 1e-3) -> None:
    diff = float(np.abs(np.asarray(got, dtype=np.float64) - np.asarray(want, dtype=np.float64)).max())
    print(f"  {name}: max abs diff {diff:.3e}")
    if diff > atol:
        failures.append(name)


for i, sample in enumerate(samples):
    scene = sample.scene
    print(f"scene {i}", flush=True)
    frames = [CameraFrame(f.image, f.target_size) for f in scene.frames_in_order()]
    vqa = model.generate_text(frames, args.question, max_new_tokens=128).text
    same = vqa == (args.golden / f"vqa_{i}.txt").read_text()
    print(f"  vqa identical: {same}")
    if not same:
        failures.append(f"vqa_{i}")

    for mode_name, mode in (("direct", "direct_planning"), ("reasoning", "reasoning_planning")):
        for planner in ("planner-sft", "planner-rl"):
            model.load_planner(Path(args.model) / planner)
            out = model.run(mode, scene=scene, num_samples=6)
            tag = planner.split("-")[1]
            check(f"traj_{tag}_{mode_name}", out.trajectories, np.load(args.golden / f"traj_{tag}_{mode_name}_{i}.npy"))
            if mode_name == "reasoning":
                want = (args.golden / f"reasoning_{i}.txt").read_text()
                if out.reasoning != want:
                    failures.append(f"reasoning_{i}")
                print(f"  reasoning identical: {out.reasoning == want}")

print("FAIL" if failures else "PASS", failures)
raise SystemExit(1 if failures else 0)
