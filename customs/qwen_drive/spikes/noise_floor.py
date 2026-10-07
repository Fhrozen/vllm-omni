"""Noise floor: reference trajectories with eager attention vs the sdpa golden (same weights, different kernels)."""

import numpy as np
import torch
from qwen_drive import QwenDriveForPlanning
from qwen_drive.benchmarks import read_scene_file

MODEL = "/workspace/extra_repos/Qwen-Drive-1.0-4B"
G = "/workspace/customs/docs/golden"
model = (
    QwenDriveForPlanning.from_pretrained(
        MODEL, planner=f"{MODEL}/planner-rl", dtype=torch.bfloat16, attn_implementation="eager"
    )
    .cuda()
    .eval()
)
samples = list(
    read_scene_file(
        "/workspace/extra_repos/Qwen-Drive-1.0/data/demo/planning_scenes.jsonl",
        image_root="/workspace/extra_repos/Qwen-Drive-1.0/data/demo",
        image_archive=None,
        num_history_points=16,
        limit=3,
    )
)
for i, s in enumerate(samples):
    got = model.run("direct_planning", scene=s.scene, num_samples=6).trajectories
    want = np.load(f"{G}/traj_rl_direct_{i}.npy")
    ade = np.linalg.norm(got[..., :2] - want[..., :2], axis=-1).mean()
    print(f"scene {i} eager-vs-sdpa ADE {ade:.4f} m max|d| {np.abs(got - want).max():.4f}", flush=True)
