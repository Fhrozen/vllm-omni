# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen-Drive planning expert: CPU unit tests plus a golden parity check against the reference package.

    pytest tests/diffusion/models/qwen_drive/test_planning_expert.py -v

The golden test needs the exported checkpoint and the E2 golden files (see customs/docs/wiki/Testing.md);
it is skipped when they are absent.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.qwen_drive.planning_expert import (
    PlannerSettings,
    PlanningExpert,
    PlanningExpertConfig,
    plan_trajectories,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = Path(os.environ.get("QD_MODEL_DIR", REPO_ROOT / "extra_repos/Qwen-Drive-1.0-4B-hf"))
GOLDEN_DIR = Path(os.environ.get("QD_GOLDEN_DIR", REPO_ROOT / "customs/docs/golden"))
GOLDEN_KV_DIR = Path(os.environ.get("QD_GOLDEN_KV_DIR", REPO_ROOT / "extra_repos/golden_kv"))


def _tiny_settings() -> PlannerSettings:
    expert = PlanningExpertConfig(
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        layers_per_kv=2,
        time_embed_dim=16,
        fourier_num_features=4,
        partial_rotary_factor=0.5,
        mrope_section=[2, 1, 1],
    )
    return PlannerSettings(num_future_points=6, num_history_points=5, expert=expert)


def _tiny_inputs(settings: PlannerSettings, seq: int = 7):
    g = torch.Generator().manual_seed(0)
    cfg = settings.expert
    cache = [
        (
            torch.randn(1, seq, cfg.num_key_value_heads, cfg.head_dim, generator=g),
            torch.randn(1, seq, cfg.num_key_value_heads, cfg.head_dim, generator=g),
        )
        for _ in range(cfg.num_kv_sources)
    ]
    return dict(
        scene_cache=cache,
        anchor=torch.full((3, 1), 20, dtype=torch.long),
        history=torch.randn(1, settings.num_history_points, 3, generator=g),
        history_velocity=torch.randn(1, settings.num_history_points, 2, generator=g),
        history_acceleration=torch.randn(1, settings.num_history_points, 2, generator=g),
        nav_command=torch.tensor([1]),
        ego_status=torch.randn(1, cfg.ego_status_dim, generator=g),
    )


@pytest.mark.core_model
@pytest.mark.cpu
class TestPlanningExpertUnits:
    def setup_method(self):
        torch.manual_seed(0)
        self.settings = _tiny_settings()
        self.expert = PlanningExpert(self.settings).eval()
        for p in self.expert.parameters():
            torch.nn.init.normal_(p, std=0.05)

    def _plan(self, num_samples: int, seed: int = 42):
        return plan_trajectories(
            self.expert, self.settings, num_samples=num_samples, num_steps=3, seed=seed, **_tiny_inputs(self.settings)
        )

    def test_shape_and_finite(self):
        out = self._plan(num_samples=2)
        assert out.shape == (2, 6, 3)
        assert torch.isfinite(out).all()

    def test_deterministic(self):
        assert torch.equal(self._plan(2), self._plan(2))

    def test_sample_k_uses_seed_plus_k(self):
        batch = self._plan(num_samples=3, seed=42)
        for k in range(3):
            single = self._plan(num_samples=1, seed=42 + k)
            torch.testing.assert_close(batch[k : k + 1], single, atol=1e-5, rtol=1e-5)

    def test_depends_on_scene_cache(self):
        inputs = _tiny_inputs(self.settings)
        base = plan_trajectories(self.expert, self.settings, num_samples=1, num_steps=3, seed=1, **inputs)
        inputs["scene_cache"] = [(k + 1.0, v) for k, v in inputs["scene_cache"]]
        changed = plan_trajectories(self.expert, self.settings, num_samples=1, num_steps=3, seed=1, **inputs)
        assert not torch.allclose(base, changed)

    def test_settings_from_hf_config(self):
        cfg = {
            "num_future_points": 50,
            "trajectory_scale": [165.0, 25.0, 1.5703125],
            "min_one_minus_t": 0.1,
            "model_type": "qwen_drive",
            "expert_config": {"hidden_size": 1024, "layers_per_kv": 4, "mrope_section": [11, 11, 10], "model_type": "x"},
        }
        s = PlannerSettings.from_hf_config(cfg)
        assert s.num_future_points == 50 and s.expert.hidden_size == 1024
        assert s.expert.num_kv_sources == 8


def _golden_available() -> bool:
    return (MODEL_DIR / "planner-sft/model.safetensors").exists() and (GOLDEN_KV_DIR / "scene_0_direct.pt").exists()


@pytest.mark.advanced_model
@pytest.mark.local_model
@pytest.mark.diffusion
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(not _golden_available(), reason="exported checkpoint / golden files not found")
@pytest.mark.parametrize("planner", ["sft", "rl"])
@pytest.mark.parametrize("mode", ["direct", "reasoning"])
@pytest.mark.parametrize("scene", [0, 1, 2])
def test_expert_matches_reference_golden(planner: str, mode: str, scene: int):
    import json

    from safetensors.torch import load_file

    settings = PlannerSettings.from_hf_config(json.loads((MODEL_DIR / "config.json").read_text()))
    expert = PlanningExpert(settings).to(device="cuda", dtype=torch.bfloat16).eval()
    weights = load_file(str(MODEL_DIR / f"planner-{planner}/model.safetensors"))
    weights = {k.removeprefix("planning_expert."): v for k, v in weights.items()}
    expert.load_state_dict({k: v.to(torch.bfloat16) for k, v in weights.items()}, strict=True)

    cached = torch.load(GOLDEN_KV_DIR / f"scene_{scene}_{mode}.pt")
    scene_cache = [(k.cuda(), v.cuda()) for k, v in cached["kv"]]
    inputs = np.load(GOLDEN_DIR / f"planner_inputs_{scene}.npz")
    out = plan_trajectories(
        expert,
        settings,
        scene_cache,
        cached["anchor"].cuda(),
        history=torch.from_numpy(inputs["history"]).cuda(),
        history_velocity=torch.from_numpy(inputs["history_velocity"]).cuda(),
        history_acceleration=torch.from_numpy(inputs["history_acceleration"]).cuda(),
        nav_command=torch.from_numpy(inputs["nav_command"]).cuda(),
        ego_status=torch.from_numpy(inputs["ego_status"]).cuda(),
        num_samples=6,
        num_steps=settings.num_inference_steps,
        seed=settings.noise_seed,
    )
    want = np.load(GOLDEN_DIR / f"traj_{planner}_{mode}_{scene}.npy")
    np.testing.assert_allclose(out.cpu().numpy(), want, atol=1e-3)


@pytest.mark.advanced_model
@pytest.mark.local_model
@pytest.mark.diffusion
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(not _golden_available(), reason="exported checkpoint / golden files not found")
@pytest.mark.parametrize("planner", ["sft", "rl"])
def test_pipeline_forward_matches_golden(planner: str):
    """Pipeline contract: sparse per-layer KV (None for GDN layers) + kv_metadata anchor + extra_args."""
    from types import SimpleNamespace

    from vllm_omni.diffusion.models.qwen_drive.pipeline_qwen_drive import QwenDrivePlannerPipeline

    od_config = SimpleNamespace(
        model=str(MODEL_DIR), model_config={"planner_subfolder": f"planner-{planner}"}, dtype=torch.bfloat16
    )
    pipe = QwenDrivePlannerPipeline(od_config=od_config)

    cached = torch.load(GOLDEN_KV_DIR / "scene_0_direct.pt")
    attention_layers = [3, 7, 11, 15, 19, 23, 27, 31]
    keys, values = [None] * 32, [None] * 32
    for idx, (k, v) in zip(attention_layers, cached["kv"], strict=True):
        keys[idx], values[idx] = k.squeeze(0).cuda(), v.squeeze(0).cuda()

    inputs = np.load(GOLDEN_DIR / "planner_inputs_0.npz")
    req = SimpleNamespace(
        past_key_values=SimpleNamespace(key_cache=keys, value_cache=values),
        kv_metadata={"rope_anchor": int(cached["anchor"][0, 0])},
        sampling_params=SimpleNamespace(
            extra_args={
                "history": inputs["history"][0],
                "history_velocity": inputs["history_velocity"][0],
                "history_acceleration": inputs["history_acceleration"][0],
                "ego_status": inputs["ego_status"][0],
                "nav_command": int(inputs["nav_command"][0]),
                "num_samples": 6,
            }
        ),
    )
    out = pipe.forward(req)
    assert out.error is None, out.error
    want = np.load(GOLDEN_DIR / f"traj_{planner}_direct_0.npy")
    np.testing.assert_allclose(out.output["trajectories"], want, atol=1e-3)
