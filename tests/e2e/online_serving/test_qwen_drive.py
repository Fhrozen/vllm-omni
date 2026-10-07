# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Online serving test for Qwen-Drive (VQA, direct planning, reasoning planning) through chat completions.

Equivalent to:
    vllm serve <hf-dir> --omni --deploy-config vllm_omni/deploy/qwen_drive_fp8.yaml \\
        --chat-template <hf-dir>/chat_template_drive.jinja --trust-remote-code
    python customs/qwen_drive/client_example.py --mode direct

Needs the exported HF repo and the E2 golden files (see customs/docs/wiki/Testing.md); skipped when absent.
Environment: QD_MODEL_DIR, QD_DEPLOY, QD_GOLDEN_DIR, QD_SCENES, QD_IMAGE_ROOT, QD_PLANNER (rl|sft), QD_MAX_ADE (m).
"""

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest
import requests

from tests.helpers.runtime import OmniServerParams

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL = os.environ.get("QD_MODEL_DIR", str(REPO_ROOT / "extra_repos/Qwen-Drive-1.0-4B-hf"))
DEPLOY = os.environ.get("QD_DEPLOY", str(REPO_ROOT / "vllm_omni/deploy/qwen_drive_fp8.yaml"))
GOLDEN = Path(os.environ.get("QD_GOLDEN_DIR", REPO_ROOT / "customs/docs/golden"))
SCENES = os.environ.get("QD_SCENES", str(REPO_ROOT / "extra_repos/Qwen-Drive-1.0/data/demo/planning_scenes.jsonl"))
IMAGE_ROOT = Path(os.environ.get("QD_IMAGE_ROOT", REPO_ROOT / "extra_repos/Qwen-Drive-1.0/data/demo"))
PLANNER = os.environ.get("QD_PLANNER", "rl")
# FP8 weights shift the trajectories by a few centimetres; BF16 should be far below this.
MAX_ADE_M = float(os.environ.get("QD_MAX_ADE", "0.15"))

pytestmark = [
    pytest.mark.advanced_model,  # real weights; core_model runs patch the deploy yaml to load_format: dummy
    pytest.mark.slow,
    pytest.mark.local_model,
    pytest.mark.skipif(
        not (Path(MODEL) / "chat_template_drive.jinja").exists() or not Path(SCENES).exists(),
        reason="exported Qwen-Drive repo or demo scenes not found",
    ),
]

test_params = [
    OmniServerParams(
        model=MODEL,
        stage_config_path=DEPLOY,
        server_args=["--chat-template", f"{MODEL}/chat_template_drive.jinja", "--trust-remote-code"],
        stage_init_timeout=900,
    ),
]


def _client_module():
    spec = importlib.util.spec_from_file_location("qd_client", REPO_ROOT / "customs/qwen_drive/client_example.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _post(server, scene: int, mode: str) -> dict:
    client = _client_module()
    record = json.loads(Path(SCENES).read_text().splitlines()[scene])
    body = client.build_request(record, IMAGE_ROOT, mode, "Describe the traffic scene and the safest action.", MODEL)
    reply = requests.post(f"http://{server.host}:{server.port}/v1/chat/completions", json=body, timeout=900)
    assert reply.status_code == 200, reply.text[:500]
    return reply.json()


def _text(data: dict) -> str:
    return " ".join((c["message"].get("content") or "") for c in data["choices"]).strip()


def _ade(trajectory, mode: str, scene: int) -> float:
    want = np.load(GOLDEN / f"traj_{PLANNER}_{mode}_{scene}.npy")
    got = np.asarray(trajectory)
    assert got.shape == want.shape == (6, 50, 3)
    return float(np.linalg.norm(got[..., :2] - want[..., :2], axis=-1).mean())


@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_qwen_drive_vqa(omni_server) -> None:
    data = _post(omni_server, scene=1, mode="vqa")
    assert len(_text(data)) > 20
    assert data.get("trajectory") is None


@pytest.mark.parametrize("omni_server", test_params, indirect=True)
@pytest.mark.parametrize("scene", [0, 1, 2])
def test_qwen_drive_direct_planning(omni_server, scene: int) -> None:
    data = _post(omni_server, scene=scene, mode="direct")
    trajectory = np.asarray(data["trajectory"])
    assert trajectory.shape == (6, 50, 3)
    assert np.isfinite(trajectory).all()
    assert _ade(trajectory, "direct", scene) < MAX_ADE_M


@pytest.mark.parametrize("omni_server", test_params, indirect=True)
@pytest.mark.parametrize("scene", [0, 1, 2])
def test_qwen_drive_reasoning_planning(omni_server, scene: int) -> None:
    data = _post(omni_server, scene=scene, mode="reasoning")
    reasoning = _text(data)
    assert reasoning and "<|" not in reasoning
    assert _ade(data["trajectory"], "reasoning", scene) < MAX_ADE_M
