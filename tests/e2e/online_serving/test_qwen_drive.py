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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import requests

from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import modify_stage_config

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


SFT_DEPLOY = (
    modify_stage_config(DEPLOY, updates={"stages": {2: {"model_config": {"planner_subfolder": "planner-sft"}}}})
    if Path(DEPLOY).exists()
    else DEPLOY
)


def _client_module():
    spec = importlib.util.spec_from_file_location("qd_client", REPO_ROOT / "customs/qwen_drive/client_example.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _body(scene: int, mode: str) -> dict:
    record = json.loads(Path(SCENES).read_text().splitlines()[scene])
    return _client_module().build_request(
        record, IMAGE_ROOT, mode, "Describe the traffic scene and the safest action.", MODEL
    )


def _url(server) -> str:
    return f"http://{server.host}:{server.port}/v1/chat/completions"


def _post(server, scene: int, mode: str) -> dict:
    reply = requests.post(_url(server), json=_body(scene, mode), timeout=900)
    assert reply.status_code == 200, reply.text[:500]
    return reply.json()


def _text(data: dict) -> str:
    return " ".join((c["message"].get("content") or "") for c in data["choices"]).strip()


def _ade(trajectory, mode: str, scene: int, planner: str = PLANNER) -> float:
    want = np.load(GOLDEN / f"traj_{planner}_{mode}_{scene}.npy")
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


@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_qwen_drive_concurrent_requests(omni_server) -> None:
    jobs = [(0, "direct"), (1, "direct"), (2, "reasoning"), (1, "vqa"), (0, "reasoning")]
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        results = list(pool.map(lambda job: _post(omni_server, scene=job[0], mode=job[1]), jobs))
    for (scene, mode), data in zip(jobs, results, strict=True):
        if mode == "vqa":
            assert len(_text(data)) > 20
        else:
            assert _ade(data["trajectory"], mode, scene) < MAX_ADE_M


@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_qwen_drive_client_abort_does_not_break_server(omni_server) -> None:
    with pytest.raises(requests.exceptions.ReadTimeout):
        requests.post(_url(omni_server), json=_body(0, "reasoning"), timeout=0.3)
    data = _post(omni_server, scene=1, mode="direct")
    assert _ade(data["trajectory"], "direct", 1) < MAX_ADE_M


@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_qwen_drive_streaming_vqa(omni_server) -> None:
    body = _body(1, "vqa") | {"stream": True}
    with requests.post(_url(omni_server), json=body, timeout=900, stream=True) as reply:
        assert reply.status_code == 200
        chunks = [line for line in reply.iter_lines() if line.startswith(b"data: ") and line != b"data: [DONE]"]
    text = "".join(
        choice["delta"].get("content") or "" for chunk in chunks for choice in json.loads(chunk[6:]).get("choices", [])
    )
    assert len(text.strip()) > 20
    assert _ade(_post(omni_server, scene=1, mode="direct")["trajectory"], "direct", 1) < MAX_ADE_M


@pytest.mark.parametrize(
    "omni_server",
    [test_params[0]._replace(stage_config_path=SFT_DEPLOY)],
    indirect=True,
)
def test_qwen_drive_planner_sft(omni_server) -> None:
    data = _post(omni_server, scene=1, mode="direct")
    assert _ade(data["trajectory"], "direct", 1, planner="sft") < MAX_ADE_M
    # The two planners must give different trajectories for the same scene.
    assert _ade(data["trajectory"], "direct", 1, planner="rl") > 1e-3
