# SPDX-FileCopyrightText: Copyright (c) 2025 vLLM-Omni contributors,
# SPDX-License-Identifier: Apache-2.0
"""Ideogram 4 online-serving coverage.

Ideogram 4 is text-to-image only: ``POST /v1/chat/completions`` takes a JSON caption and
answers with one image part per request. The checkpoint is gated, so the suite is skipped
when no local copy is available.

    IG_MODEL_DIR=/path/to/ideogram-4-fp8 pytest -s -v tests/e2e/online_serving/test_ideogram4.py -m 'core_model and diffusion' --run-level 'core_model'
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from tests.helpers.assertions import assert_image_valid
from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OnlineOmniClient

_MODEL_DIR = os.environ.get("IG_MODEL_DIR", "")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.diffusion,
    pytest.mark.local_model,
    pytest.mark.skipif(
        not Path(_MODEL_DIR).is_dir(),
        reason="Ideogram 4 is a gated checkpoint; set IG_MODEL_DIR to a local copy to run",
    ),
]

REPO_ROOT = Path(__file__).resolve().parents[3]
DEPLOY = os.environ.get("IG_DEPLOY", str(REPO_ROOT / "customs/ideogram/deploy_ideogram4_fp8.yaml"))

_SERVER = [
    pytest.param(
        OmniServerParams(model=_MODEL_DIR, stage_config_path=DEPLOY, stage_init_timeout=900),
        id="ideogram4",
        marks=hardware_marks(res={"cuda": "H100"}),
    )
]

_CAPTION = "A red fox sitting on a weathered wooden fence at golden hour"


def _request_config(omni_server: OmniServer, extra_body: dict | None = None) -> dict:
    cfg: dict = {
        "model": omni_server.model,
        "messages": [{"role": "user", "content": _CAPTION}],
    }
    if extra_body:
        cfg["extra_body"] = extra_body
    return cfg


def _single_image(omni_server: OmniServer, online_client: OnlineOmniClient, extra_body: dict):
    responses = online_client.send_diffusion_request(_request_config(omni_server, extra_body))
    assert len(responses) == 1, f"expected one response, got {len(responses)}"
    images = responses[0].images
    assert images is not None and len(images) == 1, f"expected one image, got {images}"
    return images[0]


@pytest.mark.core_model
@pytest.mark.advanced_model
def test_text_to_image_001(omni_server: OmniServer, online_client: OnlineOmniClient) -> None:
    """Baseline chat-completions round trip: one image part, correct size, non-blank."""
    image = _single_image(omni_server, online_client, {"seed": 42})
    assert_image_valid(image, width=1024, height=1024)
    # A flat canvas would mean the DiT loop or the AE decoder produced nothing usable.
    assert float(np.asarray(image.convert("RGB")).std()) > 1.0


@pytest.mark.advanced_model
def test_extra_args_request_surface(omni_server: OmniServer, online_client: OnlineOmniClient) -> None:
    """Model-specific knobs travel through ``extra_args`` and are honoured by the pipeline.

    A short step count keeps the request cheap; an explicit ``guidance_schedule`` must have
    exactly ``num_inference_steps`` entries, which is the shape the pipeline validates.
    """
    image = _single_image(
        omni_server,
        online_client,
        {
            "num_inference_steps": 8,
            "guidance_schedule": [7.0] * 8,
            "mu": 0.5,
            "std": 1.0,
            "height": 512,
            "width": 512,
            "seed": 7,
        },
    )
    assert_image_valid(image, width=512, height=512)
    assert float(np.asarray(image.convert("RGB")).std()) > 1.0
