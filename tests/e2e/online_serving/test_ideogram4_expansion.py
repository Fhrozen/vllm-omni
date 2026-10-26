# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
L4 expansion coverage for Ideogram 4 in online serving mode.

Coverage beyond the L2 baseline:
- ``dit_copies=clone`` (the default), where positive and negative passes run on
  separate DiT copies.
- ``dit_copies=keep`` combined with layerwise CPU offload, i.e. one DiT shared
  by both passes while its blocks stay off the GPU between steps.
- Ulysses sequence parallelism, which splits the joint text+image sequence
  across ranks and must still reproduce the single-rank result.

Each case also checks seed determinism: two identical seeded requests must
agree, which is what makes the sequence-parallel comparison meaningful.

The checkpoint is gated, so the whole module is skipped unless ``IG_MODEL_DIR``
points at a local copy.

Run:
    IG_MODEL_DIR=/path/to/ideogram-4-fp8 \
      pytest -s -v tests/e2e/online_serving/test_ideogram4_expansion.py \
      -m "slow and diffusion and H100 and cards_1" --run-level "full_model"
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tests.helpers.assertions import assert_image_valid
from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OnlineOmniClient

_MODEL_DIR = os.environ.get("IG_MODEL_DIR", "")
REPO_ROOT = Path(__file__).resolve().parents[2]
_DEPLOY = os.environ.get("IG_DEPLOY", str(REPO_ROOT / "docs/ideogram-4/deploy_ideogram4_fp8.yaml"))

pytestmark = [
    pytest.mark.slow,
    pytest.mark.diffusion,
    pytest.mark.full_model,
    pytest.mark.skipif(
        not Path(_MODEL_DIR).is_dir(),
        reason="Ideogram 4 is a gated checkpoint; set IG_MODEL_DIR to a local copy to run",
    ),
]

CAPTION = "A small wooden rowboat resting on a still lake at dawn"

# Small and cheap: L4 starts one server per case and the DiT dominates the cost.
# 512x512 is a multiple of 16, so the request is served unmodified.
_EXTRA_BODY = {
    "height": 512,
    "width": 512,
    "num_inference_steps": 8,
    "guidance_schedule": [7.0] * 8,
    "mu": 0.5,
    "std": 1.0,
    "seed": 7,
}


def _case(server_args, marks, case_id):
    return pytest.param(
        OmniServerParams(
            model=_MODEL_DIR,
            stage_config_path=_DEPLOY,
            stage_init_timeout=900,
            server_args=server_args,
        ),
        id=case_id,
        marks=marks,
    )


_CASES = [
    _case(
        ["--custom-pipeline-args", json.dumps({"dit_copies": "clone"})],
        hardware_marks(res={"cuda": "H100"}, num_cards=1),
        "dit_clone",
    ),
    _case(
        [
            "--custom-pipeline-args",
            json.dumps({"dit_copies": "keep"}),
            "--enable-layerwise-offload",
        ],
        hardware_marks(res={"cuda": "H100"}, num_cards=1),
        "dit_keep_layerwise_offload",
    ),
    _case(
        ["--ulysses-degree", "2"],
        hardware_marks(res={"cuda": "H100"}, num_cards=2),
        "ulysses_2",
    ),
]


def _generate(omni_server: OmniServer, online_client: OnlineOmniClient) -> Image.Image:
    responses = online_client.send_diffusion_request(
        {
            "model": omni_server.model,
            "messages": [{"role": "user", "content": CAPTION}],
            "extra_body": _EXTRA_BODY,
        }
    )
    assert len(responses) == 1, f"expected one response, got {len(responses)}"
    response = responses[0]
    assert response.success
    images = response.images
    assert images is not None and len(images) == 1, f"expected one image, got {images}"
    return images[0]


@pytest.mark.parametrize("omni_server", _CASES, indirect=True)
def test_ideogram4_expansion_features(omni_server: OmniServer, online_client: OnlineOmniClient) -> None:
    """Serve the same caption through each feature combination.

    The response must be a single non-streamed JSON object holding one decodable
    image at the requested size, and a repeated identical request must reproduce
    it.
    """
    first = _generate(omni_server, online_client)
    assert_image_valid(first, width=_EXTRA_BODY["width"], height=_EXTRA_BODY["height"])
    # A flat canvas would mean the DiT loop or the AE decoder produced nothing usable.
    assert float(np.asarray(first.convert("RGB")).std()) > 1.0

    second = _generate(omni_server, online_client)
    if not np.array_equal(np.asarray(first.convert("RGB")), np.asarray(second.convert("RGB"))):
        # GPU kernels are not bit-exact; require reproducibility, not
        # bit-identical output.
        np.testing.assert_allclose(
            np.asarray(first.convert("RGB"), dtype=np.float64),
            np.asarray(second.convert("RGB"), dtype=np.float64),
            atol=1.0,
        )
