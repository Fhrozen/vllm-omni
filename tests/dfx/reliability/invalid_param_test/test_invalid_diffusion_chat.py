# SPDX-FileCopyrightText: Copyright (c) 2025 vLLM-Omni contributors,
# SPDX-License-Identifier: Apache-2.0
"""Invalid inputs on the diffusion chat route: ``POST /v1/chat/completions`` for image-only models.

Kept separate from ``test_invalid_omni_chat.py`` because that module carries the ``omni``
model-type marker, while Ideogram 4 is a diffusion-only model served on the chat route.

Ideogram 4 answers a chat request with a single image and cannot emit incremental chunks, so
``stream: true`` must be rejected with 400 instead of producing a malformed response. The
rejection is metadata-driven (``DiffusionModelMetadata(supports_streaming=False)``), so the
test also proves the metadata reaches the serving layer. The checkpoint is gated, hence the
skip when no local copy is available.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServer, OmniServerParams, OnlineOmniClient

pytestmark = [pytest.mark.slow, pytest.mark.diffusion]

REPO_ROOT = Path(__file__).resolve().parents[3]
_MODEL_DIR = os.environ.get("IG_MODEL_DIR", "")
_DEPLOY = os.environ.get("IG_DEPLOY", str(REPO_ROOT / "docs/ideogram-4/deploy_ideogram4_fp8.yaml"))

_IDEOGRAM4_SERVER = [
    pytest.param(
        OmniServerParams(model=_MODEL_DIR, stage_config_path=_DEPLOY, stage_init_timeout=900),
        id="ideogram4",
        marks=pytest.mark.skipif(
            not Path(_MODEL_DIR).is_dir(),
            reason="Ideogram 4 is a gated checkpoint; set IG_MODEL_DIR to a local copy to run",
        ),
    )
]


def _minimal_chat_json(omni_server: OmniServer) -> dict[str, object]:
    return {
        "model": omni_server.model,
        "messages": [{"role": "user", "content": "a red fox on a wooden fence"}],
        "stream": False,
    }


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", _IDEOGRAM4_SERVER, indirect=True)
def test_chat_completions_stream_rejected_for_non_streaming_model(
    omni_server: OmniServer,
    online_client: OnlineOmniClient,
) -> None:
    body = _minimal_chat_json(omni_server)
    body["stream"] = True
    online_client.send_chat_completions_http_request(
        {"json": body, "timeout": 120},
        err_code=400,
        err_message=("does not support streaming",),
    )[0]
