# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Online-serving coverage for WeVisOne-VL segmentation output.

Stage 0 (Qwen3-VL) answers with boxes and ``<OBJ>`` markers; stage 1 (the 9.6M-parameter
pixel decoder) turns every ``<OBJ>`` row into a mask. Over ``POST /v1/chat/completions``
the masks arrive as one PNG part per object while the boxes stay in the text part, so the
pairing is positional: mask ``i`` answers box ``i`` and carries ``mask_index == i``.

Equivalent to running:
    vllm-omni serve "tencent/WeVisOne-VL-4B" --omni --port 8091

    python3 customs/wevisone/openai_request_example.py --task matting --query "the person"

Local run (the deploy yaml needs two GPUs):
    WEVISONE_MODEL_PATH=/path/to/WeVisOne-VL-4B \\
        pytest -s -v tests/e2e/online_serving/test_wevisone_vl.py \\
        -m 'advanced_model and omni and cards_2' --run-level advanced_model

CI-like run:
    VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HUB_DOWNLOAD_TIMEOUT=300 \\
        pytest -s -v tests/e2e/online_serving/test_wevisone_vl.py \\
        -m 'core_model and H100 and omni and cards_2'
Docker run on the 2x RTX 3060 development machine (stage 0 and stage 1 land on
GPU 0 and GPU 1; the CI overlay keeps stage 0 inside 12 GiB):
    bash customs/wevisone/run_e2e_test.sh tests/e2e/online_serving/test_wevisone_vl.py
    bash customs/wevisone/run_e2e_test.sh tests/e2e/online_serving/test_wevisone_vl.py -k text_only"""

from __future__ import annotations

import os

import pytest

from tests.helpers.assertions import assert_chat_mask_choices
from tests.helpers.mark import hardware_test
from tests.helpers.media import generate_synthetic_image
from tests.helpers.runtime import OmniServer, OmniServerParams, OnlineOmniClient
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.model_executor.models.wevisone.constants import SYSTEM_PROMPT, build_prompt

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = os.environ.get("WEVISONE_MODEL_PATH", "tencent/WeVisOne-VL-4B")

pytestmark = [pytest.mark.omni]

# Object counts are model-dependent, so the assertions below check the response contract
# (part shape, ``mask_index``, box/mask pairing) rather than a fixed object count.
_IMAGE_SIZE = (320, 240)

# The CI overlay is the run profile for hardware with less headroom than the
# shipped deploy yamls: on a 12 GiB card the bf16 weights leave room for only a
# few thousand KV tokens, and the multimodal dummy input alone OOMs the memory
# probe. The same overlay is used on the CI mirrors, where it is comfortably
# within budget.
_FULL_PIPELINE_PARAMS = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path("ci/wevisone_vl.yaml"),
            stage_init_timeout=900,
        ),
        id="wevisone_vl",
    )
]

_TEXT_ONLY_PIPELINE_PARAMS = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path("ci/wevisone_vl_text_only.yaml"),
            stage_init_timeout=900,
        ),
        id="wevisone_vl_text_only",
    )
]


def _chat_request_config(omni_server: OmniServer, task: str, query: str) -> dict:
    """Build ``send_chat_completions_http_request`` kwargs for one WeVisOne request.

    The task instruction travels in message content: ``task`` in ``extra_body`` is merged
    only on the diffusion serving path, so a chat request cannot rely on it.
    """
    image = generate_synthetic_image(*_IMAGE_SIZE, seed=7)
    body = {
        "model": omni_server.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image['base64']}"}},
                    {"type": "text", "text": build_prompt(task, query)},
                ],
            },
        ],
    }
    return {"json": body, "timeout": 180}


# Real weights are required: a core_model run patches every stage to dummy weights, so
# the AR stage would emit random tokens and no box span to pair a mask with.
@pytest.mark.advanced_model
@hardware_test(res={"cuda": ["H100", "RTX3060"], "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", _FULL_PIPELINE_PARAMS, indirect=True)
def test_matting_masks_are_paired_with_boxes_by_mask_index(
    omni_server: OmniServer,
    online_client: OnlineOmniClient,
    run_level: str,
) -> None:
    """Baseline mask round trip: text part carries the box, image part carries its mask."""
    request_config = _chat_request_config(omni_server, "matting", "the object")
    responses = online_client.send_chat_completions_http_request(request_config)
    assert len(responses) == 1
    assert_chat_mask_choices(
        responses[0].json_body,
        {"key_words": ["<box>", "<OBJ>"], "mask_size": _IMAGE_SIZE},
        run_level=run_level,
    )


@pytest.mark.advanced_model
@hardware_test(res={"cuda": ["H100", "RTX3060"], "rocm": "MI325"}, num_cards=1)
@pytest.mark.parametrize("omni_server", _TEXT_ONLY_PIPELINE_PARAMS, indirect=True)
def test_text_only_pipeline_never_loads_the_decoder_sidecar(
    omni_server: OmniServer,
    online_client: OnlineOmniClient,
    run_level: str,
) -> None:
    """Detection on the text-only pipeline answers with boxes and no mask parts.

    The decoder is loaded lazily, so a text-only task runs on a checkpoint that has no
    ``mask_decoder.pt``; the response must therefore be a single text choice.
    """
    request_config = _chat_request_config(omni_server, "detection", "the object")
    responses = online_client.send_chat_completions_http_request(request_config)
    assert len(responses) == 1
    content = responses[0].json_body["choices"][0]["message"]["content"]
    if run_level in {"advanced_model", "full_model"}:
        assert isinstance(content, str), "Text-only serving must keep the plain structured answer as text content"
        assert "<box>" in content.lower(), f"No box in the text-only answer: {content!r}"
        print(f"text-only structured answer is: {content}")
