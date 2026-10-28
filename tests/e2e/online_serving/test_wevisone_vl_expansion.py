# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""WeVisOne-VL expansion (L4) coverage of the mask response contract.

One feature per case:
- a multi-object answer pairs exactly one mask per box
- masks come back at the original image resolution, not the smart-resized ViT grid
- requests with different image sizes keep their own decoder canvas
- an answer with no ``<OBJ>`` query completes with no mask parts

Run locally (two GPUs, matching the ``wevisone_vl`` CI overlay):
    WEVISONE_MODEL_PATH=/path/to/WeVisOne-VL-4B \\
        pytest -s -v tests/e2e/online_serving/test_wevisone_vl_expansion.py \\
        -m 'full_model and omni and cards_2' --run-level full_model

CI-like run:
    VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HUB_DOWNLOAD_TIMEOUT=300 \\
        pytest -s -v tests/e2e/online_serving/test_wevisone_vl_expansion.py \\
        -m 'full_model and H100 and B200 and omni and cards_2' --run-level full_model

Docker run on the 2x RTX 3060 development machine:
    WVS_LEVEL=full_model bash customs/wevisone/run_e2e_test.sh \\
        tests/e2e/online_serving/test_wevisone_vl_expansion.py
"""

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
# The CI overlay is the profile that fits a 12 GiB card: the bf16 weights leave
# only a few thousand KV tokens, and the multimodal dummy input OOMs the memory
# probe on smaller cards than the shipped deploy yaml assumes.
STAGE_CONFIG_PATH = get_deploy_config_path("ci/wevisone_vl.yaml")

pytestmark = [pytest.mark.full_model, pytest.mark.omni]

_SERVER = [
    pytest.param(
        OmniServerParams(model=MODEL, stage_config_path=STAGE_CONFIG_PATH, stage_init_timeout=900),
        id="wevisone_vl",
    )
]


def _chat_request_config(omni_server: OmniServer, image_size: tuple[int, int], task: str, query: str) -> dict:
    """``send_chat_completions_http_request`` kwargs for one WeVisOne request."""
    image = generate_synthetic_image(*image_size, seed=11)
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


@hardware_test(res={"cuda": ["H100", "B200", "RTX3060"], "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", _SERVER, indirect=True)
def test_multi_object_segmentation_pairs_exactly_one_mask_per_box(
    omni_server: OmniServer,
    online_client: OnlineOmniClient,
    run_level: str,
) -> None:
    """Feature: every box in a multi-object answer gets its own mask part, in box order.

    The case asks for mattes, because a box is only a mask query when it is terminated by
    ``<OBJ>``: the plain ``referring`` and ``detection`` prompts answer with boxes and never
    reach the pixel decoder.
    """
    image_size = (480, 360)
    request_config = _chat_request_config(omni_server, image_size, "segmentation", "each colored square")
    response = online_client.send_chat_completions_http_request(request_config)[0]
    assert_chat_mask_choices(
        response.json_body,
        {"min_mask_count": 2, "mask_size": image_size, "mask_extrema": (40, 215)},
        run_level=run_level,
    )


@hardware_test(res={"cuda": ["H100", "B200", "RTX3060"], "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", _SERVER, indirect=True)
def test_masks_use_the_original_image_resolution_not_the_smart_resized_grid(
    omni_server: OmniServer,
    online_client: OnlineOmniClient,
    run_level: str,
) -> None:
    """Feature: the decoder canvas comes from the un-resized image, so masks match its size.

    ``213x149`` is not a multiple of Qwen3-VL's merge factor (28), so a response that
    followed the smart-resized grid would come back at a different size.
    """
    image_size = (213, 149)
    request_config = _chat_request_config(omni_server, image_size, "segmentation", "the object")
    response = online_client.send_chat_completions_http_request(request_config)[0]
    assert_chat_mask_choices(response.json_body, {"mask_size": image_size}, run_level=run_level)


@hardware_test(res={"cuda": ["H100", "B200", "RTX3060"], "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", _SERVER, indirect=True)
def test_requests_with_different_image_sizes_keep_their_own_canvas(
    omni_server: OmniServer,
    online_client: OnlineOmniClient,
    run_level: str,
) -> None:
    """Feature: per-request accumulation keeps one request's canvas from leaking into the next."""
    first_size = (160, 120)
    second_size = (320, 240)

    for image_size in (first_size, second_size):
        request_config = _chat_request_config(omni_server, image_size, "matting", "the object")
        response = online_client.send_chat_completions_http_request(request_config)[0]
        assert_chat_mask_choices(response.json_body, {"mask_size": image_size}, run_level=run_level)


@hardware_test(res={"cuda": ["H100", "B200", "RTX3060"], "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", _SERVER, indirect=True)
def test_answers_without_an_obj_query_complete_without_mask_parts(
    omni_server: OmniServer,
    online_client: OnlineOmniClient,
    run_level: str,
) -> None:
    """Feature: a box-only answer finishes with no masks instead of waiting for connector input.

    A box is only a mask query when ``<OBJ>`` follows its coordinates, so the ``count_box``
    prompt answers with boxes and nothing to decode. A full-payload consumer cannot finish a
    request it never received input for, so the stage processor ships an empty mask selection
    and the decoder returns an empty tensor; withholding the payload would park the request
    until the connector input-wait timeout (``VLLM_OMNI_INPUT_WAIT_TIMEOUT_S``, 600 s by
    default), which is longer than this client timeout.
    """
    image_size = (240, 180)
    request_config = _chat_request_config(omni_server, image_size, "count_box", "the object")
    response = online_client.send_chat_completions_http_request(request_config)[0]
    body = response.json_body

    choices = body.get("choices")
    assert isinstance(choices, list) and choices, "Chat response is missing choices[]"
    text = ""
    image_parts: list[dict] = []
    for choice in choices:
        content = choice.get("message", {}).get("content")
        if isinstance(content, str):
            text = text or content
        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "image_url":
                    image_parts.append(part)
                elif part.get("type") == "text" and not text:
                    text = part.get("text") or ""

    assert text, "Chat response has no text part carrying the structured output"
    print(f"structured text output is: {text}")
    if run_level in {"advanced_model", "full_model"}:
        assert "<box>" in text, "The answer carries no box span, so the request proves nothing about box-only tasks"
    assert not image_parts, f"A box-only answer must carry no mask parts, got {len(image_parts)}"
