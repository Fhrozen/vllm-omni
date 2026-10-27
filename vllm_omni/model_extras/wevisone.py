# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Model extras for WeVisOne-VL (Qwen3-VL + pixel-decoder stage).

``extra_body_params`` declares the knobs the reference
``WeVisOneVLSegPipeline.predict()`` accepts, which is the contract the example
harnesses pass through ``extra_args``. Note that ``serving_chat`` merges
declared extra-body fields on the diffusion path only, so an OpenAI chat
request carries its task instruction in the message content -- build it with
``build_prompt(task, query)`` as the examples in ``customs/wevisone/`` do.

``extra_output_params`` is deliberately empty: per-request ``metrics`` is only
populated on the diffusion serving path, so WeVisOne's structured answer
(boxes, task, prompt) is delivered as the text choice content, and each mask
is an ``image_url`` content part identified by its ``mask_index``.
"""

from typing import Any

from vllm_omni.model_executor.models.wevisone.constants import (
    EOS_TOKEN_ID_FALLBACK,
    build_prompt,
)

# Names accepted by the reference ``predict()``; ``prompt`` overrides the
# instruction built from ``task`` + ``query`` when the client supplies one.
WEVISONE_EXTRA_BODY_PARAMS: frozenset[str] = frozenset(
    {
        "task",
        "query",
        "prompt",
        "reference_boxes",
        "decode_matting",
        "min_pixels",
        "max_pixels",
        "decoder_batch_size",
    }
)

WEVISONE_EXTRA_OUTPUT_PARAMS: frozenset[str] = frozenset()


def build_x_to_text_prompt(model: str, prompt: str, has_image: bool) -> tuple[dict[str, Any], list[int] | None]:
    """Build the x_to_text prompt for WeVisOne-VL.

    The family seam has no task slot, so it defaults to ``freeform`` -- the one
    task whose instruction is the query itself. Callers that need a grounded
    task build the instruction with ``build_prompt(task, query)`` before
    sending it, as the examples in ``customs/wevisone/`` do.

    ``stop_token_ids`` pins ``<|im_end|>`` so box spans are not cut short by a
    tokenizer-level default, and ``skip_special_tokens`` must stay off for the
    text stage to keep ``<box>`` tokens in the response.
    """
    del model, has_image
    return (
        {"prompt": build_prompt("freeform", prompt), "modalities": ["text"]},
        [EOS_TOKEN_ID_FALLBACK],
    )
