# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen-Drive stage bridges."""

from __future__ import annotations

import functools
from typing import Any

from vllm.logger import init_logger

from vllm_omni.inputs.data import OmniTokensPrompt

logger = init_logger(__name__)


@functools.cache
def _chat_token_ids(tokenizer: str, trust_remote_code: bool) -> tuple[int, int, tuple[int, ...]]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=trust_remote_code)
    return (
        tok.convert_tokens_to_ids("<|im_end|>"),
        tok.convert_tokens_to_ids("<|endoftext|>"),
        tuple(tok.encode("\n", add_special_tokens=False)),
    )


def vlm_to_prefill(
    source_outputs: list[Any],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
    *,
    target_model_config: Any,
) -> list[OmniTokensPrompt]:
    """Stage 0 -> stage 1: build the prompt whose KV the planner reads.

    The planner was trained on a cache that ends with the closed assistant turn (``<|im_end|>\\n``). A direct-planning
    prompt already ends that way; for a reasoning prompt the generated reasoning and the turn ending are appended.
    Stage 0 reports its prompt with image placeholders expanded; the orchestrator hands the already-processed image
    features to stage 1 (``model_stage="thinker"``), so the expanded ids are kept as they are.
    """
    im_end, endoftext, newline = _chat_token_ids(
        target_model_config.tokenizer, bool(target_model_config.trust_remote_code)
    )
    closing = [im_end, *newline]

    result = []
    for source in source_outputs:
        ids = list(source.prompt_token_ids)
        if ids[-len(closing) :] != closing:
            content = list(source.outputs[0].token_ids)
            for position, token in enumerate(content):
                if token in (im_end, endoftext):
                    content = content[:position]
                    break
            ids = ids + content + closing
        result.append(OmniTokensPrompt(prompt_token_ids=ids))
    return result


def prefill_to_planner(
    source_outputs: list[Any],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
) -> dict[str, Any]:
    """Stage 1 -> stage 2: the planner reads the transferred KV, so its own prompt is empty.

    Without this bridge the diffusion stage would be handed the original prompt, whose processed image tensors
    cannot be serialized to it.
    """
    return {"prompt": ""}
