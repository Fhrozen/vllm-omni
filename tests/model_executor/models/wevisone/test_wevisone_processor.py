# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""L1 processor contract for the WeVisOne canvas fields (CPU, no weights).

vLLM builds the multimodal processor from ``hf_config.architectures``, so the
deploy config pins that key to our class; without the pin the base Qwen3-VL
processor runs and the pixel decoder never receives a canvas.

The canvas is added in ``_postprocess_hf_mm_data``, not ``_call_hf_processor``:
``Qwen3VLMultiModalProcessor`` overrides ``_apply_hf_processor_main`` and calls
the HF processor directly, so the latter hook never fires on that path. Only
``_postprocess_hf_mm_data`` runs on both the base and the Qwen3-VL paths.

vLLM hands the processor ``{f"{modality}s": items}``, so the pictures arrive
under ``images``; reading the singular modality name returns nothing.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import BatchFeature
from vllm.multimodal.processing import BaseMultiModalProcessor

from vllm_omni.model_executor.models.wevisone import processing

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _stub_processor(mocker):
    """A processor instance with the HF config and base field config stubbed out."""
    proc = object.__new__(processing.WeVisOneMultiModalProcessor)
    proc.info = SimpleNamespace(get_hf_config=lambda: SimpleNamespace(seg_config={"cnn_image_size": 8}))
    owner = next(c for c in type(proc).__mro__[1:] if "_get_mm_fields_config" in c.__dict__)
    mocker.patch.object(owner, "_get_mm_fields_config", return_value={})
    return proc


def test_override_matches_the_base_hook_signature() -> None:
    # vLLM calls _postprocess_hf_mm_data(mm_inputs, mm_kwargs, processed_data);
    # a wider signature is never reached, so the canvas fields would disappear.
    ours = inspect.signature(processing.WeVisOneMultiModalProcessor._postprocess_hf_mm_data)
    base = inspect.signature(BaseMultiModalProcessor._postprocess_hf_mm_data)
    assert list(ours.parameters) == list(base.parameters)


def test_canvas_and_original_size_are_added_for_image_inputs(mocker) -> None:
    image = Image.new("RGB", (6, 4), (10, 20, 30))
    proc = _stub_processor(mocker)
    processed = BatchFeature({"image_input": torch.zeros(1, 3, 4, 6)})

    out = proc._postprocess_hf_mm_data({"images": [image], "text": "hi"}, {}, processed)

    # The canvas is the bicubic square of the *un-resized* image, not the
    # smart-resized patches the vision tower sees.
    assert tuple(out[processing.CANVAS_KEY].shape) == (1, 3, 8, 8)
    assert torch.equal(out[processing.HW_KEY], torch.tensor([[4, 6]], dtype=torch.int64))


def test_the_shared_finalize_path_reaches_the_hook(mocker) -> None:
    # Qwen3VLMultiModalProcessor overrides _apply_hf_processor_main and calls the
    # HF processor directly, so _call_hf_processor never fires on that path. Both
    # the base and the Qwen3-VL implementations end in _finalize_hf_mm_data, which
    # is what actually reaches the hook.
    image = Image.new("RGB", (6, 4), (10, 20, 30))
    proc = _stub_processor(mocker)

    out = proc._finalize_hf_mm_data({"images": [image]}, {}, {})

    assert tuple(out[processing.CANVAS_KEY].shape) == (1, 3, 8, 8)


def test_the_singular_modality_name_is_also_accepted(mocker) -> None:
    image = Image.new("RGB", (6, 4), (10, 20, 30))
    proc = _stub_processor(mocker)

    out = proc._postprocess_hf_mm_data({"image": [image]}, {}, BatchFeature({"input_ids": [1]}))

    assert tuple(out[processing.CANVAS_KEY].shape) == (1, 3, 8, 8)


def test_text_only_inputs_gain_no_canvas(mocker) -> None:
    proc = _stub_processor(mocker)

    out = proc._postprocess_hf_mm_data({"text": "hello"}, {}, BatchFeature({"input_ids": [151643]}))

    assert processing.CANVAS_KEY not in out
    assert processing.HW_KEY not in out


def test_extra_keys_are_registered_as_batched_image_fields(mocker) -> None:
    hf_inputs = BatchFeature(
        {processing.CANVAS_KEY: torch.zeros(1, 3, 8, 8), processing.HW_KEY: torch.tensor([[4, 6]])}
    )
    proc = _stub_processor(mocker)

    fields = proc._get_mm_fields_config(hf_inputs, {})

    for key in (processing.CANVAS_KEY, processing.HW_KEY):
        # Kept on CPU: these are read by the stage input processor, not the
        # vision tower, so they must not be pulled onto the GPU.
        assert fields[key].modality == "image"
        assert fields[key].field.keep_on_cpu is True


def test_non_image_inputs_are_rejected_with_an_actionable_message() -> None:
    with pytest.raises(ValueError, match="require PIL image inputs"):
        processing._as_pil_image(np.zeros((2, 2, 3), dtype=np.float32))
