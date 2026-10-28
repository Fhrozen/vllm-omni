# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Light coverage for wevisone.vlm2pixel_decoder_full_payload.

Covers the two contracts that decide whether a request becomes a mask request:
boxes are reconstructed from the *generated* token ids only, and the query rows
referenced by those boxes must actually exist in the accumulated ``t_seg``.
A request that fails either check ships an empty mask selection rather than
nothing: the consumer stage cannot finish a request it never received input for,
so withholding the payload would park it until the connector input-wait timeout.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.wevisone import constants
from vllm_omni.model_executor.stage_input_processors import wevisone

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SEG_TOKEN_ID = constants.SEG_TOKEN_ID_FALLBACK
COORD_BASE = constants.COORD_TOKEN_ID_BASE_FALLBACK
BOX_OPEN = constants.BOX_OPEN_TOKEN_ID_FALLBACK
BOX_CLOSE = constants.BOX_CLOSE_TOKEN_ID_FALLBACK
EOS = constants.EOS_TOKEN_ID_FALLBACK
# A vision position in the prompt: it contributes a t_seg row but is never a
# query row, so it must not be picked up by the <OBJ> scan.
IMAGE_TOKEN = 151655

GOLDEN_COORDS = [9, 65, 753, 1000]
GOLDEN_BOX = [9 / 1000, 65 / 1000, 753 / 1000, 1000 / 1000]


def _make_request(prompt_token_ids, output_token_ids):
    return SimpleNamespace(
        request_id="wevisone-req",
        prompt_token_ids=list(prompt_token_ids),
        output_token_ids=list(output_token_ids),
        all_token_ids=list(prompt_token_ids) + list(output_token_ids),
        status=SimpleNamespace(name="FINISHED_STOPPED"),
        sampling_params=None,
    )


def _generated_ids(coords, obj_count=1):
    """One answer box followed by ``obj_count`` ``<OBJ>`` tokens, then EOS."""
    ids = [BOX_OPEN] + [COORD_BASE + coord for coord in coords]
    ids.extend([SEG_TOKEN_ID] * obj_count)
    ids.extend([BOX_CLOSE, EOS])
    return ids


def _pooling_output(t_seg_rows, *, nested=True, t_seg=None):
    if t_seg is None:
        t_seg = torch.zeros((t_seg_rows, 3), dtype=torch.float32)
    group = {
        "features": torch.arange(1 * 3 * 2 * 2, dtype=torch.float32).reshape(1, 3, 2, 2),
        "grid_hw": torch.tensor([[2, 2]], dtype=torch.int32),
        "t_seg": t_seg,
        "canvas": torch.full((1, 3, 4, 4), 0.25, dtype=torch.float32),
        "mask_hw": torch.tensor([[4, 6]], dtype=torch.int32),
    }
    # Two views of the same group: the nested dict the AR runner emits and the
    # dotted-key form produced by flatten_payload.
    return {"segmentation": group} if nested else {f"segmentation.{key}": value for key, value in group.items()}


def _generated_box_request():
    # The prompt is vision tokens only: the box and the <OBJ> both come from the
    # answer, so the parsed box refers to the row of the generated <OBJ>.
    prompt = [IMAGE_TOKEN]
    generated = _generated_ids(GOLDEN_COORDS, 1)
    return _make_request(prompt, generated)


def test_full_payload_carries_boxes_reconstructed_from_generated_ids():
    request = _generated_box_request()
    # t_seg holds one row per token the runner fed back in, so it covers every
    # position of the request once generation is complete.
    payload = wevisone.vlm2pixel_decoder_full_payload(None, _pooling_output(len(request.all_token_ids)), request)

    assert payload is not None
    assert payload["segmentation"]["boxes"].tolist() == [pytest.approx(GOLDEN_BOX)]
    assert payload["segmentation"]["boxes"].dtype == torch.float32
    assert payload["ids"]["prompt"] == request.prompt_token_ids
    assert payload["ids"]["all"] == request.all_token_ids
    assert bool(payload["meta"]["finished"].item()) is True
    assert payload["meta"]["finished"].dtype == torch.bool

    for name in ("features", "grid_hw", "t_seg", "canvas", "mask_hw"):
        tensor = payload["segmentation"][name]
        assert tensor.device.type == "cpu"


def test_prompt_boxes_are_not_treated_as_answer_boxes():
    # visual_prompt puts reference boxes in the prompt; those must not become
    # the mask contract.
    prompt_box = [BOX_OPEN] + [COORD_BASE + value for value in (100, 100, 500, 500)]
    prompt = prompt_box + [IMAGE_TOKEN]
    generated = _generated_ids(GOLDEN_COORDS, 1)
    request = _make_request(prompt, generated)

    payload = wevisone.vlm2pixel_decoder_full_payload(None, _pooling_output(len(request.all_token_ids)), request)

    assert payload is not None
    assert payload["segmentation"]["boxes"].shape == (1, 4)
    assert payload["segmentation"]["boxes"].tolist()[0] == pytest.approx(GOLDEN_BOX)


def test_flat_and_nested_pooling_views_are_equivalent():
    request = _generated_box_request()
    rows = len(request.all_token_ids)
    nested = wevisone.vlm2pixel_decoder_full_payload(None, _pooling_output(rows, nested=True), request)
    flat = wevisone.vlm2pixel_decoder_full_payload(None, _pooling_output(rows, nested=False), request)

    assert nested["segmentation"]["boxes"].tolist() == flat["segmentation"]["boxes"].tolist()
    assert set(nested) == {"ids", "segmentation", "meta"}


def test_mask_less_answer_ships_an_empty_mask_selection():
    # A count_*, ocr or freeform answer has no <OBJ> box query, so there is no
    # mask to decode. The payload still has to reach the consumer stage, which
    # answers an empty box list with an empty mask tensor.
    request = _make_request([SEG_TOKEN_ID], [EOS])
    payload = wevisone.vlm2pixel_decoder_full_payload(None, _pooling_output(1), request)

    assert payload is not None
    assert payload["segmentation"]["boxes"].shape == (0, 4)
    assert payload["segmentation"]["t_seg"].numel() == 0
    assert bool(payload["meta"]["finished"].item()) is True


def test_box_without_obj_query_is_not_a_mask_query():
    # detection and referring answers emit <box> spans with no <OBJ>, so there is
    # no query row to pair them with and no mask is produced for them.
    generated = [BOX_OPEN] + [COORD_BASE + value for value in GOLDEN_COORDS] + [BOX_CLOSE, EOS]
    request = _make_request([IMAGE_TOKEN], generated)
    payload = wevisone.vlm2pixel_decoder_full_payload(None, _pooling_output(len(request.all_token_ids)), request)

    assert payload is not None
    assert payload["segmentation"]["boxes"].shape == (0, 4)
    assert payload["segmentation"]["t_seg"].numel() == 0


def test_non_mapping_pooling_output_is_reported_as_a_gate_risk(caplog):
    request = _make_request([SEG_TOKEN_ID], [EOS])
    with caplog.at_level(logging.WARNING):
        assert wevisone.vlm2pixel_decoder_full_payload(None, [], request) is None
    assert "consumer wait gate may hang" in caplog.text


def test_query_rows_that_were_never_fed_back_are_reported_and_dropped(caplog):
    # The <OBJ> was sampled but never fed back into the model, so t_seg holds
    # only the prompt row and the parsed box has no row to point at. The builder
    # reports the gap instead of pairing the box with an unrelated row, and the
    # request ships with no masks rather than a fabricated one.
    request = _generated_box_request()
    with caplog.at_level(logging.WARNING):
        payload = wevisone.vlm2pixel_decoder_full_payload(None, _pooling_output(1), request)

    assert "no <OBJ> row was fed back in" in caplog.text
    assert payload is not None
    assert payload["segmentation"]["boxes"].shape == (0, 4)
    assert payload["segmentation"]["t_seg"].numel() == 0


def test_query_rows_are_selected_at_the_obj_positions_in_response_order():
    # Two <OBJ> tokens follow one box. Reference parity pairs query row i with
    # box i and keeps min(boxes, rows), so box 0 pairs with the earlier query and
    # the surplus row is dropped.
    request = _make_request([IMAGE_TOKEN], _generated_ids(GOLDEN_COORDS, 2))
    obj_positions = [i for i, token_id in enumerate(request.all_token_ids) if token_id == SEG_TOKEN_ID]
    assert len(obj_positions) == 2

    t_seg = torch.arange(len(request.all_token_ids) * 3, dtype=torch.float32).reshape(len(request.all_token_ids), 3)
    payload = wevisone.vlm2pixel_decoder_full_payload(None, _pooling_output(0, t_seg=t_seg), request)

    assert payload is not None
    assert torch.equal(payload["segmentation"]["t_seg"], t_seg[[obj_positions[0]]])
    assert payload["segmentation"]["boxes"].shape == (1, 4)
    assert payload["segmentation"]["boxes"].tolist() == [pytest.approx(GOLDEN_BOX)]


def test_query_rows_keep_accumulation_semantics():
    # features/grid_hw/canvas/mask_hw/boxes are complete results; t_seg is a
    # per-step delta and must stay CONCAT_DIM0 so rows accumulate.
    replace = wevisone._FULL_PAYLOAD_REPLACE_KEYS
    assert "segmentation.t_seg" not in replace
    assert {"segmentation.features", "segmentation.boxes"} <= replace


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
