# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Reference parity for the WeVisOne-VL port against the golden fixtures.

Compares the ported parsers, mask post-processing and decoder architecture
against the released checkpoint and the fixtures written by
``customs/wevisone/ref_golden.py`` (greedy decoding, fixed seed, fixed image).
The reference package itself is never imported here: ``ref_golden.py`` is the
only place that reads it, so parity is checked against recorded fixtures.

Skipped when the checkpoint or fixtures are absent. Write the fixtures once:

    customs/qwen_drive/docker_run.sh "python customs/wevisone/ref_golden.py --decode-masks"

then run locally:

    pytest tests/model_executor/models/wevisone/test_wevisone_golden_parity.py -v -s
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from vllm_omni.model_executor.models.wevisone.config import Qwen3VLSegConfig
from vllm_omni.model_executor.models.wevisone.constants import (
    BOX_CLOSE_TOKEN_ID_FALLBACK,
    BOX_OPEN_TOKEN_ID_FALLBACK,
    COORD_TOKEN_ID_BASE_FALLBACK,
    EOS_TOKEN_ID_FALLBACK,
    SEG_TOKEN_ID_FALLBACK,
    build_prompt,
    coord_token_id_base,
    parse_seg_boxes,
    parse_seg_boxes_from_ids,
)
from vllm_omni.model_executor.models.wevisone.decoder import build_mask_decoder
from vllm_omni.model_executor.models.wevisone.pixel_decoder import (
    _EXPECTED_SIDECAR_SEG_TOKEN_ID,
    _EXPECTED_SIDECAR_SPATIAL_MERGE_SIZE,
    _masks_from_logits,
)

pytestmark = [pytest.mark.local_model, pytest.mark.cpu]

REPO_ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = Path(os.environ.get("WEVISONE_MODEL_DIR", REPO_ROOT / "extra_repos" / "WeVisOne-VL-4B"))
GOLDEN_DIR = Path(os.environ.get("WEVISONE_GOLDEN_DIR", REPO_ROOT / "customs" / "wevisone" / "golden"))
GOLDEN = GOLDEN_DIR / "matting"
SIDEcar = MODEL_DIR / "mask_decoder.pt"


def _fixture_available() -> bool:
    return (GOLDEN / "result.json").is_file() and (GOLDEN / "masks" / "mask_000.npy").is_file()


def _sidecar_available() -> bool:
    return SIDEcar.is_file()


skipif_fixture = pytest.mark.skipif(not _fixture_available(), reason=f"golden fixtures not in {GOLDEN}")
skipif_sidecar = pytest.mark.skipif(not _sidecar_available(), reason=f"sidecar not in {MODEL_DIR}")


@skipif_fixture
def test_golden_prompt_and_boxes_match_the_ported_parsers() -> None:
    golden = json.loads((GOLDEN / "result.json").read_text(encoding="utf-8"))
    response = golden["response"]
    boxes = [[round(value, 3) for value in box] for box in golden["boxes"]]

    assert build_prompt(golden["task"], "the person") == golden["prompt"]
    # Text-path parser: the response the client sees in the chat choice.
    assert [[round(value, 3) for value in box] for box in parse_seg_boxes(response)] == boxes
    # Token-id path: the stage input processor rebuilds boxes from generated ids.
    coords = [9, 65, 753, 1000]
    base = coord_token_id_base(SEG_TOKEN_ID_FALLBACK)
    assert base == COORD_TOKEN_ID_BASE_FALLBACK
    generated = (
        [BOX_OPEN_TOKEN_ID_FALLBACK]
        + [base + coord for coord in coords]
        + [SEG_TOKEN_ID_FALLBACK, BOX_CLOSE_TOKEN_ID_FALLBACK, EOS_TOKEN_ID_FALLBACK]
    )
    assert [[round(value, 3) for value in box] for box in parse_seg_boxes_from_ids(generated)] == boxes


@skipif_fixture
def test_golden_mask_satisfies_the_serving_mask_contract() -> None:
    golden = json.loads((GOLDEN / "result.json").read_text(encoding="utf-8"))
    golden_mask = np.asarray(np.load(GOLDEN / "masks" / "mask_000.npy"), dtype=np.float32).squeeze()
    image_width, image_height = (int(value) for value in golden["image_size"])
    # Masks come back at the ORIGINAL image size, one channel per mask.
    assert golden_mask.shape == (image_height, image_width)
    assert golden_mask.dtype == np.float32
    assert 0.0 <= float(golden_mask.min()) and float(golden_mask.max()) <= 1.0
    # Serving encodes masks with truncation (int not round), so the PNG copy is
    # reproducible from the float fixture under that rule.
    png = np.asarray(Image.open(GOLDEN / "masks" / "mask_000.png").convert("L"), dtype=np.uint8)
    assert np.array_equal(png, np.uint8(np.clip(golden_mask, 0.0, 1.0) * 255))


@skipif_fixture
def test_mask_post_processing_reproduces_the_golden_mask() -> None:
    """The ported ``_masks_from_logits`` is applied to the golden logits.

    Recovering logits from the recorded probabilities is exact in float32, and
    resizing to the size the tensor already has is an identity, so this checks
    the shape rules, the sigmoid and the resize on real decoder output rather
    than on synthetic tensors.
    """
    mask = np.asarray(np.load(GOLDEN / "masks" / "mask_000.npy"), dtype=np.float32).squeeze()
    probabilities = torch.from_numpy(mask)
    logits = torch.log(probabilities / (1.0 - probabilities)).unsqueeze(0).unsqueeze(0)
    decoded = _masks_from_logits(logits, mask.shape[0], mask.shape[1])
    assert decoded.shape == (1, 1, mask.shape[0], mask.shape[1])
    assert decoded.dtype == torch.float32
    assert decoded.is_contiguous()
    assert torch.allclose(decoded[0, 0], probabilities, atol=1e-6)


@skipif_sidecar
def test_architecture_matches_the_released_sidecar() -> None:
    """Strict load proves the ported module has the shipped keys and shapes.

    A partial load would still produce plausible masks, which is why the runtime
    loader is strict; this is the same check expressed as a test.
    """
    checkpoint = torch.load(SIDEcar, map_location="cpu", weights_only=True)
    state_dict = checkpoint["state_dict"]
    assert len(state_dict) == 296
    assert int(checkpoint["seg_token_id"]) == _EXPECTED_SIDECAR_SEG_TOKEN_ID
    assert int(checkpoint["spatial_merge_size"]) == _EXPECTED_SIDECAR_SPATIAL_MERGE_SIZE
    assert Qwen3VLSegConfig.from_dict(dict(checkpoint["cfg"])) == Qwen3VLSegConfig()

    seg_config = Qwen3VLSegConfig()
    decoder = build_mask_decoder(seg_config, 2560, [1024] * len(seg_config.vision_feature_layers))
    decoder.load_state_dict({str(name): tensor for name, tensor in state_dict.items()}, strict=True)
    assert sum(int(tensor.numel()) for tensor in state_dict.values()) == 9609764
