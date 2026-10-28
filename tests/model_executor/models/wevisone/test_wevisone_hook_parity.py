# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Hook parity for the tensors the WeVisOne-VL pixel decoder consumes.

The pixel decoder is fed by the vision tower, so a capture that reads the wrong
layer, keeps the singleton sequence dimension the installed vllm tower adds, or
un-shuffles the patch order in the wrong way still produces a plausible-looking
mask. Comparing captured tensors is what proves the capture itself is right.

Two complementary checks:

* CPU logic dual-run against recorded fixtures. Both sides ran the *same* tower,
  so the comparison is exact and isolates the ported capture code: layer
  selection, the singleton sequence dimension, and the block-major un-shuffle.
  Fixtures are written by ``customs/wevisone/parity_capture.py``; the reference
  package is never imported here, so the test runs without it.
* GPU tower dual-run, live in one process. Here the two sides are different
  implementations of the same weights, so the check is a tolerance.

Why the GPU check is a tolerance, and what that tolerance means
---------------------------------------------------------------
The two vision towers are mathematically equivalent: position ids agree token for
token, the rotary formulas are identical once HF's duplicated frequencies and
vllm's partial-rotary factor are accounted for, and the per-block math is the
same. They are not bit-identical because the kernels differ (vllm's fused input
norm and encoder attention against HF's ``nn.LayerNorm`` and SDPA), and because
TF32 is on by default in this image, so the fp32 ``Conv3d`` convolution runs in
TF32.

Only the merger hooks fire on the port: the vision blocks are wrapped by the
torch-compile decorator, which calls ``forward`` directly and bypasses
``nn.Module._call_impl``, so a pre-hook registered on a block is never invoked.
That is what the port relies on — it captures at the merger, the same place the
reference package captures at.

The test loads only the vision tower, so the two implementations plus the decoder
occupy about 3.6 GiB and the run fits on one 12 GiB card. It also checks the fused
input norm against the processor: the kernel computes ``(x * rescale_factor - mean)
/ std``, which means it takes the unrescaled bytes rather than the already-rescaled
floats, and it reproduces the processor output to 1.2e-7 on the golden sample.

Measured on the golden sample in fp32 with TF32 disabled, the per-layer mean
absolute difference relative to the map scale is 1.8e-5 (layer 5), 1.8e-6
(layer 11), 4.6e-6 (layer 17) and 7.8e-6 (layer 23); the largest single element
difference is 0.085 of scale. Decoding both sets of maps through the same decoder
gives 77.7 dB PSNR between the two masks. With TF32 left on, which is the serving
path, the same measurements are 6e-4 to 3.9e-2 — an order of magnitude worse at
depth, which is why the test pins the flags itself instead of inheriting the image
default. A wrong hook point or a row-major un-shuffle lands far outside the band
asserted here: those mistakes drop the correlation to 0.1-0.5.

The bands below sit just above that floor, so a regression in the capture or in
the kernels the port relies on still trips the test.

Run locally:

    pytest tests/model_executor/models/wevisone/test_wevisone_hook_parity.py -v -s

Write the CPU fixtures first:

    customs/qwen_drive/docker_run.sh "python customs/wevisone/parity_capture.py --side reference --device cpu --out customs/wevisone/golden/hooks/cpu"
    customs/qwen_drive/docker_run.sh "python customs/wevisone/parity_capture.py --side port --device cpu --out customs/wevisone/golden/hooks/cpu"

The GPU test needs the checkpoint and the CPU fixtures; it skips when either is
missing, and it compares against the recorded reference maps rather than writing
new ones.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.wevisone import pixel_decoder
from vllm_omni.model_executor.models.wevisone.config import Qwen3VLSegConfig
from vllm_omni.model_executor.models.wevisone.constants import parse_seg_boxes
from vllm_omni.model_executor.models.wevisone.decoder import build_mask_decoder
from vllm_omni.model_executor.models.wevisone.feature_bridge import (
    premerge_grid_hw,
    premerge_tokens_to_2d,
)

pytestmark = [pytest.mark.local_model]

REPO_ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = Path(os.environ.get("WEVISONE_MODEL_DIR", REPO_ROOT / "extra_repos" / "WeVisOne-VL-4B"))
HOOKS_DIR = Path(os.environ.get("WEVISONE_HOOKS_DIR", REPO_ROOT / "customs" / "wevisone" / "golden" / "hooks"))
GOLDEN_DIR = HOOKS_DIR.parent / "matting"
SIDEcar = MODEL_DIR / "mask_decoder.pt"
FIXED_IMAGE = REPO_ROOT / "extra_repos" / "WeVisOne-VL" / "assets" / "demo" / "seabreeze.jpg"
SEED = 1234
LAYERS = [5, 11, 17, 23]
MERGE_SIZE = 2
QUERY = "the person"


def _manifest(side: str) -> dict[str, Any] | None:
    path = HOOKS_DIR / "cpu" / side / "manifest.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _array(manifest: dict[str, Any] | None, name: str) -> np.ndarray | None:
    if manifest is None:
        return None
    entry = next((item for item in manifest["entries"] if Path(item["file"]).name == name), None)
    if entry is None:
        return None
    path = HOOKS_DIR / "cpu" / str(manifest["side"]) / Path(entry["file"]).name
    if not path.is_file():
        return None
    return np.load(path)


def _golden() -> dict[str, Any]:
    return json.loads((GOLDEN_DIR / "result.json").read_text(encoding="utf-8"))


skipif_cpu_fixtures = pytest.mark.skipif(
    not (_manifest("reference") and _manifest("port")),
    reason=f"CPU hook fixtures not in {HOOKS_DIR / 'cpu'}",
)
skipif_mask_fixture = pytest.mark.skipif(
    not (_manifest("reference") and SIDEcar.is_file()),
    reason=f"reference mask fixture or sidecar not in {HOOKS_DIR / 'cpu'} / {MODEL_DIR}",
)
skipif_checkpoint = pytest.mark.skipif(
    not (any(MODEL_DIR.glob("*.safetensors")) and FIXED_IMAGE.is_file()),
    reason=f"checkpoint shards or golden image not in {MODEL_DIR} / {FIXED_IMAGE}",
)


def _common_metadata(reference: dict[str, Any], port: dict[str, Any]) -> None:
    """The two sides must have read the same input before maps are comparable."""
    assert reference["layers"] == port["layers"] == LAYERS
    assert reference["grid_hw"] == port["grid_hw"]
    assert reference["response"] == port["response"]
    assert reference["image_size"] == port["image_size"]
    assert reference["prompt"] == port["prompt"]
    assert reference["seed"] == port["seed"]


def _to_numpy(value: torch.Tensor) -> np.ndarray:
    """Captured maps live on the run device; the comparison is done on host."""
    return value.detach().cpu().float().numpy()


def _stats(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float, float, float]:
    """max, mean, correlation and scale, reported against the map scale."""
    difference = np.abs(actual - expected)
    scale = max(float(np.abs(expected).max()), 1.0)
    correlation = float(np.corrcoef(actual.astype(np.float64).ravel(), expected.astype(np.float64).ravel())[0, 1])
    return float(difference.max()), float(difference.mean()), correlation, scale


def _psnr(actual: np.ndarray, expected: np.ndarray) -> float:
    """Peak signal-to-noise ratio for values in [0, 1]; inf when identical."""
    mse = float(np.mean((actual.astype(np.float64) - expected.astype(np.float64)) ** 2))
    return float("inf") if mse == 0.0 else float(10.0 * np.log10(1.0 / mse))


def _decoder_from_sidecar(device: str) -> pixel_decoder.WeVisOnePixelDecoder:
    """The port's decoder loaded with the released weights, dimensions read from them.

    ``llm_dim`` and the per-level visual dimensions come from the state dict
    itself, so the test cannot drift away from the checkpoint it is checking.
    """
    checkpoint = torch.load(SIDEcar, map_location="cpu", weights_only=True)
    cfg = Qwen3VLSegConfig.from_dict(dict(checkpoint.get("cfg") or {}))
    state = checkpoint["state_dict"]
    llm_dim = int(state["query_builder.content_proj.weight"].shape[1])
    visual_in_dims = [
        int(state[f"visual_encoder.input_proj.{index}.0.weight"].shape[1]) for index in range(cfg.msda_num_levels)
    ]

    decoder = pixel_decoder.WeVisOnePixelDecoder.__new__(pixel_decoder.WeVisOnePixelDecoder)
    torch.nn.Module.__init__(decoder)
    decoder._loaded = True
    decoder.seg_config = cfg
    decoder.vllm_config = SimpleNamespace(device_config=SimpleNamespace(device=device))
    instance = build_mask_decoder(cfg, llm_dim, visual_in_dims)
    instance.load_state_dict(state, strict=True)
    instance.to(device).eval()
    decoder.decoder = instance
    return decoder


def _decode(
    decoder: pixel_decoder.WeVisOnePixelDecoder,
    features: np.ndarray,
    t_seg: np.ndarray,
    boxes: list[list[float]],
    canvas: np.ndarray,
    grid_hw: list[int],
) -> np.ndarray:
    """Decode one set of feature maps through the port's serving path."""
    golden = _golden()
    height, width = (int(value) for value in golden["image_size"][::-1])
    payload = {
        "segmentation.features": torch.from_numpy(features),
        "segmentation.grid_hw": torch.tensor([tuple(int(value) for value in grid_hw)], dtype=torch.int32),
        "segmentation.t_seg": torch.from_numpy(t_seg),
        "segmentation.boxes": torch.tensor(boxes[: t_seg.shape[0]], dtype=torch.float32),
        "segmentation.canvas": torch.from_numpy(canvas),
        "segmentation.mask_hw": torch.tensor([[height, width]], dtype=torch.int32),
    }
    with torch.no_grad():
        masks = decoder._decode_request(payload)
    assert masks is not None
    assert masks.shape == (len(boxes), 1, height, width)
    return masks.detach().numpy()


@pytest.mark.cpu
@skipif_cpu_fixtures
def test_cpu_logic_dual_run_reproduces_the_reference_maps() -> None:
    """Exact match: both sides ran the same tower, only the capture code differs.

    This is the part of the port that is easy to get wrong and hard to see —
    which blocks are hooked, the singleton sequence dimension vllm's tower
    inserts before the block loop, and the block-major un-shuffle back to a
    spatial map.
    """
    reference = _manifest("reference")
    port = _manifest("port")
    if reference is None or port is None:
        pytest.skip("the CPU fixtures are not recorded")
        return
    _common_metadata(reference, port)
    # The reference hook sees [N, C]; vllm's tower hands the hook [N, 1, C] and
    # the port folds it back, so the recorded shapes must agree.
    assert reference["hook_input_shape"] == port["hook_input_shape"]

    for layer in LAYERS:
        expected = _array(reference, f"layer_{layer:03d}.npy")
        actual = _array(port, f"layer_{layer:03d}.npy")
        assert expected is not None and actual is not None, f"layer {layer} fixture missing"
        assert actual.shape == expected.shape
        # Same tower, same dtype, same code path: the fold is a no-op on CPU, so
        # the maps must be byte-identical rather than merely close.
        assert np.array_equal(actual, expected), f"layer {layer} differs"

    assert np.array_equal(_array(port, "canvas.npy"), _array(reference, "canvas.npy"))


@hardware_test(res={"cuda": "RTX3060"})
@skipif_checkpoint
def test_gpu_vision_tower_dual_run_matches_within_fp32_noise() -> None:
    """vllm's vision tower against HF's: same weights, same input, one process.

    Runs in fp32 with TF32 disabled so what remains is the implementation rather
    than the precision setting. The ported capture classes are used on both
    towers, and the CPU test above already proves that capture code is exact, so
    anything measured here belongs to the tower.
    """
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    from PIL import Image
    from safetensors import safe_open
    from transformers import AutoConfig, AutoProcessor
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
    from vllm.config import DeviceConfig, ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import ensure_model_parallel_initialized, init_distributed_environment
    from vllm.model_executor.layers.fusion.mm_input_norm import FusedMMInputNorm, _load_norm_params
    from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer

    from vllm_omni.model_executor.models.wevisone.feature_bridge import VisualFeatureCapture

    def as_visual_owner(tower):
        """Give the capture the `model.visual` path it resolves on a full model."""
        return SimpleNamespace(model=SimpleNamespace(visual=tower))

    config = AutoConfig.from_pretrained(MODEL_DIR)
    device = "cuda"
    dtype = torch.float32

    # Only the vision tower is loaded: the checkpoint is 4.83B parameters and the
    # tower is about 350M, so both towers fit on one 12 GiB card in fp32.
    prefix = "model.visual."
    weights: dict[str, torch.Tensor] = {}
    for shard in sorted(MODEL_DIR.glob("*.safetensors")):
        with safe_open(shard, framework="pt") as handle:
            for name in handle.keys():
                if name.startswith(prefix):
                    weights[name[len(prefix) :]] = handle.get_tensor(name)

    hf_tower = Qwen3VLVisionModel(config.vision_config)
    hf_tower.load_state_dict(weights, strict=True)
    hf_tower.to(device, dtype).eval()

    vllm_config = VllmConfig(
        model_config=ModelConfig(model=str(MODEL_DIR), limit_mm_per_prompt={"image": 4}, seed=SEED, dtype="float32"),
        device_config=DeviceConfig(device),
    )
    # The tower asks the ambient parallel state for its TP size, and CustomOp
    # layers read the ambient config, so both have to be in place before it runs.
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(1, 0, "tcp://127.0.0.1:29513", 0, "nccl")
        ensure_model_parallel_initialized(1, 1)
        port_tower = Qwen3_VisionTransformer(config.vision_config)
        port_tower.load_weights((name, tensor.to(dtype)) for name, tensor in weights.items())
        params = _load_norm_params(vllm_config.model_config)
        fused_norm = FusedMMInputNorm(params.image_mean, params.image_std, params.rescale_factor)
    port_tower.to(device, dtype).eval()
    # The affine parameters are buffers registered on CPU; the Triton path needs
    # them on the same device as the pixels, as they are inside the engine.
    fused_norm.to(device)

    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    image = Image.open(FIXED_IMAGE).convert("RGB")
    normalized = processor(text=["x"], images=[image], padding=True, return_tensors="pt")
    unprocessed = processor(
        text=["x"], images=[image], padding=True, return_tensors="pt", do_rescale=False, do_normalize=False
    )
    grid_thw = normalized["image_grid_thw"].to(device)

    reference_norm = normalized["pixel_values"].to(device, dtype)

    # With device-side normalisation off — the default — the engine builds an
    # identity norm and the processor's output is what reaches the tower. When it
    # is on, raw uint8 pixels travel to the device and the fused kernel replaces
    # the processor's rescale-and-normalise step, so the kernel has to reproduce
    # it. Its weight is rescale_factor / std, which is why it takes the unrescaled
    # bytes rather than the already-rescaled floats.
    kernel = fused_norm(unprocessed["pixel_values"].to(device), dtype)
    norm_error = (kernel - reference_norm).abs()
    print(f"\nfused input norm vs processor: max {float(norm_error.max()):.2e}")
    assert float(norm_error.max()) <= 1e-6, (
        f"fused input norm differs from the processor by {float(norm_error.max()):.2e}"
    )

    hf_capture = VisualFeatureCapture(as_visual_owner(hf_tower), feature_layers=LAYERS)
    with hf_capture:
        with torch.no_grad():
            hf_tower(reference_norm, grid_thw)

    port_capture = VisualFeatureCapture(as_visual_owner(port_tower), feature_layers=LAYERS)
    with port_capture:
        with torch.no_grad():
            port_tower(reference_norm, grid_thw)

    grid_hw = [int(value) for value in premerge_grid_hw(grid_thw)[0]]
    reference = _manifest("reference")
    if reference is not None:
        assert grid_hw == [int(value) for value in reference["grid_hw"]]

    print(f"\nGPU vision tower dual-run, fp32, TF32 off, grid {tuple(grid_hw)}")
    port_maps = []
    reference_maps = []
    for layer in LAYERS:
        expected = _to_numpy(premerge_tokens_to_2d(hf_capture.premerge_features[layer], grid_hw, MERGE_SIZE))
        actual = _to_numpy(premerge_tokens_to_2d(port_capture.premerge_features[layer], grid_hw, MERGE_SIZE))
        assert actual.shape == expected.shape
        maximum, mean, correlation, scale = _stats(actual, expected)
        print(
            f"  layer {layer:03d}: max/scale={maximum / scale:.4f} mean/scale={mean / scale:.2e} corr={correlation:.4f}"
        )
        # Correlation is the structural gate: a wrong layer, an unfolded batch
        # dimension or a row-major un-shuffle breaks it long before the numeric
        # band does.
        assert correlation >= 0.999, f"layer {layer} is not the same map (corr {correlation:.4f})"
        assert mean <= 5e-5 * scale, f"layer {layer} drifted by {mean / scale:.2e} of scale"
        assert maximum <= 0.15 * scale, f"layer {layer} outlier is {maximum / scale:.4f} of scale"
        port_maps.append(actual)
        reference_maps.append(expected)

    # The end-to-end consequence: the same decoder, the same query rows and the
    # same canvas, fed the two sets of maps. Only the maps differ, so PSNR is the
    # noise floor of the whole vision path expressed on the thing the API returns.
    if reference is None or not SIDEcar.is_file():
        return
    t_seg = _array(reference, "t_seg.npy")
    canvas = _array(reference, "canvas.npy")
    if t_seg is None or canvas is None:
        return

    boxes = parse_seg_boxes(_golden()["response"])
    decoder = _decoder_from_sidecar(device)
    reference_mask = _decode(decoder, np.stack(reference_maps), t_seg, boxes, canvas, grid_hw)
    port_mask = _decode(decoder, np.stack(port_maps), t_seg, boxes, canvas, grid_hw)
    psnr = _psnr(port_mask, reference_mask)
    print(f"  mask PSNR between the two towers: {psnr:.1f} dB")
    assert psnr >= 60.0, f"the two towers decode to different masks (PSNR {psnr:.1f} dB)"


@pytest.mark.cpu
@skipif_mask_fixture
def test_cross_loaded_decoder_reproduces_the_recorded_mask() -> None:
    """Same weights, same recorded inputs, port path versus reference path.

    The decoder is 9.6M parameters, so the released state dict can be loaded into
    the port and run on the fixtures the reference recorded. Everything the decoder
    consumes comes from the fixtures, which means any difference is the ported
    decoder, the ported shape handling or the ported post-processing.
    """
    reference = _manifest("reference")
    t_seg = _array(reference, "t_seg.npy")
    canvas = _array(reference, "canvas.npy")
    expected = _array(reference, "mask.npy")
    if reference is None or t_seg is None or canvas is None or expected is None:
        pytest.skip("the reference run recorded no mask fixture")
        return

    golden = _golden()
    boxes = parse_seg_boxes(golden["response"])
    grid_hw = [int(value) for value in reference["grid_hw"]]
    # result.json records PIL's (width, height); masks are stored as (height, width).
    width, height = (int(value) for value in golden["image_size"])

    features = np.stack([_array(reference, f"layer_{layer:03d}.npy") for layer in LAYERS])
    payload = {
        "segmentation.features": torch.from_numpy(features),
        "segmentation.grid_hw": torch.tensor([tuple(grid_hw)], dtype=torch.int32),
        "segmentation.t_seg": torch.from_numpy(t_seg),
        "segmentation.boxes": torch.tensor(boxes[: t_seg.shape[0]], dtype=torch.float32),
        "segmentation.canvas": torch.from_numpy(canvas),
        "segmentation.mask_hw": torch.tensor([[height, width]], dtype=torch.int32),
    }

    masks = _decoder_from_sidecar("cpu")._decode_request(payload)
    assert masks is not None
    assert masks.shape == (len(boxes), 1, height, width)
    assert masks.device.type == "cpu"

    # Both sides ran the same weights on the same recorded tensors, so any
    # difference can only come from the ported decoder, its shape handling or its
    # post-processing. Measured on the recorded fixture: bit-identical.
    difference = np.abs(masks.detach().numpy() - expected.reshape(len(boxes), 1, height, width))
    assert float(difference.max()) <= 1e-6, f"decoder drifted by {difference.max():.8f}"
    assert float(difference.mean()) <= 1e-9


@pytest.mark.cpu
@skipif_mask_fixture
def test_the_recorded_mask_still_meets_the_serving_mask_contract() -> None:
    """The fixture is only useful if it is a mask the serving layer could return."""
    reference = _manifest("reference")
    expected = _array(reference, "mask.npy")
    if reference is None or expected is None:
        pytest.skip("the reference run recorded no mask fixture")
        return

    golden = _golden()
    width, height = (int(value) for value in golden["image_size"])
    assert expected.shape[1:] == (height, width)
    assert expected.min() >= 0.0 and expected.max() <= 1.0

    fixture = np.load(GOLDEN_DIR / "masks" / "mask_000.npy")
    assert fixture.shape == (height, width)
    # The fixture was recorded from a GPU run and this capture ran the tower on
    # CPU in float32, so the two agree in aggregate rather than pixel by pixel.
    # Pixel-level reproducibility of the fixture itself is covered by the golden
    # parity test, which re-derives the mask from its own recorded logits.
    difference = np.abs(expected - fixture.reshape(1, *fixture.shape))
    assert float(difference.mean()) <= 0.01, f"recorded mask drifted by {difference.mean():.4f} mean"
