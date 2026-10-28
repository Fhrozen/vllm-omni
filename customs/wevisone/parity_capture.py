#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Record hook-level feature dumps for WeVisOne-VL hook parity.

Runs the same fixed input (the golden matting sample: fixed image, greedy
decoding, seed 1234) through both sides and writes the tensors the pixel decoder
consumes, so ``tests/model_executor/models/wevisone/test_wevisone_hook_parity.py``
can compare them offline.

    python customs/wevisone/parity_capture.py --side reference --device cuda
    python customs/wevisone/parity_capture.py --side port --device cuda

Keep the two devices in separate output directories so the CPU logic dual-run and
the GPU tower dual-run can both be compared later:

    python customs/wevisone/parity_capture.py --side reference --device cpu --out customs/wevisone/golden/hooks/cpu
    python customs/wevisone/parity_capture.py --side port     --device cpu --out customs/wevisone/golden/hooks/cpu
    python customs/wevisone/parity_capture.py --side reference --device cuda --out customs/wevisone/golden/hooks/gpu
    python customs/wevisone/parity_capture.py --side port      --device cuda --out customs/wevisone/golden/hooks/gpu

Use the same ``--dtype`` on both sides. The checkpoint stores fp32 weights, so a
12 GiB card needs ``--dtype bfloat16``; ``float32`` only fits on CPU.

``--tf32 off`` disables TF32 for conv and matmul. The serving path leaves TF32 on,
which is why a bf16 GPU dual-run diverges by more than rounding: the fp32 Conv3d in
patch_embed runs in TF32, and the error is amplified by depth. Capturing with
``--dtype float32 --tf32 off`` keeps the noise floor near 1e-5 relative, so the
fixture comparison measures logic rather than kernels.

Both sides must be separate processes: the 4B checkpoint is 9.2 GiB in bf16, so
the reference model and the port do not fit on one 12 GiB card at the same time.

What each side exercises
------------------------
``--side reference``  the released package: HF model + its capture classes.
``--side port``       the port: vllm's vision tower + the ported capture classes.

The port's language tower cannot run outside the engine (vllm's attention layers
need an attention backend, and there is none for CPU), so ``--side port --device
cpu`` runs the ported capture logic against the reference tower instead. That is
a logic dual-run: it proves layer selection, the singleton sequence dimension and
the un-shuffle, but it cannot measure the difference between vllm's tower and HF's
tower. Only ``--side port --device cuda`` does that.

Outputs under ``--out`` (default ``customs/wevisone/golden/hooks``):
  * ``manifest.json``            run metadata and per-file entries
  * ``<side>/layer_NNN.npy``     one un-shuffled ``[C, H, W]`` float32 map per layer
  * ``<side>/t_seg.npy``         query rows fed back to the decoder, when available
  * ``<side>/canvas.npy``        the decoder canvas, ``[3, cnn_size, cnn_size]``
  * ``<side>/mask.npy``          the decoded mask, when the sidecar and a recorded
                                 response are both available
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = REPO_ROOT / "extra_repos" / "WeVisOne-VL"
FIXED_IMAGE = REFERENCE_ROOT / "assets" / "demo" / "seabreeze.jpg"
SEED = 1234
LAYERS = [5, 11, 17, 23]
MERGE_SIZE = 2
QUERY = "the person"


def _messages(prompt: str, image: Image.Image, response: str | None) -> list[dict[str, Any]]:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    if response is not None:
        # Teacher-force the recorded answer so the <OBJ> positions exist: the
        # decoder needs the query rows the model produced for them.
        messages.append({"role": "assistant", "content": response})
    return messages


def _inputs(
    processor: Any, image: Image.Image, prompt: str, device: str, response: str | None
) -> dict[str, torch.Tensor]:
    """Mirror the reference engine's input builder so both sides see the same tokens."""
    text = processor.apply_chat_template(
        _messages(prompt, image, response),
        tokenize=False,
        add_generation_prompt=response is None,
    )
    encoded = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    return {key: value.to(device) for key, value in encoded.items()}


def _save(out_dir: Path, side: str, name: str, tensor: torch.Tensor) -> dict[str, Any]:
    array = tensor.detach().to("cpu", dtype=torch.float32).contiguous().numpy()
    (out_dir / side).mkdir(parents=True, exist_ok=True)
    np.save(out_dir / side / name, array)
    return {
        "file": f"{side}/{name}",
        "shape": list(array.shape),
        "dtype": "float32",
        "checksum": float(np.abs(array).sum()),
    }


def _tf32_state() -> dict[str, bool]:
    return {
        "cudnn": bool(torch.backends.cudnn.allow_tf32),
        "matmul": bool(torch.backends.cuda.matmul.allow_tf32),
    }


def _visual_of(model: Any) -> Any:
    """The vision tower, whichever way the model exposes it."""
    return getattr(getattr(model, "model", model), "visual", None) or getattr(model, "visual", None)


def run_reference(
    model: Any,
    processor: Any,
    image: Image.Image,
    prompt: str,
    response: str | None,
    device: str,
    out_dir: Path,
    model_dir: Path,
) -> dict[str, Any]:
    from wevisone_vl.feature_bridge import (
        LastHiddenCapture,
        VisualFeatureCapture,
        extract_t_seg,
        premerge_grid_hw,
        premerge_tokens_to_2d,
    )

    batch = _inputs(processor, image, prompt, device, response)
    model.config.use_cache = False
    model.eval()
    with VisualFeatureCapture(model, feature_layers=LAYERS) as capture:
        with LastHiddenCapture(model) as hidden_capture:
            with torch.no_grad():
                model(**batch, output_hidden_states=False, return_dict=True, use_cache=False)
    hidden = hidden_capture.hidden
    if hidden is None:
        raise SystemExit("the reference model produced no last-layer hidden state")

    grid_hw = premerge_grid_hw(batch["image_grid_thw"])[0]
    entries = []
    maps = []
    for layer in LAYERS:
        # The reference decoder wants one [1, C, H, W] map per level; the saved
        # fixture keeps the single-image map without the batch dimension.
        feature_map = premerge_tokens_to_2d(capture.premerge_features[layer], grid_hw, MERGE_SIZE)
        entries.append(_save(out_dir, "reference", f"layer_{layer:03d}.npy", feature_map[0]))
        maps.append(feature_map)
    seg_token_id = int(processor.tokenizer.convert_tokens_to_ids("<OBJ>"))
    t_seg = extract_t_seg(hidden, batch["input_ids"], seg_token_id)[0]
    entries.append(_save(out_dir, "reference", "t_seg.npy", t_seg))
    cnn_size = _cnn_size(model_dir)
    canvas = _canvas(image, device, cnn_size)
    entries.append(_save(out_dir, "reference", "canvas.npy", canvas))

    boxes = None
    if response:
        from wevisone_vl.config import Qwen3VLSegConfig
        from wevisone_vl.engine import parse_seg_boxes

        boxes = parse_seg_boxes(response)
        sidecar = model_dir / "mask_decoder.pt"
        if boxes and sidecar.is_file():
            # The released decoder run on the same captured inputs. Same weights,
            # same tensors, so any difference against the port is the port itself.
            from wevisone_vl.decoder import build_mask_decoder_from_qwen

            checkpoint = torch.load(sidecar, map_location="cpu", weights_only=True)
            cfg = Qwen3VLSegConfig.from_dict(dict(checkpoint.get("cfg") or {}))
            decoder = build_mask_decoder_from_qwen(model, cfg)
            decoder.load_state_dict(checkpoint["state_dict"], strict=True)
            decoder.to(device=device, dtype=torch.float32).eval()
            count = min(len(boxes), int(t_seg.shape[0]))
            with torch.no_grad():
                logits = decoder(
                    image=canvas,
                    visual_features_2d=[feature.to(device, dtype=torch.float32) for feature in maps],
                    t_seg=t_seg[:count].to(device, dtype=torch.float32),
                    boxes01=torch.tensor(boxes[:count], device=device, dtype=torch.float32),
                    grid_hw=grid_hw,
                )["mask_logits"]
            probabilities = torch.nn.functional.interpolate(
                logits.float().sigmoid()[None],
                size=(image.height, image.width),
                mode="bilinear",
                align_corners=False,
            )[0]
            entries.append(_save(out_dir, "reference", "mask.npy", probabilities))

    return {
        "side": "reference",
        "device": device,
        "runtime": "reference package (HF model)",
        "tf32": _tf32_state(),
        "layers": LAYERS,
        "grid_hw": list(grid_hw),
        "hook_input_shape": list(capture.premerge_features[5].shape),
        "response": response,
        "boxes": boxes,
        "image_size": [image.width, image.height],
        "entries": entries,
    }


def run_port(
    model: Any,
    processor: Any,
    image: Image.Image,
    prompt: str,
    response: str | None,
    device: str,
    out_dir: Path,
    model_dir: Path,
) -> dict[str, Any]:
    from vllm_omni.model_executor.models.wevisone.feature_bridge import (
        VisualFeatureCapture as PortVisualFeatureCapture,
    )
    from vllm_omni.model_executor.models.wevisone.feature_bridge import (
        premerge_grid_hw,
    )
    from vllm_omni.model_executor.models.wevisone.feature_bridge import (
        premerge_tokens_to_2d as port_premerge_tokens_to_2d,
    )

    batch = _inputs(processor, image, prompt, device, response)
    capture = PortVisualFeatureCapture(model, feature_layers=LAYERS)
    with capture:
        with torch.no_grad():
            # Positional so the same call works for HF's (hidden_states, grid_thw)
            # and vllm's (pixel_values, grid_thw) vision forward.
            _visual_of(model)(batch["pixel_values"], batch["image_grid_thw"])

    grid_hw = premerge_grid_hw(batch["image_grid_thw"])[0]
    entries = []
    for layer in LAYERS:
        # The port has no batch dimension: a captured map belongs to one image.
        entries.append(
            _save(
                out_dir,
                "port",
                f"layer_{layer:03d}.npy",
                port_premerge_tokens_to_2d(capture.premerge_features[layer], grid_hw, MERGE_SIZE),
            )
        )
    entries.append(_save(out_dir, "port", "canvas.npy", _canvas(image, device, _cnn_size(model_dir))))
    return {
        "side": "port",
        "device": device,
        "runtime": "vllm vision tower" if device != "cpu" else "reference tower, ported capture logic",
        "tf32": _tf32_state(),
        "layers": LAYERS,
        "grid_hw": list(grid_hw),
        "hook_input_shape": list(capture.premerge_features[5].shape),
        "response": response,
        "image_size": [image.width, image.height],
        "entries": entries,
    }


def _cnn_size(model_dir: Path) -> int:
    """Canvas side as the released checkpoint declares it."""
    sidecar = model_dir / "mask_decoder.pt"
    if sidecar.is_file():
        checkpoint = torch.load(sidecar, map_location="cpu", weights_only=True)
        return int((checkpoint.get("cfg") or {}).get("cnn_image_size", 1024))
    return 1024


def _canvas(image: Image.Image, device: str, cnn_size: int) -> torch.Tensor:
    from vllm_omni.model_executor.models.wevisone.feature_bridge import assert_cnn_rgb

    canvas = torch.from_numpy(
        np.asarray(image.convert("RGB").resize((cnn_size, cnn_size), Image.BICUBIC), dtype=np.float32) / 255.0
    )
    return assert_cnn_rgb(canvas.permute(2, 0, 1).unsqueeze(0).to(device), cnn_image_size=cnn_size)


def main() -> int:
    parser = argparse.ArgumentParser(description="Write WeVisOne-VL hook parity dumps.")
    parser.add_argument("--side", choices=("reference", "port"), required=True)
    parser.add_argument("--model", default=str(REPO_ROOT / "extra_repos" / "WeVisOne-VL-4B"))
    parser.add_argument("--image", default=str(FIXED_IMAGE))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="Same precision on both sides; float32 only fits on CPU.",
    )
    parser.add_argument("--out", default=str(REPO_ROOT / "customs" / "wevisone" / "golden" / "hooks"))
    parser.add_argument("--tf32", choices=("on", "off"), default="on")
    args = parser.parse_args()

    out_dir = Path(args.out)
    dtype = getattr(torch, args.dtype)
    if args.tf32 == "off":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    image = Image.open(args.image).convert("RGB")
    prompt = f"Locate the instances that match the following description and provide the alpha mattes: {QUERY}"
    golden = REPO_ROOT / "customs" / "wevisone" / "golden" / "matting"
    response = None
    if (golden / "result.json").is_file():
        recorded = json.loads((golden / "result.json").read_text(encoding="utf-8"))
        response = recorded.get("response")

    if args.side == "reference":
        if not (REFERENCE_ROOT / "wevisone_vl").is_dir():
            raise SystemExit(f"reference package not found: {REFERENCE_ROOT}")
        sys.path.insert(0, str(REFERENCE_ROOT))
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(args.model)
        model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=dtype).to(args.device)
        manifest = run_reference(model, processor, image, prompt, response, args.device, out_dir, Path(args.model))
    else:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(args.model)
        if args.device == "cpu":
            # The ported capture logic is exercised against the reference tower;
            # the port's own language tower cannot run outside the engine.
            if not (REFERENCE_ROOT / "wevisone_vl").is_dir():
                raise SystemExit(f"reference package not found: {REFERENCE_ROOT}")
            sys.path.insert(0, str(REFERENCE_ROOT))
            from transformers import AutoModelForImageTextToText

            model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=dtype)
            manifest = run_port(model, processor, image, prompt, response, "cpu", out_dir, Path(args.model))
        else:
            from safetensors import safe_open
            from vllm.config import DeviceConfig, ModelConfig, VllmConfig, set_current_vllm_config
            from vllm.distributed.parallel_state import (
                ensure_model_parallel_initialized,
                init_distributed_environment,
            )

            from vllm_omni.model_executor.models.wevisone.wevisone_vl import (
                WeVisOneVLForConditionalGeneration,
            )

            torch.set_default_dtype(dtype)
            vllm_config = VllmConfig(
                model_config=ModelConfig(
                    model=args.model,
                    limit_mm_per_prompt={"image": 4},
                    seed=SEED,
                    dtype=args.dtype,
                ),
                device_config=DeviceConfig(args.device),
            )

            def weights():
                for shard in sorted(Path(args.model).glob("*.safetensors")):
                    with safe_open(shard, framework="pt") as handle:
                        for name in handle.keys():
                            yield name, handle.get_tensor(name)

            # The vision tower asks the ambient parallel state for its TP size, and
            # both the parallel bootstrap and CustomOp layers read the ambient config,
            # so the whole run has to happen inside the same context.
            # Layers are built on the ambient default device. Building them straight
            # on the card leaves the process at 11.42 GiB before any activation is
            # accounted for, which is past a 12 GiB GPU; constructing on CPU and
            # moving the weights over lands at 9.11 GiB instead.
            with set_current_vllm_config(vllm_config):
                init_distributed_environment(1, 0, "tcp://127.0.0.1:29511", 0)
                ensure_model_parallel_initialized(1, 1)
                model = WeVisOneVLForConditionalGeneration(vllm_config=vllm_config)
                model.load_weights(weights())
                model.to(args.device)
                manifest = run_port(model, processor, image, prompt, response, args.device, out_dir, Path(args.model))

    manifest["image"] = str(args.image)
    manifest["model"] = str(args.model)
    manifest["prompt"] = prompt
    manifest["seed"] = SEED
    manifest_path = out_dir / args.side / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
