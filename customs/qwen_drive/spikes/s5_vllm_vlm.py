"""Spike S5: serve the VLM part of the exported Qwen-Drive repo with plain vLLM (no vllm_omni).

Questions answered:
 - can a callable hf_overrides replace the nested config (QwenDriveConfig -> vlm_config)?
 - does a Qwen3_5ForConditionalGeneration subclass that strips `vlm.` load the weights and keep the mm processor?
 - does greedy VQA match the E2 golden text?
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
sys.path.insert(0, str(Path(__file__).parent))

import torch  # noqa: E402
from torchvision.transforms import InterpolationMode  # noqa: E402
from torchvision.transforms import functional as TF  # noqa: E402

MODEL = os.environ.get("QD_MODEL", "/workspace/extra_repos/Qwen-Drive-1.0-4B-hf")
GOLDEN = Path("/workspace/customs/docs/golden")


def unwrap_vlm(config):
    # vLLM first calls the override on a dummy config just to read model_type.
    vlm = getattr(config, "vlm_config", None)
    if vlm is None:
        return config
    from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5Config

    # vLLM's multimodal processor type-checks against its own Qwen3_5Config, not transformers'.
    out = Qwen3_5Config(**vlm.to_dict())
    out.architectures = ["QwenDriveVLMProbe"]
    return out


def resized_image(proc, frame, budget):
    from qwen_drive.scene import smart_resize

    image = frame.load()
    max_pixels = budget
    if frame.target_size is not None:
        width, height = frame.target_size
        image = TF.resize(image, [height, width], interpolation=InterpolationMode.BICUBIC)
        max_pixels = proc.grid_pixel_limit
    width, height = image.size
    gh, gw = smart_resize(height, width, proc.factor, proc.min_pixels, max_pixels)
    return TF.resize(image, [gh, gw], interpolation=InterpolationMode.BICUBIC)


def main() -> None:
    from qwen_drive.benchmarks import read_scene_file
    from qwen_drive.configuration_qwen_drive import QwenDriveConfig
    from qwen_drive.scene import QwenDriveProcessor
    from transformers import AutoTokenizer
    from vllm import LLM, ModelRegistry, SamplingParams

    ModelRegistry.register_model("QwenDriveVLMProbe", "qd_probe:QwenDriveVLMProbe")

    config = QwenDriveConfig.from_pretrained(MODEL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    proc = QwenDriveProcessor(tokenizer, config)
    scenes = list(
        read_scene_file(
            "/workspace/extra_repos/Qwen-Drive-1.0/data/demo/planning_scenes.jsonl",
            image_root="/workspace/extra_repos/Qwen-Drive-1.0/data/demo",
            image_archive=None,
            num_history_points=16,
            limit=2,
        )
    )

    llm = LLM(
        model=MODEL,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=float(os.environ.get("QD_GPU_UTIL", "0.85")),
        limit_mm_per_prompt={"image": 12},
        enforce_eager=True,
        hf_overrides=unwrap_vlm,
    )
    print("prefix caching:", llm.llm_engine.vllm_config.cache_config.enable_prefix_caching, flush=True)

    for i, sample in enumerate(scenes):
        frames = sample.scene.frames_in_order()
        images = [resized_image(proc, f, config.current_image_pixels) for f in frames]
        ids = proc.encode_vqa(frames, "Describe the traffic scene and the safest action.")["input_ids"][0].tolist()
        out = (
            llm.generate(
                {"prompt_token_ids": ids, "multi_modal_data": {"image": images}},
                SamplingParams(temperature=0.0, max_tokens=128),
            )[0]
            .outputs[0]
            .text.strip()
        )
        want = (GOLDEN / f"vqa_{i}.txt").read_text().strip()
        common = os.path.commonprefix([out, want])
        print(f"scene {i}: identical={out == want} common_prefix_chars={len(common)}/{len(want)}")
        print("  vllm  :", out[:160].replace("\n", " "))
        print("  golden:", want[:160].replace("\n", " "))


if __name__ == "__main__":
    main()
