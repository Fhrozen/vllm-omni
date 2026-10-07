"""E3: build a self-contained HF repo (remote code, no `qwen_drive` package, no perception head).

Pure stdlib; run on the host: python3 customs/qwen_drive/export_hf.py [--copy]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE_FILES = (
    "configuration_qwen_drive.py",
    "modeling_qwen_drive.py",
    "planning_expert.py",
    "scene.py",
    "trajectory.py",
)
CHECKPOINT_FILES = (
    ".gitattributes",
    "chat_template.jinja",
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
PLANNERS = ("planner-sft", "planner-rl")

# transformers resolves the video processor from the dir's model_type (`qwen_drive`, unknown), so name it explicitly.
VIDEO_PREPROCESSOR = {
    "video_processor_type": "Qwen3VLVideoProcessor",
    "patch_size": 16,
    "temporal_patch_size": 2,
    "merge_size": 2,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "size": {"longest_edge": 25165824, "shortest_edge": 4096},
}

RESOLVE_HELPER = '''

def _resolve_planner(planner: str | Path, base: str | Path | None) -> Path:
    """Planner directory: an existing path, a subfolder of the local model dir, or a subfolder of the hub repo."""
    planner = Path(planner)
    if planner.is_dir():
        return planner
    if base is not None and (Path(base) / planner).is_dir():
        return Path(base) / planner
    if base is None:
        raise FileNotFoundError(f"planner {str(planner)!r} not found")
    from huggingface_hub import hf_hub_download

    weights = hf_hub_download(str(base), f"{planner.as_posix()}/model.safetensors")
    return Path(weights).parent

'''

# (old, new) pairs applied to modeling_qwen_drive.py; every pair must match exactly once.
MODELING_PATCHES = (
    (
        "\n\nclass InferenceMode(str, Enum):",
        RESOLVE_HELPER + "\nclass InferenceMode(str, Enum):",
    ),
    (
        "        if planner is not None:\n            model.load_planner(planner)\n",
        "        if planner is not None:\n"
        "            base = args[0] if args else kwargs.get(\"pretrained_model_name_or_path\")\n"
        "            model.load_planner(_resolve_planner(planner, base))\n",
    ),
)

README_NOTE = """> [!Note]
> **Self-contained export.** This directory ships the modeling code (`configuration_qwen_drive.py`,
> `modeling_qwen_drive.py`, `planning_expert.py`, `scene.py`, `trajectory.py`) so the original
> `qwen_drive` package is not required. Load it with remote code and pick a planner by sub-folder:
>
> ```python
> from transformers import AutoModel
> model = AutoModel.from_pretrained(path, trust_remote_code=True, planner="planner-rl", dtype="bfloat16")
> ```
>
> The 3D perception head (`perception/`) is not included.

"""


def place(src: Path, dst: Path, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copyfile(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=ROOT / "extra_repos/Qwen-Drive-1.0-4B")
    p.add_argument("--code", type=Path, default=ROOT / "extra_repos/Qwen-Drive-1.0/src/qwen_drive")
    p.add_argument("--out", type=Path, default=ROOT / "extra_repos/Qwen-Drive-1.0-4B-hf")
    p.add_argument("--copy", action="store_true", help="copy weights instead of hard-linking them")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    for name in CODE_FILES:
        text = (args.code / name).read_text()
        if name == "modeling_qwen_drive.py":
            for old, new in MODELING_PATCHES:
                if text.count(old) != 1:
                    raise SystemExit(f"patch does not apply exactly once in {name}: {old[:60]!r}")
                text = text.replace(old, new)
        (args.out / name).write_text(text)

    for name in CHECKPOINT_FILES:
        if (args.ckpt / name).exists():
            place(args.ckpt / name, args.out / name, copy=True)
    place(args.ckpt / "model.safetensors", args.out / "model.safetensors", args.copy)
    for planner in PLANNERS:
        place(args.ckpt / planner / "model.safetensors", args.out / planner / "model.safetensors", args.copy)
        place(args.ckpt / planner / "config.json", args.out / planner / "config.json", copy=True)
    if (args.ckpt / "assets").is_dir():
        shutil.copytree(args.ckpt / "assets", args.out / "assets", dirs_exist_ok=True)

    config = json.loads((args.ckpt / "config.json").read_text())
    config["auto_map"] = {
        "AutoConfig": "configuration_qwen_drive.QwenDriveConfig",
        "AutoModel": "modeling_qwen_drive.QwenDriveForPlanning",
        "AutoModelForImageTextToText": "modeling_qwen_drive.QwenDriveForPlanning",
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.out / "video_preprocessor_config.json").write_text(json.dumps(VIDEO_PREPROCESSOR, indent=2) + "\n")
    shutil.copyfile(Path(__file__).parent / "chat_template_drive.jinja", args.out / "chat_template_drive.jinja")
    readme = (args.ckpt / "README.md").read_text()
    marker = "# Qwen-Drive-1.0-4B\n"
    head, sep, tail = readme.partition(marker)
    if not sep:
        raise SystemExit("README title not found")
    (args.out / "README.md").write_text(head + sep + "\n" + README_NOTE + tail)
    print(f"exported to {args.out}")


if __name__ == "__main__":
    main()
