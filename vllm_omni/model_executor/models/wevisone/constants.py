# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Task prompts and response parsing for the released WeVisOne-VL checkpoint.

Ported from the Tencent WeVisOne-VL reference implementation (Apache-2.0);
the prompt family must stay byte-identical to the released checkpoint.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import regex as re

SYSTEM_PROMPT = "You are a helpful assistant."

# ``<OBJ>`` is an added (non-special) tokenizer token in the released
# checkpoint. Keep the checkpoint id as a fallback for callers that do not
# have the tokenizer vocabulary available.
SEG_TOKEN = "<OBJ>"
SEG_TOKEN_ID_FALLBACK = 152678

# Added-token ids in the released tokenizer (verified against ``tokenizer.json``):
# ``<0>..<1000>`` occupy a contiguous block starting at 151669, and ``<OBJ>`` sits
# nine ids past the top of that block. Stage input processors run in the connector
# process, where no tokenizer vocabulary is available, so boxes are reconstructed
# from token ids with these constants.
COORD_TOKEN_ID_BASE_FALLBACK = 151669
BOX_OPEN_TOKEN_ID_FALLBACK = 152672
BOX_CLOSE_TOKEN_ID_FALLBACK = 152673
# Assistant turn terminator: the text parser stops at ``<|im_end|>``. The
# tokenizer adds it at 151645 (chat template), which is not an added token id in
# the coordinate block.
EOS_TOKEN_ID_FALLBACK = 151645

BOX_RE = re.compile(r"<box>(.*?)(?:</box>|$)", re.DOTALL | re.IGNORECASE)
COORD_RE = re.compile(r"<(\d{1,4})>")
SEG_RE = re.compile(r"<OBJ>", re.IGNORECASE)

TASKS = (
    "detection",
    "referring",
    "point",
    "count_box",
    "count_point",
    "segmentation",
    "matting",
    "ocr",
    "ocr_grounding",
    "visual_prompt",
    "freeform",
)


def _labels(query: str | Iterable[str]) -> list[str]:
    if isinstance(query, str):
        return [item.strip() for item in query.splitlines() if item.strip()]
    return [str(item).strip() for item in query if str(item).strip()]


def _text(query: str | Iterable[str]) -> str:
    if isinstance(query, str):
        return query.strip()
    return " ".join(str(item).strip() for item in query if str(item).strip())


def _phrase(query: str | Iterable[str]) -> str:
    text = _text(query)
    return text if not text or text[-1] in ".?!" else text + "."


def _box_tokens(boxes: Sequence[Sequence[float]], width: int, height: int) -> str:
    encoded = []
    for box in boxes:
        if len(box) < 4 or width <= 0 or height <= 0:
            continue
        values = (
            float(box[0]) / width,
            float(box[1]) / height,
            float(box[2]) / width,
            float(box[3]) / height,
        )
        bins = [max(0, min(1000, int(value * 1000))) for value in values]
        encoded.append("".join(f"<{value}>" for value in bins))
    return ",".join(encoded)


def build_prompt(
    task: str,
    query: str | Iterable[str] = "",
    *,
    reference_boxes: Sequence[Sequence[float]] | None = None,
    image_size: tuple[int, int] = (0, 0),
) -> str:
    """Build the exact instruction family used for checkpoint inference.

    Multiple detection labels can be passed as a list of strings
    or separated by newlines. A referring expression remains a single phrase
    even when it contains commas. Reference boxes are pixel-space
    ``[x1, y1, x2, y2]`` coordinates.
    """
    task = task.strip().lower()
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; choose from {', '.join(TASKS)}")
    labels = _labels(query)
    text_query = _text(query)

    if task == "detection":
        return "Locate all the instances that match the following description: " + ". ".join(labels) + "."
    if task == "referring":
        return "Locate the instances that match the following description: " + _phrase(query)
    if task == "point":
        return "point to: " + ". ".join(labels)
    if task == "count_box":
        return f"How many {text_query or 'objects'} in the image? Please count them with boxes."
    if task == "count_point":
        return f"How many {text_query or 'objects'} in the image?"
    if task in {"segmentation", "matting"}:
        return "Locate the instances that match the following description and provide the alpha mattes: " + _phrase(
            query
        )
    if task == "ocr":
        return "detect all the word in the box format"
    if task == "ocr_grounding":
        return "Locate the following text in the image: " + text_query
    if task == "visual_prompt":
        width, height = image_size
        coords = _box_tokens(reference_boxes or [], width, height)
        visual_prompt = f"<ref>object 1</ref><box>{coords}</box>"
        return "Locate all similar objects in the image based on the reference boxes: " + visual_prompt
    return text_query


def seg_token_id(tokenizer) -> int:
    """Resolve the ``<OBJ>`` id from a tokenizer, mirroring the reference."""
    vocab = tokenizer.get_vocab()
    if SEG_TOKEN not in vocab:
        raise ValueError(f"{SEG_TOKEN} is missing from the tokenizer vocabulary")
    return int(vocab[SEG_TOKEN])


def _normalize_box(numbers: Sequence[int]) -> list[float] | None:
    """Validate the last four coordinate values and normalize them to xyxy in [0, 1]."""
    if len(numbers) < 4:
        return None
    x1, y1, x2, y2 = (int(value) for value in numbers[-4:])
    if all(0 <= value <= 1000 for value in (x1, y1, x2, y2)) and x2 > x1 and y2 > y1:
        return [x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0]
    return None


def parse_seg_boxes(text: str) -> list[list[float]]:
    """Extract normalized xyxy boxes immediately followed by ``<OBJ>``."""
    text = (text or "").split("<|im_end|>", 1)[0]
    boxes: list[list[float]] = []
    for body in BOX_RE.findall(text):
        for prefix in SEG_RE.split(body)[:-1]:
            numbers = [int(value) for value in COORD_RE.findall(prefix)]
            box = _normalize_box(numbers)
            if box is not None:
                boxes.append(box)
    return boxes


def coord_token_id_base(seg_token_id: int = SEG_TOKEN_ID_FALLBACK) -> int:
    """First coordinate-token id, derived from the ``<OBJ>`` id."""
    return int(seg_token_id) - 1009


def parse_seg_boxes_from_ids(
    token_ids: Sequence[int],
    seg_token_id: int = SEG_TOKEN_ID_FALLBACK,
    coord_base: int | None = None,
) -> list[list[float]]:
    """Reconstruct the same boxes as :func:`parse_seg_boxes` from token ids.

    Only the added ``<box>``/``</box>``/``<OBJ>`` tokens and the contiguous
    coordinate block are inspected, so no tokenizer vocabulary is needed.
    Pass only the *generated* tokens: a ``visual_prompt`` request can carry
    ``<box>`` spans in its prompt, and the reference parses boxes from the
    response alone.
    """
    seg_id = int(seg_token_id)
    base = int(coord_base if coord_base is not None else coord_token_id_base(seg_id))
    boxes: list[list[float]] = []
    numbers: list[int] = []
    in_box = False
    for token_id in token_ids:
        token_id = int(token_id)
        if token_id == EOS_TOKEN_ID_FALLBACK:
            break
        if token_id == seg_id:
            if in_box:
                box = _normalize_box(numbers)
                if box is not None:
                    boxes.append(box)
            numbers = []
            continue
        if base <= token_id <= base + 1000:
            if in_box:
                numbers.append(token_id - base)
        elif token_id == BOX_OPEN_TOKEN_ID_FALLBACK:
            in_box = True
            numbers = []
        elif token_id == BOX_CLOSE_TOKEN_ID_FALLBACK:
            in_box = False
            numbers = []
    return boxes
