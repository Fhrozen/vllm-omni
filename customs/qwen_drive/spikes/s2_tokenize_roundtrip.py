"""Spike S2: does tokenizing the joined prompt text reproduce the reference token ids (so a chat template can build it)?"""

from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("/workspace/extra_repos/Qwen-Drive-1.0-4B-hf")
golden = Path("/workspace/customs/docs/golden")
for i in range(3):
    for mode in ("direct", "reasoning"):
        ids = np.load(golden / f"prompt_ids_{i}_{mode}.npy")[0].tolist()
        text = tok.decode(ids, skip_special_tokens=False)
        again = tok.encode(text, add_special_tokens=False)
        print(i, mode, "roundtrip identical:", again == ids, len(ids), len(again))
        if i == 0 and mode == "direct":
            squeezed = text.replace("<|image_pad|>", "")
            print(repr(squeezed[:300]))
            print(repr(squeezed[-420:]))
