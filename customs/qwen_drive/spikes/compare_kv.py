"""Compare the KV the planner stage received (QWEN_DRIVE_DEBUG_KV_DUMP) with the golden HF K/V of the same scene."""

import sys

import torch
import torch.nn.functional as F

got = torch.load(sys.argv[1])
want = torch.load(sys.argv[2])
print("layers received:", got["layers"], "meta:", got["meta"], "golden anchor:", want["anchor"][:, 0].tolist())
for i, ((gk, gv), (wk, wv)) in enumerate(zip(got["kv"], want["kv"])):
    wk, wv = wk.squeeze(0), wv.squeeze(0)  # golden [S, 4, 256]
    print(f"layer {got['layers'][i]}: got {tuple(gk.shape)} want {tuple(wk.shape)}", end=" ")
    n = min(gk.shape[0], wk.shape[0])
    ck = F.cosine_similarity(gk[:n].float().flatten(1), wk[:n].float().flatten(1), dim=-1)
    cv = F.cosine_similarity(gv[:n].float().flatten(1), wv[:n].float().flatten(1), dim=-1)
    print(f"cos K mean {ck.mean():.4f} min {ck.min():.4f} | V mean {cv.mean():.4f} min {cv.min():.4f}")
