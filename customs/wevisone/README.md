# WeVisOne-VL in vllm-omni

All WeVisOne-VL example files live in `customs/wevisone/`; the decision/findings record is
[customs/docs/wiki/WeVisOne-VL.md](../docs/wiki/WeVisOne-VL.md).

Reference port of `Tencent/WeChat Vision WeVisOne-VL` as a two-stage pipeline:

```
image + instruction -> Qwen3-VL thinker (text answer, captured ViT features, last-layer <OBJ> rows)
                    -> pixel decoder -> masks / alpha mattes
```

Code lives in `vllm_omni/model_executor/models/wevisone/`. The reference package is never
imported by the serving path: only `customs/wevisone/ref_golden.py` imports it, to record the
golden fixtures the port is validated against.

## Serving

```bash
customs/wevisone/serve.sh
```

Stage 0 (the Qwen3-VL thinker) runs on GPU 0 and the pixel decoder on GPU 1. On one GPU, run
the text-only pipeline instead — `--pipeline wevisone_vl_text_only` — or edit the deploy yaml
so both stages use the same device; masks always need the decoder stage.

The endpoint is the standard OpenAI-compatible `POST /v1/chat/completions`. The task is
carried in the message text, because `extra_body` knobs are merged on the diffusion path only
and `prompt_transform_func` runs after the chat template. Build the instruction with
`build_prompt(task, query)` from `vllm_omni/model_executor/models/wevisone/constants.py`;
`--prompt` sends a one-off instruction instead.

## Response shape

A mask request returns two choices: the text choice keeps the raw structured answer
(`<box>` spans survive because the stage must not strip special tokens), and the image choice
carries one `image_url` part per mask with a `mask_index`. **Mask `i` is the answer to box
`i`**, so a client pairs them by enumerating both.

```json
{
  "choices": [
    {"message": {"content": "<ref>the person</ref><box><9><65><753><1000><OBJ></box>"}},
    {"message": {"content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}, "mask_index": 0}]}}
  ]
}
```

Text-only tasks answer with boxes and produce no image parts, and they run without the
`mask_decoder.pt` sidecar being loaded at all.

## Examples

`samples.json` is the shared input set: the golden oracle and the request harnesses read the
same entries, so fixtures and served requests exercise identical prompts. `image` is
`extra_repos/WeVisOne-VL/assets/demo/seabreeze.jpg` unless an entry overrides it.

```bash
customs/wevisone/serve.sh                      # shell 1
customs/wevisone/run_samples.sh                # shell 2, runs every task and compares to golden
python customs/wevisone/visualize.py --result customs/wevisone/output/matting
```

Per task, against a running server:

| Task | Example | Masks |
|---|---|---|
| `detection` | `python customs/wevisone/openai_request_example.py --task detection --query "person\nchair"` | no |
| `referring` | `... --task referring --query "the largest object in the foreground"` | no |
| `point` | `... --task point --query "person"` | no |
| `count_box` | `... --task count_box --query "people"` | no |
| `count_point` | `... --task count_point --query "people"` | no |
| `segmentation` | `... --task segmentation --query "the person"` | yes |
| `matting` | `... --task matting --query "the person"` | yes (alpha) |
| `ocr` | `... --task ocr` | no |
| `ocr_grounding` | `... --task ocr_grounding --query "<text present in the image>"` | no |
| `visual_prompt` | `... --task visual_prompt --reference-box 120 90 460 520 --image-size 1400 933` | no |
| `freeform` | `... --task freeform --query "What is the main subject of this image?"` | no |

`--out-dir` writes `result.json` plus `masks/mask_NNN.png` in the golden shape; `--print-payload`
and `--payload-out` emit the request JSON for curl. Reference boxes are pixel-space
`[x1, y1, x2, y2]` and only `visual_prompt` reads them, which needs the image size to encode
them.

## Golden fixtures

```bash
customs/qwen_drive/docker_run.sh "python customs/wevisone/ref_golden.py --decode-masks"
```

Fixtures land under `customs/wevisone/golden/<task>/` (`result.json` + `masks/mask_NNN.npy`),
with the inputs taken from `samples.json`. `generate_samples.py --golden` compares a served
answer against a fixture: response text, parsed boxes (tolerance 0.001), and each decoded mask
against its reference array (mean tolerance 0.02). Only the `matting` fixture has been recorded
so far; the others regenerate on demand.

The seed, image, and decoding are fixed, so response text and boxes reproduce across runs. Mask
tensors do not: four reference runs on this host differed by up to 5.1e-4, which is why comparisons
use tolerances rather than exact equality.

## Files

| File | Purpose |
|---|---|
| `serve.sh` | docker wrapper that serves the two-stage pipeline on port 8091 |
| `samples.json` | shared per-task inputs (single source of truth) |
| `openai_request_example.py` | one request, prints and optionally saves the result |
| `generate_samples.py` | runs `samples.json` against the server and compares with golden |
| `run_samples.sh` | docker wrapper for `generate_samples.py` |
| `visualize.py` | tints each mask and outlines its box on the source image |
| `ref_golden.py` / `run_golden.sh` | reference oracle that writes the golden fixtures |
