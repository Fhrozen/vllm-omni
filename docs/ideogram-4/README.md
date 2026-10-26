# Ideogram 4 in vllm-omni

Native port of `ideogram-ai/ideogram-4-fp8` as a single diffusion stage:

```
caption -> Qwen3-VL language model (13 tapped layers) -> 34-layer FP8 DiT -> Flux2-style VAE -> image
```

Code lives in `vllm_omni/diffusion/models/ideogram4/`. The reference package is never
imported: `Ideogram4Pipeline` and `Ideogram4Transformer2DModel` are not Diffusers classes,
so the port reimplements the pipeline, the transformer, the logit-normal timestep schedule,
the latent norm constants, and the fp8 weight-only linear.

## Serving

```bash
docs/ideogram-4/serve.sh
```

The endpoint is the standard OpenAI-compatible `POST /v1/chat/completions`. Ideogram 4 is
text-to-image only, so a request carries one caption and the model knobs in `extra_args`.

**`stream` must be omitted or `false`.** The pipeline produces one image per request and
cannot answer a streaming request with incremental chunks; `stream: true` is rejected with
a 400 (`Ideogram4Pipeline` declares `supports_streaming=False` in
`vllm_omni/diffusion/model_metadata.py`) rather than returning a malformed response.

### Request

```json
{
  "model": "ideogram-ai/ideogram-4-fp8",
  "messages": [{"role": "user", "content": "a red fox sitting on a park bench"}],
  "extra_args": {
    "preset": "V4_QUALITY_48",
    "height": 1024,
    "width": 1024,
    "seed": 42
  }
}
```

### `extra_args`

| Key | Type | Default | Notes |
|---|---|---|---|
| `preset` | string | `V4_QUALITY_48` | `V4_QUALITY_48`, `V4_DEFAULT_20`, `V4_TURBO_12`; sets steps, guidance schedule, mu, std |
| `height` / `width` | int | 1024 | Rounded down to a multiple of 16 (`patch_size * ae_scale_factor`) |
| `seed` | int | — | Latent noise seed |
| `num_inference_steps` | int | preset | Step count; must match `guidance_schedule` length when both are given |
| `guidance_schedule` | list[float] | preset | Per-step guidance weight in **loop order** (index 0 = last, most polished step) |
| `mu` / `std` | float | preset | Logit-normal timestep distribution; omitted values are resolved per resolution (`mean = preset_mu + 0.5 * log(pixels / 512²)`) |
| `dit_copies` | `clone` \| `keep` | `clone` | Per-request override, honoured only when the server loaded both DiT copies |

`negative_prompt` is accepted by the API but ignored: CFG here runs against an
unconditional branch (zeroed conditioning), not against a negative caption.

### Response

`choices[0].message.content` is a list of image parts, one per generated image:

```json
{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."},
 "stage_durations": {...}, "peak_memory_mb": 1234.5}
```

Run the client:

```bash
python docs/ideogram-4/openai_request_example.py "a red fox sitting on a park bench"
python docs/ideogram-4/openai_request_example.py "a red fox" --payload-out payload.json
curl -s http://localhost:8091/v1/chat/completions -H 'Content-Type: application/json' -d @payload.json
```

## The `dit_copies` switch

`dit_copies` is a **startup** knob (there is no `--extra-args` CLI flag; `--custom-pipeline-args`
is the existing JSON passthrough):

```bash
vllm serve ideogram-ai/ideogram-4-fp8 --omni \
    --deploy-config docs/ideogram-4/deploy_ideogram4_fp8.yaml \
    --custom-pipeline-args '{"dit_copies": "keep"}'
```

Resolution order: `--custom-pipeline-args` > `additional_config` > `IDEOGRAM4_DIT_COPIES` > `clone`.

| Mode | What runs | Memory |
|---|---|---|
| `clone` (default) | One DiT shared by both branches; the unconditional branch sees zeroed LLM features | ~9.3 GB DiT + ~4.5 GB text encoder + ~0.3 GB VAE |
| `keep` | The separately shipped `unconditional_transformer` (different weights) | 2 × 9.3 GB, so layerwise offload is required on a 16 GB card |

`clone` is an approximation: the reference unconditional transformer genuinely has its own
weights. Golden parity must be measured in `keep` mode, and the clone-vs-keep difference must
be reported as a measured deviation, not as parity.

Per-request `dit_copies` only takes effect in `keep` mode — in `clone` mode the second copy
does not exist, and the pipeline logs a warning instead of silently switching.

## Offline example

```bash
python docs/ideogram-4/e2e_offline.py "a red fox sitting on a park bench" --output /tmp/fox.png
python docs/ideogram-4/e2e_offline.py "a red fox" --preset V4_TURBO_12 --dit-copies keep --ulysses-degree 2
```

## Files

| Path | Contents |
|---|---|
| `vllm_omni/diffusion/models/ideogram4/pipeline_ideogram4.py` | Pipeline, tokenizer/chat template, sequence layout, CFG loop, decode, post-process factory |
| `vllm_omni/diffusion/models/ideogram4/ideogram4_transformer.py` | DiT blocks, MRoPE, segment mask, sequence-parallel plan |
| `vllm_omni/diffusion/models/ideogram4/text_encoder.py` | Qwen3-VL language model only, 13 activation taps |
| `vllm_omni/diffusion/models/ideogram4/fp8_linear.py` | Weight-only fp8 linear (dequant matmul, no capability gate) |
| `vllm_omni/diffusion/models/ideogram4/scheduler.py` | Logit-normal schedule, step intervals, presets |
| `vllm_omni/diffusion/models/ideogram4/latent_norm.py` | The 128-value shift/scale constants |
| `vllm_omni/diffusion/models/ideogram4/constants.py` | Indicators, position offset, activation layers, patch sizes |
| `vllm_omni/model_extras/ideogram4.py` | Declared `extra_body` params |
| `docs/ideogram-4/deploy_ideogram4_fp8.yaml` | Single-stage deploy config |
| `tests/diffusion/models/ideogram4/test_ideogram4.py` | L1 unit tests |
| `tests/e2e/online_serving/test_ideogram4.py` | L2/L4 serving tests |
| `tests/dfx/reliability/invalid_param_test/test_invalid_diffusion_chat.py` | HTTP 400 contract test |
| `customs/docs/wiki/Ideogram-4.md` | Wiki page (decisions, findings, verification) |

The caption verifier is **not ported**: the reference `caption_verifier.py` is not part of the
snapshot, so no checks are claimed. Magic-prompt expansion and safety screening are out of scope.

## Testing

```bash
# L1: reference parity, config mapping, sequence layout, mask, SP plan, extra_args
pytest tests/diffusion/models/ideogram4/test_ideogram4.py -v -m "core_model and cpu"

# L2: chat-completions contract (JSON-only, non-streaming, image part, 400 on stream)
pytest tests/e2e/online_serving/test_ideogram4.py -v -m "core_model and diffusion"

# L4 weekly: gated model, needs VLLM_CI_HF_TOKEN
pytest tests/e2e/online_serving/test_ideogram4_expansion.py -v -m "slow and diffusion and H100 and cards_1" --run-level "full_model"

# HTTP 400 contract (weekly)
pytest tests/dfx/reliability/invalid_param_test/test_invalid_diffusion_chat.py -v -m "slow and H100 and cards_1"
```

L1 is auto-collected by `pytest tests/diffusion/models -m 'core_model and cpu'`. Ideogram 4 is
a gated checkpoint, so the serving tests are local-only (`pytest.mark.local_model`) and the
weekly Buildkite steps run the L4 expansion cases plus the invalid-param directory; see
`.buildkite/cuda/test-weekly.yml` and the `ideogram4` anchor in
`.buildkite/common/ci_source_file_dependencies.yml`.

See `customs/docs/wiki/Testing.md` for the golden-parity procedure and the memory numbers to
record.
