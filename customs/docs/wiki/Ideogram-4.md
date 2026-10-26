# Ideogram 4 in vllm-omni

Goal: serve `ideogram-ai/ideogram-4-fp8` as a native vllm-omni diffusion pipeline (no import of
the reference `ideogram4` package), request it over `/v1/chat/completions` with a JSON caption,
and keep non-streaming responses only. Target hardware: 2×16 GB with fp8 weight-only DiT.

Example code and the user-facing page live in `docs/ideogram-4/` (this is the one deliberate
exception to the "documentation lives only in `customs/docs`" rule, per the user's request).

## Status

| Phase | What | State |
|---|---|---|
| A | `constants.py`, `latent_norm.py`, `scheduler.py`, `config.py`, `fp8_linear.py` | done |
| B | `ideogram4_transformer.py` (DiT, MRoPE, segment mask, SP plan) | done |
| C | `text_encoder.py` (Qwen3-VL language model, 13 taps), `pipeline_ideogram4.py` | done |
| D | Registry entries, `supports_streaming` metadata + serving guard, `dit_copies` knob | done |
| E | `model_extras/ideogram4.py`, `docs/ideogram-4/` (README, client, offline example, deploy yaml) | done |
| F | `tests/diffusion/models/ideogram4/` (L1), `tests/e2e/online_serving/test_ideogram4.py` (L2), `test_ideogram4_expansion.py` (L4), `tests/dfx/reliability/invalid_param_test/test_invalid_diffusion_chat.py`, Buildkite wiring | done |
| — | Caption verifier | **not ported** (reference source missing from the snapshot) |

All ported modules are `py_compile`-clean; nothing has been executed on GPU yet — the container
has no running instance and the host Python has no torch. Runtime verification must run in docker.

## Decisions

1. **Native port (Path B).** `Ideogram4Pipeline` / `Ideogram4Transformer2DModel` are not Diffusers
   classes; `AutoModel.from_pretrained` cannot load the text encoder either, because
   `text_encoder/config.json` carries a non-standard `ideogram_fp8_weight_only: true` flag with no
   `quantization_config`. Using `transformers` classes is fine (precedent: `krea2`,
   `internvla_a1/adapter_qwen3_vl.py`).
2. **Non-streaming only.** `stream: true` returns a 400 driven by
   `DiffusionModelMetadata(supports_streaming=False)` — a new metadata field, not a model-name
   check — so a streaming request never produces a wrong response.
3. **`dit_copies` default `clone`** (Option B): one shared DiT, unconditional branch sees zeroed
   LLM features. `keep` loads the separately shipped `unconditional_transformer` for reference
   parity. Read order: `--custom-pipeline-args` > `additional_config` > `IDEOGRAM4_DIT_COPIES` >
   default. Per-request override works only in `keep` mode.
   `--extra-args` does not exist as a CLI flag; `--custom-pipeline-args` is the existing JSON knob.
4. **FP8 without `torch._scaled_mm`.** The reference dequantizes per forward, so no compute-capability
   gate is needed (host is sm_120). Weights stay fp8 at load; pre-dequantizing would double the DiT
   to ~18.6 GB. `Fp8Linear` buffers are persistent so the layerwise offloader can move them.
5. **Attention.** The DiT mask is block-diagonal (`segment_ids`), not causal, so `Attention` is used
   with `causal=False` and an explicit mask. The text encoder encodes causality inside the mask
   (`causal=False` + combined causal∧padding 4D mask, `extra={"attention_mask_mode": "arbitrary"}`)
   because a custom mask combined with causal masking is rejected by the flashinfer backend.
   Padding-query rows keep a diagonal so softmax never sees an all-False row; the caller zeroes them.
6. **Sequence parallelism.** `_sp_plan` lives on the transformer; `_dit_modules` is set per instance
   in `__init__` so `clone` mode does not apply the hooks twice. Rank-concatenated order equals padded
   order, so `out[:, max_text_tokens:]` after the SP gather is correct.
7. **Output contract.** `forward(req: DiffusionRequestBatch) -> list[DiffusionOutput]`, one output per
   request, because `guidance_schedule`/`mu`/`std`/`num_steps` are per-request. `[-1, 1]` float output
   matches the default `output_tensor_range`, so no declaration is needed; the post-process factory
   denormalizes through `VaeImageProcessor`.
8. **Out of scope.** Caption verifier, magic-prompt expansion, hosted API/safety screening,
   `recipes/` entry, `examples/model_display_names.yml`.

## Findings (Ideogram 4)

- `guidance_schedule` presets are stored in **loop-index order** (index 0 = last polish step), which
  is the reverse of the diffusers sigma order; the loop reads `gw_per_step[num_steps - 1 - k]`.
- `schedule(0) = 1` (noisiest) and `schedule(1) ≈ 0`, so `sigmas[j] = schedule(step_intervals[j])`
  is the usual diffusers descending order, and user-supplied `sampling_params.sigmas` is compatible.
- The checkpoint is access-restricted, so key naming is inferred from the reference loaders: text
  encoder keys are relative to the model (`language_model.layers.*`), transformer keys are
  `adaln_emb.*`, `layers.*`, `output_norm.weight`, `output_proj.weight`, and the VAE is diffusers-format.
  `weight_name_patterns=("language_model.*",)` skips the ~1.2 GB vision tower.
- `AutoWeightsLoader` cannot handle `Fp8Linear` buffers, so the pipeline dispatches weights itself
  (`load_streamed_weights`); buffers are never part of the strict expected-parameter check.
- Reference config key `adanln_dim` is a typo; the HF config uses `adaln_dim`.
- `num_steps` is not a declared extra-body param, so a request that sets it is dropped; the pipeline reads
  `sampling_params.num_inference_steps`. All examples and `customs/ideogram/samples.json` use
  `num_inference_steps` as the canonical key.

## Risks

- `clone` is an approximation: the reference unconditional transformer has genuinely different
  weights (different `lfs_sha256` on the two index JSONs). Report clone-vs-keep as a measured diff.
- `keep` mode needs 2 × 9.3 GB, so layerwise offload is mandatory on 16 GB cards (≥ 20 GB pinned host).
- Checkpoint key names are inferred, not verified against a real download.

## Verification

1. CPU unit tests: `pytest tests/diffusion/models/ideogram4/test_ideogram4.py -v -m "core_model and cpu"`.
2. Latent-norm + scheduler parity against the reference for a fixed tensor/seed.
3. `nvidia-smi` peak per GPU after loading; no OOM at 1024×1024; record numbers here.
4. e2e non-streaming via `docs/ideogram-4/serve.sh` + `docs/ideogram-4/openai_request_example.py`,
   PSNR against golden above the documented threshold.
5. `curl` with `"stream": true` must return 400 (pytest assertion).
6. `dit_copies`: resolution order + invalid-value error; then the same prompt/seed in both modes.
7. Buildkite: correct markers and suite file. L1 is auto-collected by `test-ready.yml`; L4 lives in
   `.buildkite/cuda/test-weekly.yml` (the L2 online-serving case is gated, so it is not wired into
   `test-ready.yml`).

## Testing & CI

| Level | File | Markers | Runs in CI |
|---|---|---|---|
| L1 | `tests/diffusion/models/ideogram4/test_ideogram4.py` | `core_model and cpu and diffusion` | yes, auto-collected by `pytest -sv tests/diffusion/models -m 'core_model and cpu'` |
| L2 | `tests/e2e/online_serving/test_ideogram4.py` | `slow and diffusion and local_model`, test-level `core_model`/`advanced_model` | no — gated checkpoint, skipped unless `IG_MODEL_DIR` is a local copy |
| L4 | `tests/e2e/online_serving/test_ideogram4_expansion.py` | `slow and diffusion and full_model` | `test-weekly.yml` steps `Diffusion · Ideogram 4 · H100 · Single-GPU` / `· 2-GPU` |
| L4 | `tests/dfx/reliability/invalid_param_test/test_invalid_diffusion_chat.py` | `slow and diffusion` | yes — the existing `reliability_invalid_param_h100` / `reliability_invalid_param_l4` steps run the whole directory and now include `*ideogram4`; skips without `IG_MODEL_DIR` |

Local commands:

```bash
# L1 unit tests (CPU, no checkpoint needed)
pytest -sv tests/diffusion/models/ideogram4/test_ideogram4.py -m 'core_model and cpu'

# L2 / L4 online serving (needs a local copy of the gated checkpoint)
IG_MODEL_DIR=/path/to/ideogram-4-fp8 \
  pytest -s -v tests/e2e/online_serving/test_ideogram4.py -m 'core_model' --run-level 'core_model'
IG_MODEL_DIR=/path/to/ideogram-4-fp8 \
  pytest -sv tests/e2e/online_serving/test_ideogram4_expansion.py \
  -m "slow and diffusion and H100 and cards_1" --run-level "full_model"
IG_MODEL_DIR=/path/to/ideogram-4-fp8 \
  pytest -s -v tests/dfx/reliability/invalid_param_test/ -m "slow and L4"
```

The invalid-param case lives in its own file (`test_invalid_diffusion_chat.py`) rather than
`test_invalid_omni_chat.py` because that module carries the `omni` model-type marker while
Ideogram 4 is diffusion-only.

## Sample generation

`customs/ideogram/generate_samples.py` drives the pipeline from a JSON **list** of image details,
so one file describes a whole batch of samples. `customs/ideogram/samples.json` is the checked-in
spec (five entries: default preset, non-square resolution, turbo, an explicit `guidance_schedule`,
and a multi-image entry).

```bash
python customs/ideogram/generate_samples.py                       # every entry
python customs/ideogram/generate_samples.py --only fox turbo      # a subset
python customs/ideogram/generate_samples.py --dry-run             # payloads only, no GPU
bash customs/ideogram/run_samples.sh                              # same thing inside docker
```

Keys use the same names the API takes in `extra_body`, so an entry is valid for both the offline
`Omni` path and a server request. Script-only keys are `id`, `caption`, `output`, `notes`; anything
else must be a sampling key (`height`, `width`, `seed`, `num_inference_steps`,
`num_outputs_per_prompt`, `sigmas`, `output_type`) or a declared Ideogram 4 knob (`preset`,
`guidance_schedule`, `mu`, `std`, `dit_copies`) — unknown keys are rejected before generation.

Pre-flight checks the script runs so a bad spec fails before the model loads: sizes must be
multiples of 16, `guidance_schedule` length must equal the step count, and captions must be
non-empty. `dit_copies` is a startup knob, so the startup default is passed once with
`--dit-copies`; a per-entry value only takes effect when the pipeline started in `keep` mode.
