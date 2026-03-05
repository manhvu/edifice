# Plan: Fix Pretrained Numerical Validation Tests

## Status — IMPLEMENTED, AWAITING EXLA VALIDATION

Pretrained fixtures generated (5 files in `test/fixtures/numerical/`):
- `vit_reference.safetensors` (593K) — google/vit-base-patch16-224
- `whisper_encoder_reference.safetensors` (3.9M) — openai/whisper-base (3000 mel frames)
- `convnext_reference.safetensors` (593K) — facebook/convnext-tiny-224
- `resnet_reference.safetensors` (593K) — microsoft/resnet-50
- `detr_reference.safetensors` (806K) — facebook/detr-resnet-50

Random-weight tests: **all 8 passing**.

Pretrained tests (`numerical_validation_test.exs`): **structural fixes applied, need EXLA to run**.
ConvNeXt: skipped (HF repo only has `pytorch_model.bin`, no safetensors).

## Fixes Applied

### Fix 1: Pass `.data` (plain map) instead of `Axon.ModelState` struct

`Axon.ModelState` doesn't implement the `Access` protocol. When `predict_fn` receives
a ModelState, Axon's internal `get_input/4` tries `Access.fetch(model_state, "image")`
which crashes. Passing `hf_params.data` (a plain nested map) bypasses this.

Used for: **ViT, Whisper** (all params present in HF checkpoint).

### Fix 2: `build_with_pretrained/3` — init + deep merge

For models where HF is missing params (e.g. ResNet conv bias):
1. `init_fn.(template, empty)` creates complete params with zero defaults
2. `deep_merge(full_state.data, hf_state.data)` overlays HF weights

Used for: **ResNet, DETR** (conv bias missing from HF checkpoint).

### Fix 3: Correct `Axon.build` return order

`Axon.build` returns `{init_fn, predict_fn}` — previous code had `{predict_fn, _}`.

### Fix 4: Correct input keys

- Whisper: `"mel_spectrogram"` (not `"mel"`)
- ResNet: `"input"` (NHWC)
- DETR: `"image"` (NHWC)

## Why BinaryBackend Can't Run These Tests

Full-size pretrained models are impractical on BinaryBackend:
- **ViT-base**: 768 embed, 12 layers, 197 patches → O(n²) attention for each layer
- **ResNet-50**: 50+ conv layers on 224×224 images → billions of multiply-adds
- **Whisper**: 2 conv layers on 3000-frame mel spectrogram + 6 transformer layers

These tests need EXLA: `EXLA=1 mix test --include external`

## Remaining Work (EXLA required)

1. **Run with EXLA** — `EXLA=1 mix test test/edifice/pretrained/numerical_validation_test.exs --include external`
2. **Tune tolerances** — May need `atol: 5e-4` or higher for some architectures
3. **Debug numerical mismatches** — If any test fails with max_diff > atol, investigate key map transforms
