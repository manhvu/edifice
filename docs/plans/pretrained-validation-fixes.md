# Plan: Fix Pretrained Numerical Validation Tests

## Status

Pretrained fixtures generated (5 files in `test/fixtures/numerical/`):
- `vit_reference.safetensors` (593K) — google/vit-base-patch16-224
- `whisper_encoder_reference.safetensors` (3.9M) — openai/whisper-base (3000 mel frames)
- `convnext_reference.safetensors` (593K) — facebook/convnext-tiny-224
- `resnet_reference.safetensors` (593K) — microsoft/resnet-50
- `detr_reference.safetensors` (806K) — facebook/detr-resnet-50

Random-weight tests: **all 8 passing**.

Pretrained tests (`numerical_validation_test.exs`): **0 of 4 passing** (ConvNeXt skipped).

## Root Cause

`Edifice.Pretrained.from_hub/2` returns `{model, model_state}` where `model_state` only
contains HuggingFace checkpoint params. It does NOT first init the Axon model to create a
complete param set. When `predict_fn.(params, input)` is called, Axon triggers lazy init
for missing params, which requires shape inference — and that fails in two different ways:

### Issue 1: ViT — Axon.nx shape inference failure

**Error:** `cannot broadcast tensor of dimensions {1, 196, 3, 16, 16} to {768}`

`PatchEmbed.layer/2` uses `Axon.nx` to reshape `{1, 3, 224, 224}` → `{1, 196, 768}`.
Axon can't statically infer the output shape of the closure, so when predict_fn lazy-inits
params for the downstream `Axon.dense`, it uses the pre-reshape shape.

**Fix:** Init model first with `init_fn.(template, Axon.ModelState.empty())`, then deep-merge
HF weights on top. This ensures all param shapes are correctly resolved.

### Issue 2: ResNet / DETR — Missing conv bias params

**Error:** `parameter "bias" for layer "stem_conv" was not present in the given parameter map`

HuggingFace ResNet uses `bias=False` on all Conv2d layers. Edifice ResNet uses Axon.conv
with default `use_bias: true`. The HF checkpoint has no bias keys, so ModelState is missing
bias params for every conv layer.

**Fix options (pick one):**
- **A (Best):** Add `use_bias: false` to all conv layers in `lib/edifice/convolutional/resnet.ex`
  to match HF. Conv bias is redundant when followed by BatchNorm anyway. Update ResNet unit
  tests to expect no bias params.
- **B (Quick):** Init model first, deep-merge HF weights. Default zero-init bias is harmless
  since BN follows every conv.

### Issue 3: Whisper — Wrong input key (already fixed)

Changed `"mel"` → `"mel_spectrogram"` in the test. Also need to fix `Axon.build` return
order: `{_init_fn, predict_fn}` not `{predict_fn, _}`.

### Issue 4: ConvNeXt — No safetensors on HuggingFace (skipped)

`facebook/convnext-tiny-224` only has `pytorch_model.bin`, no `model.safetensors`.
Test is `@tag :skip`. ConvNeXt is validated via random-weight tests.

## Implementation Plan

### Step 1: Add `merge_pretrained_params/3` helper

Add to `test/edifice/pretrained/numerical_validation_test.exs` (or `numerical_fixture_helper.ex`):

```elixir
defp merge_pretrained_params(model, hf_state, template) do
  {init_fn, predict_fn} = Axon.build(model, mode: :inference)
  full_state = init_fn.(template, Axon.ModelState.empty())
  merged_data = deep_merge(full_state.data, hf_state.data)
  {%{full_state | data: merged_data}, predict_fn}
end

defp deep_merge(base, overlay) when is_map(base) and is_map(overlay) do
  Map.merge(base, overlay, fn
    _k, base_v, overlay_v when is_map(base_v) and is_map(overlay_v) ->
      deep_merge(base_v, overlay_v)
    _k, _base_v, overlay_v ->
      overlay_v
  end)
end
```

This inits all params (resolving shapes correctly), then replaces values from HF.

### Step 2: Update each test

Replace the `Axon.build` + `predict_fn.(params, input)` pattern with:

```elixir
{model, hf_params} = Edifice.Pretrained.from_hub(...)
template = %{"image" => Nx.template({1, 3, 224, 224}, :f32)}
{params, predict_fn} = merge_pretrained_params(model, hf_params, template)
output = predict_fn.(params, %{"image" => input})
```

Templates per architecture:
- **ViT:** `%{"image" => Nx.template({1, 3, 224, 224}, :f32)}`
- **ResNet:** `%{"input" => Nx.template({1, 224, 224, 3}, :f32)}` (NHWC)
- **DETR:** `%{"image" => Nx.template({1, 256, 256, 3}, :f32)}` (NHWC)
- **Whisper:** `%{"mel_spectrogram" => Nx.template({1, 80, 3000}, :f32)}`

### Step 3: Fix Whisper Axon.build return order

Line 226 still has `{predict_fn, _}` — change to `{_init_fn, predict_fn}`.
(Actually use `merge_pretrained_params` so this is handled automatically.)

### Step 4: Run tests, debug numerical mismatches

After fixing the structural issues, run:
```
mix test test/edifice/pretrained/numerical_validation_test.exs --include external
```

Expect possible numerical mismatches (atol may need tuning) for:
- ResNet: zero-init bias vs no bias (should be identical, BN absorbs)
- DETR: complex pipeline, may need `atol: 5e-4` or higher
- Whisper: large model, may need `atol: 5e-4`

### Step 5: Optional — Fix ResNet `use_bias: false`

If step 2 works (zero-init bias is fine), this is optional. But it's cleaner:
- Edit `lib/edifice/convolutional/resnet.ex`: add `use_bias: false` to `Axon.conv` calls
- Verify unit tests still pass: `mix test test/edifice/convolutional/resnet_test.exs`

### Step 6: Mark TODO items done, commit

Update TODO.md: check off "Generate pretrained fixtures" and "Run pretrained validation
end-to-end". Also check off the already-done random-weight items.

## Dependencies

- Python packages already installed: `torch`, `transformers`, `safetensors`, `timm`, `numpy`, `packaging`
- `zlib` added to devenv.nix LD_LIBRARY_PATH
- Fixtures already generated in `test/fixtures/numerical/`

## Files to modify

| File | Change |
|------|--------|
| `test/edifice/pretrained/numerical_validation_test.exs` | Add `merge_pretrained_params`, update all test bodies |
| `test/support/numerical_fixture_helper.ex` | Optionally move `deep_merge` here for reuse |
| `lib/edifice/convolutional/resnet.ex` | (Optional) `use_bias: false` on all conv layers |
| `scripts/generate_numerical_fixtures.py` | Already fixed (3000 mel, fault-tolerant main) |
| `devenv.nix` | Already fixed (zlib in LD_LIBRARY_PATH, transformers+timm in venv) |
| `TODO.md` | Mark items done |
