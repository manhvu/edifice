# bench/tasks/ Fixup Plan

## What's Done

All 6 bench files are written and structurally complete:

| File | Lines | Status |
|------|-------|--------|
| `bench/tasks/task_helpers.exs` | 239 | Written — shared train loop, eval, data gen, formatting |
| `bench/tasks/sequence_classification.exs` | 160 | Written — cumsum sign prediction, 5 seq architectures |
| `bench/tasks/copy_recall.exs` | 186 | Written — template memory classification, 6 SSM/recurrent archs |
| `bench/tasks/image_classification.exs` | 246 | Written — quadrant brightness, 5 vision archs (NCHW/NHWC handling) |
| `bench/tasks/autoregressive.exs` | 192 | Written — next-token prediction on repeating grammar, 4 LM archs |
| `bench/tasks/graph_classification.exs` | 271 | Written — edge density classification, 4 graph archs (GCN/GAT/GIN/GINv2) |

Architecture → config mapping, data generation, Axon model building with task heads, and formatted output all work correctly at the module level.

## Blocking Issue: `value_and_grad` + EXLA.Backend Incompatibility

**The existing `bench/training_throughput.exs` also fails with the same error** — this is an environment-level issue, not a bug in the new bench code.

### Error
```
cannot invoke Nx function because it relies on two incompatible tensor implementations:
EXLA.Backend and Nx.Defn.Expr
```

Occurs in `Axon.Layers.layer_norm/4` when `Nx.Defn.value_and_grad` traces params as `Nx.Defn.Expr` while captured input tensors remain as `EXLA.Backend`.

### Root Cause
In Nx 0.11 + EXLA 0.11, `Nx.Defn.value_and_grad` uses the Evaluator compiler which can't mix EXLA.Backend constants (captured in closures) with Expr (traced differentiation variables). This affects any code that:
1. Sets `Nx.default_backend(EXLA.Backend)`
2. Captures EXLA tensors (inputs, targets) in a `value_and_grad` closure
3. Has layers like `layer_norm` that do element-wise ops between params (Expr) and input (EXLA)

### Confirmed Workaround
`Nx.backend_copy(tensor, Nx.BinaryBackend)` on captured inputs before entering `value_and_grad` works for the forward+grad computation:

```elixir
input_bc = Map.new(input, fn {k, v} -> {k, Nx.backend_copy(v, Nx.BinaryBackend)} end)
target_bc = Nx.backend_copy(target, Nx.BinaryBackend)

{loss, grads} = Nx.Defn.value_and_grad(ms.data, fn params ->
  state = %{ms | data: params}
  output = predict_fn.(state, input_bc)
  loss_fn.(output, target_bc)
end)
```

This was verified to work for a single `value_and_grad` call. However, when used in the full training loop, it still fails — likely because the updated `ms.data` (after SGD step) contains EXLA tensors from the grad computation that then get captured in the next iteration's closure.

### Fix Steps (in priority order)

1. **Update `TaskHelpers.train/4`**: Apply `Nx.backend_copy` to inputs AND backend-transfer the model state data to BinaryBackend before each `value_and_grad` call. After the SGD step, the new params will be BinaryBackend tensors, which is fine since they get passed as the differentiation variable next iteration.

2. **Alternative: Use `Nx.Defn.jit` wrapper**: Wrap the entire train step in `Nx.Defn.jit(step_fn, compiler: EXLA)` passing all tensors (params, input, target) as explicit arguments instead of closures. This is how `bench/sanity_check.exs` handles the same issue. More complex but avoids repeated backend copies.

3. **Alternative: Run on BinaryBackend**: Set `Nx.default_backend(Nx.BinaryBackend)` instead of EXLA. Much slower but avoids the issue entirely. Good for correctness testing.

4. **Build with `mode: :inference`**: Already implemented in the bench files. Needed to avoid state-related issues but doesn't fix the core Expr/Backend incompatibility.

## Other Items to Verify After Fix

- [ ] All 5 task scripts run without errors
- [ ] At least one architecture per task shows learning (accuracy > random baseline)
  - Sequence classification: > 50% (binary)
  - Copy/recall: > 25% (4-class)
  - Image classification: > 25% (4-class)
  - Autoregressive: > 12.5% (8-class)
  - Graph classification: > 25% (4-class)
- [ ] Vision models (ResNet, ConvNeXt) may fail `value_and_grad` due to conv layers — confirm try/rescue handles gracefully
- [ ] Tables print correctly with timing + accuracy columns
- [ ] Graph models handle multi-input (nodes + adjacency + edge_features for GINv2) correctly

## Task Adaptations from Original Plan

The original plan assumed sequence models output `[batch, seq_len, embed]` per timestep. In practice, all Edifice sequence models output `[batch, hidden_size]` (last timestep only). Tasks were adapted:

- **Sequence classification**: Predict sign of total sum (not per-timestep cumsum sign)
- **Copy/recall**: Classify which template was shown (not reproduce the sequence)
- **Autoregressive**: Predict single next token (not full sequence generation)
- All tasks use `Axon.dense(num_classes)` head on top of the architecture output
