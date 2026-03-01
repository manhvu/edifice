# Axon Graph Fusion — Analysis & Options

## The Problem

Custom calls eliminate graph breaks **within** the fused scan kernel (0.3ms
execution). But when used through `Axon.layer`, the 2-layer MinGRU still takes
~349ms instead of ~2.6ms. Where are the graph breaks?

## Investigation: How Axon Actually Compiles

Axon compiles the **entire** `predict_fn` as a single `Nx.Defn.jit()` call:

```elixir
# Axon.Loop.trainer calls Axon.build(model) which returns:
{init_fn, predict_fn} = Axon.build(model)

# predict_fn is then JIT-compiled as one unit:
Nx.Defn.jit(predict_fn, compiler: EXLA)
```

This means Axon does NOT create separate XLA computations per layer. The entire
model graph is one defn expression tree. In theory, `Nx.Shared.optional` should
work perfectly — it creates `:optional` Expr nodes that EXLA pattern-matches
into `stablehlo.custom_call` ops, all within the single compilation.

## So Where Do Graph Breaks Actually Come From?

The graph breaks come from **host-side sequential operations inside custom layer
callbacks**. Specifically, the `Enum.reduce` in the Elixir scan fallback:

```elixir
# This breaks the expression graph!
Enum.reduce(0..(seq_len - 1), {h0, []}, fn t, {h_prev, acc} ->
  z_t = Nx.slice_along_axis(z, t, 1, axis: 1)
  h_t = Nx.add(Nx.multiply(Nx.subtract(1.0, z_t), h_prev), Nx.multiply(z_t, c_t))
  {h_t, [h_t | acc]}
end)
```

`Enum.reduce` runs on the BEAM at graph-build time, unrolling the loop into N
sequential Nx operations. For `seq_len=32`, this creates 32 slice + multiply +
add chains. Each iteration depends on the previous result, so XLA cannot
parallelize them. This is **by design** — it's the correct fallback for
non-CUDA backends. But it's also why the Axon path is slow: the entire predict_fn
is one compilation, but that compilation contains a massive unrolled loop.

### Wait — the custom call path should bypass the Enum.reduce

If `custom_call_available?()` returns `true`, the `mingru_custom_call/2` path
runs, which calls `Nx.Shared.optional(:fused_mingru_scan, ...)`. Inside defn JIT
on EXLA+CUDA, this should emit a custom call and skip the fallback entirely.

The issue is that `custom_call_available?/0` calls `Code.ensure_loaded?` and
`function_exported?` — these are **runtime checks** that evaluate at graph-build
time. Inside `Axon.build(model)`, the layer callbacks execute to build the
expression graph. If the EXLA fork is installed, `custom_call_available?()` returns
true, and the custom call path is taken. The fallback `fn z, cand, h0 -> ... end`
is never executed.

So the 349ms result from the benchmark was actually measuring the **NIF bridge
path** or the **full Axon overhead**, not the custom call path. The benchmark
may not have been calling the custom call dispatch correctly through Axon.

### The real overhead: Axon.Compiler, not XLA graph breaks

Looking at Axon's compilation, the overhead comes from:
1. **Axon.Compiler traversal** — builds the expression tree by calling each layer's
   callback function, which involves pattern matching, parameter lookups, etc.
2. **First-call JIT compilation** — EXLA compiles the full XLA HLO graph on first
   execution. Subsequent calls use the cached executable.
3. **Parameter threading** — Axon passes the full parameter map through each layer.

For a single `Nx.Defn.jit(fn -> ... end)` that directly calls FusedScan, none of
this overhead exists — just the XLA compilation (cached) and kernel launch.

## Options for Improving Axon Performance

### Option A: Verify custom calls work through Axon (likely already works)

Before pursuing Axon changes, verify that the custom call path is actually being
taken when running through Axon. The benchmark showed 2.6ms for direct defn and
349ms through Axon — but the 349ms might be measuring first-call compilation, not
steady-state execution.

Test: Run the Axon model benchmark with a warmup pass, then time subsequent calls.

### Option B: Replace Enum.reduce fallbacks with Nx.while_loop

If the fallback scan is being used (custom call not available), replacing
`Enum.reduce` with `Nx.while_loop` keeps everything as Nx expressions:

```elixir
# Instead of Enum.reduce (host-side loop):
{h_final, output} = Nx.while_loop(
  {h0, Nx.broadcast(0.0, {batch, seq_len, hidden}), 0},
  fn {_, _, t} -> Nx.less(t, seq_len) end,
  fn {h_prev, output, t} ->
    z_t = Nx.slice_along_axis(z, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
    c_t = Nx.slice_along_axis(candidates, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
    h_t = (1 - z_t) * h_prev + z_t * c_t
    output = Nx.put_slice(output, [0, t, 0], Nx.new_axis(h_t, 1))
    {h_t, output, t + 1}
  end
)
```

This compiles to an XLA `while` loop — stays in the graph, no host round-trips.
However, XLA's while loop is still sequential (each iteration depends on the
previous hidden state), so it won't be faster than the unrolled version for
small `seq_len`. The benefit is eliminating host-side overhead for large `seq_len`.

### Option C: Axon fork — compile to a single XLA computation

An Axon fork could:
1. **Batch layer callbacks** — instead of calling each layer's callback during
   `Axon.build`, collect them and compose them into a single defn function.
2. **Eliminate parameter map threading** — pre-resolve parameter indices at build
   time so the execution path is a flat sequence of Nx ops.

This is a significant undertaking. Axon's architecture is designed around the
layer-callback model for flexibility. Changing this affects every model.

### Option D: Bypass Axon entirely for performance-critical models

Write the model as a pure `defn` function:

```elixir
defn mingru_2layer(x, params) do
  # Layer 1
  g1 = Nx.dot(x, params.w_gate_1) |> Nx.add(params.b_gate_1)
  c1 = Nx.dot(x, params.w_cand_1) |> Nx.add(params.b_cand_1)
  h1 = FusedScan.mingru(g1, c1)

  # Layer 2
  g2 = Nx.dot(h1, params.w_gate_2) |> Nx.add(params.b_gate_2)
  c2 = Nx.dot(h1, params.w_cand_2) |> Nx.add(params.b_cand_2)
  h2 = FusedScan.mingru(g2, c2)

  h2
end
```

This gives you the full 2.6ms performance. Training requires writing a custom
training loop with `Polaris` optimizers, but inference is straightforward.

## Recommendation

**Start with Option A** — the custom call path may already work through Axon with
proper warmup. The 349ms benchmark result likely includes first-call JIT
compilation. Steady-state Axon execution with custom calls should be much closer
to the 2.6ms direct-defn result.

If Option A confirms Axon overhead is still significant after warmup, **Option D**
(bypass Axon) is the pragmatic choice for performance-critical inference. An Axon
fork (Option C) is a large project with uncertain payoff — the simpler approach
is to use Axon for model definition/training and export to pure defn for inference.

## Files

- `docs/exla_custom_calls_results.md` — benchmark data
- `lib/edifice/cuda/fused_scan.ex` — three-tier dispatch implementation
- `bench/model_breakdown.exs` — per-layer benchmarking script
