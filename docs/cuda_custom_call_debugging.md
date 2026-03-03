# CUDA Custom Call Debugging Guide

This document describes the methodology used to debug segfaults and other failures in EXLA custom call kernels. The techniques apply to any XLA FFI-based CUDA integration.

## Architecture Overview

Custom calls flow through a 3-tier dispatch:

```
Elixir (fused_scan.ex)
  → Nx.Shared.optional(:fused_xxx, [operands], output, fallback)
    → EXLA defn.ex pattern matches :optional expression
      → value.ex builds stablehlo.custom_call
        → CUDA FFI handler (fused_xxx.cu)
          → kernel launch
```

When called outside of `EXLA.jit`/defn, `Nx.Shared.optional` runs the **fallback callback** directly — the custom call path is only exercised during EXLA JIT compilation.

## Common Failure Modes

### 1. Scalar Buffer Operand Segfault (MOST COMMON)

**Symptom**: `Segmentation fault (core dumped)` during `EXLA.jit(fn ...).(input)`. Exit code 139. No error message — process dies immediately.

**Root Cause**: Passing scalar (0-dimensional) tensors as operands to `stablehlo.custom_call` causes segfaults in XLA's MLIR compilation. This affects:
- `ffi::AnyBuffer` (used for scalar i32 flags like `causal`)
- `ffi::Buffer<ffi::F32>` (used for scalar float params like `momentum`, `leak_rate`)

**Why it happens**: XLA's FFI mechanism handles multi-dimensional buffers correctly but has an issue with 0-d buffers passed as operands. The segfault occurs during MLIR-to-HLO lowering, not during kernel execution.

**Detection**: If the kernel works via NIF bridge or BinaryBackend fallback but segfaults via EXLA JIT, suspect this issue. Count the operands in the custom call — if any are scalars, that's your culprit.

**Fix Pattern**: Hardcode the scalar value in the C++ FFI handler. Remove it from the operand list.

```cpp
// BEFORE (segfaults):
ffi::Error my_kernel_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<FFI_IO_TYPE> input,
    ffi::AnyBuffer causal_flag,        // ← scalar causes segfault
    ffi::ResultBuffer<FFI_IO_TYPE> output
) {
    int causal = reinterpret_cast<const int32_t*>(causal_flag.untyped_data())[0];
    ...
}
// Bind: .Arg<ffi::Buffer<FFI_IO_TYPE>>()  .Arg<ffi::AnyBuffer>()  .Ret<...>()

// AFTER (works):
ffi::Error my_kernel_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<FFI_IO_TYPE> input,
    ffi::ResultBuffer<FFI_IO_TYPE> output
) {
    int causal = 1;  // hardcoded
    ...
}
// Bind: .Arg<ffi::Buffer<FFI_IO_TYPE>>()  .Ret<...>()
```

Then update the Elixir side:
- `defn.ex`: Change the pattern match from `args: [input, causal]` to `args: [input, _causal]`, remove the `recur_operator(causal, ...)` call
- `value.ex`: Remove the scalar parameter, drop it from `operands` list
- `fused_scan.ex`: No change needed (the scalar is still passed in the `Nx.Shared.optional` args for the fallback, but defn.ex ignores it)

### 2. XLA 7-Operand Limit

**Symptom**: Same segfault as above, but the kernel has 8+ Args in its FFI Bind().

**Root Cause**: `stablehlo.custom_call` segfaults during MLIR compilation when a custom call has 8 or more `.Arg<>()` operands. The limit is 7 Args (empirically determined).

**Fix Pattern**: Pack tensors together. For backward kernels that need many inputs (forward_out + grad_output), concatenate them into a single packed buffer:

```elixir
# In defn.ex:
packed = Value.concatenate([forward_out, grad_output], 0)  # [2, B, T, H, d]
# Pass packed as single operand

# In the CUDA kernel:
const io_type* packed_ptr = reinterpret_cast<const io_type*>(packed.untyped_data());
const io_type* forward_out_ptr = packed_ptr;
const io_type* grad_output_ptr = packed_ptr + total_elements_per_tensor;
```

### 3. Attribute vs Buffer Mismatch

**Symptom**: `Wrong number of attributes: expected N but got 0`

**Root Cause**: The FFI handler uses `.Attr<T>("name")` but EXLA's value.ex doesn't pass named attributes — only positional `Arg` buffers and StableHLO attributes. XLA FFI `.Attr<>()` requires named attributes in the custom_call op, which `stablehlo.custom_call` doesn't easily support.

**Fix**: Convert `.Attr<>()` parameters to `.Arg<>()` buffers (pass as tensors), or hardcode them in the C++ handler.

## Debugging Methodology

### Step 1: Isolate the Dispatch Path

Write a test script that exercises each path separately:

```elixir
# Path 1: BinaryBackend fallback (always works)
result1 = FusedScan.my_kernel(input, opts)

# Path 2: EXLA JIT (custom call — may segfault)
jit_fn = EXLA.jit(fn x -> FusedScan.my_kernel(x, opts) end)
result2 = jit_fn.(input)

# Path 3: Axon pipeline (custom call through Axon build)
{init_fn, predict_fn} = Axon.build(model, compiler: EXLA)
result3 = predict_fn.(params, input)
```

If Path 1 works but Path 2 segfaults, the issue is in the custom call plumbing. If Path 2 works but Path 3 fails, the issue is in Axon layer integration.

### Step 2: Compare with a Working Kernel

Run a known-working custom call (e.g., MinGRU) in the same process to confirm the infrastructure is functional:

```elixir
# Confirm infrastructure works
mingru_jit = EXLA.jit(fn z, h -> FusedScan.mingru(z, h) end)
result_ok = mingru_jit.(z, h)  # Should pass

# Then test the failing kernel
failing_jit = EXLA.jit(fn x -> FusedScan.titans_scan(x, opts) end)
result_fail = failing_jit.(input)  # Segfaults here
```

### Step 3: Count Operands and Check Types

In the `.cu` file, count the Args in `ffi::Ffi::Bind()`:
- If any are `ffi::AnyBuffer` or `ffi::Buffer<ffi::F32>` for scalar values → scalar operand segfault
- If total Args ≥ 8 → operand limit segfault
- If any use `.Attr<>()` → attribute mismatch

### Step 4: Numerical Validation

After fixing the segfault, validate numerical correctness with **relative error** (not absolute error). Recurrent kernels amplify tiny floating point differences exponentially:

```elixir
abs_diff = Nx.abs(Nx.subtract(result_cpu, result_gpu))
denominator = Nx.add(Nx.abs(result_cpu), 1.0e-8)
rel_err = Nx.divide(abs_diff, denominator) |> Nx.reduce_max() |> Nx.to_number()
# rel_err < 1e-4 is acceptable for f32
```

Absolute error can be misleading — a diff of `687 billion` sounds terrible but is actually a relative error of `~3e-4` when values are `~10^12`.

### Step 5: Size Sweep

Test at multiple sizes to catch scaling issues:

```elixir
for {batch, seq, hidden} <- [{1,2,4}, {1,4,64}, {2,16,64}] do
  # test at each size
end
```

## Available Debugging Tools

| Tool | Path | Use For |
|------|------|---------|
| `compute-sanitizer` | `/nix/store/.../cuda-merged-12.8/bin/compute-sanitizer` | CUDA memory errors (illegal access, races). Only catches GPU-side errors, NOT host-side segfaults. |
| `cuda-gdb` | `/nix/store/.../cuda-merged-12.8/bin/cuda-gdb` | Full GPU debugger. Can break on kernel launch, inspect thread state. |
| `cuda-memcheck` | (via compute-sanitizer) | Memory leak detection. |

**Note**: Most segfaults in custom calls are **host-side** (during XLA compilation), NOT GPU-side. `compute-sanitizer` won't catch these. For host-side segfaults, `gdb` would be needed but isn't available in the NixOS environment by default.

## Resolved Issues Log

| Issue | Root Cause | Fix | Files Changed |
|-------|-----------|------|--------------|
| Titans segfault | Scalar `ffi::Buffer<ffi::F32>` momentum operand | Hardcode momentum=0.9 | titans_scan.cu, defn.ex, value.ex |
| MIRAS segfault | Scalar `ffi::Buffer<ffi::F32>` momentum operand | Hardcode momentum=0.9 | miras_scan.cu, defn.ex, value.ex |
| Flash attention segfault | Scalar `ffi::AnyBuffer` causal flag | Hardcode causal=1 | flash_attention.cu, flash_attention_backward.cu, defn.ex, value.ex |
| LASER attention segfault | Scalar `ffi::AnyBuffer` causal flag | Hardcode causal=1 | laser_attention.cu, laser_attention_backward.cu, defn.ex, value.ex |
| InfiniAttention segfault | Uses flash_attention internally | Fixed by flash attention fix | (no additional changes) |
| Reservoir EXLA | `Axon.constant` + `Nx.Shared.optional` tracing | Switch to `Axon.nx` with closure-captured weights | reservoir.ex |
| RLA attributes error | `ffi::Attr<>()` not supported by EXLA | Convert to `ffi::Arg<>()` / hardcode | rla_scan.cu, defn.ex, value.ex |
| RLA backward operands | 10 operands > 7 limit | Pack forward_out+grad_output into [2,B,T,H,d] | rla_scan_backward.cu, defn.ex, value.ex |
| Titans/MIRAS momentum | Scalar `ffi::Buffer<ffi::F32>` momentum | Pack momentum into combined tensor as extra column | titans_scan.cu, miras_scan.cu, fused_scan.ex, defn.ex |

## Current Limitations

### Causal=1 Hardcoding (Flash + LASER Attention)

The Flash Attention and LASER Attention custom call kernels hardcode `causal=1` in the C++ FFI handler. This is because passing scalar integer flags as operands to `stablehlo.custom_call` triggers the scalar buffer operand segfault described above.

**Impact:** When `causal: false` is requested, the dispatch layer in `fused_scan.ex` routes to the NIF or Elixir fallback path instead of the custom call. This is correct behavior — the NIF and fallback paths handle `causal=0` properly. The only downside is that non-causal attention doesn't benefit from staying inside the XLA graph (minor perf impact from graph break).

**When this matters:** Only affects non-causal attention through EXLA JIT. In practice, most transformer models use causal masking for autoregressive generation, so this limitation rarely triggers.

**Potential fix:** Pack causal as an extra element in one of the input tensors (similar to the momentum packing approach used for Titans/MIRAS). Not implemented because the causal flag is an integer, not a float, and the workaround would require an extra tensor or type casting. The current conditional dispatch is simpler and correct.
