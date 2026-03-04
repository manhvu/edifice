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

**Fix Pattern (preferred): Tensor packing** — Pack the scalar as a 1-element `{1}` tensor
of the same type as the model tensors. This avoids the 0-d buffer issue while still passing
the actual value through the XLA graph (no hardcoding, no graph break).

```elixir
# Elixir side (fused_scan.ex):
causal_packed = Nx.tensor([causal], type: tensor_type)  # {1} tensor, NOT scalar {}
# Pass as normal operand in Nx.Shared.optional args
```

```cpp
// CUDA side: unpack from 1-element device tensor
ffi::Error my_kernel_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<FFI_IO_TYPE> input,
    ffi::Buffer<FFI_IO_TYPE> causal_packed,   // ← 1-element tensor, NOT scalar
    ffi::ResultBuffer<FFI_IO_TYPE> output
) {
    const io_type* causal_ptr = reinterpret_cast<const io_type*>(causal_packed.untyped_data());
    io_type causal_raw;
    cudaMemcpyAsync(&causal_raw, causal_ptr, sizeof(io_type), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
#ifdef USE_BF16
    int causal = (int)__bfloat162float(causal_raw);
#else
    int causal = (int)causal_raw;
#endif
    ...
}
// Bind: .Arg<ffi::Buffer<FFI_IO_TYPE>>()  .Arg<ffi::Buffer<FFI_IO_TYPE>>()  .Ret<...>()
```

**Alternative fix (legacy): Hardcode** — Remove the scalar from the operand list entirely
and hardcode the value in C++. Simpler but loses configurability (requires conditional
dispatch on the Elixir side to fall back for non-hardcoded values).

```cpp
// Hardcoded approach (no longer recommended):
ffi::Error my_kernel_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<FFI_IO_TYPE> input,
    ffi::ResultBuffer<FFI_IO_TYPE> output
) {
    int causal = 1;  // hardcoded — only causal=1 goes through custom call
    ...
}
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
| Flash attention segfault | Scalar `ffi::AnyBuffer` causal flag | Hardcode causal=1 (initial fix) | flash_attention.cu, flash_attention_backward.cu, defn.ex, value.ex |
| LASER attention segfault | Scalar `ffi::AnyBuffer` causal flag | Hardcode causal=1 (initial fix) | laser_attention.cu, laser_attention_backward.cu, defn.ex, value.ex |
| InfiniAttention segfault | Uses flash_attention internally | Fixed by flash attention fix | (no additional changes) |
| Reservoir EXLA | `Axon.constant` + `Nx.Shared.optional` tracing | Switch to `Axon.nx` with closure-captured weights | reservoir.ex |
| RLA attributes error | `ffi::Attr<>()` not supported by EXLA | Convert to `ffi::Arg<>()` / hardcode | rla_scan.cu, defn.ex, value.ex |
| RLA backward operands | 10 operands > 7 limit | Pack forward_out+grad_output into [2,B,T,H,d] | rla_scan_backward.cu, defn.ex, value.ex |
| Titans/MIRAS momentum | Scalar `ffi::Buffer<ffi::F32>` momentum | Pack momentum into combined tensor as extra column | titans_scan.cu, miras_scan.cu, fused_scan.ex, defn.ex |
| Flash/LASER causal hardcoding | `causal=1` hardcoded, non-causal fell back to NIF | Pack causal as 1-element `{1}` tensor (tensor packing pattern) | fused_flash_attention.cu, fused_flash_attention_backward.cu, fused_laser_attention.cu, fused_laser_attention_backward.cu, fused_scan.ex |

## Current Limitations

No known scalar operand workarounds remain. All scalar parameters (momentum, causal flags)
are now passed via tensor packing (1-element `{1}` tensors or extra columns in combined tensors).

### ~~Causal=1 Hardcoding (Flash + LASER Attention)~~ — RESOLVED

**Previously:** The Flash Attention and LASER Attention custom call kernels hardcoded `causal=1` in the C++ FFI handler, causing non-causal attention to fall back to NIF (graph break).

**Fix (2026-03-04):** Pack causal as a 1-element `{1}` tensor of the model's tensor type (f32 or bf16). On the CUDA side, unpack with `cudaMemcpyAsync` D→H + type conversion. Same pattern as Titans momentum packing. The `causal == 1` guard in `fused_scan.ex` dispatch is removed — both causal and non-causal attention now stay inside the XLA graph.

**Files changed:** `fused_flash_attention.cu`, `fused_flash_attention_backward.cu`, `fused_laser_attention.cu`, `fused_laser_attention_backward.cu`, `fused_scan.ex` (6 dispatch functions + 2 docstrings).
