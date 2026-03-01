// Fused Mamba Selective Scan Kernel
//
// Implements the Mamba SSM recurrence with input-dependent discretization:
//   A_bar = exp(dt * A)
//   B_bar = dt * B
//   h_t = A_bar * h_{t-1} + B_bar * x_t    (per state dimension)
//   y_t = sum(C_t * h_t, axis=state)
//
// Thread layout: one thread per (batch, hidden) element.
// Each thread maintains state[32] in registers and scans sequentially
// through timesteps.
//
// Two compilation modes:
//   1. Standalone (-DSTANDALONE): kernel + C-linkage launch wrapper.
//   2. EXLA FFI (-DEXLA_FFI): kernel + XLA FFI handler + registration.
//
// Inputs:
//   x:  [batch, seq_len, hidden]  — input activations
//   dt: [batch, seq_len, hidden]  — discretization timesteps (clamped)
//   A:  [hidden, state]           — state transition diagonal (negative)
//   B:  [batch, seq_len, state]   — input-to-state projection
//   C:  [batch, seq_len, state]   — state-to-output projection
//
// Output:
//   out: [batch, seq_len, hidden] — scan output

#include <cuda_runtime.h>

constexpr float DT_MIN = 0.001f;
constexpr float DT_MAX = 0.1f;

// ============================================================================
// Kernel
// ============================================================================

__global__ void fused_selective_scan_kernel(
    const float* __restrict__ x,      // [B, T, H]
    const float* __restrict__ dt,     // [B, T, H]
    const float* __restrict__ A,      // [H, S]
    const float* __restrict__ B,      // [B, T, S]
    const float* __restrict__ C,      // [B, T, S]
    float* __restrict__ out,          // [B, T, H]
    int batch, int seq_len, int hidden, int state
) {
    int b = blockIdx.x;
    int h = threadIdx.x + blockIdx.y * blockDim.x;

    if (b >= batch || h >= hidden) return;

    // Initialize hidden state in registers (max state_size = 32)
    float h_state[32];
    for (int s = 0; s < state && s < 32; s++) {
        h_state[s] = 0.0f;
    }

    // Load A diagonal for this hidden dim (constant across timesteps)
    float A_diag[32];
    for (int s = 0; s < state && s < 32; s++) {
        A_diag[s] = A[h * state + s];
    }

    // Sequential scan through timesteps
    for (int t = 0; t < seq_len; t++) {
        int x_idx = b * seq_len * hidden + t * hidden + h;
        float x_t = x[x_idx];
        float dt_t = fminf(fmaxf(dt[x_idx], DT_MIN), DT_MAX);

        int bc_idx = b * seq_len * state + t * state;
        float y_t = 0.0f;

        for (int s = 0; s < state && s < 32; s++) {
            // Discretize: A_bar = exp(dt * A), B_bar = dt * B
            float A_bar = expf(dt_t * A_diag[s]);
            float B_bar = dt_t * B[bc_idx + s];
            float C_s = C[bc_idx + s];

            // Recurrence: h = A_bar * h + B_bar * x
            h_state[s] = A_bar * h_state[s] + B_bar * x_t;

            // Output: y = sum(C * h)
            y_t += C_s * h_state[s];
        }

        out[x_idx] = y_t;
    }
}

// ============================================================================
// Standalone launch wrapper (C-linkage for NIF / dlopen)
// ============================================================================

#ifndef EXLA_FFI

extern "C" {

int fused_selective_scan_launch(
    cudaStream_t stream,
    const float* x, const float* dt, const float* A,
    const float* B, const float* C,
    float* out,
    int batch, int seq_len, int hidden, int state
) {
    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);

    fused_selective_scan_kernel<<<grid, block, 0, stream>>>(
        x, dt, A, B, C, out,
        batch, seq_len, hidden, state
    );

    return (int)cudaGetLastError();
}

}  // extern "C"

#endif  // !EXLA_FFI

// ============================================================================
// XLA FFI integration (for EXLA fork)
// ============================================================================

#ifdef EXLA_FFI

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

ffi::Error fused_selective_scan_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> x,       // [B, T, H]
    ffi::Buffer<ffi::F32> dt,      // [B, T, H]
    ffi::Buffer<ffi::F32> A,       // [H, S]
    ffi::Buffer<ffi::F32> B,       // [B, T, S]
    ffi::Buffer<ffi::F32> C,       // [B, T, S]
    ffi::ResultBuffer<ffi::F32> out // [B, T, H]
) {
    // Extract dimensions from x: [batch, seq_len, hidden]
    auto x_dims = x.dimensions();
    int batch   = static_cast<int>(x_dims[0]);
    int seq_len = static_cast<int>(x_dims[1]);
    int hidden  = static_cast<int>(x_dims[2]);

    // Extract state_size from A: [hidden, state]
    auto a_dims = A.dimensions();
    int state   = static_cast<int>(a_dims[1]);

    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);

    fused_selective_scan_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const float*>(x.untyped_data()),
        reinterpret_cast<const float*>(dt.untyped_data()),
        reinterpret_cast<const float*>(A.untyped_data()),
        reinterpret_cast<const float*>(B.untyped_data()),
        reinterpret_cast<const float*>(C.untyped_data()),
        reinterpret_cast<float*>(out->untyped_data()),
        batch, seq_len, hidden, state
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    fused_selective_scan, fused_selective_scan_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()   // x
        .Arg<ffi::Buffer<ffi::F32>>()   // dt
        .Arg<ffi::Buffer<ffi::F32>>()   // A
        .Arg<ffi::Buffer<ffi::F32>>()   // B
        .Arg<ffi::Buffer<ffi::F32>>()   // C
        .Ret<ffi::Buffer<ffi::F32>>()   // out
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_selective_scan_f32", "CUDA", fused_selective_scan);

#endif  // EXLA_FFI
