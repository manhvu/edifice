// Fused Linear Scan Backward Kernel
//
// Computes gradients for the linear recurrence: h_t = a_t * h_{t-1} + b_t
//
// Reverse pass (T-1 -> 0):
//   dh = grad_output[t] + dh_acc
//   da[t] = dh * h_{t-1}
//   db[t] = dh
//   dh_acc = dh * a_t
//   grad_h0 = dh_acc (after loop)
//
// Same thread layout as the forward kernel: one thread per (batch, hidden)
// element, reverse-iterates over T timesteps. State (dh_acc) stays in
// registers — no shared memory needed.
//
// Two compilation modes:
//   1. Standalone (default): kernel + C-linkage launch wrapper for NIF.
//   2. EXLA FFI (-DEXLA_FFI): kernel + XLA FFI handler + registration.
//
// Inputs:
//   a_vals:       [B, T, H] — multiplicative coefficients (from forward)
//   h0:           [B, H]    — initial hidden state
//   forward_out:  [B, T, H] — forward pass hidden states (h_1..h_T)
//   grad_output:  [B, T, H] — upstream gradient dL/dh
//
// Outputs (concatenated for NIF, separate buffers for FFI):
//   grad_a:  [B, T, H] — gradient w.r.t. a_vals
//   grad_b:  [B, T, H] — gradient w.r.t. b_vals
//   grad_h0: [B, H]    — gradient w.r.t. initial state

#include <cuda_runtime.h>

// ============================================================================
// Kernel
// ============================================================================

__global__ void fused_linear_scan_backward_kernel(
    const float* __restrict__ a_vals,       // [B, T, H]
    const float* __restrict__ h0,           // [B, H]
    const float* __restrict__ forward_out,  // [B, T, H]
    const float* __restrict__ grad_output,  // [B, T, H]
    float* __restrict__ grad_a,             // [B, T, H]
    float* __restrict__ grad_b,             // [B, T, H]
    float* __restrict__ grad_h0,            // [B, H]
    int batch, int seq_len, int hidden
) {
    int b = blockIdx.x;
    int h = threadIdx.x + blockIdx.y * blockDim.x;

    if (b >= batch || h >= hidden) return;

    float dh_acc = 0.0f;

    for (int t = seq_len - 1; t >= 0; t--) {
        int idx = b * seq_len * hidden + t * hidden + h;

        float dh = grad_output[idx] + dh_acc;

        // h_{t-1}: use h0 for t=0, otherwise forward_out[t-1]
        float h_prev;
        if (t == 0) {
            h_prev = h0[b * hidden + h];
        } else {
            h_prev = forward_out[idx - hidden];  // forward_out[b,t-1,h]
        }

        float a = a_vals[idx];

        grad_a[idx] = dh * h_prev;
        grad_b[idx] = dh;
        dh_acc = dh * a;
    }

    grad_h0[b * hidden + h] = dh_acc;
}

// ============================================================================
// Standalone launch wrapper (C-linkage for NIF / dlopen)
// ============================================================================

#ifndef EXLA_FFI

extern "C" {

// Outputs are written to a single concatenated buffer:
//   [grad_a (B*T*H) | grad_b (B*T*H) | grad_h0 (B*H)] floats
int fused_linear_scan_backward_launch(
    cudaStream_t stream,
    const float* a_vals, const float* h0,
    const float* forward_out, const float* grad_output,
    float* output_concat,  // concatenated: grad_a + grad_b + grad_h0
    int batch, int seq_len, int hidden
) {
    int bth = batch * seq_len * hidden;
    float* grad_a  = output_concat;
    float* grad_b  = output_concat + bth;
    float* grad_h0 = output_concat + 2 * bth;

    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);

    fused_linear_scan_backward_kernel<<<grid, block, 0, stream>>>(
        a_vals, h0, forward_out, grad_output,
        grad_a, grad_b, grad_h0,
        batch, seq_len, hidden
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

ffi::Error fused_linear_scan_backward_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> a_vals,
    ffi::Buffer<ffi::F32> h0,
    ffi::Buffer<ffi::F32> forward_out,
    ffi::Buffer<ffi::F32> grad_output,
    ffi::ResultBuffer<ffi::F32> grad_a,
    ffi::ResultBuffer<ffi::F32> grad_b,
    ffi::ResultBuffer<ffi::F32> grad_h0
) {
    auto dims = a_vals.dimensions();
    int batch   = static_cast<int>(dims[0]);
    int seq_len = static_cast<int>(dims[1]);
    int hidden  = static_cast<int>(dims[2]);

    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);

    fused_linear_scan_backward_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const float*>(a_vals.untyped_data()),
        reinterpret_cast<const float*>(h0.untyped_data()),
        reinterpret_cast<const float*>(forward_out.untyped_data()),
        reinterpret_cast<const float*>(grad_output.untyped_data()),
        reinterpret_cast<float*>(grad_a->untyped_data()),
        reinterpret_cast<float*>(grad_b->untyped_data()),
        reinterpret_cast<float*>(grad_h0->untyped_data()),
        batch, seq_len, hidden
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    fused_linear_scan_backward, fused_linear_scan_backward_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()   // a_vals
        .Arg<ffi::Buffer<ffi::F32>>()   // h0
        .Arg<ffi::Buffer<ffi::F32>>()   // forward_out
        .Arg<ffi::Buffer<ffi::F32>>()   // grad_output
        .Ret<ffi::Buffer<ffi::F32>>()   // grad_a
        .Ret<ffi::Buffer<ffi::F32>>()   // grad_b
        .Ret<ffi::Buffer<ffi::F32>>()   // grad_h0
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_linear_scan_backward_f32", "CUDA", fused_linear_scan_backward);

#endif  // EXLA_FFI
