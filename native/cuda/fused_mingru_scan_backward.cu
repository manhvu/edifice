// Fused MinGRU Scan Backward Kernel
//
// Computes gradients for: h_t = (1 - z_t) * h_{t-1} + z_t * c_t
// where z_t = sigmoid(gates_t) (pre-computed on Elixir side).
//
// Reverse pass (T-1 -> 0):
//   dh = grad_output[t] + dh_acc
//   dz[t] = dh * (c_t - h_{t-1})
//   dc[t] = dh * z_t
//   dh_acc = dh * (1 - z_t)
//   grad_h0 = dh_acc (after loop)
//
// Note: This kernel computes gradients w.r.t. post-sigmoid z and candidates.
// The chain rule for raw gates (sigmoid derivative) is applied in Elixir:
//   grad_gates = grad_z * z * (1 - z)
//
// Inputs:
//   z:            [B, T, H] — post-sigmoid gate values
//   candidates:   [B, T, H] — candidate values
//   h0:           [B, H]    — initial hidden state
//   forward_out:  [B, T, H] — forward pass hidden states
//   grad_output:  [B, T, H] — upstream gradient dL/dh
//
// Outputs:
//   grad_z:    [B, T, H] — gradient w.r.t. z (post-sigmoid)
//   grad_cand: [B, T, H] — gradient w.r.t. candidates
//   grad_h0:   [B, H]    — gradient w.r.t. initial state

#include <cuda_runtime.h>

// ============================================================================
// Kernel
// ============================================================================

__global__ void fused_mingru_scan_backward_kernel(
    const float* __restrict__ z,            // [B, T, H] post-sigmoid gates
    const float* __restrict__ candidates,   // [B, T, H]
    const float* __restrict__ h0,           // [B, H]
    const float* __restrict__ forward_out,  // [B, T, H]
    const float* __restrict__ grad_output,  // [B, T, H]
    float* __restrict__ grad_z,             // [B, T, H]
    float* __restrict__ grad_cand,          // [B, T, H]
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

        float z_t = z[idx];
        float c_t = candidates[idx];

        // h_{t-1}
        float h_prev;
        if (t == 0) {
            h_prev = h0[b * hidden + h];
        } else {
            h_prev = forward_out[idx - hidden];
        }

        grad_z[idx] = dh * (c_t - h_prev);
        grad_cand[idx] = dh * z_t;
        dh_acc = dh * (1.0f - z_t);
    }

    grad_h0[b * hidden + h] = dh_acc;
}

// ============================================================================
// Standalone launch wrapper (C-linkage for NIF / dlopen)
// ============================================================================

#ifndef EXLA_FFI

extern "C" {

// Output: concatenated [grad_z (B*T*H) | grad_cand (B*T*H) | grad_h0 (B*H)]
int fused_mingru_scan_backward_launch(
    cudaStream_t stream,
    const float* z, const float* candidates,
    const float* h0, const float* forward_out,
    const float* grad_output,
    float* output_concat,
    int batch, int seq_len, int hidden
) {
    int bth = batch * seq_len * hidden;
    float* grad_z    = output_concat;
    float* grad_cand = output_concat + bth;
    float* grad_h0   = output_concat + 2 * bth;

    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);

    fused_mingru_scan_backward_kernel<<<grid, block, 0, stream>>>(
        z, candidates, h0, forward_out, grad_output,
        grad_z, grad_cand, grad_h0,
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

ffi::Error fused_mingru_scan_backward_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> z,
    ffi::Buffer<ffi::F32> candidates,
    ffi::Buffer<ffi::F32> h0,
    ffi::Buffer<ffi::F32> forward_out,
    ffi::Buffer<ffi::F32> grad_output,
    ffi::ResultBuffer<ffi::F32> grad_z,
    ffi::ResultBuffer<ffi::F32> grad_cand,
    ffi::ResultBuffer<ffi::F32> grad_h0
) {
    auto dims = z.dimensions();
    int batch   = static_cast<int>(dims[0]);
    int seq_len = static_cast<int>(dims[1]);
    int hidden  = static_cast<int>(dims[2]);

    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);

    fused_mingru_scan_backward_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const float*>(z.untyped_data()),
        reinterpret_cast<const float*>(candidates.untyped_data()),
        reinterpret_cast<const float*>(h0.untyped_data()),
        reinterpret_cast<const float*>(forward_out.untyped_data()),
        reinterpret_cast<const float*>(grad_output.untyped_data()),
        reinterpret_cast<float*>(grad_z->untyped_data()),
        reinterpret_cast<float*>(grad_cand->untyped_data()),
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
    fused_mingru_scan_backward, fused_mingru_scan_backward_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()   // z
        .Arg<ffi::Buffer<ffi::F32>>()   // candidates
        .Arg<ffi::Buffer<ffi::F32>>()   // h0
        .Arg<ffi::Buffer<ffi::F32>>()   // forward_out
        .Arg<ffi::Buffer<ffi::F32>>()   // grad_output
        .Ret<ffi::Buffer<ffi::F32>>()   // grad_z
        .Ret<ffi::Buffer<ffi::F32>>()   // grad_cand
        .Ret<ffi::Buffer<ffi::F32>>()   // grad_h0
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_mingru_scan_backward_f32", "CUDA", fused_mingru_scan_backward);

#endif  // EXLA_FFI
