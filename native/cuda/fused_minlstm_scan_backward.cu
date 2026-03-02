// Fused MinLSTM Scan Backward Kernel
//
// Computes gradients for: c_t = f'_t * c_{t-1} + i'_t * cand_t
// where f' = f/(f+i+eps), i' = i/(f+i+eps), f = sigmoid(raw_f), i = sigmoid(raw_i)
//
// The kernel works with post-sigmoid f and i values.
// Normalization gradients are computed inside the kernel.
// Chain rule for sigmoid is applied in Elixir.
//
// Reverse pass (T-1 -> 0):
//   dc = grad_output[t] + dc_acc
//   S = f + i + eps
//   f' = f / S, i' = i / S
//   df' = dc * c_{t-1}
//   di' = dc * cand_t
//   dcand[t] = dc * i'
//   dc_acc = dc * f'
//
//   Through normalization (quotient rule):
//     df = (df' * (i + eps) - di' * i) / S^2
//     di = (-df' * f + di' * (f + eps)) / S^2
//
// Inputs:
//   f:            [B, T, H] — post-sigmoid forget gate
//   i:            [B, T, H] — post-sigmoid input gate
//   candidates:   [B, T, H] — candidate values
//   h0:           [B, H]    — initial cell state
//   forward_out:  [B, T, H] — forward pass cell states
//   grad_output:  [B, T, H] — upstream gradient dL/dc
//
// Outputs:
//   grad_f:    [B, T, H] — gradient w.r.t. f (post-sigmoid)
//   grad_i:    [B, T, H] — gradient w.r.t. i (post-sigmoid)
//   grad_cand: [B, T, H] — gradient w.r.t. candidates
//   grad_h0:   [B, H]    — gradient w.r.t. initial cell state

#include <cuda_runtime.h>

constexpr float NORM_EPS = 1.0e-6f;

// ============================================================================
// Kernel
// ============================================================================

__global__ void fused_minlstm_scan_backward_kernel(
    const float* __restrict__ f,            // [B, T, H] post-sigmoid forget
    const float* __restrict__ i_gate,       // [B, T, H] post-sigmoid input
    const float* __restrict__ candidates,   // [B, T, H]
    const float* __restrict__ h0,           // [B, H]
    const float* __restrict__ forward_out,  // [B, T, H]
    const float* __restrict__ grad_output,  // [B, T, H]
    float* __restrict__ grad_f,             // [B, T, H]
    float* __restrict__ grad_i,             // [B, T, H]
    float* __restrict__ grad_cand,          // [B, T, H]
    float* __restrict__ grad_h0,            // [B, H]
    int batch, int seq_len, int hidden
) {
    int b = blockIdx.x;
    int h = threadIdx.x + blockIdx.y * blockDim.x;

    if (b >= batch || h >= hidden) return;

    float dc_acc = 0.0f;

    for (int t = seq_len - 1; t >= 0; t--) {
        int idx = b * seq_len * hidden + t * hidden + h;

        float dc = grad_output[idx] + dc_acc;

        float f_t = f[idx];
        float i_t = i_gate[idx];
        float cand_t = candidates[idx];

        // Normalization
        float S = f_t + i_t + NORM_EPS;
        float f_norm = f_t / S;
        float i_norm = i_t / S;

        // c_{t-1}
        float c_prev;
        if (t == 0) {
            c_prev = h0[b * hidden + h];
        } else {
            c_prev = forward_out[idx - hidden];
        }

        // Gradients w.r.t. normalized gates
        float df_norm = dc * c_prev;
        float di_norm = dc * cand_t;

        // Gradient w.r.t. candidates
        grad_cand[idx] = dc * i_norm;

        // Accumulate gradient flowing back through c_{t-1}
        dc_acc = dc * f_norm;

        // Through normalization: quotient rule
        // f' = f/S => df = (df' * S - f * (df'+di')) / S^2
        //           = (df' * (i+eps) - di' * f) / S^2  ... wait, let's be precise
        //
        // f' = f/(f+i+eps), i' = i/(f+i+eps)
        // d(f')/df = (S - f)/S^2 = (i+eps)/S^2
        // d(f')/di = -f/S^2
        // d(i')/df = -i/S^2
        // d(i')/di = (S - i)/S^2 = (f+eps)/S^2
        //
        // df_total = df' * d(f')/df + di' * d(i')/df
        //          = df' * (i+eps)/S^2 + di' * (-i)/S^2
        //          = (df' * (i+eps) - di' * i) / S^2
        //
        // di_total = df' * d(f')/di + di' * d(i')/di
        //          = df' * (-f)/S^2 + di' * (f+eps)/S^2
        //          = (-df' * f + di' * (f+eps)) / S^2

        float S2 = S * S;
        grad_f[idx] = (df_norm * (i_t + NORM_EPS) - di_norm * i_t) / S2;
        grad_i[idx] = (-df_norm * f_t + di_norm * (f_t + NORM_EPS)) / S2;
    }

    grad_h0[b * hidden + h] = dc_acc;
}

// ============================================================================
// Standalone launch wrapper (C-linkage for NIF / dlopen)
// ============================================================================

#ifndef EXLA_FFI

extern "C" {

// Output: concatenated [grad_f (B*T*H) | grad_i (B*T*H) | grad_cand (B*T*H) | grad_h0 (B*H)]
int fused_minlstm_scan_backward_launch(
    cudaStream_t stream,
    const float* f, const float* i_gate,
    const float* candidates,
    const float* h0, const float* forward_out,
    const float* grad_output,
    float* output_concat,
    int batch, int seq_len, int hidden
) {
    int bth = batch * seq_len * hidden;
    float* grad_f    = output_concat;
    float* grad_i    = output_concat + bth;
    float* grad_cand = output_concat + 2 * bth;
    float* grad_h0   = output_concat + 3 * bth;

    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);

    fused_minlstm_scan_backward_kernel<<<grid, block, 0, stream>>>(
        f, i_gate, candidates, h0, forward_out, grad_output,
        grad_f, grad_i, grad_cand, grad_h0,
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

ffi::Error fused_minlstm_scan_backward_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> f,
    ffi::Buffer<ffi::F32> i_gate,
    ffi::Buffer<ffi::F32> candidates,
    ffi::Buffer<ffi::F32> h0,
    ffi::Buffer<ffi::F32> forward_out,
    ffi::Buffer<ffi::F32> grad_output,
    ffi::ResultBuffer<ffi::F32> grad_f,
    ffi::ResultBuffer<ffi::F32> grad_i,
    ffi::ResultBuffer<ffi::F32> grad_cand,
    ffi::ResultBuffer<ffi::F32> grad_h0
) {
    auto dims = f.dimensions();
    int batch   = static_cast<int>(dims[0]);
    int seq_len = static_cast<int>(dims[1]);
    int hidden  = static_cast<int>(dims[2]);

    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);

    fused_minlstm_scan_backward_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const float*>(f.untyped_data()),
        reinterpret_cast<const float*>(i_gate.untyped_data()),
        reinterpret_cast<const float*>(candidates.untyped_data()),
        reinterpret_cast<const float*>(h0.untyped_data()),
        reinterpret_cast<const float*>(forward_out.untyped_data()),
        reinterpret_cast<const float*>(grad_output.untyped_data()),
        reinterpret_cast<float*>(grad_f->untyped_data()),
        reinterpret_cast<float*>(grad_i->untyped_data()),
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
    fused_minlstm_scan_backward, fused_minlstm_scan_backward_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()   // f
        .Arg<ffi::Buffer<ffi::F32>>()   // i_gate
        .Arg<ffi::Buffer<ffi::F32>>()   // candidates
        .Arg<ffi::Buffer<ffi::F32>>()   // h0
        .Arg<ffi::Buffer<ffi::F32>>()   // forward_out
        .Arg<ffi::Buffer<ffi::F32>>()   // grad_output
        .Ret<ffi::Buffer<ffi::F32>>()   // grad_f
        .Ret<ffi::Buffer<ffi::F32>>()   // grad_i
        .Ret<ffi::Buffer<ffi::F32>>()   // grad_cand
        .Ret<ffi::Buffer<ffi::F32>>()   // grad_h0
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_minlstm_scan_backward_f32", "CUDA", fused_minlstm_scan_backward);

#endif  // EXLA_FFI
