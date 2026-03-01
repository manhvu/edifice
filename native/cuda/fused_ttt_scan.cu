// Fused TTT-Linear (Test-Time Training) Scan Kernel
//
// TTT uses an inner linear model (weight matrix W) as hidden state,
// updated at each timestep via self-supervised gradient descent:
//
//   pred_t = W_{t-1} @ k_t                    — inner model forward
//   pred_normed = LayerNorm(pred_t)            — stabilize predictions
//   error_t = pred_normed - v_t                — reconstruction error
//   grad_W = eta_t * error_t @ k_t^T           — scaled gradient
//   W_t = W_{t-1} - grad_W                    — weight update
//   o_t = W_t @ q_t                           — output from updated model
//
// The LayerNorm uses learned gamma/beta parameters.
// eta_t = sigmoid(eta_raw) / inner_size (per-element learning rate gate).
//
// Thread layout: one thread per (batch, i, j) element of the output,
// but we organize as one thread per (batch, output_dim_i) since we need
// reduction over inner_size for the matmuls.
//
// Inputs:
//   q:     [batch, seq_len, inner_size]  — query projections
//   k:     [batch, seq_len, inner_size]  — key projections
//   v:     [batch, seq_len, inner_size]  — value (target) projections
//   eta:   [batch, seq_len, inner_size]  — post-sigmoid learning rate / inner_size
//   w0:    [batch, inner_size, inner_size] — initial weight matrix
//   ln_g:  [inner_size]                  — LayerNorm gamma
//   ln_b:  [inner_size]                  — LayerNorm beta
//
// Output:
//   out:   [batch, seq_len, inner_size]  — output hidden states

#include <cuda_runtime.h>

constexpr float TTT_LN_EPS = 1.0e-6f;

// Each thread block handles one batch element.
// Thread i handles output dimension i.
// W is stored in registers: W[i][j] for j in 0..inner_size-1.
// Since inner_size can be up to 64 (typical), each thread holds inner_size floats
// for its row of W.
//
// For inner_size=64, that's 64 floats = 256 bytes per thread.
// With 64 threads per block, that's 16KB registers — fits fine.

// Max inner_size we support in registers
#define TTT_MAX_INNER 128

__global__ void fused_ttt_scan_kernel(
    const float* __restrict__ q,      // [B, T, D]
    const float* __restrict__ k,      // [B, T, D]
    const float* __restrict__ v,      // [B, T, D]
    const float* __restrict__ eta,    // [B, T, D]
    const float* __restrict__ w0,     // [B, D, D]
    const float* __restrict__ ln_g,   // [D]
    const float* __restrict__ ln_b,   // [D]
    float* __restrict__ output,       // [B, T, D]
    int batch, int seq_len, int inner_size
) {
    int b = blockIdx.x;
    int i = threadIdx.x;  // output dimension index

    if (b >= batch || i >= inner_size) return;

    // Shared memory for:
    // - k_shared[inner_size]: key vector for current timestep
    // - pred_shared[inner_size]: predictions for LayerNorm reduction
    // - reduce_shared[2]: mean and variance for LayerNorm
    extern __shared__ float shared_mem[];
    float* k_shared = shared_mem;                          // [inner_size]
    float* pred_shared = shared_mem + inner_size;          // [inner_size]
    float* reduce_shared = shared_mem + 2 * inner_size;    // [2]

    // Load W[i][j] into registers (row i of weight matrix)
    float W_row[TTT_MAX_INNER];
    int w0_base = b * inner_size * inner_size + i * inner_size;
    for (int j = 0; j < inner_size; j++) {
        W_row[j] = w0[w0_base + j];
    }

    for (int t = 0; t < seq_len; t++) {
        int tk_idx = b * seq_len * inner_size + t * inner_size;

        // Load k_t into shared memory
        k_shared[i] = k[tk_idx + i];
        __syncthreads();

        // Step 1: pred_i = W[i,:] @ k  (dot product of row i with k)
        float pred_i = 0.0f;
        for (int j = 0; j < inner_size; j++) {
            pred_i += W_row[j] * k_shared[j];
        }

        // Store pred for LayerNorm reduction
        pred_shared[i] = pred_i;
        __syncthreads();

        // Step 2: LayerNorm - compute mean and variance
        // Thread 0 does the reduction (inner_size is small, typically 64)
        if (i == 0) {
            float sum = 0.0f;
            for (int j = 0; j < inner_size; j++) {
                sum += pred_shared[j];
            }
            float mean = sum / inner_size;
            reduce_shared[0] = mean;

            float var_sum = 0.0f;
            for (int j = 0; j < inner_size; j++) {
                float diff = pred_shared[j] - mean;
                var_sum += diff * diff;
            }
            reduce_shared[1] = var_sum / inner_size;
        }
        __syncthreads();

        float mean = reduce_shared[0];
        float var = reduce_shared[1];
        float inv_std = rsqrtf(var + TTT_LN_EPS);

        // Apply LayerNorm: pred_normed = gamma * (pred - mean) / std + beta
        float pred_normed = ln_g[i] * (pred_i - mean) * inv_std + ln_b[i];

        // Step 3: error = pred_normed - v
        float v_i = v[tk_idx + i];
        float error_i = pred_normed - v_i;

        // Step 4: eta-scaled error
        float eta_i = eta[tk_idx + i];
        float scaled_error_i = eta_i * error_i;

        // Step 5: Update W[i,:] -= scaled_error_i * k[:]
        // grad_W[i][j] = scaled_error_i * k[j]
        // W[i][j] -= grad_W[i][j]
        for (int j = 0; j < inner_size; j++) {
            W_row[j] -= scaled_error_i * k_shared[j];
        }

        // Step 6: Compute output o_i = W_updated[i,:] @ q
        float q_i_val;
        // Reuse k_shared to load q (sync first to ensure k reads are done)
        __syncthreads();
        k_shared[i] = q[tk_idx + i];
        __syncthreads();

        float o_i = 0.0f;
        for (int j = 0; j < inner_size; j++) {
            o_i += W_row[j] * k_shared[j];  // k_shared now holds q
        }

        output[b * seq_len * inner_size + t * inner_size + i] = o_i;
        __syncthreads();
    }
}

// ============================================================================
// Standalone launch wrapper
// ============================================================================

#ifndef EXLA_FFI

extern "C" {

int fused_ttt_scan_launch(
    cudaStream_t stream,
    const float* q, const float* k, const float* v,
    const float* eta, const float* w0,
    const float* ln_g, const float* ln_b,
    float* output,
    int batch, int seq_len, int inner_size
) {
    int threads_per_block = inner_size;  // one thread per output dim
    dim3 grid(batch);
    dim3 block(threads_per_block);
    // Shared memory: k_shared[D] + pred_shared[D] + reduce_shared[2]
    size_t smem_bytes = (2 * inner_size + 2) * sizeof(float);

    fused_ttt_scan_kernel<<<grid, block, smem_bytes, stream>>>(
        q, k, v, eta, w0, ln_g, ln_b, output,
        batch, seq_len, inner_size
    );

    return (int)cudaGetLastError();
}

}  // extern "C"

#endif

// ============================================================================
// XLA FFI integration
// ============================================================================

#ifdef EXLA_FFI

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

ffi::Error fused_ttt_scan_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> q,       // [B, T, D]
    ffi::Buffer<ffi::F32> k,       // [B, T, D]
    ffi::Buffer<ffi::F32> v,       // [B, T, D]
    ffi::Buffer<ffi::F32> eta,     // [B, T, D]
    ffi::Buffer<ffi::F32> w0,      // [B, D, D]
    ffi::Buffer<ffi::F32> ln_g,    // [D]
    ffi::Buffer<ffi::F32> ln_b,    // [D]
    ffi::ResultBuffer<ffi::F32> output  // [B, T, D]
) {
    auto q_dims = q.dimensions();
    int batch      = static_cast<int>(q_dims[0]);
    int seq_len    = static_cast<int>(q_dims[1]);
    int inner_size = static_cast<int>(q_dims[2]);

    if (inner_size > TTT_MAX_INNER) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                         "TTT inner_size exceeds max supported (128)");
    }

    int threads_per_block = inner_size;
    dim3 grid(batch);
    dim3 block(threads_per_block);
    size_t smem_bytes = (2 * inner_size + 2) * sizeof(float);

    fused_ttt_scan_kernel<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const float*>(q.untyped_data()),
        reinterpret_cast<const float*>(k.untyped_data()),
        reinterpret_cast<const float*>(v.untyped_data()),
        reinterpret_cast<const float*>(eta.untyped_data()),
        reinterpret_cast<const float*>(w0.untyped_data()),
        reinterpret_cast<const float*>(ln_g.untyped_data()),
        reinterpret_cast<const float*>(ln_b.untyped_data()),
        reinterpret_cast<float*>(output->untyped_data()),
        batch, seq_len, inner_size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    fused_ttt_scan, fused_ttt_scan_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()   // q
        .Arg<ffi::Buffer<ffi::F32>>()   // k
        .Arg<ffi::Buffer<ffi::F32>>()   // v
        .Arg<ffi::Buffer<ffi::F32>>()   // eta
        .Arg<ffi::Buffer<ffi::F32>>()   // w0
        .Arg<ffi::Buffer<ffi::F32>>()   // ln_g
        .Arg<ffi::Buffer<ffi::F32>>()   // ln_b
        .Ret<ffi::Buffer<ffi::F32>>()   // output
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_ttt_scan_f32", "CUDA", fused_ttt_scan);

#endif  // EXLA_FFI
