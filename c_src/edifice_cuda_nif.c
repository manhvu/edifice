/*
 * Edifice CUDA NIF Bridge
 *
 * Minimal C NIF that dlopen's libedifice_cuda_kernels.so and exposes
 * fused scan kernels as Erlang NIF functions. Takes device pointer
 * integers (from Nx.to_pointer) and dimensions, launches the fused
 * kernel, and returns the output pointer integer.
 *
 * Marked ERL_NIF_DIRTY_JOB_IO_BOUND so GPU blocking doesn't stall
 * the BEAM scheduler.
 */

#define _GNU_SOURCE  /* for dladdr / Dl_info */

#include <erl_nif.h>
#include <dlfcn.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>

/* CUDA runtime types — avoid pulling in full cuda_runtime.h */
typedef void* cudaStream_t;
typedef int cudaError_t;

/* Function pointer types matching the C-linkage launch wrappers */
typedef int (*mingru_launch_fn)(
    cudaStream_t stream,
    const float* gates,
    const float* candidates,
    const float* h0,
    float* output,
    int batch, int seq_len, int hidden
);

typedef int (*minlstm_launch_fn)(
    cudaStream_t stream,
    const float* forget_gates,
    const float* input_gates,
    const float* candidates,
    const float* h0,
    float* output,
    int batch, int seq_len, int hidden
);

/* cudaMalloc / cudaFree — resolved from libcudart */
typedef cudaError_t (*cuda_malloc_fn)(void** devPtr, size_t size);
typedef cudaError_t (*cuda_free_fn)(void* devPtr);
typedef cudaError_t (*cuda_device_synchronize_fn)(void);

/* Resolved function pointers (set on NIF load) */
static mingru_launch_fn  s_mingru_launch  = NULL;
static minlstm_launch_fn s_minlstm_launch = NULL;
static cuda_malloc_fn    s_cuda_malloc    = NULL;
static cuda_free_fn      s_cuda_free      = NULL;
static cuda_device_synchronize_fn s_cuda_sync = NULL;

static void* s_kernels_handle = NULL;
static void* s_cudart_handle  = NULL;

/* ========================================================================== */
/* Helpers                                                                    */
/* ========================================================================== */

static ERL_NIF_TERM make_error(ErlNifEnv* env, const char* reason) {
    return enif_make_tuple2(env,
        enif_make_atom(env, "error"),
        enif_make_string(env, reason, ERL_NIF_LATIN1));
}

static ERL_NIF_TERM make_ok_pointer(ErlNifEnv* env, uint64_t ptr) {
    return enif_make_tuple2(env,
        enif_make_atom(env, "ok"),
        enif_make_uint64(env, ptr));
}

/* ========================================================================== */
/* NIF: fused_mingru_scan                                                     */
/* ========================================================================== */

/*
 * fused_mingru_scan(gates_ptr, candidates_ptr, h0_ptr, batch, seq_len, hidden)
 *   -> {:ok, output_ptr} | {:error, reason}
 *
 * All pointer args are uint64 device pointer addresses.
 */
static ERL_NIF_TERM nif_fused_mingru_scan(
    ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    uint64_t gates_ptr, cand_ptr, h0_ptr;
    int batch, seq_len, hidden;

    if (!s_mingru_launch)
        return make_error(env, "kernel library not loaded");

    if (!enif_get_uint64(env, argv[0], &gates_ptr) ||
        !enif_get_uint64(env, argv[1], &cand_ptr) ||
        !enif_get_uint64(env, argv[2], &h0_ptr) ||
        !enif_get_int(env, argv[3], &batch) ||
        !enif_get_int(env, argv[4], &seq_len) ||
        !enif_get_int(env, argv[5], &hidden))
    {
        return enif_make_badarg(env);
    }

    /* Allocate output buffer on GPU: [batch, seq_len, hidden] * sizeof(float) */
    size_t out_bytes = (size_t)batch * seq_len * hidden * sizeof(float);
    void* out_dev = NULL;

    cudaError_t err = s_cuda_malloc(&out_dev, out_bytes);
    if (err != 0) {
        return make_error(env, "cudaMalloc failed for output buffer");
    }

    /* Launch the fused kernel on the default stream */
    int launch_err = s_mingru_launch(
        NULL,  /* default stream */
        (const float*)(uintptr_t)gates_ptr,
        (const float*)(uintptr_t)cand_ptr,
        (const float*)(uintptr_t)h0_ptr,
        (float*)out_dev,
        batch, seq_len, hidden
    );

    if (launch_err != 0) {
        s_cuda_free(out_dev);
        return make_error(env, "kernel launch failed");
    }

    /* Synchronize — blocks this dirty scheduler thread until GPU finishes */
    err = s_cuda_sync();
    if (err != 0) {
        s_cuda_free(out_dev);
        return make_error(env, "cudaDeviceSynchronize failed");
    }

    return make_ok_pointer(env, (uint64_t)(uintptr_t)out_dev);
}

/* ========================================================================== */
/* NIF: fused_minlstm_scan                                                    */
/* ========================================================================== */

/*
 * fused_minlstm_scan(forget_ptr, input_ptr, cand_ptr, h0_ptr,
 *                    batch, seq_len, hidden)
 *   -> {:ok, output_ptr} | {:error, reason}
 */
static ERL_NIF_TERM nif_fused_minlstm_scan(
    ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    uint64_t forget_ptr, input_ptr, cand_ptr, h0_ptr;
    int batch, seq_len, hidden;

    if (!s_minlstm_launch)
        return make_error(env, "kernel library not loaded");

    if (!enif_get_uint64(env, argv[0], &forget_ptr) ||
        !enif_get_uint64(env, argv[1], &input_ptr) ||
        !enif_get_uint64(env, argv[2], &cand_ptr) ||
        !enif_get_uint64(env, argv[3], &h0_ptr) ||
        !enif_get_int(env, argv[4], &batch) ||
        !enif_get_int(env, argv[5], &seq_len) ||
        !enif_get_int(env, argv[6], &hidden))
    {
        return enif_make_badarg(env);
    }

    size_t out_bytes = (size_t)batch * seq_len * hidden * sizeof(float);
    void* out_dev = NULL;

    cudaError_t err = s_cuda_malloc(&out_dev, out_bytes);
    if (err != 0) {
        return make_error(env, "cudaMalloc failed for output buffer");
    }

    int launch_err = s_minlstm_launch(
        NULL,
        (const float*)(uintptr_t)forget_ptr,
        (const float*)(uintptr_t)input_ptr,
        (const float*)(uintptr_t)cand_ptr,
        (const float*)(uintptr_t)h0_ptr,
        (float*)out_dev,
        batch, seq_len, hidden
    );

    if (launch_err != 0) {
        s_cuda_free(out_dev);
        return make_error(env, "kernel launch failed");
    }

    err = s_cuda_sync();
    if (err != 0) {
        s_cuda_free(out_dev);
        return make_error(env, "cudaDeviceSynchronize failed");
    }

    return make_ok_pointer(env, (uint64_t)(uintptr_t)out_dev);
}

/* ========================================================================== */
/* NIF: cuda_free                                                             */
/* ========================================================================== */

/*
 * cuda_free(device_ptr) -> :ok | {:error, reason}
 *
 * Frees a device pointer allocated by a fused scan NIF.
 * Needed if Nx.from_pointer doesn't take ownership.
 */
static ERL_NIF_TERM nif_cuda_free(
    ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    uint64_t ptr;

    if (!s_cuda_free)
        return make_error(env, "cudart not loaded");

    if (!enif_get_uint64(env, argv[0], &ptr))
        return enif_make_badarg(env);

    cudaError_t err = s_cuda_free((void*)(uintptr_t)ptr);
    if (err != 0)
        return make_error(env, "cudaFree failed");

    return enif_make_atom(env, "ok");
}

/* ========================================================================== */
/* NIF Load — resolve all symbols                                             */
/* ========================================================================== */

static int nif_load(ErlNifEnv* env, void** priv_data, ERL_NIF_TERM load_info) {
    (void)priv_data;
    (void)load_info;

    /* Resolve path to kernel shared library.
     * The NIF .so lives at priv/libedifice_cuda_nif.so,
     * the kernel .so lives at priv/cuda/libedifice_cuda_kernels.so.
     * We use the NIF path to derive the kernel path. */
    char kernels_path[512];

    /* Try to load libcudart first — may already be in LD_LIBRARY_PATH */
    s_cudart_handle = dlopen("libcudart.so", RTLD_NOW | RTLD_GLOBAL);
    if (!s_cudart_handle) {
        /* Try versioned name */
        s_cudart_handle = dlopen("libcudart.so.12", RTLD_NOW | RTLD_GLOBAL);
    }
    if (!s_cudart_handle) {
        /* CUDA not available — NIF loads but functions return errors */
        return 0;
    }

    s_cuda_malloc = (cuda_malloc_fn)dlsym(s_cudart_handle, "cudaMalloc");
    s_cuda_free   = (cuda_free_fn)dlsym(s_cudart_handle, "cudaFree");
    s_cuda_sync   = (cuda_device_synchronize_fn)dlsym(s_cudart_handle, "cudaDeviceSynchronize");

    if (!s_cuda_malloc || !s_cuda_free || !s_cuda_sync) {
        /* cudart symbols missing — unusual but handle gracefully */
        return 0;
    }

    /* Build path to kernel library relative to priv dir.
     * We get priv_dir from the Elixir side via the load path,
     * but here we use a fixed relative path from the NIF .so. */

    /* The NIF is loaded from priv/libedifice_cuda_nif — the .so is in priv/.
     * The kernels are in priv/cuda/. We use dlopen with a relative path
     * from the process working directory, or the Elixir side can pass the
     * path. For robustness, we try the known path pattern. */

    /* Use /proc/self/maps to find where our NIF is loaded, then derive path */
    /* Simpler approach: the Elixir side sets priv_dir, we look relative to it */
    /* Simplest: try common locations */

    /* First try: relative to CWD (mix project root) */
    s_kernels_handle = dlopen("priv/cuda/libedifice_cuda_kernels.so", RTLD_NOW);
    if (!s_kernels_handle) {
        /* Second try: find via Dl_info on our own symbol */
        Dl_info info;
        if (dladdr((void*)nif_load, &info) && info.dli_fname) {
            /* info.dli_fname = ".../priv/libedifice_cuda_nif.so" */
            const char* fname = info.dli_fname;
            const char* last_slash = strrchr(fname, '/');
            if (last_slash) {
                size_t dir_len = (size_t)(last_slash - fname);
                if (dir_len + 40 < sizeof(kernels_path)) {
                    memcpy(kernels_path, fname, dir_len);
                    strcpy(kernels_path + dir_len, "/cuda/libedifice_cuda_kernels.so");
                    s_kernels_handle = dlopen(kernels_path, RTLD_NOW);
                }
            }
        }
    }

    if (!s_kernels_handle) {
        /* Kernel library not found — NIF loads but kernel functions return errors */
        return 0;
    }

    s_mingru_launch = (mingru_launch_fn)dlsym(
        s_kernels_handle, "fused_mingru_scan_launch");
    s_minlstm_launch = (minlstm_launch_fn)dlsym(
        s_kernels_handle, "fused_minlstm_scan_launch");

    /* If symbols are missing, the individual NIF functions will return errors */
    return 0;
}

static void nif_unload(ErlNifEnv* env, void* priv_data) {
    (void)env;
    (void)priv_data;

    if (s_kernels_handle) {
        dlclose(s_kernels_handle);
        s_kernels_handle = NULL;
    }
    if (s_cudart_handle) {
        dlclose(s_cudart_handle);
        s_cudart_handle = NULL;
    }
    s_mingru_launch  = NULL;
    s_minlstm_launch = NULL;
    s_cuda_malloc    = NULL;
    s_cuda_free      = NULL;
    s_cuda_sync      = NULL;
}

/* ========================================================================== */
/* NIF Registration                                                           */
/* ========================================================================== */

static ErlNifFunc nif_funcs[] = {
    {"fused_mingru_scan",  6, nif_fused_mingru_scan,  ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"fused_minlstm_scan", 7, nif_fused_minlstm_scan, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"cuda_free",          1, nif_cuda_free,           ERL_NIF_DIRTY_JOB_IO_BOUND}
};

ERL_NIF_INIT(Elixir.Edifice.CUDA.NIF, nif_funcs, nif_load, NULL, NULL, nif_unload)
