#pragma once

// GEMM micro-kernels: C[mr x nr] += A panel (kc x mr) * B panel (kc x nr), packed layouts as
// written by gemm.cpp. Each ISA lives in its own .cpp compiled with its own -m flags, so the
// rest of the library stays baseline x86-64 and runs anywhere; gemm.cpp picks one at runtime.

namespace tinyinfer::kernels::detail {

/// `h` x `w` (<= mr x nr) is the valid part of the tile at the matrix edge.
using MicroKernel = void (*)(int kc, const float* a, const float* b, float* c, int ldc, int h,
                             int w);

/// Skinny GEMM for a handful of columns: C[m x n] += A[m x k] * Bt^T, where Bt is op(B) stored
/// transposed (n x k, row-major). A is read in place, no packing: when n is tiny every weight
/// is used n times only, so packing it would cost more than the multiply.
using SkinnyKernel = void (*)(int m, int n, int k, const float* a, int lda, const float* bt,
                              int ldbt, float* c, int ldc);

struct KernelInfo {
    const char* name;
    int mr;
    int nr;
    MicroKernel fn;
    SkinnyKernel skinny;
};

void skinny_scalar(int m, int n, int k, const float* a, int lda, const float* bt, int ldbt,
                   float* c, int ldc);

KernelInfo scalar_kernel();
#if defined(__x86_64__) || defined(_M_X64)
KernelInfo avx2_kernel();
KernelInfo avx512_kernel();
#endif

/// FMA throughput microbenchmark for the "% of peak" numbers: GFLOP/s of a loop of independent
/// register-only FMAs, one core.
double peak_gflops_avx2(long iters);
double peak_gflops_avx512(long iters);

}  // namespace tinyinfer::kernels::detail
