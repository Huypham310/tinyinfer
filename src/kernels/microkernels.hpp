#pragma once

// GEMM micro-kernels: C[mr x nr] += A panel (kc x mr) * B panel (kc x nr), packed layouts as
// written by gemm.cpp. Each ISA lives in its own .cpp compiled with its own -m flags, so the
// rest of the library stays baseline x86-64 and runs anywhere; gemm.cpp picks one at runtime.

namespace tinyinfer::kernels::detail {

/// `h` x `w` (<= mr x nr) is the valid part of the tile at the matrix edge.
using MicroKernel = void (*)(int kc, const float* a, const float* b, float* c, int ldc, int h,
                             int w);

struct KernelInfo {
    const char* name;
    int mr;
    int nr;
    MicroKernel fn;
};

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
