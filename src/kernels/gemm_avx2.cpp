// Compiled with -mavx2 -mfma (see CMakeLists.txt); only called when CPUID says AVX2 + FMA.

#include <immintrin.h>

#include <chrono>

#include "microkernels.hpp"

namespace tinyinfer::kernels::detail {

namespace {

constexpr int MR = 6;
constexpr int NR = 16;

/// 6 x 16 tile = 12 ymm accumulators, + 2 for the B row and 1 broadcast of A: 15 of the 16
/// ymm registers. Per k step: 2 loads, 6 broadcasts, 12 FMAs.
///
/// The accumulators are twelve named variables on purpose. Written as `__m256 acc[6][2]`, GCC
/// kept the array in memory (the edge path stores it) and wrote all 12 registers back to the
/// stack on *every* k step; the kernel ran at half the FMA peak, bound by the store port.
void kernel_6x16(int kc, const float* a, const float* b, float* c, int ldc, int h, int w) {
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps(), c41 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps(), c51 = _mm256_setzero_ps();
    for (int p = 0; p < kc; ++p) {
        const __m256 b0 = _mm256_loadu_ps(b);
        const __m256 b1 = _mm256_loadu_ps(b + 8);
        __m256 ai = _mm256_broadcast_ss(a + 0);
        c00 = _mm256_fmadd_ps(ai, b0, c00);
        c01 = _mm256_fmadd_ps(ai, b1, c01);
        ai = _mm256_broadcast_ss(a + 1);
        c10 = _mm256_fmadd_ps(ai, b0, c10);
        c11 = _mm256_fmadd_ps(ai, b1, c11);
        ai = _mm256_broadcast_ss(a + 2);
        c20 = _mm256_fmadd_ps(ai, b0, c20);
        c21 = _mm256_fmadd_ps(ai, b1, c21);
        ai = _mm256_broadcast_ss(a + 3);
        c30 = _mm256_fmadd_ps(ai, b0, c30);
        c31 = _mm256_fmadd_ps(ai, b1, c31);
        ai = _mm256_broadcast_ss(a + 4);
        c40 = _mm256_fmadd_ps(ai, b0, c40);
        c41 = _mm256_fmadd_ps(ai, b1, c41);
        ai = _mm256_broadcast_ss(a + 5);
        c50 = _mm256_fmadd_ps(ai, b0, c50);
        c51 = _mm256_fmadd_ps(ai, b1, c51);
        a += MR;
        b += NR;
    }
    // edge tiles go through a buffer; the full-tile case does the same adds straight into C
    alignas(32) float tmp[MR][NR] = {};
    const bool full = h == MR && w == NR;
    float* rows[MR];
    for (int i = 0; i < MR; ++i) rows[i] = full ? c + i * ldc : tmp[i];
    const __m256 acc[MR][2] = {{c00, c01}, {c10, c11}, {c20, c21},
                               {c30, c31}, {c40, c41}, {c50, c51}};
    for (int i = 0; i < MR; ++i) {
        _mm256_storeu_ps(rows[i], _mm256_add_ps(_mm256_loadu_ps(rows[i]), acc[i][0]));
        _mm256_storeu_ps(rows[i] + 8, _mm256_add_ps(_mm256_loadu_ps(rows[i] + 8), acc[i][1]));
    }
    if (!full) {
        for (int i = 0; i < h; ++i)
            for (int j = 0; j < w; ++j) c[i * ldc + j] += tmp[i][j];
    }
}

}  // namespace

KernelInfo avx2_kernel() { return {"avx2", MR, NR, kernel_6x16}; }

double peak_gflops_avx2(long iters) {
    // 10 independent accumulator chains: enough to cover FMA latency (4 cycles) x 2 ports
    __m256 x[10];
    for (int i = 0; i < 10; ++i) x[i] = _mm256_set1_ps(1.0F + 0.001F * static_cast<float>(i));
    const __m256 m = _mm256_set1_ps(0.999999F), k = _mm256_set1_ps(1e-7F);
    const auto t0 = std::chrono::steady_clock::now();
    for (long it = 0; it < iters; ++it) {
        for (auto& v : x) v = _mm256_fmadd_ps(v, m, k);
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    float sink = 0;
    for (auto& v : x) sink += _mm256_cvtss_f32(v);
    // the sink keeps the loop alive; it's never exactly -1
    return sink == -1.0F ? 0.0 : 10.0 * 8 * 2 * static_cast<double>(iters) / s / 1e9;
}

}  // namespace tinyinfer::kernels::detail
