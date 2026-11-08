// Compiled with -mavx512f -mfma (see CMakeLists.txt); only called when CPUID + XCR0 allow it.

#include <immintrin.h>

#include <chrono>

#include "microkernels.hpp"

namespace tinyinfer::kernels::detail {

namespace {

constexpr int MR = 12;
constexpr int NR = 32;

/// 12 x 32 tile = 24 zmm accumulators + 2 for B + 1 broadcast = 27 of 32 registers.
/// Twice the columns of the AVX2 kernel per register, and a taller tile so each B load is
/// reused 12 times instead of 6.
void kernel_12x32(int kc, const float* a, const float* b, float* c, int ldc, int h, int w) {
    __m512 acc[MR][2];
    for (auto& row : acc) row[0] = row[1] = _mm512_setzero_ps();
    for (int p = 0; p < kc; ++p) {
        const __m512 b0 = _mm512_loadu_ps(b);
        const __m512 b1 = _mm512_loadu_ps(b + 16);
        for (int i = 0; i < MR; ++i) {
            const __m512 ai = _mm512_set1_ps(a[i]);
            acc[i][0] = _mm512_fmadd_ps(ai, b0, acc[i][0]);
            acc[i][1] = _mm512_fmadd_ps(ai, b1, acc[i][1]);
        }
        a += MR;
        b += NR;
    }
    if (h == MR && w == NR) {
        for (int i = 0; i < MR; ++i) {
            float* ci = c + i * ldc;
            _mm512_storeu_ps(ci, _mm512_add_ps(_mm512_loadu_ps(ci), acc[i][0]));
            _mm512_storeu_ps(ci + 16, _mm512_add_ps(_mm512_loadu_ps(ci + 16), acc[i][1]));
        }
        return;
    }
    // edge tile: masked loads/stores would avoid the copy, but edges are a small share of work
    alignas(64) float tmp[MR][NR];
    for (int i = 0; i < MR; ++i) {
        _mm512_store_ps(tmp[i], acc[i][0]);
        _mm512_store_ps(tmp[i] + 16, acc[i][1]);
    }
    for (int i = 0; i < h; ++i)
        for (int j = 0; j < w; ++j) c[i * ldc + j] += tmp[i][j];
}

}  // namespace

KernelInfo avx512_kernel() { return {"avx512", MR, NR, kernel_12x32}; }

double peak_gflops_avx512(long iters) {
    __m512 x[12];
    for (int i = 0; i < 12; ++i) x[i] = _mm512_set1_ps(1.0F + 0.001F * static_cast<float>(i));
    const __m512 m = _mm512_set1_ps(0.999999F), k = _mm512_set1_ps(1e-7F);
    const auto t0 = std::chrono::steady_clock::now();
    for (long it = 0; it < iters; ++it) {
        for (auto& v : x) v = _mm512_fmadd_ps(v, m, k);
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    float sink = 0;
    for (auto& v : x) sink += _mm512_cvtss_f32(v);
    return sink == -1.0F ? 0.0 : 12.0 * 16 * 2 * static_cast<double>(iters) / s / 1e9;
}

}  // namespace tinyinfer::kernels::detail
