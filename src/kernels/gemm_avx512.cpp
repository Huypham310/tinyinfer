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

/// R rows of A against NJ columns of Bt: R * NJ zmm accumulators (16 at most), each one a
/// 16-wide partial dot product that gets reduced at the end. The template bounds are
/// compile-time so GCC fully unrolls and keeps every accumulator in a register.
template <int R, int NJ>
void skinny_block(int k, const float* a, int lda, const float* bt, int ldbt, float* c, int ldc) {
    __m512 acc[R][NJ];
    for (int r = 0; r < R; ++r)
        for (int j = 0; j < NJ; ++j) acc[r][j] = _mm512_setzero_ps();
    int p = 0;
    for (; p + 16 <= k; p += 16) {
        __m512 av[R];
        for (int r = 0; r < R; ++r) av[r] = _mm512_loadu_ps(a + r * lda + p);
        for (int j = 0; j < NJ; ++j) {
            const __m512 bv = _mm512_loadu_ps(bt + j * ldbt + p);
            for (int r = 0; r < R; ++r) acc[r][j] = _mm512_fmadd_ps(av[r], bv, acc[r][j]);
        }
    }
    if (p < k) {
        // k tail: masked loads read nothing past the end of the row
        const __mmask16 m = static_cast<__mmask16>((1U << (k - p)) - 1U);
        __m512 av[R];
        for (int r = 0; r < R; ++r) av[r] = _mm512_maskz_loadu_ps(m, a + r * lda + p);
        for (int j = 0; j < NJ; ++j) {
            const __m512 bv = _mm512_maskz_loadu_ps(m, bt + j * ldbt + p);
            for (int r = 0; r < R; ++r) acc[r][j] = _mm512_fmadd_ps(av[r], bv, acc[r][j]);
        }
    }
    for (int r = 0; r < R; ++r)
        for (int j = 0; j < NJ; ++j) c[r * ldc + j] += _mm512_reduce_add_ps(acc[r][j]);
}

template <int R>
void skinny_rows(int nj, int k, const float* a, int lda, const float* bt, int ldbt, float* c,
                 int ldc) {
    switch (nj) {
        case 1:
            skinny_block<R, 1>(k, a, lda, bt, ldbt, c, ldc);
            break;
        case 2:
            skinny_block<R, 2>(k, a, lda, bt, ldbt, c, ldc);
            break;
        case 3:
            skinny_block<R, 3>(k, a, lda, bt, ldbt, c, ldc);
            break;
        default:
            skinny_block<R, 4>(k, a, lda, bt, ldbt, c, ldc);
            break;
    }
}

void skinny_avx512(int m, int n, int k, const float* a, int lda, const float* bt, int ldbt,
                   float* c, int ldc) {
    // columns in groups of 4: n = 8 reads each row of A twice, still far cheaper than packing
    for (int j0 = 0; j0 < n; j0 += 4) {
        const int nj = n - j0 < 4 ? n - j0 : 4;
        int i = 0;
        for (; i + 4 <= m; i += 4) {
            skinny_rows<4>(nj, k, a + i * lda, lda, bt + j0 * ldbt, ldbt, c + i * ldc + j0, ldc);
        }
        for (; i < m; ++i) {
            skinny_rows<1>(nj, k, a + i * lda, lda, bt + j0 * ldbt, ldbt, c + i * ldc + j0, ldc);
        }
    }
}

}  // namespace

KernelInfo avx512_kernel() { return {"avx512", MR, NR, kernel_12x32, skinny_avx512}; }

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
