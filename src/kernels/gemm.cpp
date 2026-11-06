#include <algorithm>
#include <vector>

#include "tinyinfer/kernels.hpp"

namespace tinyinfer::kernels {

void gemm_naive(int m, int n, int k, const float* a, int lda, const float* b, int ldb, bool trans_b,
                float* c, int ldc, bool accumulate) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            // accumulate in double: this is the reference the fast kernel is compared against
            double acc = accumulate ? c[i * ldc + j] : 0.0;
            for (int p = 0; p < k; ++p) {
                const float bv = trans_b ? b[j * ldb + p] : b[p * ldb + j];
                acc += static_cast<double>(a[i * lda + p]) * bv;
            }
            c[i * ldc + j] = static_cast<float>(acc);
        }
    }
}

namespace {

// Block sizes picked for a ~32-48 KB L1d / 1 MB L2 core (Zen 5 here): a KC x NR sliver of B
// (256 x 16 floats = 16 KB) stays in L1 while the micro-kernel walks down A, and an MC x KC
// block of A (64 x 256 = 64 KB) stays in L2.
// TODO: tune MC/KC/NC per CPU instead of hard-coding (đọc cache size bằng CPUID?)
constexpr int MR = 4;
constexpr int NR = 16;
constexpr int MC = 64;
constexpr int KC = 256;
constexpr int NC = 512;

/// Copies a kc x nc block of op(B) into NR-wide column panels: panel j holds rows 0..kc-1 of
/// columns [j*NR, j*NR + NR), zero-padded past nc, so the micro-kernel reads B sequentially.
void pack_b(int kc, int nc, const float* b, int ldb, bool trans_b, int p0, int j0, float* out) {
    for (int jp = 0; jp < nc; jp += NR) {
        const int w = std::min(NR, nc - jp);
        for (int p = 0; p < kc; ++p) {
            for (int jj = 0; jj < NR; ++jj) {
                float v = 0.0F;
                if (jj < w) {
                    const int col = j0 + jp + jj;
                    const int row = p0 + p;
                    v = trans_b ? b[col * ldb + row] : b[row * ldb + col];
                }
                *out++ = v;
            }
        }
    }
}

/// Copies an mc x kc block of A into MR-tall row panels (column-major inside a panel).
void pack_a(int mc, int kc, const float* a, int lda, float* out) {
    for (int ip = 0; ip < mc; ip += MR) {
        const int h = std::min(MR, mc - ip);
        for (int p = 0; p < kc; ++p) {
            for (int ii = 0; ii < MR; ++ii) {
                *out++ = ii < h ? a[(ip + ii) * lda + p] : 0.0F;
            }
        }
    }
}

/// C[MR x NR] += A panel (kc x MR) * B panel (kc x NR). The accumulator tile lives in
/// registers; with -O3 GCC turns the jj loop into SSE / AVX vector FMAs on its own.
void micro_kernel(int kc, const float* ap, const float* bp, float* c, int ldc, int h, int w) {
    float acc[MR][NR] = {};
    for (int p = 0; p < kc; ++p) {
        const float* bv = bp + p * NR;
        for (int ii = 0; ii < MR; ++ii) {
            const float av = ap[p * MR + ii];
            for (int jj = 0; jj < NR; ++jj) acc[ii][jj] += av * bv[jj];
        }
    }
    for (int ii = 0; ii < h; ++ii) {
        for (int jj = 0; jj < w; ++jj) c[ii * ldc + jj] += acc[ii][jj];
    }
}

}  // namespace

void gemm(int m, int n, int k, const float* a, int lda, const float* b, int ldb, bool trans_b,
          float* c, int ldc, bool accumulate) {
    if (!accumulate) {
        for (int i = 0; i < m; ++i) std::fill(c + i * ldc, c + i * ldc + n, 0.0F);
    }
    if (m == 0 || n == 0 || k == 0) return;

    // packing buffers are reused across calls on this thread: no allocation in steady state
    thread_local std::vector<float> bpack, apack;
    bpack.resize(static_cast<std::size_t>(KC) * ((NC + NR - 1) / NR) * NR);
    apack.resize(static_cast<std::size_t>(KC) * ((MC + MR - 1) / MR) * MR);

    for (int j0 = 0; j0 < n; j0 += NC) {
        const int nc = std::min(NC, n - j0);
        for (int p0 = 0; p0 < k; p0 += KC) {
            const int kc = std::min(KC, k - p0);
            pack_b(kc, nc, b, ldb, trans_b, p0, j0, bpack.data());
            for (int i0 = 0; i0 < m; i0 += MC) {
                const int mc = std::min(MC, m - i0);
                pack_a(mc, kc, a + i0 * lda + p0, lda, apack.data());
                for (int jp = 0; jp < nc; jp += NR) {
                    const float* bp = bpack.data() + static_cast<std::size_t>(jp) * kc;
                    for (int ip = 0; ip < mc; ip += MR) {
                        const float* ap = apack.data() + static_cast<std::size_t>(ip) * kc;
                        micro_kernel(kc, ap, bp, c + (i0 + ip) * ldc + j0 + jp, ldc,
                                     std::min(MR, mc - ip), std::min(NR, nc - jp));
                    }
                }
            }
        }
    }
}

}  // namespace tinyinfer::kernels
