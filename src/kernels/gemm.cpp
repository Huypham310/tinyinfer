#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "microkernels.hpp"
#include "tinyinfer/cpu_features.hpp"
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

namespace detail {

namespace {

constexpr int SMR = 4;
constexpr int SNR = 16;

/// Portable fallback: plain C++ that GCC vectorizes with whatever the baseline ISA allows.
void scalar_4x16(int kc, const float* ap, const float* bp, float* c, int ldc, int h, int w) {
    float acc[SMR][SNR] = {};
    for (int p = 0; p < kc; ++p) {
        const float* bv = bp + p * SNR;
        for (int ii = 0; ii < SMR; ++ii) {
            const float av = ap[p * SMR + ii];
            for (int jj = 0; jj < SNR; ++jj) acc[ii][jj] += av * bv[jj];
        }
    }
    for (int ii = 0; ii < h; ++ii) {
        for (int jj = 0; jj < w; ++jj) c[ii * ldc + jj] += acc[ii][jj];
    }
}

}  // namespace

KernelInfo scalar_kernel() { return {"scalar", SMR, SNR, scalar_4x16}; }

}  // namespace detail

namespace {

using detail::KernelInfo;

// Block sizes for a ~48 KB L1d / 1 MB L2 core (Zen 5 here): a KC x NR sliver of B (256 x 32
// floats = 32 KB at most) stays in L1 while the micro-kernel walks down A, and an MC x KC
// block of A (96 x 256 = 96 KB) stays in L2. MC = 96 is a multiple of every kernel's MR.
// TODO: tune MC/KC/NC per CPU instead of hard-coding (đọc cache size bằng CPUID?)
constexpr int MC = 96;
constexpr int KC = 256;
constexpr int NC = 512;

/// Copies a kc x nc block of op(B) into nr-wide column panels: panel j holds rows 0..kc-1 of
/// columns [j*nr, j*nr + nr), zero-padded past nc, so the micro-kernel reads B sequentially.
void pack_b(int kc, int nc, int nr, const float* b, int ldb, bool trans_b, int p0, int j0,
            float* out) {
    for (int jp = 0; jp < nc; jp += nr) {
        const int w = std::min(nr, nc - jp);
        for (int p = 0; p < kc; ++p) {
            const int row = p0 + p;
            if (!trans_b && w == nr) {
                // contiguous row segment: one copy instead of nr strided reads
                std::memcpy(out, b + row * ldb + j0 + jp, sizeof(float) * nr);
                out += nr;
                continue;
            }
            for (int jj = 0; jj < nr; ++jj) {
                const int col = j0 + jp + jj;
                *out++ = jj < w ? (trans_b ? b[col * ldb + row] : b[row * ldb + col]) : 0.0F;
            }
        }
    }
}

/// Copies an mc x kc block of A into mr-tall row panels (column-major inside a panel).
void pack_a(int mc, int kc, int mr, const float* a, int lda, float* out) {
    for (int ip = 0; ip < mc; ip += mr) {
        const int h = std::min(mr, mc - ip);
        for (int p = 0; p < kc; ++p) {
            for (int ii = 0; ii < mr; ++ii) {
                *out++ = ii < h ? a[(ip + ii) * lda + p] : 0.0F;
            }
        }
    }
}

KernelInfo kernel_for(const std::string& isa) {
#if defined(__x86_64__) || defined(_M_X64)
    const auto& f = cpu_features();
    if (isa == "avx512" && f.avx512f && f.fma) return detail::avx512_kernel();
    if (isa == "avx2" && f.avx2 && f.fma) return detail::avx2_kernel();
#endif
    (void)isa;
    return detail::scalar_kernel();
}

KernelInfo best_kernel() {
    // TINYINFER_ISA=scalar|avx2|avx512 forces a path (benchmarks, debugging a bad kernel)
    if (const char* env = std::getenv("TINYINFER_ISA")) return kernel_for(env);
#if defined(__x86_64__) || defined(_M_X64)
    const auto& f = cpu_features();
    if (f.avx512f && f.fma) return detail::avx512_kernel();
    if (f.avx2 && f.fma) return detail::avx2_kernel();
#endif
    return detail::scalar_kernel();
}

std::atomic<const KernelInfo*>& active() {
    static const KernelInfo initial = best_kernel();
    static std::atomic<const KernelInfo*> ptr{&initial};
    return ptr;
}

}  // namespace

const char* gemm_isa() { return active().load(std::memory_order_acquire)->name; }

bool set_gemm_isa(const char* isa) {
    // one static slot per ISA, so the pointer handed out stays valid forever
    static const KernelInfo scalar = detail::scalar_kernel();
    if (std::strcmp(isa, "scalar") == 0) {
        active().store(&scalar, std::memory_order_release);
        return true;
    }
#if defined(__x86_64__) || defined(_M_X64)
    static const KernelInfo avx2 = kernel_for("avx2");
    static const KernelInfo avx512 = kernel_for("avx512");
    const KernelInfo* want = std::strcmp(isa, "avx2") == 0     ? &avx2
                             : std::strcmp(isa, "avx512") == 0 ? &avx512
                                                               : nullptr;
    // kernel_for falls back to scalar when the CPU lacks the ISA: refuse instead of lying
    if (want != nullptr && std::strcmp(want->name, isa) == 0) {
        active().store(want, std::memory_order_release);
        return true;
    }
#endif
    return false;
}

double fma_peak_gflops(const char* isa) {
#if defined(__x86_64__) || defined(_M_X64)
    const auto& f = cpu_features();
    // best of 3: the first run also pays for the core clocking up
    double best = 0.0;
    for (int r = 0; r < 3; ++r) {
        if (std::strcmp(isa, "avx512") == 0 && f.avx512f && f.fma) {
            best = std::max(best, detail::peak_gflops_avx512(20'000'000));
        } else if (std::strcmp(isa, "avx2") == 0 && f.avx2 && f.fma) {
            best = std::max(best, detail::peak_gflops_avx2(20'000'000));
        }
    }
    return best;
#else
    (void)isa;
    return 0.0;
#endif
}

void gemm(int m, int n, int k, const float* a, int lda, const float* b, int ldb, bool trans_b,
          float* c, int ldc, bool accumulate) {
    if (!accumulate) {
        for (int i = 0; i < m; ++i) std::fill(c + i * ldc, c + i * ldc + n, 0.0F);
    }
    if (m == 0 || n == 0 || k == 0) return;
    const KernelInfo& ker = *active().load(std::memory_order_acquire);
    const int mr = ker.mr, nr = ker.nr;

    // packing buffers are reused across calls on this thread: no allocation in steady state
    thread_local std::vector<float> bpack;
    bpack.resize(static_cast<std::size_t>(KC) * ((NC + nr - 1) / nr) * nr);
    float* bshared = bpack.data();
    const std::size_t apack_size = static_cast<std::size_t>(KC) * ((MC + mr - 1) / mr) * mr;
    const int m_blocks = (m + MC - 1) / MC;
    // threads only pay off once there's enough work per call (thread wake-up is ~µs)
    const bool parallel = static_cast<double>(m) * n * k > 4e6 && m_blocks > 1;
    (void)parallel;

#pragma omp parallel if (parallel)
    {
        for (int j0 = 0; j0 < n; j0 += NC) {
            const int nc = std::min(NC, n - j0);
            const int panels = (nc + nr - 1) / nr;
            for (int p0 = 0; p0 < k; p0 += KC) {
                const int kc = std::min(KC, k - p0);
                // B is packed once per (j0, p0), panels split across threads, then shared
                // read-only; the implicit barrier after the loop publishes it to everyone
#pragma omp for schedule(static)
                for (int panel = 0; panel < panels; ++panel) {
                    const int jp = panel * nr;
                    pack_b(kc, std::min(nr, nc - jp), nr, b, ldb, trans_b, p0, j0 + jp,
                           bshared + static_cast<std::size_t>(jp) * kc);
                }
                // dynamic: this laptop mixes fast and slow cores (Zen 5 / Zen 5c), a static
                // split leaves the fast ones idle waiting for the slow ones
#pragma omp for schedule(dynamic, 1)
                for (int blk = 0; blk < m_blocks; ++blk) {
                    // each thread packs its own A block: rows of C don't overlap between blocks
                    thread_local std::vector<float> apack;
                    apack.resize(apack_size);
                    const int i0 = blk * MC;
                    const int mc = std::min(MC, m - i0);
                    pack_a(mc, kc, mr, a + i0 * lda + p0, lda, apack.data());
                    for (int jp = 0; jp < nc; jp += nr) {
                        const float* bp = bshared + static_cast<std::size_t>(jp) * kc;
                        for (int ip = 0; ip < mc; ip += mr) {
                            const float* ap = apack.data() + static_cast<std::size_t>(ip) * kc;
                            ker.fn(kc, ap, bp, c + (i0 + ip) * ldc + j0 + jp, ldc,
                                   std::min(mr, mc - ip), std::min(nr, nc - jp));
                        }
                    }
                }
            }
        }
    }
}

}  // namespace tinyinfer::kernels
