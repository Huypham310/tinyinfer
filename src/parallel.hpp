#pragma once

// How many OpenMP threads a parallel region should use.
//
// "All of them" is the wrong default on this laptop (4 Zen 5 + 8 Zen 5c cores, SMT on): a region
// with little work per thread finishes before the slow cores even start, and threads that sit
// idle in one region keep spinning while the next region runs, stealing their SMT siblings.
// Measured on a 512 x 4 x 4608 GEMM: 410 us with 24 threads, 42 us with 4.

#include <algorithm>
#include <cstdlib>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tinyinfer::par {

/// Upper bound on threads per region. Default: one per physical core (logical / 2), the same
/// default ONNX Runtime uses; with SMT siblings in the team ResNet-18 at batch 32 ran 1.3x
/// slower than with 4 threads. TINYINFER_THREADS overrides it, and so does an explicit
/// OMP_NUM_THREADS. omp_set_num_threads() can still lower it (the single-core benchmarks).
inline int max_threads() {
#ifdef _OPENMP
    static const int cap = [] {
        if (const char* e = std::getenv("TINYINFER_THREADS")) return std::max(1, std::atoi(e));
        if (std::getenv("OMP_NUM_THREADS") != nullptr) return omp_get_max_threads();
        return std::max(1, omp_get_num_procs() / 2);
    }();
    return std::min(cap, omp_get_max_threads());
#else
    return 1;
#endif
}

/// Threads for `work` units of work when each thread should get at least `per_thread` of them:
/// 1 for small jobs, never more than max_threads().
inline int team_size(double work, double per_thread) {
#ifdef _OPENMP
    const double t = std::min<double>(max_threads(), work / per_thread);
    return std::max(1, static_cast<int>(t));
#else
    (void)work;
    (void)per_thread;
    return 1;
#endif
}

}  // namespace tinyinfer::par
