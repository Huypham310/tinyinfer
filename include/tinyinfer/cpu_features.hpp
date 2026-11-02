#pragma once

#include <string>

namespace tinyinfer {

/// SIMD features the current CPU *and* OS support.
///
/// A flag is only true when the instruction set is reported by CPUID and the OS saves the
/// matching register state (checked through XCR0), so it is safe to dispatch on.
struct CpuFeatures {
    bool sse42 = false;
    bool avx = false;
    bool avx2 = false;
    bool fma = false;
    bool avx512f = false;
    bool avx512bw = false;
    bool avx512vl = false;
    bool avx512vnni = false;  // int8 dot products, useful later for quantized GEMM
};

/// Detects features once and caches the result.
const CpuFeatures& cpu_features();

/// Human readable summary, e.g. "sse4.2 avx avx2 fma avx512f".
std::string to_string(const CpuFeatures& f);

}  // namespace tinyinfer
