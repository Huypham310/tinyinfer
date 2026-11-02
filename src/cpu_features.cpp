#include "tinyinfer/cpu_features.hpp"

#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define TINYINFER_X86 1
#if defined(_MSC_VER) && !defined(__clang__)
#include <immintrin.h>
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

namespace tinyinfer {
namespace {

#ifdef TINYINFER_X86

struct Regs {
    std::uint32_t eax, ebx, ecx, edx;
};

Regs cpuid(std::uint32_t leaf, std::uint32_t subleaf) {
    Regs r{};
#if defined(_MSC_VER) && !defined(__clang__)
    int out[4];
    __cpuidex(out, static_cast<int>(leaf), static_cast<int>(subleaf));
    r = {static_cast<std::uint32_t>(out[0]), static_cast<std::uint32_t>(out[1]),
         static_cast<std::uint32_t>(out[2]), static_cast<std::uint32_t>(out[3])};
#else
    __cpuid_count(leaf, subleaf, r.eax, r.ebx, r.ecx, r.edx);
#endif
    return r;
}

// XCR0 says which register files the OS saves on context switch.
std::uint64_t xgetbv0() {
#if defined(_MSC_VER) && !defined(__clang__)
    return _xgetbv(0);
#else
    // inline asm để khỏi phải bật -mxsave cho cả file
    std::uint32_t lo = 0, hi = 0;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return (static_cast<std::uint64_t>(hi) << 32) | lo;
#endif
}

constexpr bool bit(std::uint32_t reg, int n) { return (reg >> n) & 1u; }

CpuFeatures detect() {
    CpuFeatures f;
    const std::uint32_t max_leaf = cpuid(0, 0).eax;
    if (max_leaf < 1) return f;

    const Regs l1 = cpuid(1, 0);
    f.sse42 = bit(l1.ecx, 20);

    // AVX needs both the CPU flag and OSXSAVE, otherwise xgetbv itself would fault
    const bool osxsave = bit(l1.ecx, 27);
    if (!osxsave || !bit(l1.ecx, 28)) return f;

    const std::uint64_t xcr0 = xgetbv0();
    const bool os_avx = (xcr0 & 0x6) == 0x6;  // XMM + YMM state
    // AVX-512 also needs opmask (bit 5) and the two ZMM halves (bits 6, 7)
    const bool os_avx512 = (xcr0 & 0xE6) == 0xE6;
    if (!os_avx) return f;

    f.avx = true;
    f.fma = bit(l1.ecx, 12);

    if (max_leaf >= 7) {
        const Regs l7 = cpuid(7, 0);
        f.avx2 = bit(l7.ebx, 5);
        if (os_avx512) {
            f.avx512f = bit(l7.ebx, 16);
            f.avx512bw = f.avx512f && bit(l7.ebx, 30);
            f.avx512vl = f.avx512f && bit(l7.ebx, 31);
            f.avx512vnni = f.avx512f && bit(l7.ecx, 11);
        }
    }
    return f;
}

#else

CpuFeatures detect() { return {}; }  // non-x86: scalar kernels only for now

#endif

}  // namespace

const CpuFeatures& cpu_features() {
    static const CpuFeatures features = detect();
    return features;
}

std::string to_string(const CpuFeatures& f) {
    std::string out;
    auto add = [&out](bool on, const char* name) {
        if (!on) return;
        if (!out.empty()) out += ' ';
        out += name;
    };
    add(f.sse42, "sse4.2");
    add(f.avx, "avx");
    add(f.avx2, "avx2");
    add(f.fma, "fma");
    add(f.avx512f, "avx512f");
    add(f.avx512bw, "avx512bw");
    add(f.avx512vl, "avx512vl");
    add(f.avx512vnni, "avx512vnni");
    return out.empty() ? "none" : out;
}

}  // namespace tinyinfer
