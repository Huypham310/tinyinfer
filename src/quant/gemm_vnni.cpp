// Compiled with -mavx512f -mavx512bw -mavx512vnni (see CMakeLists.txt); called only when CPUID
// says VNNI is there.

#include <immintrin.h>

#include <cstdint>
#include <cstring>

#include "quant_kernels.hpp"

namespace tinyinfer::quant::detail {

/// C tile [MR x 32] += A panel (kg groups of MR rows x 4 int8) * B panel (kg groups of 32 cols
/// x 4 uint8). One vpdpbusd = 16 lanes x 4 u8*s8 products summed into int32: 64 MACs.
/// 16 accumulators + 2 B registers + 1 broadcast fit easily in the 32 zmm registers.
void kernel_vnni_8x32(int kgroups, const std::int8_t* a, const std::uint8_t* b, std::int32_t* c,
                      int ldc, int h, int w) {
    __m512i acc[kMR][2];
    for (auto& r : acc) r[0] = r[1] = _mm512_setzero_si512();
    for (int g = 0; g < kgroups; ++g) {
        const __m512i b0 = _mm512_loadu_si512(b);
        const __m512i b1 = _mm512_loadu_si512(b + 64);
        for (int r = 0; r < kMR; ++r) {
            std::int32_t word;
            std::memcpy(&word, a + r * 4, 4);  // 4 consecutive k of row r, as one 32-bit lane
            const __m512i wv = _mm512_set1_epi32(word);
            // u8 activations are the first (unsigned) operand, s8 weights the second
            acc[r][0] = _mm512_dpbusd_epi32(acc[r][0], b0, wv);
            acc[r][1] = _mm512_dpbusd_epi32(acc[r][1], b1, wv);
        }
        a += kMR * 4;
        b += kNR * 4;
    }
    alignas(64) std::int32_t tmp[kMR][kNR];
    for (int r = 0; r < kMR; ++r) {
        _mm512_store_si512(tmp[r], acc[r][0]);
        _mm512_store_si512(tmp[r] + 16, acc[r][1]);
    }
    for (int r = 0; r < h; ++r)
        for (int j = 0; j < w; ++j) c[r * ldc + j] = tmp[r][j];
}

}  // namespace tinyinfer::quant::detail
