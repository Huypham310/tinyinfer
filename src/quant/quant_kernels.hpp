#pragma once

#include <cstdint>

namespace tinyinfer::quant::detail {

// Tile of the packed int8 GEMM: MR weight rows x NR activation columns.
constexpr int kMR = 8;
constexpr int kNR = 32;

/// C[h x w] (= MR x NR at most) = packed A panel x packed B panel, overwriting C.
using Int8Kernel = void (*)(int kgroups, const std::int8_t* a, const std::uint8_t* b,
                            std::int32_t* c, int ldc, int h, int w);

void kernel_scalar_8x32(int kgroups, const std::int8_t* a, const std::uint8_t* b, std::int32_t* c,
                        int ldc, int h, int w);
#if defined(__x86_64__) || defined(_M_X64)
void kernel_vnni_8x32(int kgroups, const std::int8_t* a, const std::uint8_t* b, std::int32_t* c,
                      int ldc, int h, int w);
#endif

}  // namespace tinyinfer::quant::detail
