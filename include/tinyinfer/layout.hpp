#pragma once

#include <cstdint>

#include "tinyinfer/tensor.hpp"

namespace tinyinfer {

/// NCHW -> NCHWc ("blocked") layout used by the SIMD convolution kernels.
///
/// Channels are split into blocks of `block` (8 floats = one AVX2 register, 16 = AVX-512) and the
/// block becomes the innermost dimension: [N, C, H, W] -> [N, ceil(C/block), H, W, block]. A
/// kernel can then load the same pixel of `block` channels with one aligned vector load.
/// Channels past C in the last block are zero-filled, so they add nothing to dot products.
Tensor to_blocked(const Tensor& nchw, std::int64_t block);

/// Inverse of to_blocked; `channels` is the original C (the padding is dropped).
Tensor from_blocked(const Tensor& blocked, std::int64_t channels);

}  // namespace tinyinfer
