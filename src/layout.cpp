#include "tinyinfer/layout.hpp"

#include <stdexcept>

namespace tinyinfer {

Tensor to_blocked(const Tensor& nchw, std::int64_t block) {
    if (nchw.shape().rank() != 4) throw std::invalid_argument("to_blocked expects NCHW");
    if (block <= 0) throw std::invalid_argument("block must be positive");
    const std::int64_t n = nchw.shape()[0], c = nchw.shape()[1], h = nchw.shape()[2],
                       w = nchw.shape()[3];
    const std::int64_t cb = (c + block - 1) / block;
    const std::int64_t hw = h * w;

    Tensor out(Shape{n, cb, h, w, block}, DType::Float32);  // zero-initialized: padding is 0
    const float* src = nchw.data<float>();
    float* dst = out.data<float>();
    for (std::int64_t in = 0; in < n; ++in) {
        for (std::int64_t ic = 0; ic < c; ++ic) {
            const float* plane = src + (in * c + ic) * hw;
            // channel ic lives in block ic/block, lane ic%block
            float* base = dst + (in * cb + ic / block) * hw * block + ic % block;
            for (std::int64_t p = 0; p < hw; ++p) base[p * block] = plane[p];
        }
    }
    return out;
}

Tensor from_blocked(const Tensor& blocked, std::int64_t channels) {
    if (blocked.shape().rank() != 5) throw std::invalid_argument("from_blocked expects NCHWc");
    const std::int64_t n = blocked.shape()[0], cb = blocked.shape()[1], h = blocked.shape()[2],
                       w = blocked.shape()[3], block = blocked.shape()[4];
    if (channels <= 0 || channels > cb * block || channels <= (cb - 1) * block) {
        throw std::invalid_argument("channel count doesn't fit the blocked shape");
    }
    const std::int64_t hw = h * w;

    Tensor out(Shape{n, channels, h, w}, DType::Float32);
    const float* src = blocked.data<float>();
    float* dst = out.data<float>();
    for (std::int64_t in = 0; in < n; ++in) {
        for (std::int64_t ic = 0; ic < channels; ++ic) {
            const float* base = src + (in * cb + ic / block) * hw * block + ic % block;
            float* plane = dst + (in * channels + ic) * hw;
            for (std::int64_t p = 0; p < hw; ++p) plane[p] = base[p * block];
        }
    }
    return out;
}

}  // namespace tinyinfer
