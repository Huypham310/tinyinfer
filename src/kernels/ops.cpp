#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "tinyinfer/kernels.hpp"

namespace tinyinfer::kernels {

namespace {

void need_rank4(const Tensor& x, const char* op) {
    if (x.shape().rank() != 4) throw std::invalid_argument(std::string(op) + " expects NCHW input");
}

}  // namespace

Tensor max_pool2d(const Tensor& x, const Pool2dParams& p) {
    need_rank4(x, "max_pool2d");
    const int n = static_cast<int>(x.shape()[0]), c = static_cast<int>(x.shape()[1]);
    const int h = static_cast<int>(x.shape()[2]), w = static_cast<int>(x.shape()[3]);
    const int oh = conv_out_size(h, p.kernel[0], p.stride[0], p.pads[0], p.pads[2], 1);
    const int ow = conv_out_size(w, p.kernel[1], p.stride[1], p.pads[1], p.pads[3], 1);
    if (oh <= 0 || ow <= 0) throw std::invalid_argument("max_pool2d: window larger than input");
    Tensor y(Shape{n, c, oh, ow}, DType::Float32);
    const float* xd = x.data<float>();
    float* yd = y.data<float>();
    for (int plane = 0; plane < n * c; ++plane) {
        const float* src = xd + static_cast<std::size_t>(plane) * h * w;
        float* dst = yd + static_cast<std::size_t>(plane) * oh * ow;
        for (int oy = 0; oy < oh; ++oy) {
            const int y0 = std::max(oy * p.stride[0] - p.pads[0], 0);
            const int y1 = std::min(oy * p.stride[0] - p.pads[0] + p.kernel[0], h);
            for (int ox = 0; ox < ow; ++ox) {
                const int x0 = std::max(ox * p.stride[1] - p.pads[1], 0);
                const int x1 = std::min(ox * p.stride[1] - p.pads[1] + p.kernel[1], w);
                float m = -std::numeric_limits<float>::infinity();
                for (int iy = y0; iy < y1; ++iy)
                    for (int ix = x0; ix < x1; ++ix) m = std::max(m, src[iy * w + ix]);
                dst[oy * ow + ox] = m;
            }
        }
    }
    return y;
}

Tensor global_avg_pool(const Tensor& x) {
    need_rank4(x, "global_avg_pool");
    const auto n = x.shape()[0], c = x.shape()[1];
    const auto pixels = x.shape()[2] * x.shape()[3];
    Tensor y(Shape{n, c, 1, 1}, DType::Float32);
    const float* xd = x.data<float>();
    float* yd = y.data<float>();
    for (std::int64_t plane = 0; plane < n * c; ++plane) {
        double sum = 0.0;  // 8x8 = 64 values per plane, but keep it exact for big inputs too
        for (std::int64_t i = 0; i < pixels; ++i) sum += xd[plane * pixels + i];
        yd[plane] = static_cast<float>(sum / static_cast<double>(pixels));
    }
    return y;
}

Tensor batch_norm(const Tensor& x, const Tensor& scale, const Tensor& bias, const Tensor& mean,
                  const Tensor& var, float eps) {
    need_rank4(x, "batch_norm");
    const auto n = x.shape()[0], c = x.shape()[1];
    const auto pixels = x.shape()[2] * x.shape()[3];
    for (const Tensor* t : {&scale, &bias, &mean, &var}) {
        if (t->numel() != c) throw std::invalid_argument("batch_norm: parameters must have C values");
    }
    Tensor y(x.shape(), DType::Float32);
    const float* xd = x.data<float>();
    float* yd = y.data<float>();
    for (std::int64_t ci = 0; ci < c; ++ci) {
        // fold into one multiply-add per element: y = x * a + b
        const float a = scale.data<float>()[ci] / std::sqrt(var.data<float>()[ci] + eps);
        const float b = bias.data<float>()[ci] - mean.data<float>()[ci] * a;
        for (std::int64_t ni = 0; ni < n; ++ni) {
            const std::int64_t off = (ni * c + ci) * pixels;
            for (std::int64_t i = 0; i < pixels; ++i) yd[off + i] = xd[off + i] * a + b;
        }
    }
    return y;
}

Tensor relu(const Tensor& x) {
    Tensor y(x.shape(), DType::Float32);
    const float* xd = x.data<float>();
    float* yd = y.data<float>();
    for (std::int64_t i = 0; i < x.numel(); ++i) yd[i] = xd[i] > 0.0F ? xd[i] : 0.0F;
    return y;
}

Tensor add(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("add: shapes " + a.shape().to_string() + " and " +
                                    b.shape().to_string() + " differ");
    }
    Tensor y(a.shape(), DType::Float32);
    const float* ad = a.data<float>();
    const float* bd = b.data<float>();
    float* yd = y.data<float>();
    for (std::int64_t i = 0; i < a.numel(); ++i) yd[i] = ad[i] + bd[i];
    return y;
}

Tensor softmax(const Tensor& x) {
    if (x.shape().rank() == 0) throw std::invalid_argument("softmax needs at least one axis");
    const auto last = x.shape()[x.shape().rank() - 1];
    const auto rows = x.numel() / last;
    Tensor y(x.shape(), DType::Float32);
    const float* xd = x.data<float>();
    float* yd = y.data<float>();
    for (std::int64_t r = 0; r < rows; ++r) {
        const float* in = xd + r * last;
        float* out = yd + r * last;
        const float mx = *std::max_element(in, in + last);
        double sum = 0.0;
        for (std::int64_t i = 0; i < last; ++i) {
            out[i] = std::exp(in[i] - mx);
            sum += out[i];
        }
        for (std::int64_t i = 0; i < last; ++i) out[i] = static_cast<float>(out[i] / sum);
    }
    return y;
}

Tensor gemm_op(const Tensor& a, const Tensor& b, const Tensor* c, bool trans_b, float alpha,
               float beta) {
    if (a.shape().rank() != 2 || b.shape().rank() != 2) {
        throw std::invalid_argument("gemm expects 2-D inputs");
    }
    const int m = static_cast<int>(a.shape()[0]), k = static_cast<int>(a.shape()[1]);
    const int bk = static_cast<int>(trans_b ? b.shape()[1] : b.shape()[0]);
    const int n = static_cast<int>(trans_b ? b.shape()[0] : b.shape()[1]);
    if (bk != k) throw std::invalid_argument("gemm: inner dimensions differ");
    Tensor y(Shape{m, n}, DType::Float32);
    float* yd = y.data<float>();
    gemm(m, n, k, a.data<float>(), k, b.data<float>(), trans_b ? k : n, trans_b, yd, n);
    if (alpha != 1.0F) {
        for (std::int64_t i = 0; i < y.numel(); ++i) yd[i] *= alpha;
    }
    if (c != nullptr) {
        // only the bias shapes exporters actually write: [N] or [1, N]
        if (c->numel() != n) throw std::invalid_argument("gemm: C must broadcast as a row of N");
        const float* cd = c->data<float>();
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j) yd[i * n + j] += beta * cd[j];
    }
    return y;
}

}  // namespace tinyinfer::kernels
