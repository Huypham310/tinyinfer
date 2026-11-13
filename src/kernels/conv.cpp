#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "../parallel.hpp"
#include "tinyinfer/kernels.hpp"

namespace tinyinfer::kernels {

int conv_out_size(int in, int kernel, int stride, int pad_begin, int pad_end, int dilation) {
    const int span = dilation * (kernel - 1) + 1;
    return (in + pad_begin + pad_end - span) / stride + 1;
}

namespace {

// column-matrix elements per im2col thread (~2 MB of writes)
constexpr double kIm2colPerThread = 1 << 19;

struct ConvShape {
    int n, c, h, w, oc, kh, kw, oh, ow, cg, ocg;
};

ConvShape check(const Tensor& x, const Tensor& w, const Tensor* bias, const Conv2dParams& p) {
    if (x.shape().rank() != 4 || w.shape().rank() != 4) {
        throw std::invalid_argument("conv2d expects 4-D input and weight");
    }
    ConvShape s{};
    s.n = static_cast<int>(x.shape()[0]);
    s.c = static_cast<int>(x.shape()[1]);
    s.h = static_cast<int>(x.shape()[2]);
    s.w = static_cast<int>(x.shape()[3]);
    s.oc = static_cast<int>(w.shape()[0]);
    s.cg = static_cast<int>(w.shape()[1]);
    s.kh = static_cast<int>(w.shape()[2]);
    s.kw = static_cast<int>(w.shape()[3]);
    if (p.groups < 1 || s.c != s.cg * p.groups || s.oc % p.groups != 0) {
        throw std::invalid_argument("conv2d: " + std::to_string(s.c) + " input channels, weight " +
                                    w.shape().to_string() + ", groups " +
                                    std::to_string(p.groups) + " don't fit together");
    }
    if (bias != nullptr && bias->numel() != s.oc) {
        throw std::invalid_argument("conv2d: bias must have one value per output channel");
    }
    s.ocg = s.oc / p.groups;
    s.oh = conv_out_size(s.h, s.kh, p.stride[0], p.pads[0], p.pads[2], p.dilation[0]);
    s.ow = conv_out_size(s.w, s.kw, p.stride[1], p.pads[1], p.pads[3], p.dilation[1]);
    if (s.oh <= 0 || s.ow <= 0) throw std::invalid_argument("conv2d: kernel larger than input");
    return s;
}

/// Unrolls the receptive fields of one image's channel group into a (cg*kh*kw) x (oh*ow)
/// matrix, so the convolution becomes one GEMM with the (ocg) x (cg*kh*kw) weight matrix.
/// `ld` is the row stride of `cols`: several images can sit side by side in one wide matrix.
void im2col(const float* x, const ConvShape& s, const Conv2dParams& p, float* cols,
            std::size_t ld) {
    for (int ci = 0; ci < s.cg; ++ci) {
        const float* plane = x + static_cast<std::size_t>(ci) * s.h * s.w;
        for (int ky = 0; ky < s.kh; ++ky) {
            for (int kx = 0; kx < s.kw; ++kx) {
                float* row = cols + (static_cast<std::size_t>(ci) * s.kh * s.kw + ky * s.kw + kx) *
                                        ld;
                for (int oy = 0; oy < s.oh; ++oy) {
                    const int iy = oy * p.stride[0] - p.pads[0] + ky * p.dilation[0];
                    float* out = row + oy * s.ow;
                    if (iy < 0 || iy >= s.h) {
                        std::fill(out, out + s.ow, 0.0F);  // whole row is padding
                        continue;
                    }
                    for (int ox = 0; ox < s.ow; ++ox) {
                        const int ix = ox * p.stride[1] - p.pads[1] + kx * p.dilation[1];
                        out[ox] = (ix >= 0 && ix < s.w) ? plane[iy * s.w + ix] : 0.0F;
                    }
                }
            }
        }
    }
}

void add_bias(float* y, const Tensor* bias, const ConvShape& s) {
    if (bias == nullptr) return;
    const float* b = bias->data<float>();
    const int pixels = s.oh * s.ow;
    for (int n = 0; n < s.n; ++n) {
        for (int o = 0; o < s.oc; ++o) {
            float* plane = y + (static_cast<std::size_t>(n) * s.oc + o) * pixels;
            for (int i = 0; i < pixels; ++i) plane[i] += b[o];
        }
    }
}

}  // namespace

Tensor conv2d(const Tensor& x, const Tensor& w, const Tensor* bias, const Conv2dParams& p) {
    const ConvShape s = check(x, w, bias, p);
    Tensor y(Shape{s.n, s.oc, s.oh, s.ow}, DType::Float32);
    const int pixels = s.oh * s.ow;
    const int kdim = s.cg * s.kh * s.kw;
    // a 1x1, stride-1, unpadded conv *is* a GEMM on the input as stored: skip the copy
    const bool pointwise = s.kh == 1 && s.kw == 1 && p.stride[0] == 1 && p.stride[1] == 1 &&
                           p.pads[0] == 0 && p.pads[1] == 0 && p.pads[2] == 0 && p.pads[3] == 0;
    // Images per GEMM: one image at a time leaves the late layers with n = oh*ow = 4 columns
    // whatever the batch, so put several images' columns side by side instead. Capped so the
    // column matrix stays around 8 MB (layer 1 of ResNet-18 at 64x64: ~13 images).
    constexpr std::size_t kColsBudget = std::size_t{2} << 20;  // floats
    const std::size_t per_image = static_cast<std::size_t>(kdim) * pixels;
    const int chunk = std::max(1, std::min(s.n, static_cast<int>(kColsBudget / per_image)));

    const float* xd = x.data<float>();
    const float* wd = w.data<float>();
    float* yd = y.data<float>();
    if (chunk == 1) {
        std::vector<float> cols(pointwise ? 0 : per_image);
        for (int n = 0; n < s.n; ++n) {
            for (int g = 0; g < p.groups; ++g) {
                const float* xg = xd + (static_cast<std::size_t>(n) * s.c + g * s.cg) * s.h * s.w;
                const float* b = xg;
                if (!pointwise) {
                    im2col(xg, s, p, cols.data(), pixels);
                    b = cols.data();
                }
                const float* wg = wd + static_cast<std::size_t>(g) * s.ocg * kdim;
                float* yg = yd + (static_cast<std::size_t>(n) * s.oc + g * s.ocg) * pixels;
                gemm(s.ocg, pixels, kdim, wg, kdim, b, pixels, false, yg, pixels);
            }
        }
        add_bias(yd, bias, s);
        return y;
    }

    std::vector<float> cols(per_image * chunk);
    std::vector<float> out(static_cast<std::size_t>(s.ocg) * pixels * chunk);
    for (int n0 = 0; n0 < s.n; n0 += chunk) {
        const int cnt = std::min(chunk, s.n - n0);
        const std::size_t ld = static_cast<std::size_t>(cnt) * pixels;
        // images write disjoint column blocks, so im2col splits across threads for free. Only a
        // few though: it's memory-bound, and 24 threads here then 4 in the GEMM left 20 of them
        // spinning on SMT siblings: batch-32 layer1 ran 3x slower than single-threaded.
        const int im2col_threads = std::min(
            cnt, par::team_size(static_cast<double>(ld) * kdim, kIm2colPerThread));
        (void)im2col_threads;
        for (int g = 0; g < p.groups; ++g) {
#pragma omp parallel for schedule(static) num_threads(im2col_threads) if (im2col_threads > 1)
            for (int i = 0; i < cnt; ++i) {
                const float* xg =
                    xd + (static_cast<std::size_t>(n0 + i) * s.c + g * s.cg) * s.h * s.w;
                im2col(xg, s, p, cols.data() + static_cast<std::size_t>(i) * pixels, ld);
            }
            const float* wg = wd + static_cast<std::size_t>(g) * s.ocg * kdim;
            const int cols_n = static_cast<int>(ld);
            gemm(s.ocg, cols_n, kdim, wg, kdim, cols.data(), cols_n, false, out.data(), cols_n);
            // GEMM output is [ocg, image, pixel]; NCHW wants [image, oc, pixel]
            for (int o = 0; o < s.ocg; ++o) {
                for (int i = 0; i < cnt; ++i) {
                    const float* src = out.data() + o * ld + static_cast<std::size_t>(i) * pixels;
                    float* dst =
                        yd + ((static_cast<std::size_t>(n0 + i) * s.oc + g * s.ocg + o) * pixels);
                    std::copy(src, src + pixels, dst);
                }
            }
        }
    }
    add_bias(yd, bias, s);
    return y;
}

Tensor conv2d_naive(const Tensor& x, const Tensor& w, const Tensor* bias, const Conv2dParams& p) {
    const ConvShape s = check(x, w, bias, p);
    Tensor y(Shape{s.n, s.oc, s.oh, s.ow}, DType::Float32);
    const float* xd = x.data<float>();
    const float* wd = w.data<float>();
    float* yd = y.data<float>();
    for (int n = 0; n < s.n; ++n)
        for (int o = 0; o < s.oc; ++o) {
            const int g = o / s.ocg;
            for (int oy = 0; oy < s.oh; ++oy)
                for (int ox = 0; ox < s.ow; ++ox) {
                    double acc = bias ? bias->data<float>()[o] : 0.0;
                    for (int ci = 0; ci < s.cg; ++ci)
                        for (int ky = 0; ky < s.kh; ++ky)
                            for (int kx = 0; kx < s.kw; ++kx) {
                                const int iy = oy * p.stride[0] - p.pads[0] + ky * p.dilation[0];
                                const int ix = ox * p.stride[1] - p.pads[1] + kx * p.dilation[1];
                                if (iy < 0 || iy >= s.h || ix < 0 || ix >= s.w) continue;
                                const float xv =
                                    xd[((static_cast<std::size_t>(n) * s.c + g * s.cg + ci) * s.h +
                                        iy) * s.w + ix];
                                const float wv =
                                    wd[((static_cast<std::size_t>(o) * s.cg + ci) * s.kh + ky) *
                                           s.kw + kx];
                                acc += static_cast<double>(xv) * wv;
                            }
                    yd[((static_cast<std::size_t>(n) * s.oc + o) * s.oh + oy) * s.ow + ox] =
                        static_cast<float>(acc);
                }
        }
    return y;
}

}  // namespace tinyinfer::kernels
