#include "tinyinfer/quant.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "quant_kernels.hpp"
#include "tinyinfer/cpu_features.hpp"

namespace tinyinfer::quant {

using detail::kMR;
using detail::kNR;

QParams choose_u8(float lo, float hi) {
    lo = std::min(lo, 0.0F);
    hi = std::max(hi, 0.0F);
    if (hi - lo < 1e-12F) return {1.0F, 0};  // all zeros: any scale works
    QParams q;
    q.scale = (hi - lo) / 255.0F;
    q.zero_point = static_cast<int>(std::lround(-lo / q.scale));
    q.zero_point = std::clamp(q.zero_point, 0, 255);
    return q;
}

std::uint8_t quantize_u8(float x, QParams q) noexcept {
    const long v = std::lround(x / q.scale) + q.zero_point;
    return static_cast<std::uint8_t>(std::clamp(v, 0L, 255L));
}

void gemm_s8u8_ref(int m, int n, int k, const std::int8_t* a, const std::uint8_t* b,
                   std::int32_t* c) {
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            std::int32_t acc = 0;
            for (int p = 0; p < k; ++p) acc += static_cast<std::int32_t>(a[i * k + p]) * b[p * n + j];
            c[i * n + j] = acc;
        }
}

namespace detail {

void kernel_scalar_8x32(int kgroups, const std::int8_t* a, const std::uint8_t* b, std::int32_t* c,
                        int ldc, int h, int w) {
    std::int32_t acc[kMR][kNR] = {};
    for (int g = 0; g < kgroups; ++g) {
        for (int r = 0; r < kMR; ++r)
            for (int j = 0; j < kNR; ++j)
                for (int t = 0; t < 4; ++t)
                    acc[r][j] += static_cast<std::int32_t>(a[r * 4 + t]) * b[j * 4 + t];
        a += kMR * 4;
        b += kNR * 4;
    }
    for (int r = 0; r < h; ++r)
        for (int j = 0; j < w; ++j) c[r * ldc + j] = acc[r][j];
}

}  // namespace detail

namespace {

detail::Int8Kernel best_kernel() {
#if defined(__x86_64__) || defined(_M_X64)
    const auto& f = cpu_features();
    if (f.avx512f && f.avx512bw && f.avx512vnni) return detail::kernel_vnni_8x32;
#endif
    return detail::kernel_scalar_8x32;
}

std::atomic<detail::Int8Kernel>& active() {
    static std::atomic<detail::Int8Kernel> k{best_kernel()};
    return k;
}

/// Packs kpad x n uint8 (row-major, rows past k are zero) columns [j0, j0 + NR) into
/// k-groups of 4: out[g][j][t] = b[4g + t][j0 + j].
void pack_b(const std::uint8_t* b, int k, int n, int kgroups, int j0, std::uint8_t* out) {
    const int w = std::min(kNR, n - j0);
    for (int g = 0; g < kgroups; ++g)
        for (int j = 0; j < kNR; ++j)
            for (int t = 0; t < 4; ++t) {
                const int row = 4 * g + t;
                *out++ = (j < w && row < k) ? b[row * n + j0 + j] : 0;
            }
}

/// C[oc x n] = packed weights x B[k x n], one parallel loop over column panels.
void gemm_packed(const QConv& q, const std::uint8_t* b, int n, std::int32_t* c) {
    const detail::Int8Kernel kernel = active().load(std::memory_order_relaxed);
    const int kgroups = q.kpad / 4;
    const int col_panels = (n + kNR - 1) / kNR;
    const int row_panels = (q.oc + kMR - 1) / kMR;
    const std::size_t panel_bytes = static_cast<std::size_t>(kgroups) * kMR * 4;
    const bool parallel = col_panels >= 4 && static_cast<double>(q.oc) * n * q.k > 4e6;
    (void)parallel;
#pragma omp parallel for schedule(dynamic, 1) if (parallel)
    for (int jp = 0; jp < col_panels; ++jp) {
        thread_local std::vector<std::uint8_t> bp;
        bp.resize(static_cast<std::size_t>(kgroups) * kNR * 4);
        pack_b(b, q.k, n, kgroups, jp * kNR, bp.data());
        for (int p = 0; p < row_panels; ++p) {
            kernel(kgroups, q.w_packed.data() + p * panel_bytes, bp.data(),
                   c + static_cast<std::size_t>(p) * kMR * n + jp * kNR, n,
                   std::min(kMR, q.oc - p * kMR), std::min(kNR, n - jp * kNR));
        }
    }
}

}  // namespace

const char* int8_isa() {
    return active().load() == detail::kernel_scalar_8x32 ? "scalar" : "vnni";
}

bool set_int8_isa(const char* isa) {
    if (std::strcmp(isa, "scalar") == 0) {
        active().store(detail::kernel_scalar_8x32);
        return true;
    }
#if defined(__x86_64__) || defined(_M_X64)
    const auto& f = cpu_features();
    if (std::strcmp(isa, "vnni") == 0 && f.avx512f && f.avx512bw && f.avx512vnni) {
        active().store(detail::kernel_vnni_8x32);
        return true;
    }
#endif
    return false;
}

QConv quantize_conv(const Tensor& w, const Tensor* bias, const kernels::Conv2dParams& p,
                    QParams input) {
    if (p.groups != 1) throw std::invalid_argument("int8 conv supports groups = 1 only");
    QConv q;
    q.oc = static_cast<int>(w.shape()[0]);
    q.ic = static_cast<int>(w.shape()[1]);
    q.kh = static_cast<int>(w.shape()[2]);
    q.kw = static_cast<int>(w.shape()[3]);
    q.k = q.ic * q.kh * q.kw;
    q.kpad = (q.k + 3) / 4 * 4;
    q.params = p;
    q.input = input;
    q.w_scale.resize(q.oc);
    q.w_sum.assign(q.oc, 0);
    const float* wd = w.data<float>();

    // symmetric per-output-channel scales: each filter uses the full [-127, 127]
    std::vector<std::int8_t> wq(static_cast<std::size_t>(q.oc) * q.k);
    for (int o = 0; o < q.oc; ++o) {
        float amax = 0.0F;
        for (int j = 0; j < q.k; ++j) amax = std::max(amax, std::abs(wd[o * q.k + j]));
        const float s = amax > 0 ? amax / 127.0F : 1.0F;
        q.w_scale[o] = s;
        for (int j = 0; j < q.k; ++j) {
            const auto v = static_cast<std::int8_t>(
                std::clamp(std::lround(wd[o * q.k + j] / s), -127L, 127L));
            wq[static_cast<std::size_t>(o) * q.k + j] = v;
            q.w_sum[o] += v;
        }
    }

    // pack: [row panel][k group][row in panel][4 bytes], zero-padded in both directions
    const int kgroups = q.kpad / 4;
    const int panels = (q.oc + kMR - 1) / kMR;
    q.w_packed.assign(static_cast<std::size_t>(panels) * kgroups * kMR * 4, 0);
    std::int8_t* out = q.w_packed.data();
    for (int pnl = 0; pnl < panels; ++pnl)
        for (int g = 0; g < kgroups; ++g)
            for (int r = 0; r < kMR; ++r)
                for (int t = 0; t < 4; ++t) {
                    const int o = pnl * kMR + r, j = 4 * g + t;
                    *out++ = (o < q.oc && j < q.k) ? wq[static_cast<std::size_t>(o) * q.k + j] : 0;
                }
    if (bias != nullptr) q.bias.assign(bias->data<float>(), bias->data<float>() + q.oc);
    return q;
}

Tensor conv2d_int8(const Tensor& x, const QConv& q) {
    const auto& p = q.params;
    const int n = static_cast<int>(x.shape()[0]), c = static_cast<int>(x.shape()[1]);
    const int h = static_cast<int>(x.shape()[2]), w = static_cast<int>(x.shape()[3]);
    if (c != q.ic) throw std::invalid_argument("conv2d_int8: channel mismatch");
    const int oh = kernels::conv_out_size(h, q.kh, p.stride[0], p.pads[0], p.pads[2], p.dilation[0]);
    const int ow = kernels::conv_out_size(w, q.kw, p.stride[1], p.pads[1], p.pads[3], p.dilation[1]);
    const int pixels = oh * ow;
    const auto pad = static_cast<std::uint8_t>(q.input.zero_point);  // the uint8 that means 0.0

    Tensor y(Shape{n, q.oc, oh, ow}, DType::Float32);
    thread_local std::vector<std::uint8_t> xq, cols;
    thread_local std::vector<std::int32_t> acc;
    xq.resize(static_cast<std::size_t>(c) * h * w);
    cols.resize(static_cast<std::size_t>(q.k) * pixels);
    acc.resize(static_cast<std::size_t>(q.oc) * pixels);

    for (int img = 0; img < n; ++img) {
        const float* xs = x.data<float>() + static_cast<std::size_t>(img) * c * h * w;
        for (std::size_t i = 0; i < xq.size(); ++i) xq[i] = quantize_u8(xs[i], q.input);
        // im2col on uint8; out-of-image taps get the zero point, not 0
        for (int ci = 0; ci < c; ++ci)
            for (int ky = 0; ky < q.kh; ++ky)
                for (int kx = 0; kx < q.kw; ++kx) {
                    std::uint8_t* row = cols.data() +
                                        (static_cast<std::size_t>(ci) * q.kh * q.kw + ky * q.kw + kx) * pixels;
                    for (int oy = 0; oy < oh; ++oy) {
                        const int iy = oy * p.stride[0] - p.pads[0] + ky * p.dilation[0];
                        for (int ox = 0; ox < ow; ++ox) {
                            const int ix = ox * p.stride[1] - p.pads[1] + kx * p.dilation[1];
                            row[oy * ow + ox] = (iy >= 0 && iy < h && ix >= 0 && ix < w)
                                                    ? xq[(static_cast<std::size_t>(ci) * h + iy) * w + ix]
                                                    : pad;
                        }
                    }
                }
        gemm_packed(q, cols.data(), pixels, acc.data());
        // dequantize: sum((xq - zp) * wq) = acc - zp * sum(wq), times s_x * s_w[o]
        float* ys = y.data<float>() + static_cast<std::size_t>(img) * q.oc * pixels;
        for (int o = 0; o < q.oc; ++o) {
            const float s = q.input.scale * q.w_scale[o];
            const std::int32_t corr = q.input.zero_point * q.w_sum[o];
            const float b = q.bias.empty() ? 0.0F : q.bias[o];
            const std::int32_t* a = acc.data() + static_cast<std::size_t>(o) * pixels;
            for (int j = 0; j < pixels; ++j) ys[o * pixels + j] = static_cast<float>(a[j] - corr) * s + b;
        }
    }
    return y;
}

QuantPlan calibrate(const Graph& g, const std::string& input_name,
                    const std::vector<Tensor>& batches) {
    const auto& nodes = g.nodes();
    std::vector<std::pair<float, float>> ranges(nodes.size(), {std::numeric_limits<float>::max(),
                                                               std::numeric_limits<float>::lowest()});
    Executor ex(g);
    ex.observe_conv_inputs(&ranges);
    for (const Tensor& b : batches) ex.run({{input_name, b}});

    QuantPlan plan(nodes.size());
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        const auto& n = nodes[i];
        if (n.kind != OpKind::Conv || n.conv.groups != 1) continue;
        const auto& v = g.values();
        const Tensor& w = v[n.inputs[1]].data;
        const Tensor* bias = n.inputs.size() > 2 && n.inputs[2] >= 0 ? &v[n.inputs[2]].data : nullptr;
        plan[i] = quantize_conv(w, bias, n.conv, choose_u8(ranges[i].first, ranges[i].second));
    }
    return plan;
}

}  // namespace tinyinfer::quant
