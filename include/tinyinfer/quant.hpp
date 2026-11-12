#pragma once

// INT8 post-training quantization for convolutions.
//
// Scheme (the one ONNX Runtime / oneDNN use on x86):
//  - weights: symmetric int8, one scale per output channel (s_w[o] = max|W[o]| / 127);
//  - activations: asymmetric uint8 with a zero point, one scale per tensor, from min / max
//    observed on calibration images (x ~ s_x * (q - zp));
//  - the convolution runs as an int8 GEMM accumulating in int32 (AVX-512 VNNI vpdpbusd: u8 x s8
//    four at a time), then is dequantized to float with s_x * s_w[o] and the bias added.
// Everything between convolutions (Add, pooling, the classifier) stays float.

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "tinyinfer/graph.hpp"
#include "tinyinfer/kernels.hpp"
#include "tinyinfer/tensor.hpp"

namespace tinyinfer::quant {

struct QParams {
    float scale = 1.0F;
    int zero_point = 0;  // the uint8 value that stands for 0.0
};

/// uint8 parameters covering [lo, hi]. The range is widened to include 0 so that zero padding
/// (and ReLU's zeros) are represented exactly.
QParams choose_u8(float lo, float hi);
std::uint8_t quantize_u8(float x, QParams q) noexcept;

/// C[m x n] = A[m x k] (int8) * B[k x n] (uint8), int32 accumulation. Plain loops: the
/// reference the packed kernels are tested against.
void gemm_s8u8_ref(int m, int n, int k, const std::int8_t* a, const std::uint8_t* b,
                   std::int32_t* c);

/// Weights of one conv, quantized and pre-packed for the int8 GEMM (done once, at load time).
struct QConv {
    int oc = 0, ic = 0, kh = 0, kw = 0;
    int k = 0;       // ic * kh * kw
    int kpad = 0;    // k rounded up to a multiple of 4 (vpdpbusd works on groups of 4)
    std::vector<std::int8_t> w_packed;  // MR-row panels, k in groups of 4, zero-padded
    std::vector<float> w_scale;         // per output channel
    std::vector<std::int32_t> w_sum;    // sum over k of the int8 weights, for the zero point
    std::vector<float> bias;            // empty if none
    kernels::Conv2dParams params;
    QParams input;
};

QConv quantize_conv(const Tensor& w, const Tensor* bias, const kernels::Conv2dParams& p,
                    QParams input);
/// Float in, float out; groups must be 1.
Tensor conv2d_int8(const Tensor& x, const QConv& q);

/// Which int8 GEMM kernel conv2d_int8 uses: "vnni" (AVX-512 VNNI) or "scalar".
const char* int8_isa();
bool set_int8_isa(const char* isa);

/// Per-node quantized convolutions for a graph (nullopt where a node stays float).
using QuantPlan = std::vector<std::optional<QConv>>;

/// Runs the float graph on the calibration batches, records the input range of every Conv,
/// and quantizes each (non-grouped) Conv with it.
QuantPlan calibrate(const Graph& g, const std::string& input_name,
                    const std::vector<Tensor>& batches);

}  // namespace tinyinfer::quant
