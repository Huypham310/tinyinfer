#pragma once

// Scalar reference kernels for the operators ResNet-style ONNX models need.
//
// Everything is float32, NCHW, contiguous. These are the baselines: the SIMD phase replaces the
// hot loops (GEMM micro-kernel, im2col) with AVX2 / AVX-512 versions and checks them against
// these. The `*_naive` functions are the textbook loops and exist only as test references.

#include <array>
#include <cstdint>

#include "tinyinfer/tensor.hpp"

namespace tinyinfer::kernels {

// ---- GEMM -------------------------------------------------------------------------------------

/// C[M x N] (+)= A[M x K] * op(B), row-major. op(B) is B[K x N], or B[N x K] transposed when
/// `trans_b` (the layout of a PyTorch Linear weight). `accumulate` adds to C instead of
/// overwriting it.
void gemm_naive(int m, int n, int k, const float* a, int lda, const float* b, int ldb, bool trans_b,
                float* c, int ldc, bool accumulate = false);

/// Same contract, cache-blocked: B is packed into KC x NC panels and A into MC x KC blocks so the
/// inner loops stream through contiguous memory, and a 4 x 16 register tile accumulates C
/// (the Goto / BLIS loop order).
void gemm(int m, int n, int k, const float* a, int lda, const float* b, int ldb, bool trans_b,
          float* c, int ldc, bool accumulate = false);

/// Micro-kernel `gemm` currently uses: "avx512", "avx2" or "scalar". Picked from CPUID on first
/// use (the env var TINYINFER_ISA overrides it).
const char* gemm_isa();
/// Switch the micro-kernel; false (and no change) if this CPU can't run that ISA.
bool set_gemm_isa(const char* isa);
/// Measured single-core FMA throughput (GFLOP/s) of a register-only loop for "avx2" or
/// "avx512": the practical peak the GEMM is compared against. 0 if unsupported.
double fma_peak_gflops(const char* isa);

// ---- Convolution --------------------------------------------------------------------------------

struct Conv2dParams {
    std::array<int, 2> stride{1, 1};
    std::array<int, 4> pads{0, 0, 0, 0};  // top, left, bottom, right (ONNX order: begins, ends)
    std::array<int, 2> dilation{1, 1};
    int groups = 1;
};

/// Output spatial size of a convolution / pooling window along one axis.
int conv_out_size(int in, int kernel, int stride, int pad_begin, int pad_end, int dilation);

/// x [N, C, H, W], w [OC, C / groups, KH, KW], optional bias [OC] -> [N, OC, OH, OW].
/// im2col (one column per output pixel) + gemm per image and group.
Tensor conv2d(const Tensor& x, const Tensor& w, const Tensor* bias, const Conv2dParams& p);
/// Direct seven-loop convolution, for tests.
Tensor conv2d_naive(const Tensor& x, const Tensor& w, const Tensor* bias, const Conv2dParams& p);

// ---- Everything else ResNet-18 uses -----------------------------------------------------------

struct Pool2dParams {
    std::array<int, 2> kernel{2, 2};
    std::array<int, 2> stride{2, 2};
    std::array<int, 4> pads{0, 0, 0, 0};
};

/// Padding never wins the max (it counts as -inf, like ONNX MaxPool).
Tensor max_pool2d(const Tensor& x, const Pool2dParams& p);
/// Mean over H and W -> [N, C, 1, 1] (GlobalAveragePool, ReduceMean over axes 2, 3).
Tensor global_avg_pool(const Tensor& x);

/// Inference batch norm: y = (x - mean) / sqrt(var + eps) * scale + bias, per channel.
Tensor batch_norm(const Tensor& x, const Tensor& scale, const Tensor& bias, const Tensor& mean,
                  const Tensor& var, float eps);

Tensor relu(const Tensor& x);
/// Elementwise a + b for equal shapes (residual connections).
Tensor add(const Tensor& a, const Tensor& b);
/// Softmax over the last axis, max-subtracted so large logits don't overflow.
Tensor softmax(const Tensor& x);

/// ONNX Gemm: y = alpha * a * op(b) + beta * c, with c broadcast over rows ([N] or [1, N]).
Tensor gemm_op(const Tensor& a, const Tensor& b, const Tensor* c, bool trans_b, float alpha = 1.0F,
               float beta = 1.0F);

}  // namespace tinyinfer::kernels
