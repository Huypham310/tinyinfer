#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "tinyinfer/cpu_features.hpp"
#include "tinyinfer/kernels.hpp"

using namespace tinyinfer;
using namespace tinyinfer::kernels;

namespace {

Tensor random_tensor(Shape s, std::uint32_t seed) {
    Tensor t(std::move(s), DType::Float32);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0F, 1.0F);
    float* d = t.data<float>();
    for (std::int64_t i = 0; i < t.numel(); ++i) d[i] = u(rng);
    return t;
}

void expect_close(const Tensor& a, const Tensor& b, float tol) {
    ASSERT_EQ(a.shape(), b.shape());
    const float* x = a.data<float>();
    const float* y = b.data<float>();
    float worst = 0.0F;
    for (std::int64_t i = 0; i < a.numel(); ++i) worst = std::max(worst, std::abs(x[i] - y[i]));
    EXPECT_LE(worst, tol);
}

}  // namespace

// sizes straddle the MR / NR / MC / KC / NC block edges on purpose
class GemmSizes : public ::testing::TestWithParam<std::tuple<int, int, int>> {};

TEST_P(GemmSizes, BlockedMatchesNaive) {
    const auto [m, n, k] = GetParam();
    const std::string original = gemm_isa();
    for (const char* isa : {"scalar", "avx2", "avx512"}) {
    if (!set_gemm_isa(isa)) continue;  // CPU doesn't have it
    for (bool trans_b : {false, true}) {
        const Tensor a = random_tensor(Shape{m, k}, 1);
        const Tensor b = trans_b ? random_tensor(Shape{n, k}, 2) : random_tensor(Shape{k, n}, 2);
        const int ldb = trans_b ? k : n;
        Tensor want(Shape{m, n}, DType::Float32), got(Shape{m, n}, DType::Float32);
        gemm_naive(m, n, k, a.data<float>(), k, b.data<float>(), ldb, trans_b, want.data<float>(), n);
        gemm(m, n, k, a.data<float>(), k, b.data<float>(), ldb, trans_b, got.data<float>(), n);
        // float accumulation vs the double reference: error grows like sqrt(k) * eps
        expect_close(got, want, 1e-5F * static_cast<float>(std::sqrt(k) + 1) * 4);
    }
    }
    set_gemm_isa(original.c_str());
}

TEST(Gemm, IsaSelection) {
    const std::string original = gemm_isa();
    EXPECT_TRUE(set_gemm_isa("scalar"));
    EXPECT_STREQ(gemm_isa(), "scalar");
    EXPECT_FALSE(set_gemm_isa("neon"));
    EXPECT_STREQ(gemm_isa(), "scalar");  // unchanged after a refusal
    // whatever CPUID allows must also be selectable
    const auto& f = cpu_features();
    EXPECT_EQ(set_gemm_isa("avx2"), f.avx2 && f.fma);
    EXPECT_EQ(set_gemm_isa("avx512"), f.avx512f && f.fma);
    set_gemm_isa(original.c_str());
}

INSTANTIATE_TEST_SUITE_P(Kernels, GemmSizes,
                         ::testing::Values(std::make_tuple(1, 1, 1), std::make_tuple(3, 5, 7),
                                           std::make_tuple(4, 16, 256), std::make_tuple(5, 17, 257),
                                           std::make_tuple(64, 64, 64), std::make_tuple(65, 513, 300),
                                           std::make_tuple(130, 40, 600), std::make_tuple(7, 1000, 3)));

TEST(Gemm, AccumulateAddsToC) {
    const float a[] = {1, 2, 3, 4};  // 2x2
    const float b[] = {1, 0, 0, 1};  // identity
    float c[] = {10, 10, 10, 10};
    gemm(2, 2, 2, a, 2, b, 2, false, c, 2, true);
    EXPECT_FLOAT_EQ(c[0], 11);
    EXPECT_FLOAT_EQ(c[3], 14);
    gemm(2, 2, 2, a, 2, b, 2, false, c, 2, false);  // overwrite
    EXPECT_FLOAT_EQ(c[3], 4);
}

struct ConvCase {
    int n, c, h, w, oc, k;
    Conv2dParams p;
};

class ConvCases : public ::testing::TestWithParam<ConvCase> {};

TEST_P(ConvCases, Im2colMatchesDirect) {
    const ConvCase cc = GetParam();
    const Tensor x = random_tensor(Shape{cc.n, cc.c, cc.h, cc.w}, 3);
    const Tensor w = random_tensor(Shape{cc.oc, cc.c / cc.p.groups, cc.k, cc.k}, 4);
    const Tensor b = random_tensor(Shape{cc.oc}, 5);
    expect_close(conv2d(x, w, &b, cc.p), conv2d_naive(x, w, &b, cc.p), 1e-4F);
    expect_close(conv2d(x, w, nullptr, cc.p), conv2d_naive(x, w, nullptr, cc.p), 1e-4F);
}

INSTANTIATE_TEST_SUITE_P(
    Kernels, ConvCases,
    ::testing::Values(
        // ResNet-18 shapes: 3x3 stem, 3x3 stride 2, 1x1 stride-2 downsample, 1x1 pointwise
        ConvCase{1, 3, 64, 64, 64, 3, {{1, 1}, {1, 1, 1, 1}, {1, 1}, 1}},
        ConvCase{2, 16, 15, 15, 32, 3, {{2, 2}, {1, 1, 1, 1}, {1, 1}, 1}},
        ConvCase{1, 16, 16, 16, 32, 1, {{2, 2}, {0, 0, 0, 0}, {1, 1}, 1}},
        ConvCase{2, 24, 9, 7, 40, 1, {{1, 1}, {0, 0, 0, 0}, {1, 1}, 1}},
        // asymmetric padding, dilation, grouped and depthwise
        ConvCase{1, 8, 10, 12, 8, 3, {{1, 2}, {0, 1, 2, 0}, {1, 1}, 1}},
        ConvCase{1, 8, 12, 12, 16, 3, {{1, 1}, {2, 2, 2, 2}, {2, 2}, 1}},
        ConvCase{2, 8, 9, 9, 12, 3, {{1, 1}, {1, 1, 1, 1}, {1, 1}, 4}},
        ConvCase{1, 16, 8, 8, 16, 3, {{1, 1}, {1, 1, 1, 1}, {1, 1}, 16}}));

TEST(Conv, RejectsMismatchedShapes) {
    const Tensor x = random_tensor(Shape{1, 4, 8, 8}, 1);
    const Tensor w = random_tensor(Shape{8, 3, 3, 3}, 2);  // expects 3 input channels
    EXPECT_THROW(conv2d(x, w, nullptr, {}), std::invalid_argument);
    const Tensor big = random_tensor(Shape{8, 4, 9, 9}, 2);  // kernel larger than the image
    EXPECT_THROW(conv2d(x, big, nullptr, {}), std::invalid_argument);
    EXPECT_EQ(conv_out_size(64, 3, 2, 1, 1, 1), 32);
    EXPECT_EQ(conv_out_size(7, 3, 1, 0, 0, 2), 3);
}

TEST(Pool, MaxPoolIgnoresPaddingAndGlobalAverage) {
    // 1 x 1 x 2 x 2 of negatives: with padding, a naive "pad with 0" max would wrongly return 0
    Tensor x = Tensor::from(Shape{1, 1, 2, 2}, std::vector<float>{-4, -3, -2, -1});
    Pool2dParams p{{3, 3}, {2, 2}, {1, 1, 1, 1}};
    const Tensor y = max_pool2d(x, p);
    ASSERT_EQ(y.shape(), (Shape{1, 1, 1, 1}));
    EXPECT_FLOAT_EQ(y.data<float>()[0], -1);

    Tensor z = Tensor::from(Shape{1, 2, 2, 2}, std::vector<float>{1, 2, 3, 4, 10, 10, 10, 30});
    const Tensor g = global_avg_pool(z);
    EXPECT_FLOAT_EQ(g.data<float>()[0], 2.5F);
    EXPECT_FLOAT_EQ(g.data<float>()[1], 15.0F);
}

TEST(Ops, BatchNormReluAddSoftmax) {
    Tensor x = Tensor::from(Shape{1, 2, 1, 2}, std::vector<float>{1, 3, -2, 2});
    const Tensor scale = Tensor::from(Shape{2}, std::vector<float>{2, 1});
    const Tensor bias = Tensor::from(Shape{2}, std::vector<float>{0, 1});
    const Tensor mean = Tensor::from(Shape{2}, std::vector<float>{1, 0});
    const Tensor var = Tensor::from(Shape{2}, std::vector<float>{4, 1});
    const Tensor bn = batch_norm(x, scale, bias, mean, var, 0.0F);
    // channel 0: (x - 1) / 2 * 2 + 0 ; channel 1: x * 1 + 1
    EXPECT_FLOAT_EQ(bn.data<float>()[1], 2.0F);
    EXPECT_FLOAT_EQ(bn.data<float>()[2], -1.0F);

    const Tensor r = relu(bn);
    EXPECT_FLOAT_EQ(r.data<float>()[2], 0.0F);
    EXPECT_FLOAT_EQ(add(r, r).data<float>()[1], 4.0F);
    EXPECT_THROW(add(r, scale), std::invalid_argument);

    // huge logits: without subtracting the max, exp overflows to inf
    const Tensor logits = Tensor::from(Shape{2, 3}, std::vector<float>{1000, 1001, 1002, 0, 0, 0});
    const Tensor sm = softmax(logits);
    EXPECT_NEAR(sm.data<float>()[2], 0.6652409F, 1e-6);
    EXPECT_NEAR(sm.data<float>()[3], 1.0F / 3, 1e-6);
}

TEST(Ops, GemmOpWithBiasAndAlpha) {
    // y = 0.5 * x W^T + b, W stored [out, in] like a Linear layer
    const Tensor x = Tensor::from(Shape{1, 2}, std::vector<float>{1, 2});
    const Tensor w = Tensor::from(Shape{3, 2}, std::vector<float>{1, 0, 0, 1, 1, 1});
    const Tensor b = Tensor::from(Shape{3}, std::vector<float>{10, 20, 30});
    const Tensor y = gemm_op(x, w, &b, true, 0.5F, 1.0F);
    ASSERT_EQ(y.shape(), (Shape{1, 3}));
    EXPECT_FLOAT_EQ(y.data<float>()[0], 10.5F);
    EXPECT_FLOAT_EQ(y.data<float>()[2], 31.5F);
    EXPECT_THROW(gemm_op(x, w, &b, false), std::invalid_argument);  // inner dims 2 vs 3
}
