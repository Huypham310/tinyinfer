#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "tinyinfer/cpu_features.hpp"
#include "tinyinfer/graph.hpp"
#include "tinyinfer/onnx/model.hpp"
#include "tinyinfer/quant.hpp"

using namespace tinyinfer;
using namespace tinyinfer::quant;

namespace {

Tensor random_tensor(Shape s, std::uint32_t seed, float lo = -1.0F, float hi = 1.0F) {
    Tensor t(std::move(s), DType::Float32);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(lo, hi);
    for (std::int64_t i = 0; i < t.numel(); ++i) t.data<float>()[i] = u(rng);
    return t;
}

}  // namespace

TEST(Quant, U8ParamsAlwaysRepresentZero) {
    const QParams relu = choose_u8(0.0F, 6.0F);
    EXPECT_EQ(relu.zero_point, 0);
    EXPECT_NEAR(relu.scale, 6.0F / 255, 1e-7);
    const QParams sym = choose_u8(-2.0F, 2.0F);
    EXPECT_NEAR(sym.zero_point, 128, 1);
    // a range that doesn't contain 0 is widened so zero padding stays exact
    const QParams shifted = choose_u8(0.5F, 1.5F);
    EXPECT_EQ(shifted.zero_point, 0);
    EXPECT_EQ(quantize_u8(0.0F, shifted), 0);
    // round trip error is at most half a step; values outside the range saturate
    for (float x : {-2.0F, -0.3F, 0.0F, 0.7F, 1.99F}) {
        const auto q = quantize_u8(x, sym);
        EXPECT_NEAR((static_cast<float>(q) - static_cast<float>(sym.zero_point)) * sym.scale, x,
                    sym.scale / 2 + 1e-6F);
    }
    EXPECT_EQ(quantize_u8(100.0F, sym), 255);
    EXPECT_EQ(quantize_u8(-100.0F, sym), 0);
}

class Int8Conv : public ::testing::TestWithParam<std::tuple<const char*, int, int, int, int>> {};

TEST_P(Int8Conv, MatchesFloatConvWithinQuantizationError) {
    const auto [isa, c, oc, hw, k] = GetParam();
    if (!set_int8_isa(isa)) GTEST_SKIP() << isa << " not available on this CPU";
    const Tensor x = random_tensor(Shape{2, c, hw, hw}, 1, -1.0F, 2.0F);
    const Tensor w = random_tensor(Shape{oc, c, k, k}, 2, -0.5F, 0.5F);
    const Tensor b = random_tensor(Shape{oc}, 3);
    kernels::Conv2dParams p;
    p.pads = {k / 2, k / 2, k / 2, k / 2};
    const QConv q = quantize_conv(w, &b, p, choose_u8(-1.0F, 2.0F));
    const Tensor got = conv2d_int8(x, q);
    const Tensor want = kernels::conv2d(x, w, &b, p);
    ASSERT_EQ(got.shape(), want.shape());
    float worst = 0.0F, range = 0.0F;
    for (std::int64_t i = 0; i < want.numel(); ++i) {
        worst = std::max(worst, std::abs(got.data<float>()[i] - want.data<float>()[i]));
        range = std::max(range, std::abs(want.data<float>()[i]));
    }
    // 8-bit inputs and weights: errors of a few tenths of a percent of the output range
    EXPECT_LT(worst, 0.02F * range);
    set_int8_isa("vnni") || set_int8_isa("scalar");
}

INSTANTIATE_TEST_SUITE_P(
    Quant, Int8Conv,
    ::testing::Values(std::make_tuple("scalar", 3, 8, 9, 3), std::make_tuple("vnni", 3, 8, 9, 3),
                      // odd sizes: k not a multiple of 4, oc not a multiple of 8, n not of 32
                      std::make_tuple("vnni", 5, 13, 7, 3), std::make_tuple("vnni", 64, 64, 8, 3),
                      std::make_tuple("vnni", 16, 40, 6, 1)));

TEST(Int8Conv, BatchedMatchesOneImageAtATime) {
    // 32 ch x 3x3 on 56x56 is ~0.9 MB of columns per image: 2 images per chunk, then a tail of 1
    const Tensor x = random_tensor(Shape{3, 32, 56, 56}, 21, -1.0F, 2.0F);
    const Tensor w = random_tensor(Shape{8, 32, 3, 3}, 22, -0.5F, 0.5F);
    kernels::Conv2dParams p;
    p.pads = {1, 1, 1, 1};
    const QConv q = quantize_conv(w, nullptr, p, choose_u8(-1.0F, 2.0F));
    const Tensor all = conv2d_int8(x, q);
    const std::int64_t per = all.numel() / 3;
    for (int i = 0; i < 3; ++i) {
        Tensor one(Shape{1, 32, 56, 56}, DType::Float32);
        std::copy_n(x.data<float>() + i * one.numel(), one.numel(), one.data<float>());
        const Tensor y = conv2d_int8(one, q);
        // integer accumulation + the same float ops: identical, not just close
        for (std::int64_t j = 0; j < per; ++j) {
            ASSERT_EQ(y.data<float>()[j], all.data<float>()[i * per + j]) << "image " << i;
        }
    }
}

TEST(Quant, VnniKernelIsBitExactAgainstTheReference) {
    if (!set_int8_isa("vnni")) GTEST_SKIP() << "no AVX-512 VNNI";
    // the int8 GEMM is exact integer math: packed VNNI and plain loops must agree to the bit
    const Tensor w = random_tensor(Shape{19, 7, 3, 3}, 4);
    kernels::Conv2dParams p;
    const Tensor x = random_tensor(Shape{1, 7, 10, 10}, 5, 0.0F, 3.0F);
    const QParams in = choose_u8(0.0F, 3.0F);
    const QConv q = quantize_conv(w, nullptr, p, in);
    const Tensor vnni = conv2d_int8(x, q);
    set_int8_isa("scalar");
    const Tensor scalar = conv2d_int8(x, q);
    for (std::int64_t i = 0; i < vnni.numel(); ++i) {
        ASSERT_EQ(vnni.data<float>()[i], scalar.data<float>()[i]) << i;
    }
    set_int8_isa("vnni");

    std::vector<std::int8_t> a = {1, -2, 3, 127, -127, 0};  // 2 x 3
    std::vector<std::uint8_t> bm = {255, 0, 1, 2, 3, 4};    // 3 x 2
    std::vector<std::int32_t> c(4);
    gemm_s8u8_ref(2, 2, 3, a.data(), bm.data(), c.data());
    EXPECT_EQ(c[0], 1 * 255 - 2 * 1 + 3 * 3);
    EXPECT_EQ(c[3], 127 * 0 - 127 * 2 + 0 * 4);
}

TEST(Quant, CalibratedGraphStaysCloseToFloat) {
    const auto model = onnx::load_model(std::string(TINYINFER_TEST_DATA_DIR) + "/conv_bn_head.onnx");
    Graph g(model, {{"x", Shape{4, 3, 16, 16}}});
    std::vector<Tensor> calib = {random_tensor(Shape{4, 3, 16, 16}, 11, -2.0F, 2.0F),
                                 random_tensor(Shape{4, 3, 16, 16}, 12, -2.0F, 2.0F)};
    const QuantPlan plan = calibrate(g, "x", calib);
    ASSERT_TRUE(plan[0].has_value());  // the folded Conv(+BN+Relu) got quantized
    EXPECT_GT(plan[0]->input.zero_point, 0);  // its input (the image) is signed

    const Tensor x = random_tensor(Shape{4, 3, 16, 16}, 13, -2.0F, 2.0F);
    Executor fp(g), q8(g);
    q8.set_int8(&plan);
    const Tensor a = fp.run({{"x", x}})[0], b = q8.run({{"x", x}})[0];
    for (std::int64_t i = 0; i < a.numel(); ++i) {
        EXPECT_NEAR(a.data<float>()[i], b.data<float>()[i], 0.02F);  // softmax probabilities
    }
}
