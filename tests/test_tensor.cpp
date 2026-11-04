#include <gtest/gtest.h>

#include <cstdint>
#include <numeric>
#include <vector>

#include "tinyinfer/layout.hpp"
#include "tinyinfer/tensor.hpp"

using namespace tinyinfer;

TEST(AlignedBuffer, IsAlignedZeroedAndMovable) {
    for (std::size_t bytes : {1u, 63u, 64u, 1000u}) {
        AlignedBuffer b(bytes);
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(b.data()) % kAlignment, 0u);
        for (std::size_t i = 0; i < bytes; ++i) ASSERT_EQ(b.data()[i], std::byte{0});
    }
    AlignedBuffer a(128);
    std::byte* p = a.data();
    AlignedBuffer moved(std::move(a));
    EXPECT_EQ(moved.data(), p);
    EXPECT_EQ(a.data(), nullptr);  // NOLINT(bugprone-use-after-move): checking the moved-from state
}

TEST(Shape, NumelStridesAndPrinting) {
    Shape s{2, 3, 4};
    EXPECT_EQ(s.numel(), 24);
    EXPECT_EQ(s.contiguous_strides(), (std::vector<std::int64_t>{12, 4, 1}));
    EXPECT_EQ(s.to_string(), "[2, 3, 4]");
    EXPECT_EQ(Shape{}.numel(), 1);  // scalar
    EXPECT_EQ((Shape{3, 0}).numel(), 0);
    EXPECT_THROW((Shape{2, -1}), std::invalid_argument);
}

TEST(Tensor, StorageIsAlignedAndTyped) {
    Tensor t(Shape{1, 3, 5, 5}, DType::Float32);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(t.data<float>()) % kAlignment, 0u);
    EXPECT_EQ(t.nbytes(), 75u * 4u);
    EXPECT_THROW((void)t.data<std::int8_t>(), std::invalid_argument);
}

TEST(Tensor, FromValuesAndIndexing) {
    auto t = Tensor::from(Shape{2, 3}, std::vector<float>{0, 1, 2, 3, 4, 5});
    EXPECT_EQ(t.at<float>({1, 2}), 5.0f);
    EXPECT_EQ(t.at<float>({0, 1}), 1.0f);
    EXPECT_THROW((void)t.at<float>({2, 0}), std::out_of_range);
    EXPECT_THROW((void)t.at<float>({0}), std::out_of_range);
    EXPECT_THROW(Tensor::from(Shape{2, 2}, std::vector<float>{1, 2, 3}), std::invalid_argument);
}

TEST(Tensor, ReshapeIsAViewCloneIsACopy) {
    auto t = Tensor::from(Shape{2, 3}, std::vector<std::int32_t>{0, 1, 2, 3, 4, 5});
    Tensor flat = t.reshape(Shape{6});
    EXPECT_TRUE(flat.shares_storage_with(t));
    flat.data<std::int32_t>()[5] = 99;
    EXPECT_EQ(t.at<std::int32_t>({1, 2}), 99);

    Tensor copy = t.clone();
    EXPECT_FALSE(copy.shares_storage_with(t));
    copy.data<std::int32_t>()[0] = -1;
    EXPECT_EQ(t.at<std::int32_t>({0, 0}), 0);
    EXPECT_THROW((void)t.reshape(Shape{4}), std::invalid_argument);
}

TEST(Layout, BlockedRoundTripWithChannelPadding) {
    // 10 channels with block 8 -> 2 blocks, the second one half padding
    const std::int64_t n = 2, c = 10, h = 3, w = 4;
    std::vector<float> values(static_cast<std::size_t>(n * c * h * w));
    std::iota(values.begin(), values.end(), 1.0f);
    auto nchw = Tensor::from(Shape{n, c, h, w}, values);

    Tensor blocked = to_blocked(nchw, 8);
    EXPECT_EQ(blocked.shape(), (Shape{2, 2, 3, 4, 8}));
    // element (n=1, c=9, h=2, w=3) -> block 1, lane 1
    EXPECT_EQ(blocked.at<float>({1, 1, 2, 3, 1}), nchw.at<float>({1, 9, 2, 3}));
    // lanes past channel 9 are padding
    EXPECT_EQ(blocked.at<float>({0, 1, 0, 0, 2}), 0.0f);
    EXPECT_EQ(blocked.at<float>({1, 1, 2, 3, 7}), 0.0f);

    Tensor back = from_blocked(blocked, c);
    ASSERT_EQ(back.shape(), nchw.shape());
    for (std::int64_t i = 0; i < back.numel(); ++i) {
        ASSERT_EQ(back.data<float>()[i], nchw.data<float>()[i]) << "at " << i;
    }
}

TEST(Layout, RejectsBadArguments) {
    Tensor t(Shape{1, 4, 2, 2}, DType::Float32);
    EXPECT_THROW(to_blocked(t, 0), std::invalid_argument);
    EXPECT_THROW(to_blocked(t.reshape(Shape{4, 4}), 8), std::invalid_argument);
    Tensor b = to_blocked(t, 8);  // 4 channels in one block of 8
    EXPECT_THROW(from_blocked(b, 9), std::invalid_argument);
    EXPECT_THROW(from_blocked(b, 0), std::invalid_argument);
}
