#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

#include "proto_writer.hpp"
#include "tinyinfer/onnx/wire.hpp"

using tinyinfer::onnx::ParseError;
using tinyinfer::onnx::WireReader;
using tinyinfer::onnx::WireType;

TEST(Wire, VarintsRoundTrip) {
    const std::uint64_t values[] = {0,   1,       127, 128, 300, 16384, 0xFFFFFFFFull,
                                     std::numeric_limits<std::uint64_t>::max()};
    for (auto v : values) {
        test_proto::Writer w;
        w.varint(v);
        WireReader r(w.data(), w.size());
        EXPECT_EQ(r.varint(), v);
        EXPECT_TRUE(r.done());
    }
}

TEST(Wire, KnownEncodings) {
    // the example from the protobuf docs: 150 -> 96 01
    const std::uint8_t b150[] = {0x96, 0x01};
    EXPECT_EQ(WireReader(b150, 2).varint(), 150u);
    // int64 -1 is ten bytes: nine 0xFF then 0x01
    test_proto::Writer w;
    w.varint(static_cast<std::uint64_t>(std::int64_t{-1}));
    EXPECT_EQ(w.size(), 10u);
    EXPECT_EQ(WireReader(w.data(), w.size()).int64(), -1);
}

TEST(Wire, TagsAndPayloads) {
    test_proto::Writer inner;
    inner.int_field(1, 42);
    test_proto::Writer w;
    w.int_field(3, 7).float_field(4, 2.5F).bytes_field(5, "hi").message(6, inner);

    WireReader r(w.data(), w.size());
    auto f = r.next();
    EXPECT_EQ(f.number, 3u);
    EXPECT_EQ(f.type, WireType::Varint);
    EXPECT_EQ(r.int64(), 7);
    f = r.next();
    EXPECT_EQ(f.type, WireType::Fixed32);
    EXPECT_FLOAT_EQ(r.f32(), 2.5F);
    f = r.next();
    EXPECT_EQ(r.bytes(), "hi");
    f = r.next();
    WireReader m = r.message();
    EXPECT_EQ(m.next().number, 1u);
    EXPECT_EQ(m.int64(), 42);
    EXPECT_TRUE(m.done());
    EXPECT_TRUE(r.done());
}

TEST(Wire, SkipsEveryWireType) {
    test_proto::Writer w;
    w.int_field(1, 123456789).float_field(2, 1.0F).bytes_field(3, "skipped");
    w.tag(4, 1).raw(std::string(8, '\0'));  // fixed64
    w.int_field(9, 5);
    WireReader r(w.data(), w.size());
    for (int i = 0; i < 4; ++i) {
        auto f = r.next();
        r.skip(f.type);
    }
    EXPECT_EQ(r.next().number, 9u);
    EXPECT_EQ(r.int64(), 5);
}

TEST(Wire, RejectsTruncatedAndBadInput) {
    const std::uint8_t unfinished[] = {0x80, 0x80};  // continuation bit, then nothing
    EXPECT_THROW(WireReader(unfinished, 2).varint(), ParseError);

    std::uint8_t too_long[11];
    std::fill(std::begin(too_long), std::end(too_long), 0xFF);
    EXPECT_THROW(WireReader(too_long, 11).varint(), ParseError);

    test_proto::Writer len;
    len.tag(1, 2).varint(100).raw("short");  // claims 100 bytes, has 5
    WireReader r(len.data(), len.size());
    r.next();
    EXPECT_THROW(r.bytes(), ParseError);

    test_proto::Writer group;
    group.tag(1, 3);  // start-group: deprecated, we refuse it
    EXPECT_THROW(WireReader(group.data(), group.size()).next(), ParseError);

    const std::uint8_t field_zero[] = {0x00};
    EXPECT_THROW(WireReader(field_zero, 1).next(), ParseError);

    const std::uint8_t three[] = {1, 2, 3};
    EXPECT_THROW(WireReader(three, 3).fixed32(), ParseError);
}
