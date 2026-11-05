#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

#include "proto_writer.hpp"
#include "tinyinfer/onnx/model.hpp"

using namespace tinyinfer;
using namespace tinyinfer::onnx;

namespace {

std::filesystem::path data_dir() { return TINYINFER_TEST_DATA_DIR; }

std::vector<std::uint8_t> read_bytes(const std::filesystem::path& p) {
    std::ifstream in(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

}  // namespace

TEST(Onnx, ConvReluFixture) {
    const Model m = load_model(data_dir() / "conv_relu.onnx");
    EXPECT_EQ(m.ir_version, 8);
    EXPECT_EQ(m.opset(), 17);
    EXPECT_EQ(m.producer_name, "make_test_models");

    const Graph& g = m.graph;
    EXPECT_EQ(g.name, "conv_relu");
    ASSERT_EQ(g.nodes.size(), 2u);
    const Node& conv = g.nodes[0];
    EXPECT_EQ(conv.op_type, "Conv");
    EXPECT_EQ(conv.name, "conv0");
    EXPECT_EQ(conv.inputs, (std::vector<std::string>{"x", "W", "B"}));
    EXPECT_EQ(conv.outputs, (std::vector<std::string>{"c"}));
    EXPECT_EQ(conv.attr_ints("kernel_shape", {}), (std::vector<std::int64_t>{3, 3}));
    EXPECT_EQ(conv.attr_ints("pads", {}), (std::vector<std::int64_t>{1, 1, 1, 1}));
    EXPECT_EQ(conv.attr_int("group", 0), 1);
    EXPECT_EQ(conv.attr_string("auto_pad", ""), "NOTSET");
    // missing attributes fall back, wrong-typed ones are an error
    EXPECT_EQ(conv.attr_ints("dilations", {1, 1}), (std::vector<std::int64_t>{1, 1}));
    EXPECT_THROW((void)conv.attr_int("pads", 0), ParseError);

    ASSERT_EQ(g.inputs.size(), 1u);
    EXPECT_EQ(g.inputs[0].shape_string(), "[N, 3, 8, 8]");
    EXPECT_FALSE(g.inputs[0].dims[0].is_static());
    EXPECT_EQ(g.outputs[0].name, "y");

    // W came as raw_data, B as float_data
    const Initializer* w = g.find_initializer("W");
    ASSERT_NE(w, nullptr);
    EXPECT_EQ(w->value.shape(), (Shape{4, 3, 3, 3}));
    EXPECT_FLOAT_EQ(w->value.data<float>()[0], 0.0F);
    EXPECT_FLOAT_EQ(w->value.data<float>()[107], 1.07F);
    const Initializer* b = g.find_initializer("B");
    ASSERT_NE(b, nullptr);
    EXPECT_FLOAT_EQ(b->value.data<float>()[3], -1.0F);
    EXPECT_EQ(g.parameter_count(), 108 + 4);
    EXPECT_EQ(g.find_initializer("nope"), nullptr);
}

TEST(Onnx, MixedTypesFixture) {
    const Model m = load_model(data_dir() / "mixed_types.onnx");
    const Graph& g = m.graph;

    const Tensor& shape = g.find_initializer("shape")->value;
    EXPECT_EQ(shape.dtype(), DType::Int64);
    EXPECT_EQ(shape.data<std::int64_t>()[1], 4);
    const Tensor& ids = g.find_initializer("ids")->value;
    EXPECT_EQ(ids.dtype(), DType::Int32);
    EXPECT_EQ(ids.data<std::int32_t>()[1], -8);  // negative int32 survives the varint round trip
    const Tensor& qw = g.find_initializer("qw")->value;
    EXPECT_EQ(qw.dtype(), DType::Int8);
    EXPECT_EQ(qw.data<std::int8_t>()[0], -128);
    const Tensor& cmax = g.find_initializer("cmax")->value;
    EXPECT_EQ(cmax.shape().rank(), 0u);
    EXPECT_FLOAT_EQ(cmax.data<float>()[0], 6.0F);

    const Node& gemm = g.nodes[1];
    EXPECT_FLOAT_EQ(gemm.attr_float("alpha", 1.0F), 0.5F);
    EXPECT_EQ(gemm.attr_int("transB", 1), 0);
    // omitted optional input keeps its slot as ""
    EXPECT_EQ(g.nodes[2].inputs, (std::vector<std::string>{"g", "", "cmax"}));

    // all six names are graph inputs, but only x has to be fed
    EXPECT_EQ(g.inputs.size(), 6u);
    const auto feeds = g.runtime_inputs();
    ASSERT_EQ(feeds.size(), 1u);
    EXPECT_EQ(feeds[0]->name, "x");
}

TEST(Onnx, HandBuiltModelWithUnpackedAndUnknownFields) {
    using test_proto::Writer;
    // TensorProto with unpacked dims and float_data: legal proto2-style encoding
    Writer tensor;
    tensor.int_field(1, 2).int_field(1, 2).int_field(2, 1);
    for (float f : {1.0F, 2.0F, 3.0F, 4.0F}) tensor.float_field(4, f);
    tensor.bytes_field(8, "w").int_field(99, 1);  // unknown field 99 is skipped
    Writer node;
    node.bytes_field(1, "x").bytes_field(1, "w").bytes_field(2, "y").bytes_field(4, "MatMul");
    Writer graph;
    graph.message(1, node).message(5, tensor).bytes_field(2, "g");
    Writer opset;
    opset.int_field(2, 21);
    Writer model;
    model.int_field(1, 10).message(8, opset).message(7, graph).bytes_field(6, "doc");

    const Model m = parse_model(model.data(), model.size());
    EXPECT_EQ(m.opset(), 21);
    const Tensor& w = m.graph.find_initializer("w")->value;
    EXPECT_EQ(w.shape(), (Shape{2, 2}));
    EXPECT_FLOAT_EQ(w.at<float>({1, 0}), 3.0F);
    EXPECT_EQ(m.graph.nodes[0].op_type, "MatMul");
}

TEST(Onnx, RejectsBadTensors) {
    using test_proto::Writer;
    auto parse_init = [](const Writer& tensor) {
        Writer graph;
        graph.message(5, tensor);
        Writer model;
        model.message(7, graph);
        return parse_model(model.data(), model.size());
    };
    Writer wrong_count;  // shape [3] but two floats
    wrong_count.int_field(1, 3).int_field(2, 1).float_field(4, 1.0F).float_field(4, 2.0F);
    EXPECT_THROW(parse_init(wrong_count), ParseError);

    Writer huge;  // 2^30 x 2^30 elements: must fail before trying to allocate
    huge.int_field(1, std::int64_t{1} << 30).int_field(1, std::int64_t{1} << 30).int_field(2, 1);
    EXPECT_THROW(parse_init(huge), ParseError);

    Writer doubles;  // DOUBLE (11) isn't supported
    doubles.int_field(1, 1).int_field(2, 11);
    EXPECT_THROW(parse_init(doubles), ParseError);

    Writer external;
    external.int_field(1, 1).int_field(2, 1).int_field(14, 1);
    EXPECT_THROW(parse_init(external), ParseError);

    Writer raw_mismatch;  // shape [2] float needs 8 bytes
    raw_mismatch.int_field(1, 2).int_field(2, 1).bytes_field(9, "1234567");
    EXPECT_THROW(parse_init(raw_mismatch), ParseError);
}

TEST(Onnx, NotAModel) {
    EXPECT_THROW(load_model(data_dir() / "does_not_exist.onnx"), ParseError);
    test_proto::Writer w;
    w.int_field(1, 8);  // ir_version only, no graph
    EXPECT_THROW(parse_model(w.data(), w.size()), ParseError);
}

TEST(Onnx, TruncatedFilesNeverCrash) {
    const auto bytes = read_bytes(data_dir() / "conv_relu.onnx");
    ASSERT_GT(bytes.size(), 100u);
    int parsed = 0;
    for (std::size_t n = 0; n < bytes.size(); ++n) {
        try {
            parse_model(bytes.data(), n);
            ++parsed;  // a cut exactly at a top-level field boundary is still a valid message
        } catch (const ParseError&) {
        }
    }
    // only cuts after the graph field can succeed
    EXPECT_LT(parsed, 40);
}

TEST(Onnx, RandomByteFlipsOnlyThrowParseError) {
    const auto original = read_bytes(data_dir() / "mixed_types.onnx");
    std::mt19937 rng(1234);
    std::uniform_int_distribution<std::size_t> pos(0, original.size() - 1);
    std::uniform_int_distribution<int> byte(0, 255);
    int failures = 0;
    for (int trial = 0; trial < 3000; ++trial) {
        auto bytes = original;
        for (int k = 0; k < 3; ++k) bytes[pos(rng)] = static_cast<std::uint8_t>(byte(rng));
        try {
            parse_model(bytes.data(), bytes.size());
        } catch (const ParseError&) {
            ++failures;
        }
        // anything else (bad_alloc, out_of_range, a crash) fails the test
    }
    EXPECT_GT(failures, 0);
}

TEST(Onnx, TerraSenseResNet18IfAvailable) {
    // the real model is 43 MB, so it's not in the repo; point TINYINFER_RESNET18 at it
    const char* path = std::getenv("TINYINFER_RESNET18");
    if (path == nullptr) GTEST_SKIP() << "TINYINFER_RESNET18 not set";
    const Model m = load_model(path);
    EXPECT_EQ(m.graph.nodes.size(), 49u);
    EXPECT_EQ(m.graph.parameter_count(), 11176846);
    EXPECT_EQ(m.graph.runtime_inputs().size(), 1u);
}
