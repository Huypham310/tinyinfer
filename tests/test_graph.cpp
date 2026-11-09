#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "proto_writer.hpp"
#include "tinyinfer/graph.hpp"
#include "tinyinfer/onnx/model.hpp"

using namespace tinyinfer;

namespace {

std::filesystem::path data_dir() { return TINYINFER_TEST_DATA_DIR; }

Tensor read_floats(const std::string& path, Shape shape) {
    Tensor t(std::move(shape), DType::Float32);
    std::ifstream in(path, std::ios::binary);
    in.read(reinterpret_cast<char*>(t.data<float>()), static_cast<std::streamsize>(t.nbytes()));
    if (!in) ADD_FAILURE() << "could not read " << path;
    return t;
}

float max_abs_diff(const Tensor& a, const Tensor& b) {
    EXPECT_EQ(a.shape(), b.shape());
    float worst = 0.0F;
    for (std::int64_t i = 0; i < a.numel(); ++i) {
        worst = std::max(worst, std::abs(a.data<float>()[i] - b.data<float>()[i]));
    }
    return worst;
}

/// Runs a fixture with its saved ONNX Runtime reference; returns the max abs difference.
float parity(const std::string& name, const Shape& in_shape, const Shape& out_shape,
             const std::string& input_name, Graph::Stats* stats = nullptr) {
    const auto model = onnx::load_model(data_dir() / (name + ".onnx"));
    Graph g(model, {{input_name, in_shape}});
    if (stats) *stats = g.stats();
    const Tensor x = read_floats((data_dir() / (name + ".input.bin")).string(), in_shape);
    const Tensor want = read_floats((data_dir() / (name + ".output.bin")).string(), out_shape);
    Executor ex(g);
    const auto out = ex.run({{input_name, x}});
    EXPECT_EQ(out.size(), 1u);
    return max_abs_diff(out[0], want);
}

}  // namespace

TEST(Graph, ConvReluMatchesOnnxRuntime) {
    Graph::Stats s;
    EXPECT_LT(parity("conv_relu", Shape{2, 3, 8, 8}, Shape{2, 4, 8, 8}, "x", &s), 1e-5F);
    EXPECT_EQ(s.relu_fused, 1);
}

TEST(Graph, MixedTypesClipReshapeGemm) {
    // Reshape to a constant shape, Gemm without transB, Clip with the min input omitted
    EXPECT_LT(parity("mixed_types", Shape{2, 6}, Shape{3, 2}, "x"), 1e-5F);
}

TEST(Graph, BatchNormIsFoldedIntoTheConv) {
    Graph::Stats s;
    EXPECT_LT(parity("conv_bn_head", Shape{3, 3, 16, 16}, Shape{3, 5}, "x", &s), 1e-5F);
    EXPECT_EQ(s.bn_folded, 1);
    EXPECT_EQ(s.relu_fused, 1);
    // Conv(+BN+Relu), GlobalAveragePool, Reshape, Gemm, Softmax: 7 ONNX nodes -> 5
    const auto model = onnx::load_model(data_dir() / "conv_bn_head.onnx");
    Graph g(model, {{"x", Shape{1, 3, 16, 16}}});
    ASSERT_EQ(g.nodes().size(), 5u);
    EXPECT_EQ(g.nodes()[0].kind, OpKind::Conv);
    EXPECT_TRUE(g.nodes()[0].fused_relu);
}

TEST(Graph, SymbolicBatchNeedsAShape) {
    const auto model = onnx::load_model(data_dir() / "conv_relu.onnx");
    EXPECT_THROW(Graph(model, {}), GraphError);  // "N" is symbolic
    Graph g(model, {{"x", Shape{5, 3, 8, 8}}});
    EXPECT_EQ(g.values()[g.outputs()[0]].shape, (Shape{5, 4, 8, 8}));
    Executor ex(g);
    const Tensor wrong(Shape{1, 3, 8, 8}, DType::Float32);
    EXPECT_THROW(ex.run({{"x", wrong}}), GraphError);
    EXPECT_THROW(ex.run({}), GraphError);
}

namespace {

/// Hand-built graph: nodes listed in the *wrong* order, and an op we don't support.
std::string chain_model(bool shuffled, const char* last_op = "Relu") {
    using test_proto::Writer;
    auto node = [](const char* op, const char* in, const char* out) {
        Writer n;
        n.bytes_field(1, in).bytes_field(2, out).bytes_field(4, op);
        return n;
    };
    auto value = [](const char* name) {
        Writer dim1, dim2, shape, tensor_type, type, v;
        dim1.int_field(1, 2);
        dim2.int_field(1, 4);
        shape.message(1, dim1).message(1, dim2);
        tensor_type.int_field(1, 1).message(2, shape);
        type.message(1, tensor_type);
        v.bytes_field(1, name).message(2, type);
        return v;
    };
    Writer graph;
    std::vector<Writer> nodes = {node("Relu", "x", "a"), node("Identity", "a", "b"),
                                 node(last_op, "b", "y")};
    if (shuffled) std::swap(nodes[0], nodes[2]);
    for (const auto& n : nodes) graph.message(1, n);
    graph.message(11, value("x")).message(12, value("y"));
    Writer model;
    model.int_field(1, 8).message(7, graph);
    return model.str();
}

}  // namespace

TEST(Graph, TopologicalSortFixesNodeOrder) {
    const std::string bytes = chain_model(true);
    const auto model = onnx::parse_model(reinterpret_cast<const std::uint8_t*>(bytes.data()),
                                         bytes.size());
    Graph g(model, {});
    ASSERT_EQ(g.nodes().size(), 3u);
    EXPECT_EQ(g.values()[g.nodes()[0].inputs[0]].name, "x");
    EXPECT_EQ(g.values()[g.nodes()[2].output].name, "y");

    Tensor x = Tensor::from(Shape{2, 4}, std::vector<float>{-1, 2, -3, 4, 5, -6, 7, -8});
    const auto out = Executor(g).run({{"x", x}});
    EXPECT_FLOAT_EQ(out[0].data<float>()[0], 0.0F);
    EXPECT_FLOAT_EQ(out[0].data<float>()[1], 2.0F);
}

TEST(Graph, UnsupportedOpsFailAtLoadTime) {
    const std::string bytes = chain_model(false, "LSTM");
    const auto model = onnx::parse_model(reinterpret_cast<const std::uint8_t*>(bytes.data()),
                                         bytes.size());
    try {
        Graph g(model, {});
        FAIL() << "expected GraphError";
    } catch (const GraphError& e) {
        EXPECT_NE(std::string(e.what()).find("LSTM"), std::string::npos);
    }
}

TEST(Graph, LivenessFreesIntermediates) {
    // a chain only ever needs ~2 activations alive; keeping everything would need all of them
    const auto model = onnx::load_model(data_dir() / "conv_bn_head.onnx");
    Graph g(model, {{"x", Shape{8, 3, 16, 16}}});
    EXPECT_LT(g.stats().peak_live_bytes, g.stats().activation_bytes + 8 * 3 * 16 * 16 * 4);
    Executor ex(g);
    const Tensor x(Shape{8, 3, 16, 16}, DType::Float32);
    ex.run({{"x", x}});
    EXPECT_GT(ex.last_peak_bytes(), 0u);
    EXPECT_LE(ex.last_peak_bytes(), g.stats().peak_live_bytes);
}

TEST(Graph, TerraSenseResNet18MatchesOnnxRuntime) {
    // TINYINFER_RESNET18 = model path, TINYINFER_RESNET18_REF = stem of the
    // tools/ort_reference.py output (shape 1,3,64,64)
    const char* model_path = std::getenv("TINYINFER_RESNET18");
    const char* ref = std::getenv("TINYINFER_RESNET18_REF");
    if (model_path == nullptr || ref == nullptr) GTEST_SKIP() << "ResNet-18 reference not set";
    const auto model = onnx::load_model(model_path);
    Graph g(model, {{"input", Shape{1, 3, 64, 64}}});
    const Tensor x = read_floats(std::string(ref) + ".input.bin", Shape{1, 3, 64, 64});
    const Tensor want = read_floats(std::string(ref) + ".output.bin", Shape{1, 10});
    const auto out = Executor(g).run({{"input", x}});
    EXPECT_LT(max_abs_diff(out[0], want), 1e-3F);
    EXPECT_EQ(g.stats().relu_fused, 17);
}
