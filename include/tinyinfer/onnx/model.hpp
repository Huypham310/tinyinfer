#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "tinyinfer/onnx/wire.hpp"
#include "tinyinfer/tensor.hpp"

namespace tinyinfer::onnx {

/// Subset of onnx.AttributeProto.AttributeType we understand. Values match the proto enum.
enum class AttrType : std::uint8_t {
    Undefined = 0,
    Float = 1,
    Int = 2,
    String = 3,
    Tensor = 4,
    Graph = 5,
    Floats = 6,
    Ints = 7,
    Strings = 8,
};

const char* attr_type_name(AttrType t) noexcept;

struct Attribute {
    std::string name;
    AttrType type = AttrType::Undefined;
    float f = 0.0F;
    std::int64_t i = 0;
    std::string s;
    tinyinfer::Tensor t;
    std::vector<float> floats;
    std::vector<std::int64_t> ints;
    std::vector<std::string> strings;
};

struct Node {
    std::string name;
    std::string op_type;
    std::string domain;  // "" = the default ai.onnx domain
    std::vector<std::string> inputs;  // "" marks an omitted optional input
    std::vector<std::string> outputs;
    std::vector<Attribute> attributes;

    [[nodiscard]] const Attribute* find_attr(std::string_view attr) const noexcept;
    [[nodiscard]] std::int64_t attr_int(std::string_view attr, std::int64_t fallback) const;
    [[nodiscard]] float attr_float(std::string_view attr, float fallback) const;
    [[nodiscard]] std::string attr_string(std::string_view attr, std::string fallback) const;
    [[nodiscard]] std::vector<std::int64_t> attr_ints(std::string_view attr,
                                                      std::vector<std::int64_t> fallback) const;
};

/// One dimension of a declared shape: either a number or a symbolic name like "batch".
struct Dim {
    std::int64_t value = -1;  // -1 when symbolic or unknown
    std::string param;
    [[nodiscard]] bool is_static() const noexcept { return value >= 0; }
};

/// Graph input/output declaration (onnx.ValueInfoProto with a tensor type).
struct ValueInfo {
    std::string name;
    int elem_type = 0;  // onnx.TensorProto.DataType, 1 = FLOAT
    std::vector<Dim> dims;
    bool has_shape = false;

    [[nodiscard]] std::string shape_string() const;
};

struct Initializer {
    std::string name;
    tinyinfer::Tensor value;
};

struct Graph {
    std::string name;
    std::vector<Node> nodes;
    std::vector<Initializer> initializers;
    std::vector<ValueInfo> inputs;
    std::vector<ValueInfo> outputs;

    [[nodiscard]] const Initializer* find_initializer(std::string_view wanted) const noexcept;
    /// Inputs the caller has to feed. Older exporters (IR < 4) also list every weight as a
    /// graph input; those are filtered out here.
    [[nodiscard]] std::vector<const ValueInfo*> runtime_inputs() const;
    [[nodiscard]] std::int64_t parameter_count() const noexcept;
    [[nodiscard]] std::size_t parameter_bytes() const noexcept;
};

struct Model {
    std::int64_t ir_version = 0;
    std::string producer_name;
    std::string producer_version;
    std::map<std::string, std::int64_t> opsets;  // domain -> version ("" = ai.onnx)
    Graph graph;

    [[nodiscard]] std::int64_t opset() const noexcept;
};

/// Parses a serialized onnx.ModelProto. Throws ParseError on malformed input.
Model parse_model(const std::uint8_t* data, std::size_t size);
Model load_model(const std::filesystem::path& path);

/// onnx.TensorProto -> Tensor (float32 / int32 / int64 / int8; raw_data or typed fields).
tinyinfer::Tensor parse_tensor(WireReader r, std::string* name = nullptr);

}  // namespace tinyinfer::onnx
