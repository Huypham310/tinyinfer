// ONNX ModelProto -> tinyinfer structs, decoded straight from the protobuf wire format.
//
// Field numbers come from onnx/onnx.proto (IR version 10). Anything we don't use is skipped,
// so newer files with extra fields still load.

#include "tinyinfer/onnx/model.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>

namespace tinyinfer::onnx {

namespace {

// onnx.TensorProto.DataType values we can hold in a Tensor
constexpr int kFloat = 1;
constexpr int kInt8 = 3;
constexpr int kInt32 = 6;
constexpr int kInt64 = 7;

template <class Push>
void read_repeated_varint(WireReader& r, WireType type, Push&& push) {
    // proto3 packs repeated scalars by default, but the spec says parsers must accept both forms
    if (type == WireType::Bytes) {
        WireReader packed = r.message();
        while (!packed.done()) push(packed.int64());
    } else if (type == WireType::Varint) {
        push(r.int64());
    } else {
        throw ParseError("expected a varint field");
    }
}

template <class Push>
void read_repeated_float(WireReader& r, WireType type, Push&& push) {
    if (type == WireType::Bytes) {
        WireReader packed = r.message();
        while (!packed.done()) push(packed.f32());
    } else if (type == WireType::Fixed32) {
        push(r.f32());
    } else {
        throw ParseError("expected a float field");
    }
}

std::string as_string(std::string_view v) { return std::string(v); }

DType dtype_from_onnx(int t, const std::string& name) {
    switch (t) {
        case kFloat:
            return DType::Float32;
        case kInt8:
            return DType::Int8;
        case kInt32:
            return DType::Int32;
        case kInt64:
            return DType::Int64;
        default:
            throw ParseError("tensor '" + name + "' has unsupported data type " +
                             std::to_string(t));
    }
}

template <class T, class Src>
void copy_typed(Tensor& t, const std::vector<Src>& src, const std::string& name) {
    if (static_cast<std::int64_t>(src.size()) != t.numel()) {
        throw ParseError("tensor '" + name + "' has " + std::to_string(src.size()) +
                         " values for shape " + t.shape().to_string());
    }
    T* dst = t.data<T>();
    for (std::size_t i = 0; i < src.size(); ++i) dst[i] = static_cast<T>(src[i]);
}

void copy_raw(Tensor& t, std::string_view raw, const std::string& name) {
    if (raw.size() != t.nbytes()) {
        throw ParseError("tensor '" + name + "' raw_data has " + std::to_string(raw.size()) +
                         " bytes, shape " + t.shape().to_string() + " needs " +
                         std::to_string(t.nbytes()));
    }
    if (raw.empty()) return;
    // raw_data is little-endian, same as the host; one memcpy into the aligned buffer
    void* dst = nullptr;
    switch (t.dtype()) {
        case DType::Float32:
            dst = t.data<float>();
            break;
        case DType::Int8:
            dst = t.data<std::int8_t>();
            break;
        case DType::Int32:
            dst = t.data<std::int32_t>();
            break;
        case DType::Int64:
            dst = t.data<std::int64_t>();
            break;
    }
    std::memcpy(dst, raw.data(), raw.size());
}

Attribute parse_attribute(WireReader r) {
    Attribute a;
    bool has_t = false;
    while (!r.done()) {
        const auto f = r.next();
        switch (f.number) {
            case 1:
                a.name = as_string(r.bytes());
                break;
            case 2:
                a.f = r.f32();
                break;
            case 3:
                a.i = r.int64();
                break;
            case 4:
                a.s = as_string(r.bytes());
                break;
            case 5:
                a.t = parse_tensor(r.message());
                has_t = true;
                break;
            case 7:
                read_repeated_float(r, f.type, [&](float v) { a.floats.push_back(v); });
                break;
            case 8:
                read_repeated_varint(r, f.type, [&](std::int64_t v) { a.ints.push_back(v); });
                break;
            case 9:
                a.strings.push_back(as_string(r.bytes()));
                break;
            case 20:
                a.type = static_cast<AttrType>(r.int64());
                break;
            default:  // g (6), graphs, sparse tensors, doc strings...
                r.skip(f.type);
        }
    }
    if (a.type == AttrType::Undefined) {
        // very old exporters leave `type` unset; guess from whichever field is filled
        if (has_t) a.type = AttrType::Tensor;
        else if (!a.ints.empty()) a.type = AttrType::Ints;
        else if (!a.floats.empty()) a.type = AttrType::Floats;
        else if (!a.strings.empty()) a.type = AttrType::Strings;
        else if (!a.s.empty()) a.type = AttrType::String;
    }
    return a;
}

Node parse_node(WireReader r) {
    Node n;
    while (!r.done()) {
        const auto f = r.next();
        switch (f.number) {
            case 1:
                n.inputs.push_back(as_string(r.bytes()));
                break;
            case 2:
                n.outputs.push_back(as_string(r.bytes()));
                break;
            case 3:
                n.name = as_string(r.bytes());
                break;
            case 4:
                n.op_type = as_string(r.bytes());
                break;
            case 5:
                n.attributes.push_back(parse_attribute(r.message()));
                break;
            case 7:
                n.domain = as_string(r.bytes());
                break;
            default:
                r.skip(f.type);
        }
    }
    if (n.op_type.empty()) throw ParseError("node '" + n.name + "' has no op_type");
    return n;
}

void parse_shape(WireReader r, ValueInfo& v) {
    v.has_shape = true;
    while (!r.done()) {
        const auto f = r.next();
        if (f.number != 1) {
            r.skip(f.type);
            continue;
        }
        Dim d;
        WireReader dr = r.message();
        while (!dr.done()) {
            const auto df = dr.next();
            if (df.number == 1) d.value = dr.int64();
            else if (df.number == 2) d.param = as_string(dr.bytes());
            else dr.skip(df.type);
        }
        v.dims.push_back(std::move(d));
    }
}

ValueInfo parse_value_info(WireReader r) {
    ValueInfo v;
    while (!r.done()) {
        const auto f = r.next();
        if (f.number == 1) {
            v.name = as_string(r.bytes());
        } else if (f.number == 2) {
            // TypeProto: we only look at tensor_type (1); sequences/maps are not supported
            WireReader tr = r.message();
            while (!tr.done()) {
                const auto tf = tr.next();
                if (tf.number != 1) {
                    tr.skip(tf.type);
                    continue;
                }
                WireReader tt = tr.message();
                while (!tt.done()) {
                    const auto ttf = tt.next();
                    if (ttf.number == 1) v.elem_type = static_cast<int>(tt.int64());
                    else if (ttf.number == 2) parse_shape(tt.message(), v);
                    else tt.skip(ttf.type);
                }
            }
        } else {
            r.skip(f.type);
        }
    }
    return v;
}

Graph parse_graph(WireReader r) {
    Graph g;
    while (!r.done()) {
        const auto f = r.next();
        switch (f.number) {
            case 1:
                g.nodes.push_back(parse_node(r.message()));
                break;
            case 2:
                g.name = as_string(r.bytes());
                break;
            case 5: {
                Initializer init;
                init.value = parse_tensor(r.message(), &init.name);
                g.initializers.push_back(std::move(init));
                break;
            }
            case 11:
                g.inputs.push_back(parse_value_info(r.message()));
                break;
            case 12:
                g.outputs.push_back(parse_value_info(r.message()));
                break;
            case 15:
                throw ParseError("sparse initializers are not supported");
            default:  // value_info (13) hints are recomputed later, doc strings etc.
                r.skip(f.type);
        }
    }
    return g;
}

}  // namespace

Tensor parse_tensor(WireReader r, std::string* name_out) {
    std::vector<std::int64_t> dims;
    int data_type = 0;
    std::string name;
    std::string_view raw;
    bool has_raw = false;
    bool external = false;
    std::vector<float> floats;
    std::vector<std::int64_t> ints;  // int32_data and int64_data both decode to this

    while (!r.done()) {
        const auto f = r.next();
        switch (f.number) {
            case 1:
                read_repeated_varint(r, f.type, [&](std::int64_t v) { dims.push_back(v); });
                break;
            case 2:
                data_type = static_cast<int>(r.int64());
                break;
            case 4:
                read_repeated_float(r, f.type, [&](float v) { floats.push_back(v); });
                break;
            case 5:
            case 7:
                read_repeated_varint(r, f.type, [&](std::int64_t v) { ints.push_back(v); });
                break;
            case 8:
                name = as_string(r.bytes());
                break;
            case 9:
                raw = r.bytes();
                has_raw = true;
                break;
            case 13:
                external = true;
                r.skip(f.type);
                break;
            case 14:
                external = external || r.int64() == 1;
                break;
            default:
                r.skip(f.type);
        }
    }
    if (external) {
        // TODO: h? tr? external data (model > 2GB luu weight ra file riêng)
        throw ParseError("tensor '" + name + "' stores its data in an external file");
    }
    const DType dtype = dtype_from_onnx(data_type, name);
    // validate the element count against the data we actually have *before* allocating:
    // a corrupted dims field could otherwise ask for terabytes
    std::uint64_t numel = 1;
    for (auto d : dims) {
        if (d < 0) throw ParseError("tensor '" + name + "' has a negative dimension");
        if (d != 0 && numel > (std::uint64_t{1} << 40) / static_cast<std::uint64_t>(d)) {
            throw ParseError("tensor '" + name + "' is too large");
        }
        numel *= static_cast<std::uint64_t>(d);
    }
    const std::uint64_t have =
        has_raw ? raw.size() / dtype_size(dtype) : (data_type == kFloat ? floats.size() : ints.size());
    if (have != numel || (has_raw && raw.size() % dtype_size(dtype) != 0)) {
        throw ParseError("tensor '" + name + "' has data for " + std::to_string(have) +
                         " elements, its shape needs " + std::to_string(numel));
    }

    Tensor t(Shape(dims), dtype);
    if (has_raw) {
        copy_raw(t, raw, name);
    } else if (data_type == kFloat) {
        copy_typed<float>(t, floats, name);
    } else if (data_type == kInt8) {
        copy_typed<std::int8_t>(t, ints, name);
    } else if (data_type == kInt32) {
        copy_typed<std::int32_t>(t, ints, name);
    } else {
        copy_typed<std::int64_t>(t, ints, name);
    }
    if (name_out != nullptr) *name_out = std::move(name);
    return t;
}

Model parse_model(const std::uint8_t* data, std::size_t size) {
    WireReader r(data, size);
    Model m;
    bool has_graph = false;
    while (!r.done()) {
        const auto f = r.next();
        switch (f.number) {
            case 1:
                m.ir_version = r.int64();
                break;
            case 2:
                m.producer_name = as_string(r.bytes());
                break;
            case 3:
                m.producer_version = as_string(r.bytes());
                break;
            case 7:
                m.graph = parse_graph(r.message());
                has_graph = true;
                break;
            case 8: {
                WireReader op = r.message();
                std::string domain;
                std::int64_t version = 0;
                while (!op.done()) {
                    const auto of = op.next();
                    if (of.number == 1) domain = as_string(op.bytes());
                    else if (of.number == 2) version = op.int64();
                    else op.skip(of.type);
                }
                m.opsets[domain] = version;
                break;
            }
            default:
                r.skip(f.type);
        }
    }
    if (!has_graph) throw ParseError("not an ONNX model: no graph found");
    return m;
}

Model load_model(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw ParseError("cannot open " + path.string());
    const std::vector<char> buf((std::istreambuf_iterator<char>(in)),
                                std::istreambuf_iterator<char>());
    return parse_model(reinterpret_cast<const std::uint8_t*>(buf.data()), buf.size());
}

const char* attr_type_name(AttrType t) noexcept {
    switch (t) {
        case AttrType::Float:
            return "float";
        case AttrType::Int:
            return "int";
        case AttrType::String:
            return "string";
        case AttrType::Tensor:
            return "tensor";
        case AttrType::Graph:
            return "graph";
        case AttrType::Floats:
            return "floats";
        case AttrType::Ints:
            return "ints";
        case AttrType::Strings:
            return "strings";
        default:
            return "undefined";
    }
}

const Attribute* Node::find_attr(std::string_view attr) const noexcept {
    for (const auto& a : attributes) {
        if (a.name == attr) return &a;
    }
    return nullptr;
}

namespace {
const Attribute* typed_attr(const Node& n, std::string_view attr, AttrType want) {
    const Attribute* a = n.find_attr(attr);
    if (a != nullptr && a->type != want) {
        throw ParseError(n.op_type + " '" + n.name + "': attribute '" + std::string(attr) +
                         "' is " + attr_type_name(a->type) + ", expected " +
                         attr_type_name(want));
    }
    return a;
}
}  // namespace

std::int64_t Node::attr_int(std::string_view attr, std::int64_t fallback) const {
    const Attribute* a = typed_attr(*this, attr, AttrType::Int);
    return a != nullptr ? a->i : fallback;
}

float Node::attr_float(std::string_view attr, float fallback) const {
    const Attribute* a = typed_attr(*this, attr, AttrType::Float);
    return a != nullptr ? a->f : fallback;
}

std::string Node::attr_string(std::string_view attr, std::string fallback) const {
    const Attribute* a = typed_attr(*this, attr, AttrType::String);
    return a != nullptr ? a->s : fallback;
}

std::vector<std::int64_t> Node::attr_ints(std::string_view attr,
                                          std::vector<std::int64_t> fallback) const {
    const Attribute* a = typed_attr(*this, attr, AttrType::Ints);
    return a != nullptr ? a->ints : fallback;
}

std::string ValueInfo::shape_string() const {
    if (!has_shape) return "?";
    std::ostringstream os;
    os << '[';
    for (std::size_t i = 0; i < dims.size(); ++i) {
        if (i) os << ", ";
        if (dims[i].is_static()) os << dims[i].value;
        else os << (dims[i].param.empty() ? "?" : dims[i].param);
    }
    os << ']';
    return os.str();
}

const Initializer* Graph::find_initializer(std::string_view wanted) const noexcept {
    for (const auto& i : initializers) {
        if (i.name == wanted) return &i;
    }
    return nullptr;
}

std::vector<const ValueInfo*> Graph::runtime_inputs() const {
    std::set<std::string_view> weights;
    for (const auto& i : initializers) weights.insert(i.name);
    std::vector<const ValueInfo*> out;
    for (const auto& in : inputs) {
        if (weights.count(in.name) == 0) out.push_back(&in);
    }
    return out;
}

std::int64_t Graph::parameter_count() const noexcept {
    std::int64_t n = 0;
    for (const auto& i : initializers) n += i.value.numel();
    return n;
}

std::size_t Graph::parameter_bytes() const noexcept {
    std::size_t n = 0;
    for (const auto& i : initializers) n += i.value.nbytes();
    return n;
}

std::int64_t Model::opset() const noexcept {
    // "ai.onnx" is the long name of the default domain
    for (const char* d : {"", "ai.onnx"}) {
        auto it = opsets.find(d);
        if (it != opsets.end()) return it->second;
    }
    return 0;
}

}  // namespace tinyinfer::onnx
