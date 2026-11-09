// ONNX model -> executable graph: op mapping, topological sort, shape inference, BatchNorm
// folding, Relu fusion and liveness.

#include "tinyinfer/graph.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <set>

namespace tinyinfer {

const char* op_name(OpKind k) noexcept {
    switch (k) {
        case OpKind::Conv:
            return "Conv";
        case OpKind::Relu:
            return "Relu";
        case OpKind::Add:
            return "Add";
        case OpKind::MaxPool:
            return "MaxPool";
        case OpKind::GlobalAveragePool:
            return "GlobalAveragePool";
        case OpKind::Reshape:
            return "Reshape";
        case OpKind::Gemm:
            return "Gemm";
        case OpKind::BatchNorm:
            return "BatchNormalization";
        case OpKind::Softmax:
            return "Softmax";
        case OpKind::Clip:
            return "Clip";
        case OpKind::Identity:
            return "Identity";
    }
    return "?";
}

namespace {

std::array<int, 2> pair_attr(const onnx::Node& n, const char* name, int fallback) {
    const auto v = n.attr_ints(name, {fallback, fallback});
    if (v.size() != 2) throw GraphError(n.name + ": only 2-D " + name + " is supported");
    return {static_cast<int>(v[0]), static_cast<int>(v[1])};
}

std::array<int, 4> pads_attr(const onnx::Node& n) {
    const std::string auto_pad = n.attr_string("auto_pad", "NOTSET");
    if (auto_pad != "NOTSET" && auto_pad != "VALID") {
        // SAME_UPPER / SAME_LOWER need the input size to resolve; PyTorch never emits them
        throw GraphError(n.name + ": auto_pad " + auto_pad + " is not supported");
    }
    const auto v = n.attr_ints("pads", {0, 0, 0, 0});
    if (v.size() != 4) throw GraphError(n.name + ": only 2-D pads are supported");
    return {static_cast<int>(v[0]), static_cast<int>(v[1]), static_cast<int>(v[2]),
            static_cast<int>(v[3])};
}

std::size_t bytes_of(const Shape& s) { return static_cast<std::size_t>(s.numel()) * 4; }

}  // namespace

int Graph::value_id(const std::string& name) {
    auto it = ids_.find(name);
    if (it != ids_.end()) return it->second;
    const int id = static_cast<int>(values_.size());
    values_.push_back({name, {}, false, {}, -1, -1});
    ids_[name] = id;
    return id;
}

Graph::Graph(const onnx::Model& model, const std::map<std::string, Shape>& input_shapes) {
    for (const auto& init : model.graph.initializers) {
        const int id = value_id(init.name);
        values_[id].constant = true;
        values_[id].data = init.value;
        values_[id].shape = init.value.shape();
    }
    for (const auto* in : model.graph.runtime_inputs()) {
        const int id = value_id(in->name);
        auto it = input_shapes.find(in->name);
        if (it != input_shapes.end()) {
            values_[id].shape = it->second;
        } else {
            std::vector<std::int64_t> dims;
            for (const auto& d : in->dims) {
                if (!d.is_static()) {
                    throw GraphError("input '" + in->name + "' has symbolic dim '" + d.param +
                                     "': pass its shape explicitly");
                }
                dims.push_back(d.value);
            }
            values_[id].shape = Shape(dims);
        }
        inputs_.push_back(id);
    }
    build(model);
    for (const auto& out : model.graph.outputs) outputs_.push_back(value_id(out.name));

    topo_sort();
    infer_shapes();
    fold_batch_norm();
    fuse_relu();
    compact();
    liveness();
}

void Graph::build(const onnx::Model& model) {
    for (const auto& n : model.graph.nodes) {
        if (!n.domain.empty() && n.domain != "ai.onnx") {
            throw GraphError(n.name + ": custom domain '" + n.domain + "' is not supported");
        }
        if (n.outputs.size() != 1 && n.op_type != "BatchNormalization") {
            throw GraphError(n.name + ": " + n.op_type + " with several outputs is not supported");
        }
        GraphNode g{};
        g.name = n.name.empty() ? n.op_type : n.name;
        for (const auto& in : n.inputs) g.inputs.push_back(in.empty() ? -1 : value_id(in));
        g.output = value_id(n.outputs[0]);

        const std::string& op = n.op_type;
        if (op == "Conv") {
            g.kind = OpKind::Conv;
            g.conv.stride = pair_attr(n, "strides", 1);
            g.conv.dilation = pair_attr(n, "dilations", 1);
            g.conv.pads = pads_attr(n);
            g.conv.groups = static_cast<int>(n.attr_int("group", 1));
        } else if (op == "Relu") {
            g.kind = OpKind::Relu;
        } else if (op == "Add") {
            g.kind = OpKind::Add;
        } else if (op == "MaxPool") {
            g.kind = OpKind::MaxPool;
            if (n.attr_int("ceil_mode", 0) != 0) throw GraphError(g.name + ": ceil_mode=1");
            if (n.attr_ints("dilations", {1, 1}) != std::vector<std::int64_t>{1, 1}) {
                throw GraphError(g.name + ": dilated pooling is not supported");
            }
            g.pool.kernel = pair_attr(n, "kernel_shape", 1);
            g.pool.stride = pair_attr(n, "strides", 1);
            g.pool.pads = pads_attr(n);
        } else if (op == "GlobalAveragePool") {
            g.kind = OpKind::GlobalAveragePool;
        } else if (op == "ReduceMean") {
            // opset 18 moved `axes` from an attribute to an optional second input
            std::vector<std::int64_t> axes = n.attr_ints("axes", {});
            if (axes.empty() && g.inputs.size() > 1 && g.inputs[1] >= 0) {
                const auto& v = values_[g.inputs[1]];
                if (!v.constant) throw GraphError(g.name + ": axes must be a constant");
                const auto* a = v.data.data<std::int64_t>();
                axes.assign(a, a + v.data.numel());
            }
            std::set<std::int64_t> norm;
            for (auto a : axes) norm.insert(a < 0 ? a + 4 : a);
            if (norm != std::set<std::int64_t>{2, 3}) {
                throw GraphError(g.name + ": ReduceMean only over the spatial axes (2, 3)");
            }
            g.kind = OpKind::GlobalAveragePool;
            g.keepdims = n.attr_int("keepdims", 1) != 0;
            g.inputs.resize(1);
        } else if (op == "Flatten") {
            if (n.attr_int("axis", 1) != 1) throw GraphError(g.name + ": Flatten axis != 1");
            g.kind = OpKind::Reshape;
            g.shape_spec = {0, -1};
        } else if (op == "Reshape") {
            g.kind = OpKind::Reshape;
            const auto& v = values_.at(g.inputs.at(1));
            if (!v.constant) throw GraphError(g.name + ": the target shape must be a constant");
            const auto* s = v.data.data<std::int64_t>();
            g.shape_spec.assign(s, s + v.data.numel());
            g.allow_zero = n.attr_int("allowzero", 0) != 0;
            g.inputs.resize(1);
        } else if (op == "Gemm") {
            g.kind = OpKind::Gemm;
            if (n.attr_int("transA", 0) != 0) throw GraphError(g.name + ": transA=1");
            g.trans_b = n.attr_int("transB", 0) != 0;
            g.alpha = n.attr_float("alpha", 1.0F);
            g.beta = n.attr_float("beta", 1.0F);
        } else if (op == "BatchNormalization") {
            g.kind = OpKind::BatchNorm;
            g.eps = n.attr_float("epsilon", 1e-5F);
        } else if (op == "Softmax") {
            g.kind = OpKind::Softmax;
            g.alpha = static_cast<float>(n.attr_int("axis", -1));  // checked in infer_shapes
        } else if (op == "Clip") {
            g.kind = OpKind::Clip;
            // opset >= 11: min / max are optional constant inputs; before that, attributes
            g.clip_min = n.attr_float("min", g.clip_min);
            g.clip_max = n.attr_float("max", g.clip_max);
            for (std::size_t i = 1; i < g.inputs.size() && i < 3; ++i) {
                if (g.inputs[i] < 0) continue;
                const auto& v = values_[g.inputs[i]];
                if (!v.constant) throw GraphError(g.name + ": Clip bounds must be constants");
                (i == 1 ? g.clip_min : g.clip_max) = v.data.data<float>()[0];
            }
            g.inputs.resize(1);
        } else if (op == "Identity" || op == "Dropout") {
            g.kind = OpKind::Identity;  // Dropout is the identity at inference time
            g.inputs.resize(1);
        } else {
            throw GraphError("unsupported operator " + op + " (node '" + g.name + "')");
        }
        nodes_.push_back(std::move(g));
    }
}

void Graph::topo_sort() {
    // Kahn's algorithm on "node needs value": constants and graph inputs are ready up front
    std::vector<bool> ready(values_.size(), false);
    for (std::size_t v = 0; v < values_.size(); ++v) ready[v] = values_[v].constant;
    for (int v : inputs_) ready[v] = true;
    std::vector<std::vector<int>> waiting(values_.size());
    std::vector<int> missing(nodes_.size(), 0);
    std::deque<int> queue;
    for (std::size_t i = 0; i < nodes_.size(); ++i) {
        for (int in : nodes_[i].inputs) {
            if (in >= 0 && !ready[in]) {
                ++missing[i];
                waiting[in].push_back(static_cast<int>(i));
            }
        }
        if (missing[i] == 0) queue.push_back(static_cast<int>(i));
    }
    std::vector<GraphNode> order;
    order.reserve(nodes_.size());
    while (!queue.empty()) {
        const int i = queue.front();
        queue.pop_front();
        order.push_back(nodes_[i]);
        const int out = nodes_[i].output;
        if (ready[out]) throw GraphError("value '" + values_[out].name + "' is produced twice");
        ready[out] = true;
        for (int j : waiting[out]) {
            if (--missing[j] == 0) queue.push_back(j);
        }
    }
    if (order.size() != nodes_.size()) {
        for (std::size_t i = 0; i < nodes_.size(); ++i) {
            if (missing[i] > 0) {
                throw GraphError("node '" + nodes_[i].name +
                                 "' waits for a value nothing produces (or the graph has a cycle)");
            }
        }
    }
    nodes_ = std::move(order);
    for (std::size_t i = 0; i < nodes_.size(); ++i) values_[nodes_[i].output].producer = static_cast<int>(i);
}

void Graph::infer_shapes() {
    for (auto& n : nodes_) {
        auto in = [&](std::size_t i) -> const Shape& { return values_.at(n.inputs.at(i)).shape; };
        Shape out;
        switch (n.kind) {
            case OpKind::Conv: {
                const Shape &x = in(0), &w = in(1);
                if (x.rank() != 4 || w.rank() != 4) throw GraphError(n.name + ": Conv needs NCHW");
                const int oh = kernels::conv_out_size(static_cast<int>(x[2]), static_cast<int>(w[2]),
                                                      n.conv.stride[0], n.conv.pads[0],
                                                      n.conv.pads[2], n.conv.dilation[0]);
                const int ow = kernels::conv_out_size(static_cast<int>(x[3]), static_cast<int>(w[3]),
                                                      n.conv.stride[1], n.conv.pads[1],
                                                      n.conv.pads[3], n.conv.dilation[1]);
                if (x[1] != w[1] * n.conv.groups) {
                    throw GraphError(n.name + ": input has " + std::to_string(x[1]) +
                                     " channels, weight expects " +
                                     std::to_string(w[1] * n.conv.groups));
                }
                out = Shape{x[0], w[0], oh, ow};
                break;
            }
            case OpKind::Relu:
            case OpKind::Clip:
            case OpKind::Identity:
            case OpKind::BatchNorm:
                out = in(0);
                break;
            case OpKind::Softmax: {
                const auto rank = static_cast<int>(in(0).rank());
                const int axis = static_cast<int>(n.alpha);
                if (axis != -1 && axis != rank - 1) throw GraphError(n.name + ": Softmax not on the last axis");
                out = in(0);
                break;
            }
            case OpKind::Add:
                if (in(0) != in(1)) {
                    throw GraphError(n.name + ": Add of " + in(0).to_string() + " and " +
                                     in(1).to_string() + " (broadcasting is not supported)");
                }
                out = in(0);
                break;
            case OpKind::MaxPool: {
                const Shape& x = in(0);
                out = Shape{x[0], x[1],
                            kernels::conv_out_size(static_cast<int>(x[2]), n.pool.kernel[0],
                                                   n.pool.stride[0], n.pool.pads[0], n.pool.pads[2], 1),
                            kernels::conv_out_size(static_cast<int>(x[3]), n.pool.kernel[1],
                                                   n.pool.stride[1], n.pool.pads[1], n.pool.pads[3], 1)};
                break;
            }
            case OpKind::GlobalAveragePool:
                out = n.keepdims ? Shape{in(0)[0], in(0)[1], 1, 1} : Shape{in(0)[0], in(0)[1]};
                break;
            case OpKind::Reshape: {
                const Shape& x = in(0);
                std::vector<std::int64_t> dims;
                std::int64_t known = 1;
                int infer = -1;
                for (std::size_t i = 0; i < n.shape_spec.size(); ++i) {
                    std::int64_t d = n.shape_spec[i];
                    if (d == 0 && !n.allow_zero) d = x[i];  // 0 = copy the input dim
                    if (d == -1) {
                        if (infer >= 0) throw GraphError(n.name + ": two -1 dims in Reshape");
                        infer = static_cast<int>(i);
                    } else {
                        known *= d;
                    }
                    dims.push_back(d);
                }
                if (infer >= 0) {
                    if (known == 0 || x.numel() % known != 0) throw GraphError(n.name + ": bad Reshape");
                    dims[infer] = x.numel() / known;
                }
                out = Shape(dims);
                if (out.numel() != x.numel()) throw GraphError(n.name + ": Reshape changes the size");
                break;
            }
            case OpKind::Gemm: {
                const Shape &a = in(0), &b = in(1);
                if (a.rank() != 2 || b.rank() != 2) throw GraphError(n.name + ": Gemm needs 2-D inputs");
                const auto k = n.trans_b ? b[1] : b[0];
                if (a[1] != k) throw GraphError(n.name + ": Gemm inner dims " + std::to_string(a[1]) +
                                                " vs " + std::to_string(k));
                out = Shape{a[0], n.trans_b ? b[0] : b[1]};
                break;
            }
        }
        values_[n.output].shape = out;
    }
}

namespace {

std::vector<int> consumers_count(const std::vector<GraphNode>& nodes, std::size_t values,
                                 const std::vector<int>& outputs) {
    std::vector<int> count(values, 0);
    for (const auto& n : nodes) {
        if (n.dead) continue;
        for (int in : n.inputs) {
            if (in >= 0) ++count[in];
        }
    }
    for (int o : outputs) ++count[o];  // a graph output is "used" by the caller
    return count;
}

}  // namespace

void Graph::fold_batch_norm() {
    // Conv -> BatchNorm with constant statistics is one Conv with rescaled weights:
    //   y = s * (conv(x, W) + b - mean) + beta,  s = scale / sqrt(var + eps)
    //     = conv(x, s * W) + (s * (b - mean) + beta)
    const auto uses = consumers_count(nodes_, values_.size(), outputs_);
    for (auto& bn : nodes_) {
        if (bn.kind != OpKind::BatchNorm) continue;
        const int x = bn.inputs[0];
        const int p = values_[x].producer;
        bool params_const = bn.inputs.size() >= 5;
        for (std::size_t i = 1; params_const && i < 5; ++i) params_const = values_[bn.inputs[i]].constant;
        if (p < 0 || nodes_[p].kind != OpKind::Conv || uses[x] != 1 || !params_const) continue;
        GraphNode& conv = nodes_[p];
        const Tensor& w = values_[conv.inputs[1]].data;
        const auto oc = w.shape()[0];
        const auto per = w.numel() / oc;
        const float* scale = values_[bn.inputs[1]].data.data<float>();
        const float* beta = values_[bn.inputs[2]].data.data<float>();
        const float* mean = values_[bn.inputs[3]].data.data<float>();
        const float* var = values_[bn.inputs[4]].data.data<float>();
        Tensor nw = w.clone();  // never edit an initializer in place: it may be shared
        Tensor nb(Shape{oc}, DType::Float32);
        const bool has_bias = conv.inputs.size() > 2 && conv.inputs[2] >= 0;
        for (std::int64_t o = 0; o < oc; ++o) {
            const float s = scale[o] / std::sqrt(var[o] + bn.eps);
            float* row = nw.data<float>() + o * per;
            for (std::int64_t j = 0; j < per; ++j) row[j] *= s;
            const float b = has_bias ? values_[conv.inputs[2]].data.data<float>()[o] : 0.0F;
            nb.data<float>()[o] = s * (b - mean[o]) + beta[o];
        }
        const int wid = value_id(conv.name + ".folded_weight");
        const int bid = value_id(conv.name + ".folded_bias");
        values_[wid] = {values_[wid].name, nw.shape(), true, nw, -1, -1};
        values_[bid] = {values_[bid].name, nb.shape(), true, nb, -1, -1};
        conv.inputs = {conv.inputs[0], wid, bid};
        values_[x].producer = -1;
        conv.output = bn.output;  // the conv now writes what the BN used to
        values_[bn.output].producer = p;
        bn.dead = true;
        ++stats_.bn_folded;
    }
}

void Graph::fuse_relu() {
    const auto uses = consumers_count(nodes_, values_.size(), outputs_);
    for (auto& relu : nodes_) {
        if (relu.dead || relu.kind != OpKind::Relu) continue;
        const int x = relu.inputs[0];
        const int p = values_[x].producer;
        if (p < 0 || uses[x] != 1) continue;  // someone else needs the pre-activation value
        GraphNode& prod = nodes_[p];
        if (prod.dead || prod.fused_relu ||
            (prod.kind != OpKind::Conv && prod.kind != OpKind::Add && prod.kind != OpKind::Gemm)) {
            continue;
        }
        prod.fused_relu = true;
        prod.output = relu.output;
        values_[relu.output].producer = p;
        relu.dead = true;
        ++stats_.relu_fused;
    }
}

void Graph::compact() {
    std::vector<GraphNode> live;
    for (auto& n : nodes_) {
        if (!n.dead) live.push_back(std::move(n));
    }
    nodes_ = std::move(live);
    for (auto& v : values_) {
        if (!v.constant) v.producer = -1;
    }
    for (std::size_t i = 0; i < nodes_.size(); ++i) values_[nodes_[i].output].producer = static_cast<int>(i);
}

void Graph::liveness() {
    for (std::size_t i = 0; i < nodes_.size(); ++i) {
        for (int in : nodes_[i].inputs) {
            if (in >= 0) values_[in].last_use = static_cast<int>(i);
        }
    }
    const int never = static_cast<int>(nodes_.size());
    for (int o : outputs_) values_[o].last_use = never;

    // simulate the executor: allocate each output, then free inputs read for the last time
    std::size_t live = 0;
    for (int in : inputs_) live += bytes_of(values_[in].shape);
    stats_.peak_live_bytes = live;
    for (std::size_t i = 0; i < nodes_.size(); ++i) {
        const auto& n = nodes_[i];
        const std::size_t out = n.kind == OpKind::Reshape ? 0 : bytes_of(values_[n.output].shape);
        stats_.activation_bytes += out;
        live += out;
        stats_.peak_live_bytes = std::max(stats_.peak_live_bytes, live);
        for (int in : n.inputs) {
            if (in < 0 || values_[in].constant || values_[in].last_use != static_cast<int>(i)) continue;
            const int p = values_[in].producer;
            if (p >= 0 && nodes_[p].kind == OpKind::Reshape) continue;  // a view, owns nothing
            live -= bytes_of(values_[in].shape);
        }
    }
}

}  // namespace tinyinfer
