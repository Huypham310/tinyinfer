#include <algorithm>
#include <chrono>

#include "tinyinfer/graph.hpp"
#include "tinyinfer/kernels.hpp"

namespace tinyinfer {

namespace {

std::size_t bytes(const Tensor& t) { return t.nbytes(); }

}  // namespace

std::vector<Tensor> Executor::run(const std::map<std::string, Tensor>& feeds) {
    const auto& values = g_.values();
    std::vector<Tensor> slot(values.size());
    for (std::size_t v = 0; v < values.size(); ++v) {
        if (values[v].constant) slot[v] = values[v].data;
    }
    std::size_t live = 0;
    for (int in : g_.inputs()) {
        auto it = feeds.find(values[in].name);
        if (it == feeds.end()) throw GraphError("missing input '" + values[in].name + "'");
        if (it->second.shape() != values[in].shape) {
            throw GraphError("input '" + values[in].name + "' is " + it->second.shape().to_string() +
                             ", the graph was built for " + values[in].shape.to_string());
        }
        slot[in] = it->second;
        live += bytes(it->second);
    }
    peak_ = live;

    const auto& nodes = g_.nodes();
    if (profile_) node_ms_.assign(nodes.size(), 0.0);
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        const GraphNode& n = nodes[i];
        const auto t0 = std::chrono::steady_clock::now();
        auto in = [&](std::size_t k) -> const Tensor& { return slot[n.inputs[k]]; };
        auto opt = [&](std::size_t k) -> const Tensor* {
            return k < n.inputs.size() && n.inputs[k] >= 0 ? &slot[n.inputs[k]] : nullptr;
        };
        Tensor out;
        switch (n.kind) {
            case OpKind::Conv:
                out = kernels::conv2d(in(0), in(1), opt(2), n.conv);
                break;
            case OpKind::Relu:
                out = kernels::relu(in(0));
                break;
            case OpKind::Add:
                out = kernels::add(in(0), in(1));
                break;
            case OpKind::MaxPool:
                out = kernels::max_pool2d(in(0), n.pool);
                break;
            case OpKind::GlobalAveragePool:
                out = kernels::global_avg_pool(in(0));
                if (!n.keepdims) out = out.reshape(values[n.output].shape);
                break;
            case OpKind::Reshape:
                out = in(0).reshape(values[n.output].shape);  // view, shares the storage
                break;
            case OpKind::Gemm:
                out = kernels::gemm_op(in(0), in(1), opt(2), n.trans_b, n.alpha, n.beta);
                break;
            case OpKind::BatchNorm:
                out = kernels::batch_norm(in(0), in(1), in(2), in(3), in(4), n.eps);
                break;
            case OpKind::Softmax:
                out = kernels::softmax(in(0));
                break;
            case OpKind::Clip:
                out = kernels::clip(in(0), n.clip_min, n.clip_max);
                break;
            case OpKind::Identity:
                out = in(0);
                break;
        }
        if (n.fused_relu) kernels::relu_inplace(out);
        if (profile_) {
            node_ms_[i] =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        }
        if (n.kind != OpKind::Reshape && n.kind != OpKind::Identity) live += bytes(out);
        peak_ = std::max(peak_, live);
        slot[n.output] = std::move(out);

        // drop every input this node read for the last time; views keep their storage alive
        for (int v : n.inputs) {
            if (v < 0 || values[v].constant || values[v].last_use != static_cast<int>(i)) continue;
            if (!slot[v].shares_storage_with(slot[n.output])) live -= bytes(slot[v]);
            slot[v] = Tensor{};
        }
    }

    std::vector<Tensor> outputs;
    for (int o : g_.outputs()) outputs.push_back(slot[o]);
    return outputs;
}

}  // namespace tinyinfer
