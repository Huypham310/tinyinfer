#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <utility>
#include <string>
#include <vector>

#include "tinyinfer/kernels.hpp"
#include "tinyinfer/onnx/model.hpp"
#include "tinyinfer/tensor.hpp"

namespace tinyinfer {

namespace quant {
struct QConv;
}

/// Operators the executor can run. Anything else in a model is a GraphError at load time,
/// not a surprise halfway through inference.
enum class OpKind : std::uint8_t {
    Conv,
    Relu,
    Add,
    MaxPool,
    GlobalAveragePool,  // also ReduceMean over the two spatial axes
    Reshape,            // also Flatten; a view, no copy
    Gemm,
    BatchNorm,
    Softmax,
    Clip,
    Identity,
};

const char* op_name(OpKind k) noexcept;

class GraphError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct GraphNode {
    OpKind kind;
    std::string name;
    std::vector<int> inputs;  // value ids; -1 for an omitted optional input
    int output = -1;
    // per-op parameters, filled from the ONNX attributes
    kernels::Conv2dParams conv;
    kernels::Pool2dParams pool;
    bool trans_b = false;
    float alpha = 1.0F, beta = 1.0F, eps = 1e-5F;
    float clip_min = -3.4e38F, clip_max = 3.4e38F;
    bool keepdims = true;                // GlobalAveragePool from ReduceMean(keepdims=0)
    std::vector<std::int64_t> shape_spec;  // Reshape target, ONNX rules (-1 infers, 0 copies)
    bool allow_zero = false;
    bool fused_relu = false;  // Conv / Add / Gemm followed by a Relu that was folded in
    bool dead = false;        // removed by a fusion pass, dropped by compact()
};

struct GraphValue {
    std::string name;
    Shape shape;
    bool constant = false;
    Tensor data;          // constants only
    int producer = -1;    // node index, -1 for inputs and constants
    int last_use = -1;    // index (in execution order) of the last node reading it
};

/// Executable graph: nodes in topological order, every shape known, constants resolved.
///
/// Built from an onnx::Model plus concrete input shapes (symbolic dims like "batch" get a
/// number here). Construction runs, in order: topological sort, shape inference, BatchNorm
/// folding into the preceding Conv, Conv/Add/Gemm + Relu fusion, and liveness analysis.
class Graph {
public:
    Graph(const onnx::Model& model, const std::map<std::string, Shape>& input_shapes);

    [[nodiscard]] const std::vector<GraphNode>& nodes() const noexcept { return nodes_; }
    [[nodiscard]] const std::vector<GraphValue>& values() const noexcept { return values_; }
    [[nodiscard]] const std::vector<int>& inputs() const noexcept { return inputs_; }
    [[nodiscard]] const std::vector<int>& outputs() const noexcept { return outputs_; }

    struct Stats {
        int bn_folded = 0;
        int relu_fused = 0;
        std::size_t activation_bytes = 0;  // every intermediate kept alive (no reuse)
        std::size_t peak_live_bytes = 0;   // max bytes alive at once when freed after last use
    };
    [[nodiscard]] const Stats& stats() const noexcept { return stats_; }

private:
    int value_id(const std::string& name);
    void build(const onnx::Model& model);
    void topo_sort();
    void infer_shapes();
    void fold_batch_norm();
    void fuse_relu();
    void compact();
    void liveness();

    std::vector<GraphNode> nodes_;
    std::vector<GraphValue> values_;
    std::map<std::string, int> ids_;
    std::vector<int> inputs_, outputs_;
    Stats stats_;
};

/// Runs a Graph. Intermediate tensors are released right after their last use (from the
/// liveness analysis), so peak memory is the widest point of the network, not the sum.
class Executor {
public:
    explicit Executor(const Graph& graph) : g_(graph) {}

    /// Feeds the graph inputs by name and returns the graph outputs in declaration order.
    std::vector<Tensor> run(const std::map<std::string, Tensor>& feeds);
    /// Bytes of activations alive at the worst point of the last run().
    [[nodiscard]] std::size_t last_peak_bytes() const noexcept { return peak_; }

    /// With profiling on, run() records the wall time of every node (milliseconds).
    void set_profiling(bool on) noexcept { profile_ = on; }
    [[nodiscard]] const std::vector<double>& node_ms() const noexcept { return node_ms_; }

    /// Calibration: widen ranges[i] to the min / max of node i's first input, for every Conv.
    void observe_conv_inputs(std::vector<std::pair<float, float>>* ranges) noexcept {
        ranges_ = ranges;
    }
    /// Run the Conv nodes that have an entry in `plan` with int8 kernels (see quant.hpp).
    void set_int8(const std::vector<std::optional<quant::QConv>>* plan) noexcept { plan_ = plan; }

private:
    const Graph& g_;
    std::size_t peak_ = 0;
    bool profile_ = false;
    std::vector<double> node_ms_;
    std::vector<std::pair<float, float>>* ranges_ = nullptr;
    const std::vector<std::optional<quant::QConv>>* plan_ = nullptr;
};

}  // namespace tinyinfer
