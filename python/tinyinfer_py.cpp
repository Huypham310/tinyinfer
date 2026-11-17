// Python bindings (pybind11). Session wraps Model + Graph + Executor so a caller can load an
// ONNX file once and run numpy arrays through it. Calibration stores a QuantPlan on the Session
// and flips every non-grouped Conv to the INT8 kernels for later run() calls.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "tinyinfer/cpu_features.hpp"
#include "tinyinfer/graph.hpp"
#include "tinyinfer/kernels.hpp"
#include "tinyinfer/onnx/model.hpp"
#include "tinyinfer/quant.hpp"
#include "tinyinfer/version.hpp"

namespace py = pybind11;
using tinyinfer::DType;
using tinyinfer::Executor;
using tinyinfer::Graph;
using tinyinfer::Shape;
using tinyinfer::Tensor;

namespace {

/// Contiguous float32 N-D array -> Tensor. Refuses non-C-contiguous or wrong dtype so a silent
/// copy doesn't hide a layout bug in the caller.
Tensor tensor_from_numpy(const py::array& arr) {
    if (arr.dtype().is(py::dtype::of<float>()) == false) {
        throw std::invalid_argument("expected float32 ndarray");
    }
    auto info = arr.request();
    if (info.ndim < 1) throw std::invalid_argument("empty ndarray");
    // require C-contiguous: strides must match a dense row-major layout
    py::array_t<float, py::array::c_style | py::array::forcecast> contig(arr);
    info = contig.request();
    std::vector<std::int64_t> dims(info.shape.begin(), info.shape.end());
    Tensor t(Shape{dims}, DType::Float32);
    std::memcpy(t.data<float>(), info.ptr, t.nbytes());
    return t;
}

py::array_t<float> numpy_from_tensor(const Tensor& t) {
    if (t.dtype() != DType::Float32) throw std::invalid_argument("only float32 tensors are exposed");
    const auto& dims = t.shape().dims();
    std::vector<py::ssize_t> shape(dims.begin(), dims.end());
    py::array_t<float> out(shape);
    std::memcpy(out.mutable_data(), t.data<float>(), t.nbytes());
    return out;
}

class Session {
public:
    Session(const std::string& path, std::int64_t batch) {
        model_ = std::make_unique<tinyinfer::onnx::Model>(tinyinfer::onnx::load_model(path));
        const auto feeds = model_->graph.runtime_inputs();
        if (feeds.size() != 1) {
            throw std::invalid_argument("Session supports single-input models only");
        }
        input_name_ = feeds[0]->name;
        std::vector<std::int64_t> dims;
        for (const auto& d : feeds[0]->dims) dims.push_back(d.is_static() ? d.value : batch);
        graph_ = std::make_unique<Graph>(*model_, std::map<std::string, Shape>{{input_name_, Shape{dims}}});
        executor_ = std::make_unique<Executor>(*graph_);
        batch_ = batch;
    }

    py::dict info() const {
        py::dict d;
        d["ir_version"] = model_->ir_version;
        d["opset"] = model_->opset();
        d["nodes"] = graph_->nodes().size();
        d["bn_folded"] = graph_->stats().bn_folded;
        d["relu_fused"] = graph_->stats().relu_fused;
        d["activation_mib"] =
            static_cast<double>(graph_->stats().activation_bytes) / (1024.0 * 1024.0);
        d["peak_live_mib"] =
            static_cast<double>(graph_->stats().peak_live_bytes) / (1024.0 * 1024.0);
        d["input"] = input_name_;
        d["batch"] = batch_;
        d["int8"] = plan_.has_value();
        d["gemm_isa"] = tinyinfer::kernels::gemm_isa();
        d["int8_isa"] = tinyinfer::quant::int8_isa();
        return d;
    }

    py::array_t<float> run(const py::array& input) {
        Tensor x = tensor_from_numpy(input);
        if (x.shape().rank() != 4) throw std::invalid_argument("input must be NCHW float32");
        if (x.shape()[0] != batch_) {
            throw std::invalid_argument("batch mismatch: session built for " +
                                        std::to_string(batch_) + ", got " +
                                        std::to_string(x.shape()[0]));
        }
        auto outs = executor_->run({{input_name_, x}});
        last_peak_ = executor_->last_peak_bytes();
        return numpy_from_tensor(outs[0]);
    }

    /// Calibrate INT8 on a list of float32 batches (same shape as run()). Replaces any prior plan.
    void calibrate(const py::list& batches) {
        std::vector<Tensor> xs;
        xs.reserve(batches.size());
        for (const auto& item : batches) xs.push_back(tensor_from_numpy(py::cast<py::array>(item)));
        plan_ = tinyinfer::quant::calibrate(*graph_, input_name_, xs);
        executor_->set_int8(&*plan_);
    }

    void clear_int8() {
        plan_.reset();
        executor_->set_int8(nullptr);
    }

    std::size_t last_peak_bytes() const noexcept { return last_peak_; }

private:
    std::unique_ptr<tinyinfer::onnx::Model> model_;
    std::unique_ptr<Graph> graph_;
    std::unique_ptr<Executor> executor_;
    std::optional<tinyinfer::quant::QuantPlan> plan_;
    std::string input_name_;
    std::int64_t batch_ = 1;
    std::size_t last_peak_ = 0;
};

}  // namespace

PYBIND11_MODULE(_tinyinfer, m) {
    m.doc() = "TinyInfer: CPU ONNX inference engine (C++17, AVX2/AVX-512, INT8)";
    m.attr("__version__") = tinyinfer::kVersion;

    m.def(
        "cpu_features",
        [] {
            const auto& f = tinyinfer::cpu_features();
            py::dict d;
            d["sse42"] = f.sse42;
            d["avx"] = f.avx;
            d["avx2"] = f.avx2;
            d["fma"] = f.fma;
            d["avx512f"] = f.avx512f;
            d["avx512bw"] = f.avx512bw;
            d["avx512vl"] = f.avx512vl;
            d["avx512vnni"] = f.avx512vnni;
            d["summary"] = tinyinfer::to_string(f);
            return d;
        },
        "SIMD features this process can safely use");

    m.def("gemm_isa", &tinyinfer::kernels::gemm_isa);
    m.def("set_gemm_isa", &tinyinfer::kernels::set_gemm_isa, py::arg("isa"));
    m.def("int8_isa", &tinyinfer::quant::int8_isa);
    m.def("set_int8_isa", &tinyinfer::quant::set_int8_isa, py::arg("isa"));

    py::class_<Session>(m, "Session")
        .def(py::init<const std::string&, std::int64_t>(), py::arg("path"), py::arg("batch") = 1,
             "Load an ONNX model; symbolic dims (e.g. batch) are bound to `batch`")
        .def("info", &Session::info)
        .def("run", &Session::run, py::arg("input"),
             "Run one float32 NCHW batch; returns the model output as float32")
        .def("calibrate", &Session::calibrate, py::arg("batches"),
             "Record Conv input ranges and switch those nodes to INT8")
        .def("clear_int8", &Session::clear_int8)
        .def_property_readonly("last_peak_bytes", &Session::last_peak_bytes);
}
