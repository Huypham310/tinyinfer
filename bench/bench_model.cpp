// End-to-end ResNet-18 (the TerraSense model, 3 x 64 x 64 -> 10 classes), fp32 vs int8.
// The model isn't in the repo: point TINYINFER_RESNET18 at the .onnx file, otherwise these
// benchmarks report "skipped".

#include <benchmark/benchmark.h>

#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "tinyinfer/graph.hpp"
#include "tinyinfer/onnx/model.hpp"
#include "tinyinfer/quant.hpp"

using tinyinfer::DType;
using tinyinfer::Shape;
using tinyinfer::Tensor;

namespace {

Tensor random_images(std::int64_t n, unsigned seed) {
    Tensor t(Shape{n, 3, 64, 64}, DType::Float32);
    std::mt19937 rng(seed);
    // roughly what normalized EuroSAT pixels look like
    std::normal_distribution<float> d(0.0F, 1.0F);
    for (std::int64_t i = 0; i < t.numel(); ++i) t.data<float>()[i] = d(rng);
    return t;
}

const tinyinfer::onnx::Model* model() {
    static const std::unique_ptr<tinyinfer::onnx::Model> m = [] {
        const char* path = std::getenv("TINYINFER_RESNET18");
        if (path == nullptr) return std::unique_ptr<tinyinfer::onnx::Model>();
        return std::make_unique<tinyinfer::onnx::Model>(tinyinfer::onnx::load_model(path));
    }();
    return m.get();
}

void BM_ResNet18(benchmark::State& st, bool int8) {
    const auto* m = model();
    if (m == nullptr) {
        st.SkipWithError("set TINYINFER_RESNET18=path/to/model.onnx");
        return;
    }
    const auto batch = static_cast<std::int64_t>(st.range(0));
    const std::string in = m->graph.runtime_inputs().at(0)->name;
    tinyinfer::Graph g(*m, {{in, Shape{batch, 3, 64, 64}}});
    tinyinfer::Executor ex(g);
    tinyinfer::quant::QuantPlan plan;
    if (int8) {
        // random calibration is fine for timing; accuracy numbers come from `tinyinfer eval`
        plan = tinyinfer::quant::calibrate(g, in, {random_images(batch, 1)});
        ex.set_int8(&plan);
    }
    const Tensor x = random_images(batch, 2);
    for (auto _ : st) {
        auto out = ex.run({{in, x}});
        benchmark::DoNotOptimize(out[0].data<float>());
    }
    st.SetItemsProcessed(st.iterations() * batch);
    st.counters["peak_act_MiB"] = static_cast<double>(ex.last_peak_bytes()) / (1024.0 * 1024.0);
}

BENCHMARK_CAPTURE(BM_ResNet18, fp32, false)
    ->Arg(1)
    ->Arg(8)
    ->Arg(32)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_ResNet18, int8, true)
    ->Arg(1)
    ->Arg(8)
    ->Arg(32)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

}  // namespace
