// Kernel-level benchmarks: GEMM per ISA, the few-columns GEMMs batch-1 convs turn into, and
// every distinct 3x3 conv layer of ResNet-18 at 64x64 input, fp32 vs int8.

#include <benchmark/benchmark.h>

#include <random>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "tinyinfer/kernels.hpp"
#include "tinyinfer/quant.hpp"

namespace k = tinyinfer::kernels;
namespace q = tinyinfer::quant;
using tinyinfer::DType;
using tinyinfer::Shape;
using tinyinfer::Tensor;

namespace {

Tensor random_tensor(Shape s, unsigned seed, float lo = -1.0F, float hi = 1.0F) {
    Tensor t(std::move(s), DType::Float32);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(lo, hi);
    for (std::int64_t i = 0; i < t.numel(); ++i) t.data<float>()[i] = u(rng);
    return t;
}

void flops_counter(benchmark::State& st, double flops_per_iter) {
    st.counters["FLOPS"] =
        benchmark::Counter(flops_per_iter, benchmark::Counter::kIsIterationInvariantRate);
}

/// Square n x n x n GEMM on one core with a fixed micro-kernel, like `tinyinfer bench-gemm`.
void BM_Gemm(benchmark::State& st, const char* isa) {
    const int n = static_cast<int>(st.range(0));
    const std::string before = k::gemm_isa();
    if (!k::set_gemm_isa(isa)) {
        st.SkipWithError("ISA not supported on this CPU");
        return;
    }
#ifdef _OPENMP
    const int threads = omp_get_max_threads();
    omp_set_num_threads(1);
#endif
    std::vector<float> a(static_cast<std::size_t>(n) * n, 0.1F), b(a.size(), 0.2F), c(a.size());
    for (auto _ : st) {
        k::gemm(n, n, n, a.data(), n, b.data(), n, false, c.data(), n);
        benchmark::DoNotOptimize(c.data());
    }
    flops_counter(st, 2.0 * n * n * n);
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif
    k::set_gemm_isa(before.c_str());
}

/// 512 x n x 4608: a 512-channel 3x3 conv with n output pixels. n <= 16 takes the skinny
/// kernel, 17 and up the packed one, so the crossover shows up directly in this sweep.
void BM_GemmFewColumns(benchmark::State& st) {
    const int m = 512, n = static_cast<int>(st.range(0)), kk = 4608;
    const Tensor a = random_tensor(Shape{m, kk}, 1);
    const Tensor b = random_tensor(Shape{kk, n}, 2);
    std::vector<float> c(static_cast<std::size_t>(m) * n);
    for (auto _ : st) {
        k::gemm(m, n, kk, a.data<float>(), kk, b.data<float>(), n, false, c.data(), n);
        benchmark::DoNotOptimize(c.data());
    }
    flops_counter(st, 2.0 * m * n * kk);
}

struct Layer {
    const char* name;
    int c, oc, hw, stride;
};

// the 3x3 convs of ResNet-18 at a 64x64 input (the stem is 7x7, benchmarked separately)
const Layer kLayers[] = {
    {"layer1_64x16x16", 64, 64, 16, 1},
    {"layer2_128x8x8", 128, 128, 8, 1},
    {"layer3_256x4x4", 256, 256, 4, 1},
    {"layer4_512x2x2", 512, 512, 2, 1},
};

void BM_Conv(benchmark::State& st, Layer l, bool int8) {
    const int batch = static_cast<int>(st.range(0));
    const Tensor x = random_tensor(Shape{batch, l.c, l.hw, l.hw}, 3, 0.0F, 2.0F);
    const Tensor w = random_tensor(Shape{l.oc, l.c, 3, 3}, 4, -0.1F, 0.1F);
    k::Conv2dParams p;
    p.stride = {l.stride, l.stride};
    p.pads = {1, 1, 1, 1};
    const q::QConv qc = q::quantize_conv(w, nullptr, p, q::choose_u8(0.0F, 2.0F));
    for (auto _ : st) {
        Tensor y = int8 ? q::conv2d_int8(x, qc) : k::conv2d(x, w, nullptr, p);
        benchmark::DoNotOptimize(y.data<float>());
    }
    const double out_hw = static_cast<double>(l.hw / l.stride) * (l.hw / l.stride);
    flops_counter(st, 2.0 * batch * l.oc * out_hw * l.c * 9);
    st.SetItemsProcessed(st.iterations() * batch);
}

void BM_ConvStem(benchmark::State& st) {
    const int batch = static_cast<int>(st.range(0));
    const Tensor x = random_tensor(Shape{batch, 3, 64, 64}, 5);
    const Tensor w = random_tensor(Shape{64, 3, 7, 7}, 6, -0.1F, 0.1F);
    k::Conv2dParams p;
    p.stride = {2, 2};
    p.pads = {3, 3, 3, 3};
    for (auto _ : st) {
        Tensor y = k::conv2d(x, w, nullptr, p);
        benchmark::DoNotOptimize(y.data<float>());
    }
    flops_counter(st, 2.0 * batch * 64 * 32 * 32 * 3 * 49);
    st.SetItemsProcessed(st.iterations() * batch);
}

const int kRegistered = [] {
    for (const char* isa : {"scalar", "avx2", "avx512"}) {
        benchmark::RegisterBenchmark((std::string("BM_Gemm/") + isa).c_str(), BM_Gemm, isa)
            ->RangeMultiplier(2)
            ->Range(128, 1024)
            ->Unit(benchmark::kMillisecond);
    }
    benchmark::RegisterBenchmark("BM_GemmFewColumns", BM_GemmFewColumns)
        ->ArgsProduct({{1, 2, 4, 8, 16, 17, 24, 32, 64}})
        ->Unit(benchmark::kMicrosecond);
    for (const Layer& l : kLayers) {
        for (bool int8 : {false, true}) {
            const std::string name =
                std::string("BM_Conv/") + l.name + (int8 ? "/int8" : "/fp32");
            benchmark::RegisterBenchmark(name.c_str(), BM_Conv, l, int8)
                ->Arg(1)
                ->Arg(32)
                ->Unit(benchmark::kMicrosecond);
        }
    }
    benchmark::RegisterBenchmark("BM_ConvStem/fp32", BM_ConvStem)
        ->Arg(1)
        ->Arg(32)
        ->Unit(benchmark::kMicrosecond);
    return 0;
}();

}  // namespace
