// Command line driver: CPU info and model inspection for now; running models comes with the
// kernels and the graph executor.

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "tinyinfer/cpu_features.hpp"
#include "tinyinfer/kernels.hpp"
#include "tinyinfer/onnx/model.hpp"
#include "tinyinfer/version.hpp"

namespace {

void print_usage() {
    std::cout << "usage: tinyinfer [--info] [--version]\n"
                 "       tinyinfer inspect <model.onnx> [--nodes]\n"
                 "       tinyinfer bench-gemm\n"
                 "  --info     show detected CPU SIMD features\n"
                 "  --version  print version\n"
                 "  inspect    parse an ONNX model and summarize it (--nodes lists every node)\n";
}

const char* elem_type_name(int t) {
    switch (t) {
        case 1:
            return "float";
        case 3:
            return "int8";
        case 6:
            return "int32";
        case 7:
            return "int64";
        default:
            return "other";
    }
}

int inspect(const char* path, bool list_nodes) {
    namespace ox = tinyinfer::onnx;
    const auto start = std::chrono::steady_clock::now();
    ox::Model m;
    try {
        m = ox::load_model(path);
    } catch (const ox::ParseError& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
            .count();

    const auto& g = m.graph;
    std::cout << "model:   ir " << m.ir_version << ", opset " << m.opset() << ", producer "
              << (m.producer_name.empty() ? "?" : m.producer_name) << ' ' << m.producer_version
              << '\n';
    std::cout << "graph:   " << (g.name.empty() ? "(unnamed)" : g.name) << ", " << g.nodes.size()
              << " nodes, " << g.initializers.size() << " initializers, "
              << g.parameter_count() << " params (" << std::fixed << std::setprecision(1)
              << static_cast<double>(g.parameter_bytes()) / (1024.0 * 1024.0) << " MiB)\n";
    for (const auto* in : g.runtime_inputs()) {
        std::cout << "input:   " << in->name << ' ' << elem_type_name(in->elem_type) << ' '
                  << in->shape_string() << '\n';
    }
    for (const auto& out : g.outputs) {
        std::cout << "output:  " << out.name << ' ' << elem_type_name(out.elem_type) << ' '
                  << out.shape_string() << '\n';
    }

    std::map<std::string, int> ops;
    for (const auto& n : g.nodes) ++ops[n.op_type];
    std::vector<std::pair<std::string, int>> sorted(ops.begin(), ops.end());
    std::stable_sort(sorted.begin(), sorted.end(),
                     [](const auto& a, const auto& b) { return a.second > b.second; });
    std::cout << "ops:    ";
    for (const auto& [op, count] : sorted) std::cout << ' ' << op << " x" << count;
    std::cout << '\n';

    if (list_nodes) {
        for (const auto& n : g.nodes) {
            std::cout << "  " << std::left << std::setw(20) << n.op_type << n.name << " (";
            for (std::size_t i = 0; i < n.inputs.size(); ++i) {
                std::cout << (i ? ", " : "") << n.inputs[i];
            }
            std::cout << ") ->";
            for (const auto& o : n.outputs) std::cout << ' ' << o;
            std::cout << '\n';
        }
    }
    std::cout << "parsed in " << std::setprecision(1) << ms << " ms\n";
    return 0;
}

/// Best-of-`reps` GFLOP/s for an n x n x n GEMM (2 n^3 flops).
template <class F>
double gflops(int n, int reps, F&& run) {
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        run();
        best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
                                  .count());
    }
    return 2.0 * n * n * n / best / 1e9;
}

int bench_gemm() {
    namespace k = tinyinfer::kernels;
    const std::string original = k::gemm_isa();
    const double peak2 = k::fma_peak_gflops("avx2");
    const double peak512 = k::fma_peak_gflops("avx512");
    std::cout << std::fixed << std::setprecision(1) << "FMA peak, one core: avx2 " << peak2
              << " GFLOP/s, avx512 " << peak512 << " GFLOP/s\n"
              << "GFLOP/s, one core (n x n x n):\n"
              << "    n    naive   scalar     avx2   avx512   avx512 % of peak\n";
#ifdef _OPENMP
    omp_set_num_threads(1);  // the per-ISA table is single-core
#endif
    for (int n : {128, 256, 512, 1024, 2048}) {
        std::vector<float> a(static_cast<std::size_t>(n) * n), b(a.size()), c(a.size());
        for (std::size_t i = 0; i < a.size(); ++i) {
            a[i] = static_cast<float>(i % 7) * 0.1F;
            b[i] = static_cast<float>(i % 5) * 0.2F;
        }
        std::cout << std::setw(5) << n;
        if (n <= 512) {
            const double naive = gflops(n, 3, [&] {
                k::gemm_naive(n, n, n, a.data(), n, b.data(), n, false, c.data(), n);
            });
            std::cout << std::setw(9) << naive;
        } else {
            std::cout << std::setw(9) << "-";  // minutes at this size, skip
        }
        double last = 0.0;
        for (const char* isa : {"scalar", "avx2", "avx512"}) {
            if (!k::set_gemm_isa(isa)) {
                std::cout << std::setw(9) << "n/a";
                continue;
            }
            last = gflops(n, n >= 2048 ? 2 : 5, [&] {
                k::gemm(n, n, n, a.data(), n, b.data(), n, false, c.data(), n);
            });
            std::cout << std::setw(9) << last;
        }
        std::cout << std::setw(14) << (peak512 > 0 ? 100.0 * last / peak512 : 0.0) << "%\n";
    }
    k::set_gemm_isa(original.c_str());

#ifdef _OPENMP
    // multithreaded scaling at one size; physical cores matter more than logical ones
    const int n = 2048;
    std::vector<float> a(static_cast<std::size_t>(n) * n, 0.1F), b(a.size(), 0.2F), c(a.size());
    std::cout << "\nOpenMP scaling, n = " << n << " (" << k::gemm_isa() << "):\n";
    double one = 0.0;
    for (int threads : {1, 2, 4, 8, 12, 24}) {
        omp_set_num_threads(threads);
        const double g = gflops(n, 3, [&] {
            k::gemm(n, n, n, a.data(), n, b.data(), n, false, c.data(), n);
        });
        if (threads == 1) one = g;
        std::cout << std::setw(4) << threads << " threads: " << std::setw(7) << g
                  << " GFLOP/s  (" << std::setprecision(1) << g / one << "x)\n";
    }
    omp_set_num_threads(omp_get_num_procs());
#endif
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && std::strcmp(argv[1], "bench-gemm") == 0) return bench_gemm();
    if (argc < 2) {
        print_usage();
        return 0;
    }
    if (std::strcmp(argv[1], "inspect") == 0) {
        if (argc < 3) {
            print_usage();
            return 1;
        }
        const bool nodes = argc > 3 && std::strcmp(argv[3], "--nodes") == 0;
        return inspect(argv[2], nodes);
    }
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--version") == 0) {
            std::cout << tinyinfer::kVersion << '\n';
        } else if (std::strcmp(argv[i], "--info") == 0) {
            std::cout << "tinyinfer " << tinyinfer::kVersion << '\n'
                      << "simd: " << tinyinfer::to_string(tinyinfer::cpu_features()) << '\n';
        } else {
            std::cerr << "unknown option: " << argv[i] << '\n';
            print_usage();
            return 1;
        }
    }
    return 0;
}
