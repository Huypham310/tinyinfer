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

#include "tinyinfer/cpu_features.hpp"
#include "tinyinfer/onnx/model.hpp"
#include "tinyinfer/version.hpp"

namespace {

void print_usage() {
    std::cout << "usage: tinyinfer [--info] [--version]\n"
                 "       tinyinfer inspect <model.onnx> [--nodes]\n"
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

}  // namespace

int main(int argc, char** argv) {
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
