// Command line driver. For now it only reports what the engine sees on this machine;
// model loading and benchmarking get added as those parts are built.

#include <cstring>
#include <iostream>

#include "tinyinfer/cpu_features.hpp"
#include "tinyinfer/version.hpp"

namespace {

void print_usage() {
    std::cout << "usage: tinyinfer [--info] [--version]\n"
                 "  --info     show detected CPU SIMD features\n"
                 "  --version  print version\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 0;
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
