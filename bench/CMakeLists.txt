include(FetchContent)

FetchContent_Declare(
  benchmark
  URL https://github.com/google/benchmark/archive/refs/tags/v1.9.1.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP ON
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
# benchmark's configure step compiles *and runs* tiny feature probes (try_run); Windows Smart
# App Control blocks freshly built exes at random, which kills the configure. Every compiler
# we target has these two, so skip the probes.
set(HAVE_STD_REGEX ON CACHE BOOL "" FORCE)
set(HAVE_STEADY_CLOCK ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)

add_executable(tinyinfer_bench bench_kernels.cpp bench_model.cpp)
target_link_libraries(tinyinfer_bench PRIVATE tinyinfer tinyinfer_warnings benchmark::benchmark_main)
