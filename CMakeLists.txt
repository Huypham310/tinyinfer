cmake_minimum_required(VERSION 3.24)

project(tinyinfer VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(TINYINFER_BUILD_TESTS "Build unit tests" ON)
option(TINYINFER_NATIVE "Compile with -march=native" OFF)
option(TINYINFER_OPENMP "Multithread GEMM with OpenMP" ON)
option(TINYINFER_BUILD_BENCH "Build Google Benchmark suite" OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Warnings stay on for everything we own; third-party targets get their own flags.
add_library(tinyinfer_warnings INTERFACE)
if(MSVC)
  target_compile_options(tinyinfer_warnings INTERFACE /W4 /permissive-)
else()
  target_compile_options(tinyinfer_warnings INTERFACE -Wall -Wextra -Wpedantic -Wshadow)
endif()

configure_file(include/tinyinfer/version.hpp.in
               ${CMAKE_CURRENT_BINARY_DIR}/generated/tinyinfer/version.hpp @ONLY)

add_library(tinyinfer
  src/cpu_features.cpp
  src/executor.cpp
  src/graph.cpp
  src/kernels/conv.cpp
  src/kernels/gemm.cpp
  src/kernels/gemm_avx2.cpp
  src/kernels/gemm_avx512.cpp
  src/kernels/ops.cpp
  src/layout.cpp
  src/onnx/model.cpp
  src/quant/gemm_vnni.cpp
  src/quant/quant.cpp
  src/onnx/wire.cpp
  src/shape.cpp
  src/tensor.cpp
)
target_include_directories(tinyinfer
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/generated
)
target_link_libraries(tinyinfer PRIVATE tinyinfer_warnings)
if(TINYINFER_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(tinyinfer PUBLIC OpenMP::OpenMP_CXX)
  endif()
endif()
if(TINYINFER_NATIVE AND NOT MSVC)
  target_compile_options(tinyinfer PRIVATE -march=native)
endif()

# Only the micro-kernel files get the wide ISAs; everything else stays baseline x86-64, and
# gemm.cpp only calls them after CPUID says they're safe.
if(MSVC)
  set_source_files_properties(src/kernels/gemm_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
  set_source_files_properties(src/kernels/gemm_avx512.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
  set_source_files_properties(src/quant/gemm_vnni.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
else()
  set_source_files_properties(src/kernels/gemm_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
  set_source_files_properties(src/kernels/gemm_avx512.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f;-mfma")
  set_source_files_properties(src/quant/gemm_vnni.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vnni")
endif()
if(MINGW)
  # the Windows x64 ABI only guarantees a 16-byte aligned stack, and GCC can't realign it for
  # 32/64-byte vector spills there; have the assembler emit unaligned moves instead
  target_compile_options(tinyinfer PRIVATE -Wa,-muse-unaligned-vector-move)
endif()

add_executable(tinyinfer_cli tools/tinyinfer_cli.cpp)
set_target_properties(tinyinfer_cli PROPERTIES OUTPUT_NAME tinyinfer)
target_link_libraries(tinyinfer_cli PRIVATE tinyinfer tinyinfer_warnings)

if(TINYINFER_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(TINYINFER_BUILD_BENCH)
  add_subdirectory(bench)
endif()
