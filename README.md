# TinyInfer

A small CPU inference engine for convolutional networks, written in C++17.

TinyInfer loads an ONNX model, builds its own graph representation, plans memory up front, and
runs it with hand-written kernels. The goal is to understand where inference time actually goes
on a modern CPU: every kernel starts as a naive loop and is then improved step by step
(cache blocking, AVX2 / AVX-512 SIMD, OpenMP threads, operator fusion, INT8 quantization),
with each step measured against a roofline model of the machine and against ONNX Runtime.

The reference workload is the ResNet-18 land-use classifier from
[TerraSense](https://github.com/Huypham310/terrasense).

> Status: early work in progress.

## Planned features

- Tensor type with aligned storage and NCHW / blocked layouts
- ONNX loader (protobuf) for the operators ResNet-style models need
- Kernels: GEMM, im2col + direct convolution, pooling, batch norm, ReLU, add, softmax
- SIMD paths for AVX2 and AVX-512 chosen at runtime from CPUID
- Graph passes: conv + bn + relu fusion, constant folding, static buffer reuse
- INT8 post-training quantization with per-channel weight scales
- Benchmarks (Google Benchmark) and roofline analysis
- Python bindings via pybind11

## Tensors

- `AlignedBuffer`: zero-initialized storage aligned to 64 bytes (one cache line, one AVX-512
  register) and padded to a multiple of 64, so vector loops can always load a full register.
- `Tensor`: dense row-major tensor (`float32`, `int8`, `int32`, `int64`) that shares its
  storage on copy; `reshape` is a view, `clone` a real copy, and typed access checks the dtype.
- `to_blocked` / `from_blocked`: NCHW <-> NCHWc. Channels are grouped in blocks of 8 (AVX2) or
  16 (AVX-512) that become the innermost dimension, so a convolution kernel can load the same
  pixel of a whole channel block with one aligned vector load. Missing channels in the last block
  are zero, so they don't change dot products.

## Build

Requirements: CMake 3.24+, Ninja, and GCC 12+ or Clang 16+ (MSVC should work but is not the
main target).

```bash
cmake --preset release
cmake --build --preset release
ctest --preset release
./build/release/tinyinfer --info
```

`--info` prints the SIMD features the engine detected on your CPU.

## License

MIT
