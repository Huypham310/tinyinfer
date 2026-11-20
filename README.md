# TinyInfer

[![CI](https://github.com/Huypham310/tinyinfer/actions/workflows/ci.yml/badge.svg)](https://github.com/Huypham310/tinyinfer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![C++17](https://img.shields.io/badge/C%2B%2B-17-informational)

A small CPU inference engine for convolutional networks, written in C++17.

TinyInfer loads an ONNX model, builds its own graph representation, plans memory up front, and
runs it with hand-written kernels. The goal is to understand where inference time actually goes
on a modern CPU: every kernel started as a naive loop and was improved step by step
(cache blocking, AVX2 / AVX-512 SIMD, OpenMP threads, operator fusion, INT8 quantization),
with each step measured against the machine's FMA peak and against ONNX Runtime.

The reference workload is the ResNet-18 land-use classifier from
[TerraSense](https://github.com/Huypham310/terrasense). It is **not** a replacement for ONNX
Runtime: on that model TinyInfer is still ~2.6-2.9x slower (see [Benchmarks](#benchmarks)),
but it matches ONNX Runtime's fp32 accuracy exactly and every step of the way is explained.

## Features

- **Tensors**: 64-byte aligned storage, `float32` / `int8` / `int32` / `int64`, NCHW and blocked
  NCHWc layouts ([Tensors](#tensors))
- **ONNX loader** with its own protobuf wire reader, no libprotobuf, fuzz-style corruption tests
  ([ONNX loading](#onnx-loading))
- **Kernels**: GEMM (Goto / BLIS blocking), im2col conv with batching, pooling, batch norm,
  ReLU, add, softmax ([Kernels](#kernels))
- **SIMD** micro-kernels for AVX2 and AVX-512 picked at runtime from CPUID, ~90% of the core's
  FMA peak, plus OpenMP threading with work-based team sizes ([SIMD and threads](#simd-and-threads))
- **Graph passes**: topological sort, shape inference, BatchNorm folding, ReLU fusion, liveness-based
  buffer release ([Running a model](#running-a-model))
- **INT8 post-training quantization** with per-channel weight scales and an AVX-512 VNNI kernel
  ([INT8 quantization](#int8-quantization))
- **Benchmarks** (Google Benchmark + end-to-end comparison with ONNX Runtime)
  ([Benchmarks](#benchmarks))
- **Python bindings** via pybind11, `pip install .` ([Python](#python))
- **Tests and CI**: GoogleTest + pytest, ~90% line coverage of `src/`, GitHub Actions and a Docker
  check image ([Tests](#tests), [CI and Docker](#ci-and-docker))

## Architecture

```mermaid
flowchart LR
    subgraph load["Load"]
        onnx["model.onnx"] --> wire["onnx::WireReader<br/>(protobuf wire format)"]
        wire --> model["onnx::Model<br/>nodes + initializers"]
    end

    subgraph plan["Graph (build time)"]
        model --> graph["Graph<br/>topo sort, shape inference"]
        graph --> passes["BN folding,<br/>ReLU fusion,<br/>liveness"]
    end

    subgraph run["Executor (run time)"]
        passes --> exec["Executor<br/>node by node,<br/>free after last use"]
        exec --> conv["conv2d<br/>(batched im2col)"]
        conv --> gemm["gemm<br/>packed Goto / BLIS<br/>or skinny n &lt;= 32"]
        exec --> ops["pool / add / relu /<br/>softmax / gemm_op"]
    end

    subgraph isa["Kernels (ISA picked at runtime)"]
        cpuid["CPUID<br/>cpu_features()"] --> pick{"best ISA"}
        pick --> avx512["AVX-512 12x32"]
        pick --> avx2["AVX2 + FMA 6x16"]
        pick --> scalar["scalar 4x16"]
        gemm --> pick
    end

    subgraph int8["INT8 path (optional)"]
        calib["calibrate():<br/>Conv input ranges"] --> qconv["QConv<br/>s8 weights, u8 activations"]
        qconv --> vnni["VNNI vpdpbusd<br/>(scalar fallback)"]
    end

    exec -. "Conv nodes in the plan" .-> qconv

    py["Python Session<br/>(pybind11)"] --> graph
    py --> exec
    py --> calib
    cli["tinyinfer CLI<br/>inspect / run / eval / bench-gemm"] --> graph
    cli --> exec
```

Deeper notes (memory planning, GEMM blocking, threading, the INT8 dequantization maths) are in
[docs/architecture.md](docs/architecture.md).

## Design decisions

- **No libprotobuf.** A model file only needs a handful of message types, and a small hand-written
  wire reader (~170 lines with its header) keeps the build to CMake + a compiler. It also lets
  the loader bounds-check every read and hand out zero-copy views into the file buffer.
- **ISA chosen at runtime.** Only the micro-kernel files are compiled with `-mavx2` / `-mavx512f`;
  `cpu_features()` reads CPUID on first use, so one binary runs on any x86-64 CPU and
  `TINYINFER_ISA=scalar|avx2|avx512` can force a path for benchmarks and debugging.
- **Skinny GEMM for n <= 32.** Late ResNet convs at batch 1 are GEMMs with N = 4 or 16, where a
  packed 32-wide tile is mostly padding. Those shapes skip packing and compute dot products
  straight off the weights; `BM_GemmFewColumns` put the crossover at n = 32.
- **OpenMP team sizes from the work, not the core count.** Every parallel region asks
  `par::team_size(work, per_thread)` and the cap is one thread per physical core. On this
  CPU a small GEMM took 410 us with 24 threads and 42 us with 4, because idle threads kept
  spinning on the SMT siblings of the busy ones.
- **INT8 with VNNI.** Symmetric per-channel int8 weights, asymmetric uint8 activations, int32
  accumulation with `vpdpbusd` (64 MACs per instruction), and a scalar kernel over the same
  packed layout that is tested to agree bit for bit. Only convolutions are quantized; everything
  between them stays float.

## Tensors

- `AlignedBuffer`: zero-initialized storage aligned to 64 bytes (one cache line, one AVX-512
  register) and padded to a multiple of 64, so vector loops can always load a full register.
- `Tensor`: dense row-major tensor (`float32`, `int8`, `int32`, `int64`) that shares its
  storage on copy; `reshape` is a view, `clone` a real copy, and typed access checks the dtype.
- `to_blocked` / `from_blocked`: NCHW <-> NCHWc. Channels are grouped in blocks of 8 (AVX2) or
  16 (AVX-512) that become the innermost dimension, so a convolution kernel can load the same
  pixel of a whole channel block with one aligned vector load. Missing channels in the last block
  are zero, so they don't change dot products.

## ONNX loading

There is no libprotobuf dependency: `onnx::WireReader` decodes the protobuf wire format
directly (varint tags, length-delimited fields as zero-copy views into the file buffer) and
`onnx::load_model` maps `ModelProto` / `GraphProto` / `NodeProto` / `TensorProto` onto plain
structs. Weights stored as `raw_data` are copied with one `memcpy` into aligned tensors;
`float_data` / `int32_data` / `int64_data` (packed or not) are supported too.

- Every read is bounds-checked and tensor sizes are validated against the bytes actually present
  *before* allocating, so corrupted files fail with `ParseError` instead of crashing or asking
  for terabytes. The tests cut a model at every byte offset and flip random bytes 3,000 times.
- Weights listed as graph inputs (old exporters) are filtered out of `runtime_inputs()`.
- Not supported: external data files (models > 2 GB), sparse initializers, subgraphs.

```text
$ tinyinfer inspect terrasense/artifacts/model.onnx
model:   ir 10, opset 20, producer pytorch 2.14.1+cpu
graph:   main_graph, 49 nodes, 44 initializers, 11176846 params (42.6 MiB)
input:   input float [batch, 3, 64, 64]
output:  logits float [batch, 10]
ops:     Conv x20 Relu x17 Add x8 Gemm x1 MaxPool x1 ReduceMean x1 Reshape x1
parsed in 70.4 ms
```

That is the TerraSense ResNet-18 (EuroSAT, BatchNorm already folded into the convolutions by
the exporter). Test fixtures are generated by `tools/make_test_models.py` (needs `pip install
onnx`); set `TINYINFER_RESNET18=/path/to/model.onnx` to also run the full-model test.

## Kernels

float32 NCHW kernels for everything ResNet-18 needs (the scalar versions are the baseline and
fallback; the SIMD GEMM micro-kernels are in [SIMD and threads](#simd-and-threads)): `gemm`, `conv2d` (im2col + GEMM,
with stride / asymmetric padding / dilation / groups), `max_pool2d`, `global_avg_pool`,
inference `batch_norm`, `relu`, `add`, `softmax` and ONNX `Gemm` (`gemm_op`).

- **GEMM** follows the Goto / BLIS loop order: B is packed into 256 x NR panels (one sliver
  stays in L1 while the micro-kernel walks down A), A into 96 x 256 blocks (in L2), and an
  MR x NR accumulator tile stays in registers (4 x 16 scalar, 6 x 16 AVX2, 12 x 32 AVX-512). Packing buffers are thread-local and reused, so
  steady-state calls don't allocate. Edges are zero-padded in the packed copies, so any size
  works.
- **conv2d** unrolls receptive fields with im2col and calls the GEMM once per image and group;
  1x1 stride-1 convs skip the copy since the input already *is* the GEMM operand.
- Every kernel is checked against a textbook reference (`gemm_naive`, seven-loop
  `conv2d_naive`) on sizes that straddle the block edges, ResNet-18's conv shapes, asymmetric
  padding, dilation, grouped and depthwise convs.

```text
$ tinyinfer bench-gemm          # one core, GCC 16 -O3, baseline x86-64 (SSE2), Ryzen AI 9 HX 370
    n    naive GFLOP/s   blocked GFLOP/s   speedup
  128            4.22             49.17     11.7x
  256            3.23             49.47     15.3x
  512            2.06             48.27     23.4x
 1024            0.66             30.48     46.4x
```

The naive loop falls off a cliff as soon as a column of B stops fitting in cache (every inner
step jumps a full row); the blocked scalar one holds ~49 GFLOP/s up to 512 and drops at 1024
in this build. The AVX2 / AVX-512 micro-kernels below hold their speed at that size.

## SIMD and threads

The GEMM micro-kernel is picked at runtime from CPUID (override with `TINYINFER_ISA=scalar|avx2|avx512`):

| ISA | Tile | Registers | Built with |
|---|---|---|---|
| scalar | 4 x 16 | whatever GCC auto-vectorizes for baseline x86-64 | default flags |
| AVX2 + FMA | 6 x 16 | 12 ymm accumulators + 2 B + 1 broadcast (15 / 16) | `-mavx2 -mfma` on that file only |
| AVX-512 | 12 x 32 | 24 zmm accumulators + 2 B + 1 broadcast (27 / 32) | `-mavx512f -mfma` on that file only |

Only the micro-kernel files are compiled with wide ISAs, so the library still runs on any x86-64 CPU. `bench-gemm` also measures the practical FMA peak with a register-only loop, so "% of peak" is against what this core actually does, not a datasheet number.

```text
$ tinyinfer bench-gemm            # Ryzen AI 9 HX 370, one core unless noted
FMA peak, one core: avx2 158.4 GFLOP/s, avx512 159.3 GFLOP/s
    n    naive   scalar     avx2   avx512   avx512 % of peak
  256      3.4     49.6    133.1    134.3          84.3%
  512      2.1     49.7    139.5    143.6          90.2%
 1024        -     48.1    137.9    140.6          88.3%
 2048        -     47.6    133.0    134.2          84.3%

OpenMP scaling, n = 2048 (avx512):
   1 threads:   135.4 GFLOP/s  (1.0x)
   2 threads:   260.4 GFLOP/s  (1.9x)
   4 threads:   464.6 GFLOP/s  (3.4x)
   8 threads:   311.4 GFLOP/s  (2.3x)
  24 threads:   538.7 GFLOP/s  (4.0x)
```

- **~90% of the core's FMA peak** at n = 512 with either vector kernel, ~50x the textbook triple loop.
- **AVX-512 doesn't beat AVX2 here**: both peaks are 159 GFLOP/s because mobile Zen 5 executes 512-bit ops on 256-bit units. The wider kernel still helps a little (fewer instructions and broadcasts per flop).
- **The AVX2 kernel first ran at half speed.** Written with `__m256 acc[6][2]`, GCC kept the accumulator array in memory (the edge-tile path stores it) and wrote all 12 registers back to the stack on every k step, so the loop was bound by the store port: 66 GFLOP/s with the data in L1. Twelve named accumulator variables fixed it: 155 GFLOP/s in L1. Found by reading the generated assembly.
- **Threads**: B is packed once per block by all threads together, then each thread packs and multiplies its own row blocks (`schedule(dynamic)`). Scaling is near-linear on the 4 big Zen 5 cores; past that the chip's other 8 cores are Zen 5c in a separate cluster with its own L3, and 8 threads land partly on SMT siblings sharing FMA units, so 8 threads is *slower* than 4. All 24 threads reach ~540-650 GFLOP/s.

## Running a model

`Graph` turns an `onnx::Model` plus concrete input shapes into an executable graph; `Executor` runs it.

- **Op mapping** for Conv, Relu, Add, MaxPool, GlobalAveragePool / ReduceMean (spatial axes, `axes` as an attribute or, from opset 18, a constant input), Flatten / Reshape (views, no copy), Gemm, BatchNormalization, Softmax, Clip, Identity / Dropout. Anything else is a `GraphError` at load time naming the op, not a crash mid-inference.
- **Topological sort** (Kahn), so node order in the file doesn't matter; **shape inference** for every value, with symbolic dims like `batch` bound at construction.
- **BatchNorm folding**: Conv followed by BatchNormalization becomes one Conv with weights scaled by gamma / sqrt(var + eps) and a shifted bias (on copies; initializers are never edited in place).
- **Relu fusion** into a preceding Conv / Add / Gemm when nothing else reads the pre-activation value: one less pass over memory and one less buffer.
- **Liveness**: each intermediate is released right after its last reader, so peak memory is the widest point of the network instead of the sum of all activations.

```text
$ tinyinfer run terrasense/artifacts/model.onnx --ref build/ref/resnet18
32 ops after fusion (0 BN folded, 17 ReLU fused), gemm isa avx512
input [1, 3, 64, 64] -> output [1, 10]
latency: median 13.66 ms, min 11.90 ms over 50 runs
activations: 1.07 MiB if all kept, peak 0.31 MiB with liveness
max |diff| vs reference: 2.38e-07
```

The reference is ONNX Runtime's output for the same input (`tools/ort_reference.py`); the C++ parity tests cover the three small fixtures (Conv + Relu, Reshape / Gemm / Clip, Conv + BN + Relu + pooling + Gemm + Softmax) and the full ResNet-18 when `TINYINFER_RESNET18` / `TINYINFER_RESNET18_REF` are set.

**Correct first, then fast.** The output above is from the first end-to-end version (current
numbers are in [Benchmarks](#benchmarks)). Back then ONNX Runtime ran the model in ~1.8 ms and TinyInfer in ~13.7 ms. `TINYINFER_PROFILE=1` prints per-op times, and the late convolutions dominated: 512 -> 512 channels on a 2 x 2 feature map is a GEMM with M = 512, N = 4, K = 4608, which wastes most of a 12 x 32 tile and re-packs 9.4 MB of weights on every call (~10 GFLOP/s). Profiling also caught a real bug: the GEMM went multithreaded on total work, and these shapes ran **3x slower with 24 threads than with one** (two barriers per k-block, 18 k-blocks). Gating threads on the work between barriers took the model from 34.6 to 13.7 ms. The fixes for the rest are in [Benchmarks](#benchmarks).

## INT8 quantization

Post-training quantization of the convolutions (`tinyinfer::quant`), the scheme ONNX Runtime and oneDNN use on x86:

- **Weights**: symmetric int8 with one scale per output channel, packed once at load time into 8-row panels with k in groups of 4.
- **Activations**: asymmetric uint8 with a zero point, one scale per tensor from the min / max observed on calibration images. The range always includes 0, so zero padding is exact; im2col pads with the zero point, not with 0.
- **Kernel**: AVX-512 VNNI. `vpdpbusd` multiplies 4 uint8 activations by 4 int8 weights and adds them into an int32 lane, 64 MACs per instruction; an 8 x 32 tile keeps 16 accumulators in registers. A scalar kernel over the same packed layout is the fallback, and a test checks the two agree **bit for bit**.
- **Dequantization**: `y = (acc - zp * sum(w)) * s_x * s_w[o] + bias`; everything between convolutions (Add, pooling, the classifier) stays float.

`tools/export_eval_set.py` exports TerraSense's preprocessed EuroSAT test split (2,700 images) and 256 validation images for calibration:

```text
$ tinyinfer eval terrasense/artifacts/model.onnx --data build/eval --batch 100
2700 test images, batch 100, int8 kernel vnni, calibration 256 images in 2.31 s
  fp32: accuracy 98.19%, 10.89 ms/image
  int8: accuracy 98.07%, 4.38 ms/image (2.49x), same top-1 as fp32 on 99.81%
```

- The fp32 accuracy is exactly the 98.19% TerraSense reports with ONNX Runtime on the same split: the whole engine, end to end, on 2,700 real images.
- INT8 costs 0.12 points (ONNX Runtime's static INT8 on this model: 98.11%) and ran 2.5x faster than TinyInfer's fp32 *at the time*. The bench phase then made fp32 much faster (skinny GEMM, batched im2col) and the gap shrank; see [Benchmarks](#benchmarks).

## Benchmarks

All numbers in this section come from `scripts/bench_linux.sh`, run in a `python:3.12` container
(Linux, GCC 12) on the same laptop (Ryzen AI 9 HX 370: 4 Zen 5 + 8 Zen 5c cores, SMT), with
ONNX Runtime 1.31 installed in the same container:

```bash
docker run --rm -v "$PWD:/src" -v /path/to/terrasense/artifacts:/model:ro -w /src \
    python:3.12 bash scripts/bench_linux.sh
```

(Windows Smart App Control kept blocking the freshly built benchmark exe, so the container is
also the easiest way to get repeatable runs.) Raw Google Benchmark output:
[`bench/results/linux.json`](bench/results/linux.json).

### ResNet-18 end to end (TerraSense model, 3 x 64 x 64, 4 threads)

| batch | before (INT8 phase) | now | speed-up | ONNX Runtime | vs ORT |
|---|---|---|---|---|---|
| 1 | 14.46 ms | 5.12 ms | 2.8x | 1.76 ms | 2.9x slower |
| 8 | 96.0 ms (12.0 ms/img) | 26.8 ms (3.3 ms/img) | 3.6x | 9.4 ms | 2.8x slower |
| 32 | 399 ms (12.5 ms/img) | 90.0 ms (2.8 ms/img) | 4.4x | 34.4 ms | 2.6x slower |

"Before" is the engine as it was right after the INT8 work (before the changes below), built and run in the same container. Medians;
both engines on 4 threads, which is the fastest setting for both at batch 1 on this CPU
(ONNX Runtime's default of 12 threads takes 3.1 ms). On the 2,700-image EuroSAT eval
(`tinyinfer eval`, batch 100) fp32 now takes 3.40 ms/image in the container (10.89 natively on
Windows before these changes), accuracy unchanged at 98.19%.

### What changed

1. **Skinny GEMM for few columns.** At batch 1 the late convs are GEMMs with N = 4 (layer 4) or
   16 (layer 3). The packed path wastes most of a 32-wide tile and re-packs MBs of weights per
   call; the new kernel computes C = A * B^T as dot products straight off the weights (4 rows x 4
   columns = 16 zmm accumulators, masked loads for the k tail). `BM_GemmFewColumns` (512 x n x 4608)
   put the cutoff at n = 32:

   | n | 1 | 4 | 8 | 16 | 32 | 64 (packed) |
   |---|---|---|---|---|---|---|
   | GFLOP/s | 44 | 183 | 239 | 300 | 169 | 212 |

   The packed kernel managed ~62 GFLOP/s at n = 32 before the switch.
2. **Batched im2col.** Convolutions used to run one image at a time, so N stayed at 4 for the
   late layers whatever the batch. Now several images' columns sit side by side in one matrix
   (capped at ~8 MB), and one GEMM covers them all; the int8 conv got the same treatment.
3. **OpenMP team sizes.** The biggest surprise. A batch-32 64-channel conv took 32.7 ms with the
   default 24 threads, 10.4 ms with 1 and 4.9 ms with 4: a memory-bound im2col region with all
   24 threads followed by a 4-thread GEMM left 20 threads spinning on the SMT siblings of the
   working ones. Every parallel region now asks `par::team_size(work, per_thread)` for a team
   that fits its work, capped at one thread per physical core (ONNX Runtime's default too;
   `TINYINFER_THREADS` overrides it). The same fix took the skinny GEMM at n = 1 from 410 to 42 us.

### Kernels (Google Benchmark, `tinyinfer_bench`)

| benchmark | result |
|---|---|
| GEMM 512^3, one core: scalar / AVX2 / AVX-512 | 7.8 / 107 / 121 GFLOP/s |
| conv layer 4 (512 ch, 2 x 2), batch 1: fp32 / int8 | 149 / 358 us |
| conv layer 4, batch 32: fp32 / int8 | 3.51 / 1.96 ms (172 / 308 GFLOP/s) |
| conv layer 3 (256 ch, 4 x 4), batch 32: fp32 / int8 | 4.84 / 2.63 ms |

The scalar GEMM is 6x slower here than with MinGW GCC 14 on Windows (49 GFLOP/s): GCC 12 doesn't
vectorize the 4 x 16 fallback loop as well. The SIMD kernels are within 15-20% of the Windows
numbers.

### Where ONNX Runtime still wins

The remaining ~2.8x is mostly in the first layers: the 7 x 7 stem and the 64-channel 3 x 3
convs run at 50-95 GFLOP/s because im2col writes (and the GEMM re-reads) a copy of the input
9-12x its size. ONNX Runtime (MLAS) uses NCHWc blocked layouts and direct convolution for
these. INT8 only beats fp32 on the wide layers: at layer 4 batch 1 it's 2.4x *slower*, because
the u8 quantize + im2col pass costs more than the VNNI GEMM saves on 4 columns, and end to end
it's now about as fast as fp32 (eval: 3.56 vs 3.40 ms/image).

## Python

`pip install .` builds the engine as a pybind11 module (scikit-build-core drives CMake with
`TINYINFER_BUILD_PYTHON=ON`):

```python
from tinyinfer import Session, cpu_features

s = Session("model.onnx", batch=1)
print(s.info())            # nodes, BN folded, ReLU fused, peak live MiB, ISA
out = s.run(images)        # float32 NCHW ndarray -> float32 logits

s.calibrate([images])      # record Conv ranges, switch those nodes to INT8
out8 = s.run(images)
```

```text
$ python -c "..."          # Linux container, same ResNet-18 as above
fp32 batch1: median 4.50 ms, peak 320 KiB
int8 batch8: median 41.3 ms (5.16 ms/img)
```

Crossing the Python boundary costs almost nothing next to the model: 4.50 ms vs 5.12 ms for the
same batch-1 run through the C++ CLI (noise). The pytest cases are listed under [Tests](#tests).

## Tests

```bash
cmake --preset release && cmake --build --preset release
ctest --preset release                      # C++ (GoogleTest)
pip install . pytest && pytest              # Python bindings
```

- **C++ (gtest)**: tensors / layouts, the protobuf wire reader and ONNX loader (including cutting a
  model at every byte and flipping random bytes), every kernel against its naive reference, graph
  passes, the executor, and INT8. Three small ONNX fixtures under `tests/data` come with
  ONNX Runtime's output for the same input, so `Graph` + `Executor` are checked against ORT, not
  just against themselves. Edge cases covered on purpose: `clip` / `relu_inplace`, executor
  profiling and range observation, load-time errors for unsupported attributes, `groups != 1`
  refused by the INT8 path, `set_int8_isa` refusing unknown / unsupported kernels.
- **Python (pytest)**: `Session.run` against the same ORT golden outputs (`conv_relu`,
  `conv_bn_head`), the 2-D `mixed_types` model loading but being refused by `run`, the INT8
  ISA override and refuse path, and scalar vs VNNI INT8 giving the same output.
- Models that need external files (`TINYINFER_RESNET18`, `TINYINFER_RESNET18_REF`) are skipped
  when not set; everything else is self-contained.

### Coverage

`-DTINYINFER_COVERAGE=ON` builds the library and tests at `-O0 -g --coverage` (GCC or Clang;
it is a Debug build on purpose). `scripts/check_linux.sh` does the whole loop: configure, run
ctest, collect line coverage with lcov, keep only `src/` (no googletest, tests or tools), fail
under 85 %, then build the wheel and run pytest:

```bash
docker run --rm -v "$PWD:/src" -w /src python:3.12 bash scripts/check_linux.sh
# HTML=1 also writes an lcov HTML report; MIN_COVERAGE=90 changes the gate; SKIP_PYTHON=1 skips pytest
```

Measured in that container (GCC 12, Ryzen AI 9 HX 370 with AVX-512 + VNNI):
**`src/` line coverage 90.6 % (1,603 of 1,770 lines), 94.2 % of functions** (it was 89.7 % before
the extra gap tests). The lowest files are `graph.cpp` (83 %, mostly shape-inference error
branches), `gemm_avx512.cpp` and `gemm_avx2.cpp` (~87 %, edge-tile paths) and `wire.cpp` (86 %).
The AVX-512 and VNNI kernel files only count when the CPU actually has those instructions, so on
an older CPU the number is lower (roughly 5 points); that is why the gate is 85 %, not 90 %.

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

The `bench` preset builds the Google Benchmark suite (`build/bench/bench/tinyinfer_bench`; set
`TINYINFER_RESNET18` to the model for the end-to-end cases). Threads default to one per physical
core; `TINYINFER_THREADS=N` changes that (e.g. 24 to reproduce the `bench-gemm` scaling table).

## CI and Docker

GitHub Actions (`.github/workflows/ci.yml`, free `ubuntu-24.04` runners only) runs on every pull
request and on pushes to `main`:

| job | what it does |
|---|---|
| `format` | `clang-format --dry-run -Werror` over every tracked `.cpp` / `.hpp` (clang-format 18.1.8 from pip, pinned so the result doesn't depend on the runner image) |
| `cpp-test (release)` | `cmake --preset release`, build, `ctest --preset release`, then `tinyinfer --info` as a smoke test |
| `python-test` | `pip install .` (scikit-build-core builds the pybind11 module) and `pytest`, on Python 3.10 and 3.12 |
| `coverage (docker)` | builds the `Dockerfile` and runs it: coverage build, ctest, lcov gate (85 % of `src/` lines), pytest |

The models that need external files (`TINYINFER_RESNET18`, `TINYINFER_RESNET18_REF`) are not
available on the runners, so those tests and benchmarks are skipped there.

The `Dockerfile` (`python:3.12-bookworm` + CMake, Ninja, lcov; GCC 12 comes with the image) is a
throwaway build / check environment, not a runtime image. Its default command is
`scripts/check_linux.sh`:

```bash
docker build -t tinyinfer .
docker run --rm tinyinfer                        # coverage gate + pytest
docker run --rm -e MIN_COVERAGE=90 tinyinfer     # stricter gate
docker run --rm -e SKIP_PYTHON=1 tinyinfer       # C++ only
```

To run the formatting check locally the same way CI does:

```bash
pip install clang-format==18.1.8
git ls-files '*.cpp' '*.hpp' '*.h' | xargs clang-format --dry-run -Werror   # add -i to fix in place
```

## Limitations

- Tuned for x86-64 only: the SIMD and VNNI kernels are x86 intrinsics, anything else runs the
  scalar kernels.
- Inference only, float32 + INT8 convolutions, `groups = 1` for INT8, batch size fixed per
  `Graph` / `Session`.
- Operator coverage is what ResNet-style models need; anything else is a load-time `GraphError`.
- Still ~2.6-2.9x slower than ONNX Runtime on ResNet-18 (see
  [Where ONNX Runtime still wins](#where-onnx-runtime-still-wins) for why).

## License

MIT, see [LICENSE](LICENSE).
