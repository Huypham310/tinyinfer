# TinyInfer architecture notes

The [README](../README.md) has the overview, the diagram and the measurements. This file is the
longer "how it fits together" companion: what each layer owns, and the reasoning behind the
decisions that were not obvious at the time.

## Layers

| layer | files | owns |
|---|---|---|
| tensors | `aligned.hpp`, `tensor.hpp`, `layout.hpp`, `shape.hpp` | aligned storage, dtypes, NCHW <-> NCHWc |
| ONNX | `onnx/wire.*`, `onnx/model.*` | protobuf wire decoding, `ModelProto` -> plain structs |
| graph | `graph.hpp`, `graph.cpp`, `executor.cpp` | op mapping, shape inference, fusion, liveness, execution |
| kernels | `kernels/*`, `parallel.hpp`, `cpu_features.cpp` | GEMM, conv, pooling, element-wise ops, ISA dispatch, thread sizing |
| quantization | `quant/*`, `quant.hpp` | calibration, int8 packing, VNNI GEMM, dequantization |
| front ends | `tools/tinyinfer_cli.cpp`, `python/tinyinfer_py.cpp` | CLI (`inspect`, `run`, `eval`, `bench-gemm`) and the pybind11 `Session` |

Dependencies only point downwards: kernels know nothing about graphs, the graph knows nothing
about the CLI or Python, and the Python module is a thin wrapper over `Graph`, `Executor` and
`quant::calibrate`.

## From file to executable graph

1. **Decode.** `onnx::WireReader` walks the protobuf wire format (varint tags, length-delimited
   fields). Strings and `raw_data` are views into the file buffer until a tensor is built, so
   parsing does one `memcpy` per weight tensor. Every length is checked against the bytes
   that remain *before* anything is allocated.
2. **Map.** `Graph` turns each `NodeProto` into a `GraphNode` with an `OpKind` and typed
   parameters (`Conv2dParams`, `Pool2dParams`, `alpha` / `beta`, ...). An unsupported op is a
   `GraphError` naming the op, raised at construction rather than during inference.
3. **Sort and infer.** Kahn's algorithm gives a topological order (the file order is not
   trusted). Shapes are propagated for every value; symbolic dimensions such as `batch` are
   bound to the concrete input shapes passed to the constructor.
4. **Fold and fuse.** `Conv -> BatchNormalization` becomes one `Conv` with weights scaled by
   `gamma / sqrt(var + eps)` and a shifted bias. The new weights are copies; initializers in the
   `Model` are never modified. `Relu` is merged into a preceding `Conv` / `Add` / `Gemm` only
   when no other node reads the pre-activation value. Merged nodes are marked `dead` and
   removed by `compact()`.
5. **Liveness.** For each value the index of its last reader is recorded (`last_use`).

The TerraSense ResNet-18 comes out of the exporter with BatchNorm already folded, so there the
BN pass has nothing to do (`0 BN folded`), but the small `conv_bn_head` test fixture exercises it.

## Executing

`Executor::run` visits the nodes in order, calls the matching kernel, and drops every input
tensor whose `last_use` is the current node. Peak memory is therefore the widest cut through the
network (0.31 MiB at batch 1 for ResNet-18 versus 1.07 MiB if every activation were kept).
`Reshape` / `Flatten` return views, so they cost nothing and keep the storage alive.

Two optional hooks live on the executor and keep the hot path free of branches in the common
case: profiling (per-node wall time, `TINYINFER_PROFILE=1`) and an INT8 plan
(`set_int8`). Calibration reuses the same hooks: it runs the float graph once per calibration
batch with `observe_conv_inputs` recording the min / max of every Conv input.

## Kernels

### GEMM

`gemm(m, n, k, A, B, C, trans_b, accumulate)` is the only compute-heavy primitive; convolutions
and `Gemm` nodes are expressed through it.

- **Packed path** (n > 32). Goto / BLIS loop nest with `NC = 512`, `KC = 256`, `MC = 96`. A
  `KC x NR` sliver of B stays in L1 while the micro-kernel walks down a packed block of A in
  L2. The packed copies are zero-padded, so edge tiles reuse the full micro-kernel and any size
  works. Packing buffers are `thread_local` and reused.
- **Skinny path** (n <= 32). No packing: `C = A * B^T` as dot products over rows of A and rows of
  `B^T`, with masked loads for the k tail. This is what the last ResNet stages need at small
  batch, where N is the number of output pixels (4 or 16).
- **Micro-kernels.** One per ISA, selected through a small `KernelInfo` record (name, MR, NR,
  packed kernel, skinny kernel), so `gemm()` itself is ISA-agnostic:

  | ISA | tile (MR x NR) | accumulators |
  |---|---|---|
  | scalar | 4 x 16 | compiler-vectorized |
  | AVX2 + FMA | 6 x 16 | 12 ymm |
  | AVX-512 | 12 x 32 | 24 zmm |

  The accumulators are plain named variables, not an array: with `__m256 acc[6][2]` GCC spilled
  the array to the stack on every k step and the kernel ran at half speed.

### ISA dispatch

`cpu_features()` runs CPUID once. `gemm()` takes the active `KernelInfo` through an atomic
pointer, defaulting to the best supported ISA; `TINYINFER_ISA` or `set_gemm_isa()` can pick another
one. `set_gemm_isa` refuses an ISA the CPU does not have instead of silently falling back, so a
benchmark can never report numbers for the wrong kernel. Only the files holding micro-kernels are
compiled with `-mavx2` / `-mavx512f`, which is why the same binary starts on any x86-64 CPU.

### Convolution

`conv2d` unrolls receptive fields with im2col and calls GEMM. 1x1 stride-1 convolutions skip the
copy. Several images are unrolled side by side into one wide matrix (capped at ~8 MB) so the
GEMM sees a large N even for late layers; for those layers at batch 1 the skinny path takes over.

### Threading

OpenMP is used inside kernels, never across graph nodes. Each parallel region sizes its own team
with `par::team_size(work, per_thread)`, bounded by `max_threads()` (one per physical core;
`TINYINFER_THREADS` or `OMP_NUM_THREADS` override it). Two measurements shaped this:

- GEMM threads synchronize twice per k-block, so what matters is the work *between* barriers,
  not the total. Gating on total work made `512 x 4 x 4608` run 3x slower with 24 threads
  than with one.
- Threads that are idle in one region still spin and steal their SMT siblings' execution
  resources in the next. Capping the team at the physical core count (and shrinking it for small
  regions) fixed a batch-32 64-channel conv that took 32.7 ms with 24 threads and 4.9 ms with 4.

## INT8 path

`quant::calibrate` runs the float graph on calibration batches, records the input range of every
non-grouped `Conv`, and builds a `QConv` for each: int8 weights with one scale per output
channel, packed once into MR-row panels with k in groups of 4 (the layout `vpdpbusd` wants).

Activations are quantized to uint8 with a zero point. The range is widened to include 0 so zero
padding and ReLU outputs are exactly representable, and im2col pads with the zero point. Because
the activation has a zero point, the int32 accumulator carries a term that has to be removed:

```text
acc  = sum_k  w_q[o,k] * x_q[k]
y[o] = (acc - zp * sum_k w_q[o,k]) * s_x * s_w[o] + bias[o]
```

`sum_k w_q[o,k]` (`w_sum`) is computed once at quantization time. `vpdpbusd` multiplies u8 by s8
and accumulates in int32; there is no s8 x s8 form, which is why activations are the unsigned
operand. The scalar fallback uses the same packed layout and a test requires it to match the VNNI
kernel bit for bit, so the fallback doubles as a reference.

Only convolutions run in INT8. `Add`, pooling and the classifier stay float, which avoids
requantization between layers at the cost of a quantize / dequantize around each conv. That cost
is why INT8 is slower than fp32 on the narrow late layers at batch 1 and only pays off on wide
ones (numbers in the README).

## Python

`Session(path, batch=N)` builds a `Graph` for that batch size and owns an `Executor`. `run`
accepts a float32 NCHW array and returns float32 logits; `calibrate([images])` runs
`quant::calibrate` and attaches the resulting plan so subsequent `run` calls use INT8 for the
quantized convolutions. `gemm_isa`, `int8_isa` and their setters expose the same overrides as
the environment variables, which the tests use to compare scalar and VNNI output.

## Testing strategy

- Every kernel is compared with a deliberately naive reference (`gemm_naive`, a seven-loop
  `conv2d_naive`, `gemm_s8u8_ref`) on sizes that straddle block edges.
- The loader is tested with a model cut at every byte and with random byte flips; the expected
  result is `ParseError`, never a crash or a huge allocation.
- Three small ONNX fixtures ship with ONNX Runtime's output for the same input, so the graph and
  executor are checked against another engine and not only against themselves.
- Coverage is measured on `src/` only, in a container, with a gate below the measured value (see
  README) because the AVX-512 / VNNI files only count on CPUs that run them.

## Known limits and ideas

- Convolution still goes through im2col; a direct or NCHWc blocked convolution for the stem and
  the 64-channel layers is the largest remaining gap to ONNX Runtime.
- `MC` / `KC` / `NC` are constants tuned for a 48 KB L1d / 1 MB L2 core; reading cache sizes at
  startup would be more portable.
- Only convolutions are quantized, and only per-tensor for activations.
- No ARM / NEON kernels.
