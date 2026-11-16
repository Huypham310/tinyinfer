#!/usr/bin/env bash
# Build, test and benchmark in a plain Linux container, next to ONNX Runtime on the same CPU.
#
#   docker run --rm -v "$PWD:/src" -v /path/to/terrasense/artifacts:/model:ro -w /src \
#       python:3.12 bash scripts/bench_linux.sh
#
# (I run it this way because Windows Smart App Control keeps blocking freshly built exes.)
# Expects /model/model.onnx; build/eval from tools/export_eval_set.py is used if present.
set -euo pipefail

MODEL=${MODEL:-/model/model.onnx}
B=/tmp/build

pip install -q cmake ninja onnxruntime numpy 2>/dev/null
cmake -S . -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release -DTINYINFER_BUILD_BENCH=ON >/dev/null
cmake --build "$B" 2>&1 | tail -1

echo "== unit tests"
TINYINFER_RESNET18="$MODEL" "$B/tests/tinyinfer_tests" --gtest_brief=1 | tail -2

echo "== tinyinfer run (batch 1)"
"$B/tinyinfer" run "$MODEL" --iters 300
for b in 8 32; do
    echo "== tinyinfer run (batch $b)"
    "$B/tinyinfer" run "$MODEL" --batch "$b" --iters 30 | grep latency
done

if [ -f build/eval/test.images.bin ]; then
    echo "== tinyinfer eval"
    "$B/tinyinfer" eval "$MODEL" --data build/eval --batch 100
fi

echo "== onnxruntime"
python tools/bench_ort.py "$MODEL" --batch 1 8 32 --threads 0 4

echo "== google benchmark"
mkdir -p bench/results
TINYINFER_RESNET18="$MODEL" "$B/bench/tinyinfer_bench" --benchmark_min_time=0.3s \
    --benchmark_out=bench/results/linux.json --benchmark_out_format=json
