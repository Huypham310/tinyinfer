"""Latency / throughput of ONNX Runtime on a model, to compare against `tinyinfer run`.

    python tools/bench_ort.py model.onnx --batch 1 8 32 --threads 0 4

Same protocol as the C++ side: 5 warm-up runs, then the median of `--iters` timed runs on a
fixed random input. threads = 0 means ONNX Runtime's default (one per physical core).
"""

import argparse
import statistics
import time

import numpy as np
import onnxruntime as ort


def bench(path: str, batch: int, threads: int, iters: int) -> float:
    """Median milliseconds per run of one batch."""
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    x = np.random.default_rng(0).standard_normal((batch, 3, 64, 64)).astype(np.float32)
    for _ in range(5):
        sess.run(None, {inp.name: x})
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, {inp.name: x})
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model")
    p.add_argument("--batch", type=int, nargs="+", default=[1, 8, 32])
    p.add_argument("--threads", type=int, nargs="+", default=[0])
    p.add_argument("--iters", type=int, default=50)
    args = p.parse_args()
    print(f"onnxruntime {ort.__version__}")
    print(f"{'threads':>7} {'batch':>5} {'ms/batch':>9} {'ms/image':>9} {'img/s':>8}")
    for t in args.threads:
        for b in args.batch:
            # fewer timed runs for big batches, they're slow and stable anyway
            ms = bench(args.model, b, t, max(10, args.iters // b))
            label = "default" if t == 0 else str(t)
            print(f"{label:>7} {b:>5} {ms:>9.2f} {ms / b:>9.3f} {1000.0 * b / ms:>8.0f}")


if __name__ == "__main__":
    main()
