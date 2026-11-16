"""Time Session.run on an ONNX model (default: TINYINFER_RESNET18).

    TINYINFER_RESNET18=path/to/model.onnx python python/examples/bench_session.py
"""

import os
import statistics
import time

import numpy as np

from tinyinfer import Session


def main() -> None:
    path = os.environ.get("TINYINFER_RESNET18")
    if not path:
        raise SystemExit("set TINYINFER_RESNET18 to an ONNX model")
    batch = int(os.environ.get("BATCH", "1"))
    s = Session(path, batch=batch)
    x = np.random.default_rng(0).standard_normal((batch, 3, 64, 64), dtype=np.float32)
    for _ in range(5):
        s.run(x)
    times = []
    for _ in range(max(20, 100 // batch)):
        t0 = time.perf_counter()
        s.run(x)
        times.append((time.perf_counter() - t0) * 1000.0)
    med = statistics.median(times)
    print(s.info())
    print(f"fp32 batch {batch}: median {med:.2f} ms ({med / batch:.2f} ms/img), "
          f"peak {s.last_peak_bytes / 1024:.1f} KiB")


if __name__ == "__main__":
    main()
