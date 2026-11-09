"""Reference input/output for a model, computed with ONNX Runtime (for parity checks).

    python tools/ort_reference.py model.onnx out/resnet18 --shape 1,3,64,64

Writes out/resnet18.input.bin and out/resnet18.output.bin (raw float32). Big models like the
TerraSense ResNet-18 stay out of the repo, so their references are generated on demand.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model")
    p.add_argument("out_stem")
    p.add_argument("--shape", required=True, help="comma-separated input shape")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    shape = tuple(int(s) for s in args.shape.split(","))
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    x = np.random.default_rng(args.seed).standard_normal(shape).astype(np.float32)
    (y,) = sess.run(None, {sess.get_inputs()[0].name: x})
    Path(args.out_stem).parent.mkdir(parents=True, exist_ok=True)
    x.tofile(f"{args.out_stem}.input.bin")
    y.astype(np.float32).tofile(f"{args.out_stem}.output.bin")
    print(f"{shape} -> {y.shape}, written to {args.out_stem}.*.bin")


if __name__ == "__main__":
    main()
