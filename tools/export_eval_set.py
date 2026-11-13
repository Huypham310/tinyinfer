"""Export preprocessed EuroSAT images for `tinyinfer eval`, using TerraSense's own data pipeline.

    python tools/export_eval_set.py --terrasense ../terrasense --out build/eval

Writes, as raw little-endian arrays:
  test.images.bin  float32 [N, 3, 64, 64]  the held-out test split, normalized like training
  test.labels.bin  int64   [N]
  calib.images.bin float32 [M, 3, 64, 64]  validation images for INT8 calibration
Run it with TerraSense's environment (it imports `terrasense`).
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--terrasense", default="../terrasense")
    p.add_argument("--out", default="build/eval")
    p.add_argument("--calib", type=int, default=256, help="calibration images (from val)")
    args = p.parse_args()

    sys.path.insert(0, str(Path(args.terrasense, "ml", "src").resolve()))
    from terrasense.config import TrainConfig
    from terrasense.data.loaders import build_dataloaders

    cfg = TrainConfig(model="resnet18", num_workers=0, data_dir=Path(args.terrasense, "data"))
    data = build_dataloaders(cfg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    xs, ys = [], []
    for x, y in data.test:
        xs.append(x.numpy())
        ys.append(y.numpy())
    np.concatenate(xs).astype(np.float32).tofile(out / "test.images.bin")
    np.concatenate(ys).astype(np.int64).tofile(out / "test.labels.bin")

    calib, n = [], 0
    for x, _ in data.val:  # never the test split: calibration must not see it
        calib.append(x.numpy())
        n += len(x)
        if n >= args.calib:
            break
    np.concatenate(calib)[: args.calib].astype(np.float32).tofile(out / "calib.images.bin")
    print(f"test: {sum(len(y) for y in ys)} images, calib: {min(n, args.calib)} images -> {out}")


if __name__ == "__main__":
    main()
