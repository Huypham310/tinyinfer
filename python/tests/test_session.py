"""Bindings tests. Uses the tiny ONNX fixtures under tests/data (committed)."""

from pathlib import Path

import numpy as np
import pytest

from tinyinfer import Session, cpu_features, gemm_isa, set_gemm_isa

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"


def test_cpu_features_and_isa_override():
    feats = cpu_features()
    assert "summary" in feats and isinstance(feats["avx2"], bool)
    original = gemm_isa()
    assert set_gemm_isa("scalar")
    assert gemm_isa() == "scalar"
    assert set_gemm_isa(original)


def test_conv_relu_matches_numpy_reference():
    # same weights as tools/make_test_models.py::conv_relu
    s = Session(str(DATA / "conv_relu.onnx"), batch=2)
    info = s.info()
    assert info["nodes"] >= 1 and info["relu_fused"] >= 1
    assert info["batch"] == 2 and info["int8"] is False

    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 3, 8, 8), dtype=np.float32)
    y = s.run(x)
    assert y.shape == (2, 4, 8, 8)
    assert y.dtype == np.float32
    # a few ReLU zeros somewhere, and nothing NaN
    assert np.isfinite(y).all()
    assert (y >= 0).all()


def test_rejects_wrong_batch_and_dtype():
    s = Session(str(DATA / "conv_relu.onnx"), batch=1)
    with pytest.raises(ValueError, match="batch"):
        s.run(np.zeros((2, 3, 8, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="float32"):
        s.run(np.zeros((1, 3, 8, 8), dtype=np.float64))


def test_int8_calibration_changes_output_a_little():
    path = DATA / "conv_bn_head.onnx"
    if not path.exists():
        pytest.skip("conv_bn_head.onnx fixture missing")
    s = Session(str(path), batch=2)
    rng = np.random.default_rng(1)
    batches = [rng.uniform(-1, 2, size=(2, 3, 16, 16)).astype(np.float32) for _ in range(3)]
    fp = s.run(batches[0])
    s.calibrate(batches)
    assert s.info()["int8"] is True
    q8 = s.run(batches[0])
    assert q8.shape == fp.shape
    # quantized should be close, not bit-identical
    rel = np.max(np.abs(q8 - fp)) / (np.max(np.abs(fp)) + 1e-6)
    assert 0 < rel < 0.05
    s.clear_int8()
    assert s.info()["int8"] is False
    assert np.allclose(s.run(batches[0]), fp)


def test_peak_memory_is_reported():
    s = Session(str(DATA / "conv_relu.onnx"), batch=1)
    s.run(np.zeros((1, 3, 8, 8), dtype=np.float32))
    assert s.last_peak_bytes > 0
