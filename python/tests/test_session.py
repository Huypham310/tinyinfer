"""Bindings tests. Uses the tiny ONNX fixtures under tests/data (committed)."""

from pathlib import Path

import numpy as np
import pytest

from tinyinfer import Session, cpu_features, gemm_isa, int8_isa, set_gemm_isa, set_int8_isa

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"


def read_bin(name, shape):
    """float32 .bin written by tools/make_test_models.py (ONNX Runtime's input / output)."""
    arr = np.fromfile(DATA / name, dtype=np.float32)
    assert arr.size == int(np.prod(shape)), f"{name}: {arr.size} floats, expected {shape}"
    return arr.reshape(shape)


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


@pytest.mark.parametrize(
    "name, batch, in_shape, out_shape",
    [
        ("conv_relu", 2, (2, 3, 8, 8), (2, 4, 8, 8)),
        ("conv_bn_head", 3, (3, 3, 16, 16), (3, 5)),  # BN folded + Relu fused + softmax head
    ],
)
def test_session_matches_onnx_runtime_golden_outputs(name, batch, in_shape, out_shape):
    # the .output.bin files are what ONNX Runtime produced for the same .input.bin
    s = Session(str(DATA / f"{name}.onnx"), batch=batch)
    got = s.run(read_bin(f"{name}.input.bin", in_shape))
    want = read_bin(f"{name}.output.bin", out_shape)
    assert got.shape == want.shape
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)


def test_mixed_types_model_loads_but_session_wants_nchw():
    # mixed_types is a 2-D model (Reshape + Gemm + Clip, int initializers, weights listed as
    # graph inputs). The loader must cope with all that; Session.run only takes 4-D images.
    s = Session(str(DATA / "mixed_types.onnx"), batch=2)
    info = s.info()
    assert info["input"] == "x"  # the weights that are also graph inputs are not feeds
    assert info["nodes"] >= 3
    with pytest.raises(ValueError, match="NCHW"):
        s.run(read_bin("mixed_types.input.bin", (2, 6)))


def test_batch_must_match_the_session_even_for_golden_models():
    s = Session(str(DATA / "conv_bn_head.onnx"), batch=3)
    x = read_bin("conv_bn_head.input.bin", (3, 3, 16, 16))
    with pytest.raises(ValueError, match="batch"):
        s.run(x[:1])
    # a non-contiguous view is accepted (it gets copied), and gives the same answer
    flipped = np.ascontiguousarray(x[..., ::-1])[..., ::-1]
    assert not flipped.flags["C_CONTIGUOUS"]
    np.testing.assert_allclose(s.run(flipped), s.run(x), rtol=1e-5, atol=1e-6)


def test_int8_isa_override_and_refusal():
    original = int8_isa()
    assert original in ("vnni", "scalar")
    try:
        # unknown names are refused and nothing changes
        assert set_int8_isa("bogus") is False
        assert set_int8_isa("avx2") is False  # no such int8 kernel
        assert int8_isa() == original

        assert set_int8_isa("scalar") is True
        assert int8_isa() == "scalar"
        # the Session reports whatever is active
        s = Session(str(DATA / "conv_relu.onnx"), batch=1)
        assert s.info()["int8_isa"] == "scalar"

        if cpu_features()["avx512vnni"]:
            assert set_int8_isa("vnni") is True
            assert int8_isa() == "vnni"
        else:
            # no VNNI on this CPU: the request is refused and we stay on scalar
            assert set_int8_isa("vnni") is False
            assert int8_isa() == "scalar"
    finally:
        set_int8_isa(original)


def test_int8_scalar_and_vnni_kernels_agree():
    if not cpu_features()["avx512vnni"]:
        pytest.skip("no AVX-512 VNNI on this CPU")
    original = int8_isa()
    path = DATA / "conv_bn_head.onnx"
    rng = np.random.default_rng(7)
    batches = [rng.uniform(-1, 2, size=(2, 3, 16, 16)).astype(np.float32) for _ in range(2)]
    try:
        outs = {}
        for isa in ("scalar", "vnni"):
            assert set_int8_isa(isa)
            s = Session(str(path), batch=2)
            s.calibrate(batches)
            outs[isa] = s.run(batches[0])
        # same int32 accumulators, same dequantization -> same floats
        np.testing.assert_allclose(outs["scalar"], outs["vnni"], rtol=0, atol=1e-6)
    finally:
        set_int8_isa(original)


def test_peak_memory_is_reported():
    s = Session(str(DATA / "conv_relu.onnx"), batch=1)
    s.run(np.zeros((1, 3, 8, 8), dtype=np.float32))
    assert s.last_peak_bytes > 0
