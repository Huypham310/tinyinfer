"""Generates the small ONNX files under tests/data used by the parser tests.

Needs the `onnx` package (pip install onnx). The outputs are committed, so this only has to be
rerun when the fixtures change:

    python tools/make_test_models.py
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

OUT = Path(__file__).resolve().parents[1] / "tests" / "data"


def conv_relu() -> onnx.ModelProto:
    """Conv(3->4, 3x3, pad 1) + Relu with a symbolic batch dim.

    W is stored as raw_data, B as float_data (the two ways exporters write floats).
    """
    w = (np.arange(4 * 3 * 3 * 3, dtype=np.float32) * 0.01).reshape(4, 3, 3, 3)
    weight = helper.make_tensor("W", TensorProto.FLOAT, w.shape, w.tobytes(), raw=True)
    bias = helper.make_tensor("B", TensorProto.FLOAT, [4], [0.5, -0.5, 1.0, -1.0])
    conv = helper.make_node(
        "Conv",
        ["x", "W", "B"],
        ["c"],
        name="conv0",
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[1, 1],
        group=1,
        auto_pad="NOTSET",
    )
    relu = helper.make_node("Relu", ["c"], ["y"], name="relu0")
    graph = helper.make_graph(
        [conv, relu],
        "conv_relu",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 3, 8, 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 4, 8, 8])],
        initializer=[weight, bias],
    )
    model = helper.make_model(
        graph, producer_name="make_test_models", opset_imports=[helper.make_opsetid("", 17)]
    )
    model.ir_version = 8
    return model


def mixed_types() -> onnx.ModelProto:
    """Integer initializers, float attrs, an omitted optional input, and weights that are also
    listed as graph inputs (what IR < 4 exporters did)."""
    shape = helper.make_tensor("shape", TensorProto.INT64, [2], [3, 4])  # int64_data
    ids = helper.make_tensor("ids", TensorProto.INT32, [3], [7, -8, 9])  # int32_data
    q = np.array([-128, 0, 127, 5], dtype=np.int8)
    qw = helper.make_tensor("qw", TensorProto.INT8, [2, 2], q.tobytes(), raw=True)
    gw = np.ones((4, 2), dtype=np.float32)
    gemm_w = helper.make_tensor("gw", TensorProto.FLOAT, [4, 2], gw.tobytes(), raw=True)
    scalar = helper.make_tensor("cmax", TensorProto.FLOAT, [], [6.0])  # rank 0

    nodes = [
        helper.make_node("Reshape", ["x", "shape"], ["r"], name="reshape"),
        helper.make_node("Gemm", ["r", "gw"], ["g"], name="gemm", alpha=0.5, transB=0),
        # Clip(x, min, max) with min omitted -> "" in the input list
        helper.make_node("Clip", ["g", "", "cmax"], ["y"], name="clip"),
    ]
    inits = [shape, ids, qw, gemm_w, scalar]
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 6])]
    inputs += [helper.make_tensor_value_info(t.name, t.data_type, list(t.dims)) for t in inits]
    graph = helper.make_graph(
        nodes,
        "mixed",
        inputs,
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 2])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    return model


def conv_bn_head() -> onnx.ModelProto:
    """Conv -> BatchNormalization -> Relu -> GlobalAveragePool -> Flatten -> Gemm -> Softmax.

    Everything the executor's BatchNorm folding and Relu fusion passes touch, plus a classifier
    head, with non-trivial BN statistics so a wrong fold shows up in the outputs.
    """
    rng = np.random.default_rng(0)
    w = rng.normal(0, 0.2, (8, 3, 3, 3)).astype(np.float32)
    inits = [
        helper.make_tensor("w", TensorProto.FLOAT, w.shape, w.tobytes(), raw=True),
        helper.make_tensor("b", TensorProto.FLOAT, [8], rng.normal(0, 0.1, 8).tolist()),
        helper.make_tensor("scale", TensorProto.FLOAT, [8], rng.uniform(0.5, 1.5, 8).tolist()),
        helper.make_tensor("shift", TensorProto.FLOAT, [8], rng.normal(0, 0.3, 8).tolist()),
        helper.make_tensor("mean", TensorProto.FLOAT, [8], rng.normal(0, 0.5, 8).tolist()),
        helper.make_tensor("var", TensorProto.FLOAT, [8], rng.uniform(0.2, 2.0, 8).tolist()),
    ]
    fc = rng.normal(0, 0.3, (5, 8)).astype(np.float32)
    inits.append(helper.make_tensor("fc", TensorProto.FLOAT, fc.shape, fc.tobytes(), raw=True))
    inits.append(helper.make_tensor("fcb", TensorProto.FLOAT, [5], [0.1, -0.1, 0.0, 0.2, -0.2]))
    nodes = [
        helper.make_node("Conv", ["x", "w", "b"], ["c"], strides=[2, 2], pads=[1, 1, 1, 1]),
        helper.make_node("BatchNormalization", ["c", "scale", "shift", "mean", "var"], ["bn"],
                         epsilon=1e-3),  # fmt: skip
        helper.make_node("Relu", ["bn"], ["r"]),
        helper.make_node("GlobalAveragePool", ["r"], ["g"]),
        helper.make_node("Flatten", ["g"], ["f"]),
        helper.make_node("Gemm", ["f", "fc", "fcb"], ["logits"], transB=1),
        helper.make_node("Softmax", ["logits"], ["y"], axis=-1),
    ]
    graph = helper.make_graph(
        nodes,
        "conv_bn_head",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 3, 16, 16])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 5])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def write_reference(path: Path, input_shape: tuple[int, ...], seed: int = 0) -> None:
    """Runs the model with ONNX Runtime on a seeded random input; saves input and output as
    raw little-endian float32 next to the model, for the C++ parity tests."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(seed).standard_normal(input_shape).astype(np.float32)
    (y,) = sess.run(None, {sess.get_inputs()[0].name: x})
    stem = path.with_suffix("")
    x.tofile(f"{stem}.input.bin")
    y.astype(np.float32).tofile(f"{stem}.output.bin")
    print(f"  reference {input_shape} -> {y.shape}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fixtures = [
        ("conv_relu", conv_relu, (2, 3, 8, 8)),
        ("mixed_types", mixed_types, (2, 6)),
        ("conv_bn_head", conv_bn_head, (3, 3, 16, 16)),
    ]
    for name, build, shape in fixtures:
        model = build()
        onnx.checker.check_model(model)
        path = OUT / f"{name}.onnx"
        onnx.save(model, path)
        print(f"wrote {path} ({path.stat().st_size} bytes)")
        write_reference(path, shape)


if __name__ == "__main__":
    main()
