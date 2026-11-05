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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, build in [("conv_relu", conv_relu), ("mixed_types", mixed_types)]:
        model = build()
        onnx.checker.check_model(model)
        path = OUT / f"{name}.onnx"
        onnx.save(model, path)
        print(f"wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
