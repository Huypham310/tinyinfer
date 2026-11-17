"""Python bindings for the TinyInfer CPU ONNX engine.

    >>> from tinyinfer import Session, cpu_features
    >>> s = Session("model.onnx", batch=1)
    >>> out = s.run(images)          # float32 NCHW ndarray -> float32 logits
    >>> s.calibrate([images])        # switch Conv nodes to INT8
    >>> out8 = s.run(images)
"""

from tinyinfer._tinyinfer import (
    Session,
    __version__,
    cpu_features,
    gemm_isa,
    int8_isa,
    set_gemm_isa,
    set_int8_isa,
)

__all__ = [
    "Session",
    "__version__",
    "cpu_features",
    "gemm_isa",
    "int8_isa",
    "set_gemm_isa",
    "set_int8_isa",
]
