"""bitsandbytes quantization for JAX, as CuTe DSL kernels.

Quantized tensors are :class:`QuantizedArray` pytrees, made by :func:`quantize`
or loaded from a HuggingFace bnb-4bit checkpoint with
:meth:`QuantizedArray.from_dict`, and decoded with
:meth:`QuantizedArray.dequantize` — bitwise equal to bitsandbytes either way.
The kernels themselves, on flat arrays, are in :mod:`bnb_jax.kernels`.
"""

from ._matmul import matmul
from .array import QType, QuantizedArray, quantize
from .kernels import create_dynamic_map

__all__ = [
    "QType",
    "QuantizedArray",
    "create_dynamic_map",
    "matmul",
    "quantize",
]
