"""QuantizedArray: a JAX pytree holding a bitsandbytes-quantized tensor."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from .kernels import (
    create_dynamic_map,
    dequantize_blockwise,
    dequantize_nf4,
    quantize_blockwise,
    quantize_nf4,
)

__all__ = ["QType", "QuantizedArray", "quantize"]

QType = Literal["nf4", "8bit"]
"""``"nf4"``: 4-bit NormalFloat, two codes per byte (bitsandbytes'
``quantize_4bit``). ``"8bit"``: 8-bit blockwise with a 256-entry codebook
(``quantize_blockwise``)."""

# bitsandbytes' double quantization: absmax is itself 8-bit quantized in
# blocks of this size, around its mean.
_NESTED_BLOCKSIZE = 256

_DTYPES = {"float16": jnp.float16, "bfloat16": jnp.bfloat16, "float32": jnp.float32}


@jax.tree_util.register_dataclass
@dataclass
class QuantizedArray:
    """A blockwise-quantized tensor: codes, per-block scales, and the
    metadata to decode them.

    Element ``i`` decodes as ``table[code_i] * absmax[i // blocksize] + offset``,
    with ``table`` the NF4 table or the 8-bit codebook ``code``.

    This is a pytree: ``data``, ``absmax``, ``code`` and ``offset`` are leaves;
    ``qtype``, ``blocksize``, ``shape`` and ``dtype`` are static.

    Attributes:
        data: uint8 codes for the flattened tensor: ``ceil(size / 2)`` bytes
            for ``"nf4"`` (high nibble first), ``size`` for ``"8bit"``.
        absmax: float32 scale per block, or itself a ``"8bit"``
            QuantizedArray when double-quantized.
        code: the 256-entry float32 codebook (``"8bit"`` only).
        offset: float32 scalar added after decoding. Only set on the absmax
            of a double-quantized array, which is quantized around its mean.
        qtype: ``"nf4"`` or ``"8bit"``.
        blocksize: elements per absmax block.
        shape: shape of the decoded tensor.
        dtype: default dtype :meth:`dequantize` decodes to.
    """

    data: jax.Array
    absmax: jax.Array | QuantizedArray
    code: jax.Array | None
    offset: jax.Array | None
    qtype: QType = field(metadata={"static": True})
    blocksize: int = field(metadata={"static": True})
    shape: tuple[int, ...] = field(metadata={"static": True})
    dtype: Any = field(default=jnp.dtype(jnp.bfloat16), metadata={"static": True})

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return math.prod(self.shape)

    @property
    def nbytes(self) -> int:
        """Bytes held by all the leaves: codes, scales, codebook."""
        return sum(x.size * x.dtype.itemsize for x in jax.tree.leaves(self))

    @property
    def double_quantized(self) -> bool:
        return isinstance(self.absmax, QuantizedArray)

    def astype(self, dtype) -> QuantizedArray:
        """Return a copy whose default dequantization dtype is ``dtype``."""
        return replace(self, dtype=jnp.dtype(dtype))

    def dequantize(self, dtype=None) -> jax.Array:
        """Decode to a dense array of ``dtype`` (default: ``self.dtype``),
        bitwise equal to bitsandbytes' dequantization."""
        dtype = self.dtype if dtype is None else dtype
        absmax = self.absmax
        if isinstance(absmax, QuantizedArray):
            absmax = absmax.dequantize(jnp.float32)
        if self.qtype == "nf4":
            out = dequantize_nf4(self.data, absmax, blocksize=self.blocksize,
                                 dtype=dtype, n=self.size)
        elif self.qtype == "8bit":
            assert self.code is not None, "8bit QuantizedArray without a codebook"
            out = dequantize_blockwise(self.data, absmax, self.code,
                                       blocksize=self.blocksize, dtype=dtype)
        else:
            raise NotImplementedError(f"dequantization of {self.qtype!r} is not implemented")
        if self.offset is not None:
            out = out + self.offset.astype(out.dtype)
        return out.reshape(self.shape)

    def __repr__(self) -> str:
        nested = ", double_quantized" if self.double_quantized else ""
        return (f"QuantizedArray({self.qtype}, shape={self.shape}, "
                f"dtype={jnp.dtype(self.dtype).name}, blocksize={self.blocksize}{nested})")

    @classmethod
    def from_dict(cls, weight: jax.Array, qs_dict: dict[str, Any]) -> QuantizedArray:
        """Wrap a weight from a HuggingFace bnb-4bit state dict.

        ``weight`` is the packed uint8 tensor; ``qs_dict`` its quant state
        entries with the weight's key prefix stripped (``absmax``,
        ``quant_map``, ``nested_*``, and either the unpacked metadata or the
        packed ``quant_state.bitsandbytes__nf4`` JSON tensor).
        """
        qs_dict = dict(qs_dict)  # make a copy to avoid mutating input
        packed = [k for k, v in qs_dict.items() if "quant_state" in k and isinstance(v, jax.Array)]
        if "quant_type" not in qs_dict:
            if len(packed) != 1:
                raise ValueError(f"expected exactly one packed quant_state item, found {packed}")
            qs_dict.update(_unpack_tensor_to_dict(qs_dict.pop(packed[0])))
        qs_dict = {k.split(".")[-1]: v for k, v in qs_dict.items()}  # strip prefixes

        if qs_dict["quant_type"] != "nf4":
            raise NotImplementedError(f"quant_type {qs_dict['quant_type']!r} is not supported")

        absmax = qs_dict["absmax"]
        if "nested_absmax" in qs_dict:
            assert qs_dict["nested_dtype"] == "float32", qs_dict["nested_dtype"]
            absmax = cls(
                data=absmax.ravel(),
                absmax=qs_dict["nested_absmax"].ravel(),
                code=qs_dict["nested_quant_map"].ravel(),
                offset=jnp.array(float(qs_dict["nested_offset"]), jnp.float32),
                qtype="8bit",
                blocksize=int(qs_dict["nested_blocksize"]),
                shape=tuple(absmax.shape),
                dtype=jnp.dtype(jnp.float32),
            )
        else:
            absmax = absmax.ravel()

        # quant_map, the NF4 table, is not kept: like bitsandbytes' kernel,
        # dequantization uses the built-in one
        return cls(
            data=weight.ravel(),
            absmax=absmax,
            code=None,
            offset=None,
            qtype="nf4",
            blocksize=int(qs_dict["blocksize"]),
            shape=tuple(int(s) for s in qs_dict["shape"]),
            dtype=jnp.dtype(_DTYPES[qs_dict["dtype"]]),
        )


def quantize(
    array: jax.Array,
    qtype: QType = "nf4",
    blocksize: int = 64,
    *,
    code: jax.Array | None = None,
    double_quant: bool = False,
    dtype=None,
) -> QuantizedArray:
    """Quantize an array, bitwise as bitsandbytes does.

    Args:
        array: float16, bfloat16 or float32 input, any shape.
        qtype: ``"nf4"`` (``quantize_4bit``) or ``"8bit"`` (``quantize_blockwise``).
        blocksize: power of two; bitsandbytes supports 32-4096 for nf4 and
            64-4096 for 8bit.
        code: the codebook for ``"8bit"`` (default :func:`create_dynamic_map`).
        double_quant: also quantize the absmax values, as bitsandbytes'
            ``compress_statistics=True``. The offset is their mean, whose float32
            rounding can differ from torch's, so this part is close to
            bitsandbytes but not bitwise.
        dtype: default dequantization dtype (default: ``array.dtype``).
    """
    x = jnp.ravel(array)
    if qtype == "nf4":
        data, absmax = quantize_nf4(x, blocksize=blocksize)
        code = None
    elif qtype == "8bit":
        code = create_dynamic_map() if code is None else jnp.asarray(code, jnp.float32).ravel()
        data, absmax = quantize_blockwise(x, code, blocksize=blocksize)
    else:
        raise NotImplementedError(f"quantization to {qtype!r} is not implemented")

    if double_quant:
        offset = jnp.mean(absmax)
        absmax = quantize(absmax - offset, "8bit", _NESTED_BLOCKSIZE)
        absmax = replace(absmax, offset=offset)

    return QuantizedArray(
        data=data,
        absmax=absmax,
        code=code,
        offset=None,
        qtype=qtype,
        blocksize=blocksize,
        shape=tuple(array.shape),
        dtype=jnp.dtype(array.dtype if dtype is None else dtype),
    )


def _unpack_tensor_to_dict(tensor: jax.Array) -> dict[str, Any]:
    return json.loads(bytes(np.array(tensor)).decode("utf-8"))
