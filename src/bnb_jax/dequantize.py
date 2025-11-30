"""JAX primitive wrappers for bitsandbytes dequantization kernels."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from dataclasses import dataclass, field, replace

from .backend import ensure_initialized

@jax.tree_util.register_dataclass
@dataclass
class QuantizedArray:
    array: jax.Array
    absmax: jax.Array | QuantizedArray
    code: jax.Array | None
    quantize_type: str | None = field(metadata=dict(static=True))
    blocksize: int = field(metadata=dict(static=True))
    shape: tuple[int, ...] = field(metadata=dict(static=True))
    dtype: jnp.dtype = field(metadata=dict(static=True))

    def astype(self, dtype: jnp.dtype) -> QuantizedArray:
        return replace(self, dtype=dtype)
    def dequantize(self, dtype: jnp.dtype | None = None) -> jax.Array:
        return dequantize(self, dtype=dtype)
    def __jax_array__(self) -> jax.Array:
        return dequantize(self)

def dequantize(array: QuantizedArray, dtype: jnp.dtype | None = None) -> jax.Array:
    """Dequantize a QuantizedArray to the specified dtype."""
    if isinstance(array.absmax, QuantizedArray):
        absmax = dequantize(array.absmax, dtype=jnp.float32)
    else:
        absmax = array.absmax
    if array.quantize_type == "nf4":
        return dequantize_nf4(
            array.array,
            absmax,
            blocksize=array.blocksize,
            dtype=dtype or array.dtype,
        ).reshape(array.shape)
    elif array.quantize_type is None:
        assert array.code is not None, "Codebook must be provided for 8-bit dequantization."
        return dequantize_blockwise(
            array.array,
            absmax,
            array.code,
            blocksize=array.blocksize,
            dtype=dtype or array.dtype,
        ).reshape(array.shape)
    else:
        raise NotImplementedError(f"Dequantization not implemented for type {array.quantize_type}")


def dequantize_nf4(
    quantized: jax.Array,
    absmax: jax.Array,
    blocksize: int = 64,
    dtype: jnp.dtype = jnp.float16,
) -> jax.Array:
    ensure_initialized()
    # Each byte contains 2 4-bit values
    n = quantized.shape[0] * 2
    if dtype not in (jnp.float16, jnp.bfloat16, jnp.float32):
        raise NotImplementedError(f"dequantize_nf4 not implemented for dtype {dtype}")
    return jax.ffi.ffi_call(
        "bnb_dequant_nf4",
        jax.ShapeDtypeStruct((n,), dtype=dtype),
        vmap_method="broadcast_all"
    )(quantized, absmax, blocksize=blocksize)

def dequantize_blockwise(
    quantized: jax.Array,
    absmax: jax.Array,
    code: jax.Array,
    blocksize: int = 64,
    dtype: jnp.dtype = jnp.float16,
) -> jax.Array:
    """Dequantize an 8-bit blockwise quantized array."""
    ensure_initialized()
    # Each byte contains 1 8-bit value
    n = quantized.shape[0]

    if dtype not in (jnp.float16, jnp.bfloat16, jnp.float32):
        raise NotImplementedError(f"dequantize_blockwise not implemented for dtype {dtype}")

    return jax.ffi.ffi_call(
        "bnb_dequant_blockwise",
        jax.ShapeDtypeStruct((n,), dtype=dtype),
        vmap_method="broadcast_all"
    )(quantized, absmax, code, blocksize=blocksize)