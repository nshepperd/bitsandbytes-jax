"""JAX primitive wrappers for bitsandbytes dequantization kernels."""
import jax
import jax.numpy as jnp

from .backend import ensure_initialized

def dequantize_nf4(
    quantized: jax.Array,
    absmax: jax.Array,
    blocksize: int = 64,
    dtype: jnp.dtype = jnp.float16,
) -> jax.Array:
    ensure_initialized()
    # Each byte contains 2 4-bit values
    n = quantized.size * 2
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
    n = quantized.size
    if dtype not in (jnp.float16, jnp.bfloat16, jnp.float32):
        raise NotImplementedError(f"dequantize_blockwise not implemented for dtype {dtype}")
    return jax.ffi.ffi_call(
        "bnb_dequant_blockwise",
        jax.ShapeDtypeStruct((n,), dtype=dtype),
        vmap_method="broadcast_all"
    )(quantized, absmax, code, blocksize=blocksize)
