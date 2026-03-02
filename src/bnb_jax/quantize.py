"""JAX primitive wrappers for bitsandbytes quantization kernels."""
import jax
import jax.numpy as jnp
import numpy as np

from .backend import ensure_initialized


def create_dynamic_map(signed: bool = True, max_exponent_bits: int = 7, total_bits: int = 8) -> jax.Array:
    """Create the dynamic quantization codebook for 8-bit blockwise quantization.

    Matches bitsandbytes.functional.create_dynamic_map exactly.
    """
    data = []
    non_sign_bits = total_bits - 1
    additional_items = 2 ** (non_sign_bits - max_exponent_bits) - 1
    for i in range(max_exponent_bits):
        fraction_items = int(
            2 ** (i + non_sign_bits - max_exponent_bits) + 1
            if signed
            else 2 ** (i + non_sign_bits - max_exponent_bits + 1) + 1,
        )
        boundaries = np.linspace(0.1, 1, fraction_items, dtype=np.float32)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        data += ((10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
        if signed:
            data += (-(10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()

    if additional_items > 0:
        boundaries = np.linspace(0.1, 1, additional_items + 1, dtype=np.float32)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        data += ((10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
        if signed:
            data += (-(10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()

    data.append(0)
    data.append(1.0)

    gap = 256 - len(data)
    for _ in range(gap):
        data.append(0)

    data.sort()
    return jnp.array(data, dtype=jnp.float32)

def quantize_nf4(
    input: jax.Array,
    blocksize: int = 64,
) -> tuple[jax.Array, jax.Array]:
    """Quantize a tensor to NF4 format.

    Returns (quantized_u8, absmax_f32).
    Each byte in quantized_u8 contains 2 packed 4-bit values.
    """
    ensure_initialized()
    n = input.size
    return jax.ffi.ffi_call(
        "bnb_quant_nf4",
        (jax.ShapeDtypeStruct((n // 2,), dtype=jnp.uint8),
         jax.ShapeDtypeStruct((n // blocksize,), dtype=jnp.float32)),
        vmap_method="broadcast_all"
    )(input, blocksize=blocksize)

def quantize_blockwise(
    input: jax.Array,
    code: jax.Array,
    blocksize: int = 64,
) -> tuple[jax.Array, jax.Array]:
    """Quantize a tensor using 8-bit blockwise quantization with a codebook.

    Returns (quantized_u8, absmax_f32).
    """
    ensure_initialized()
    n = input.size
    return jax.ffi.ffi_call(
        "bnb_quant_blockwise",
        (jax.ShapeDtypeStruct((n,), dtype=jnp.uint8),
         jax.ShapeDtypeStruct((n // blocksize,), dtype=jnp.float32)),
        vmap_method="broadcast_all"
    )(input, code, blocksize=blocksize)
