"""bitsandbytes NF4 / 8-bit blockwise dequantization as a CuTe DSL kernel.

Element i of a blockwise-quantized array decodes as

    out[i] = T(table[q[i]] * absmax[i // blocksize])

where ``table`` is the 16-entry NF4 lookup table (4-bit codes packed two per
byte, high nibble first) or the caller's 256-entry codebook (8-bit), and the
multiply is float32 with denormals flushed, exactly as in bitsandbytes'
``kDequantizeBlockwise``. The result is bitwise equal to bitsandbytes'.

Grid mapping: one thread per 8 elements, which is one 32-bit word of NF4
codes or two of 8-bit codes, and one 16-byte (bf16/f16) or 32-byte (f32)
store. The table is staged in shared memory once per CTA.
"""
from __future__ import annotations

import cutejax
import cutlass
import jax
import jax.numpy as jnp
import numpy as np
from cutlass import cute

from ._ptx import mul_ftz

NF4_TABLE = np.array([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
], dtype=np.float32)

_CTA = 256
_UNIT = 8          # elements per thread
_DTYPES = (jnp.float16, jnp.bfloat16, jnp.float32)


@cute.kernel
def _dequant_kernel(gW: cute.Tensor, gA: cute.Tensor, gT: cute.Tensor, gO: cute.Tensor,
                    n_units: cutlass.Int32,
                    NF4: cutlass.Constexpr[bool], LOG2_BS: cutlass.Constexpr[int]):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    table_size = 16 if NF4 else 256

    smem = cutlass.memory.SmemAllocator()
    sT = smem.allocate_tensor(cutlass.Float32, cute.make_layout(table_size), byte_alignment=16)
    if tid < table_size:
        sT[tid] = gT[tid]
    cute.arch.sync_threads()

    u = bid * _CTA + tid
    if u < n_units:
        # blocksize >= 8, so the unit's 8 elements share one absmax
        a = gA[u >> (LOG2_BS - 3)]
        out = cute.make_rmem_tensor(_UNIT, gO.element_type)
        if cutlass.const_expr(NF4):
            w = gW[u, 0]
            for j in cutlass.range_constexpr(4):
                out[2 * j] = gO.element_type(mul_ftz(sT[(w >> (8 * j + 4)) & 0x0F], a))
                out[2 * j + 1] = gO.element_type(mul_ftz(sT[(w >> (8 * j)) & 0x0F], a))
        else:
            words = cute.make_rmem_tensor(2, cutlass.Int32)
            cute.autovec_copy(gW[u, None], words)
            for j in cutlass.range_constexpr(_UNIT):
                q = (words[j >> 2] >> (8 * (j & 3))) & 0xFF
                out[j] = gO.element_type(mul_ftz(sT[q], a))
        cute.autovec_copy(out, gO[u, None])


@cute.jit
def _dequant_launch(stream, gQ: cute.Tensor, gA: cute.Tensor, gT: cute.Tensor, gO: cute.Tensor,
                    nf4: cutlass.Constexpr[bool], log2_bs: cutlass.Constexpr[int]):
    n_units = gO.shape[0] // _UNIT
    words = 1 if nf4 else 2
    # the codes as 32-bit words and the output as 8-element rows: the static
    # inner extents make every load and store one vector access
    gW = cute.make_tensor(cute.recast_ptr(gQ.iterator, dtype=cutlass.Int32),
                          cute.make_layout((n_units, words), stride=(words, 1)))
    gO8 = cute.make_tensor(gO.iterator, cute.make_layout((n_units, _UNIT), stride=(_UNIT, 1)))
    n_cta = (n_units + _CTA - 1) // _CTA
    _dequant_kernel(gW, gA, gT, gO8, n_units, nf4, log2_bs).launch(
        grid=[n_cta, 1, 1], block=[_CTA, 1, 1], stream=stream)


def _log2_blocksize(blocksize: int) -> int:
    if blocksize < 8 or blocksize & (blocksize - 1):
        raise ValueError(f"blocksize must be a power of two >= 8, got {blocksize}")
    return blocksize.bit_length() - 1


def _dequantize(quantized, absmax, table, n, blocksize, dtype, nf4):
    if jnp.dtype(dtype) not in map(jnp.dtype, _DTYPES):
        raise NotImplementedError(f"dequantization to {dtype} is not implemented")
    log2_bs = _log2_blocksize(blocksize)
    assert quantized.dtype == jnp.uint8, quantized.dtype
    assert absmax.dtype == jnp.float32, absmax.dtype
    if absmax.size * blocksize < n:
        raise ValueError(f"{absmax.size} absmax values do not cover {n} elements "
                         f"with blocksize {blocksize}")
    assert n < 2**31, "too many elements for int32 indexing"
    # whole 8-element units only; the padding decodes against the last
    # (partial) block's absmax and is sliced off
    n_pad = -(-n // _UNIT) * _UNIT
    q = quantized.ravel()
    q_bytes = n_pad // 2 if nf4 else n_pad
    if q.size != q_bytes:
        q = jnp.pad(q[:q_bytes], (0, q_bytes - min(q.size, q_bytes)))
    out = cutejax.call(
        _dequant_launch,
        jax.ShapeDtypeStruct((n_pad,), dtype),
        q, absmax.ravel(), table,
        nf4=nf4, log2_bs=log2_bs,
        vmap_method="sequential",
    )
    return out[:n] if n_pad != n else out


def dequantize_nf4(
    quantized: jax.Array,
    absmax: jax.Array,
    blocksize: int = 64,
    dtype: jnp.dtype = jnp.float16,
    n: int | None = None,
) -> jax.Array:
    """Dequantize packed NF4 codes (two per byte) to a flat ``dtype`` array
    of ``n`` elements (default ``2 * quantized.size``; one fewer when the
    last byte holds a single code)."""
    if n is None:
        n = quantized.size * 2
    assert -(-n // 2) == quantized.size, f"{quantized.size} bytes do not hold {n} codes"
    return _dequantize(quantized, absmax, jnp.asarray(NF4_TABLE),
                       n, blocksize, dtype, nf4=True)


def dequantize_blockwise(
    quantized: jax.Array,
    absmax: jax.Array,
    code: jax.Array,
    blocksize: int = 64,
    dtype: jnp.dtype = jnp.float16,
) -> jax.Array:
    """Dequantize an 8-bit blockwise quantized array with codebook ``code``."""
    code = jnp.asarray(code, jnp.float32).ravel()
    assert code.size == 256, f"codebook must have 256 entries, got {code.size}"
    return _dequantize(quantized, absmax, code,
                       quantized.size, blocksize, dtype, nf4=False)
