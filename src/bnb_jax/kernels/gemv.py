"""Fused NF4 dequant-matmul (warp-GEMV): ``y = x @ W.T`` with W kept quantized.

Mapping: each warp computes R consecutive output rows of W (N, K), R = 2 or 4
(see _rows_per_warp). Lane w walks the rows in 32-element chunks
c = w, w + 32, ...: per row, one 16-byte load of codes per chunk, which across
the warp is 512 contiguous bytes, and the chunk's x is loaded once for all R
rows. blocksize >= 32 and K % 32 == 0 put each chunk inside one absmax block,
so a lane reads one absmax per row per chunk.

The codes go through the 16-entry NF4 table in shared memory, filled once per
CTA. Entry i lands in bank i, so 16 distinct values across 32 lanes never
collide and equal values broadcast: one conflict-free LDS per element.

Numerics: per chunk and batch row, ``t = sum table[q] * x`` in float32, then
``acc += absmax * t``. The scale is applied once per chunk instead of once per
element, and no weight is rounded to bf16, so this is closer to the exact
product than ``x @ w.dequantize(bf16).T`` and not bitwise equal to it. With a
one-hot x the output is the dequantized weight, bitwise.

A double-quantized absmax is decoded in the kernel, from its 8-bit codes, the
nested scales and the offset, exactly as dequantization does.
"""
from __future__ import annotations

import cutejax
import cutlass
import jax
import jax.numpy as jnp
from cutlass import cute

from ._ptx import clamp_hi, mul_ftz
from .dequantize import NF4_TABLE, _log2_blocksize

_WARPS = 8         # per CTA
_CTA = 32 * _WARPS
_CHUNK = 32         # elements per lane per step: 16 bytes of codes

# Above this the accumulators spill (see _rows_per_warp) and the GEMV stops
# beating dequantize-then-matmul: on an RTX 5070 Ti at (4096, 14336) it takes
# 162us at M=20 against the fallback's 342us, and ~300us at M=24 against ~340.
# See bench/bench_gemv.py.
GEMV_MAX_M = 20


@cute.kernel
def _gemv_kernel(gW16: cute.Tensor, gX8: cute.Tensor, gA: cute.Tensor,
                 gQ2: cute.Tensor, gA2: cute.Tensor, gC2: cute.Tensor, gOff: cute.Tensor,
                 gT: cute.Tensor, gO: cute.Tensor,
                 n_rows: cutlass.Int32, cpr: cutlass.Int32,
                 M: cutlass.Constexpr[int], R: cutlass.Constexpr[int],
                 LOG2_BS: cutlass.Constexpr[int],
                 NESTED: cutlass.Constexpr[bool], LOG2_BS2: cutlass.Constexpr[int]):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()

    smem = cutlass.memory.SmemAllocator()
    sT = smem.allocate_tensor(cutlass.Float32, cute.make_layout(16), byte_alignment=16)
    if tid < 16:
        sT[tid] = gT[tid]
    cute.arch.sync_threads()

    n0 = (bid * _WARPS + (tid >> 5)) * R
    lane = tid & 31

    # rows past the end load the last row and drop their results, so that
    # every lane keeps in step for the shuffles
    rows = [clamp_hi(n0 + r, n_rows - 1) for r in range(R)]
    acc = cute.make_rmem_tensor(M * R, cutlass.Float32)
    acc.fill(0.0)
    offset = cutlass.Float32(0.0)
    if cutlass.const_expr(NESTED):
        offset = gOff[0]

    for c in cutlass.range(lane, cpr, 32):
        # decode once per row, shared by every batch row; high nibble first
        q = cute.make_rmem_tensor(R * _CHUNK, cutlass.Float32)
        a = cute.make_rmem_tensor(R, cutlass.Float32)
        for r in cutlass.range_constexpr(R):
            chunk = rows[r] * cpr + c
            words = cute.make_rmem_tensor(4, cutlass.Int32)
            cute.autovec_copy(gW16[chunk, None], words)

            blk = (chunk * _CHUNK) >> LOG2_BS
            if cutlass.const_expr(NESTED):
                q2 = cutlass.Int32(gQ2[blk]) & 0xFF  # Uint8 -> Int32 sign-extends
                a[r] = mul_ftz(gC2[q2], gA2[blk >> LOG2_BS2]) + offset
            else:
                a[r] = gA[blk]

            for wi in cutlass.range_constexpr(4):
                w = words[wi]
                for j in cutlass.range_constexpr(4):
                    q[r * _CHUNK + 8 * wi + 2 * j] = sT[(w >> (8 * j + 4)) & 0x0F]
                    q[r * _CHUNK + 8 * wi + 2 * j + 1] = sT[(w >> (8 * j)) & 0x0F]

        # each x load and convert is shared by the R rows
        for m in cutlass.range_constexpr(M):
            t = cute.make_rmem_tensor(R, cutlass.Float32)
            for r in cutlass.range_constexpr(R):
                t[r] = cutlass.Float32(0.0)
            for i in cutlass.range_constexpr(_CHUNK // 8):
                xv = cute.make_rmem_tensor(8, gX8.element_type)
                cute.autovec_copy(gX8[(m * cpr + c) * 4 + i, None], xv)
                for k in cutlass.range_constexpr(8):
                    xf = cutlass.Float32(xv[k])
                    for r in cutlass.range_constexpr(R):
                        t[r] = t[r] + q[r * _CHUNK + 8 * i + k] * xf
            for r in cutlass.range_constexpr(R):
                acc[m * R + r] = acc[m * R + r] + a[r] * t[r]

    for m in cutlass.range_constexpr(M):
        for r in cutlass.range_constexpr(R):
            v = acc[m * R + r]
            v = v + cute.arch.shuffle_sync_bfly(v, 16)
            v = v + cute.arch.shuffle_sync_bfly(v, 8)
            v = v + cute.arch.shuffle_sync_bfly(v, 4)
            v = v + cute.arch.shuffle_sync_bfly(v, 2)
            v = v + cute.arch.shuffle_sync_bfly(v, 1)
            # nested: the DSL stages each `if`, and `and` of two traced
            # booleans is not one it can stage
            if n0 + r < n_rows:  # noqa: SIM102
                if lane == 0:
                    gO[m, n0 + r] = gO.element_type(v)


@cute.jit
def _gemv_launch(stream, gU: cute.Tensor, gX: cute.Tensor, gA: cute.Tensor,
                 gQ2: cute.Tensor, gA2: cute.Tensor, gC2: cute.Tensor, gOff: cute.Tensor,
                 gT: cute.Tensor, gO: cute.Tensor,
                 rows_per_warp: cutlass.Constexpr[int],
                 log2_bs: cutlass.Constexpr[int], nested: cutlass.Constexpr[bool],
                 log2_bs2: cutlass.Constexpr[int]):
    m = gX.shape[0]
    n_rows = gO.shape[1]
    cpr = gX.shape[1] // _CHUNK
    # codes and x as flat runs of 16 bytes: the static inner extents make each
    # lane's loads provably aligned, so each is one LDG.128
    gW16 = cute.make_tensor(cute.recast_ptr(gU.iterator, dtype=cutlass.Int32),
                            cute.make_layout((n_rows * cpr, 4), stride=(4, 1)))
    gX8 = cute.make_tensor(gX.iterator, cute.make_layout((m * cpr * 4, 8), stride=(8, 1)))
    rows_per_cta = _WARPS * rows_per_warp
    n_cta = (n_rows + rows_per_cta - 1) // rows_per_cta
    _gemv_kernel(gW16, gX8, gA, gQ2, gA2, gC2, gOff, gT, gO, n_rows, cpr,
                 m, rows_per_warp, log2_bs, nested, log2_bs2).launch(
        grid=[n_cta, 1, 1], block=[_CTA, 1, 1], stream=stream)


def gemv_nf4(
    x: jax.Array,
    data: jax.Array,
    absmax: jax.Array,
    n_rows: int,
    blocksize: int,
    *,
    nested: tuple[jax.Array, jax.Array, jax.Array, int] | None = None,
    rows_per_warp: int | None = None,
) -> jax.Array:
    """``x @ W.T`` for an NF4 weight W (n_rows, K), never materialized.

    Args:
        x: float16 or bfloat16 ``(M, K)``, ``K % 32 == 0``; the result is
            ``(M, n_rows)`` of the same dtype.
        data: the packed NF4 codes of the row-major flattened W.
        absmax: float32 scale per block, or, if ``nested`` is given, the
            uint8 codes of the double-quantized scales.
        blocksize: power of two >= 32.
        nested: ``(absmax2, code2, offset, blocksize2)`` of a double-quantized
            absmax.
        rows_per_warp: output rows each warp computes, sharing its x loads
            between them (default: tuned for M).
    """
    m, k = x.shape
    if jnp.dtype(x.dtype) not in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16)):
        raise NotImplementedError(f"gemv_nf4 of {x.dtype} x is not implemented")
    if k % _CHUNK:
        raise ValueError(f"K must be a multiple of {_CHUNK}, got {k}")
    log2_bs = _log2_blocksize(blocksize)
    if blocksize < _CHUNK:
        raise ValueError(f"blocksize must be >= {_CHUNK}, got {blocksize}")
    assert data.dtype == jnp.uint8 and data.size * 2 == n_rows * k, (data.shape, n_rows, k)
    assert n_rows * k < 2**31, "too many elements for int32 indexing"

    dummy = jnp.zeros((1,), jnp.float32)
    if nested is None:
        assert absmax.dtype == jnp.float32, absmax.dtype
        args = (absmax.ravel(), jnp.zeros((1,), jnp.uint8), dummy, dummy, dummy)
        log2_bs2 = 0
    else:
        absmax2, code2, offset, blocksize2 = nested
        assert absmax.dtype == jnp.uint8, absmax.dtype
        args = (dummy, absmax.ravel(), absmax2.ravel().astype(jnp.float32),
                code2.ravel().astype(jnp.float32), jnp.reshape(offset, (1,)).astype(jnp.float32))
        log2_bs2 = _log2_blocksize(blocksize2)

    return cutejax.call(
        _gemv_launch,
        jax.ShapeDtypeStruct((m, n_rows), x.dtype),
        data.ravel(), x, *args, jnp.asarray(NF4_TABLE),
        in_specs=[None, cutejax.ArraySpec(static_dims=(0,)), None, None, None, None, None, None],
        out_specs=cutejax.ArraySpec(static_dims=(0,)),
        rows_per_warp=_rows_per_warp(m) if rows_per_warp is None else rows_per_warp,
        log2_bs=log2_bs, nested=nested is not None, log2_bs2=log2_bs2,
        vmap_method="sequential",
    )


def _rows_per_warp(m: int) -> int:
    # A warp re-reads the whole x row for every output row it computes, so
    # with one row per warp each batch row costs 4x the weight's bytes in L1
    # traffic (bf16 x against half-byte codes). Sharing x across R rows cuts
    # that R-fold, at R * 32 registers of decoded codes plus M * R
    # accumulators. Measured on an RTX 5070 Ti (4096x14336, us per call):
    # R=2 is best at M <= 2; R=4 at M=4..20 (M=8: 208us at R=1, 85us at R=4);
    # R=8 spills at M >= 4, and R=4 spills from M=24.
    return 2 if m <= 2 else 4
