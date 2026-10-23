"""``x @ w.T`` for a quantized ``w``, picking the route.

A fused GEMV keeps the weight quantized in HBM and never materializes it, but
only pays up to a batch size and only covers NF4; everything else
dequantizes the weight and does a dense matmul. Callers should not need to
know which applies.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .array import QuantizedArray
from .kernels.gemv import _CHUNK, GEMV_MAX_M, gemv_nf4

__all__ = ["matmul"]


def matmul(x: jax.Array, w: QuantizedArray) -> jax.Array:
    """``x @ w.T`` for a 2-D quantized weight ``w`` (N, K).

    ``x`` is ``(..., K)``; the result is ``(..., N)`` in ``x.dtype``. Uses the
    fused NF4 GEMV when it applies -- an NF4 weight, bfloat16 or float16 ``x``,
    at most :data:`~bnb_jax.kernels.gemv.GEMV_MAX_M` rows of ``x`` once its
    batch dims are flattened, ``K % 32 == 0`` -- and ``x @ w.dequantize(x.dtype).T``
    otherwise.

    The GEMV accumulates in float32 and applies each block's absmax once,
    without rounding the weights to ``x.dtype``, so it is closer to the exact
    product than the fallback, though not bitwise equal to it.
    """
    if w.ndim != 2:
        raise ValueError(f"w must be a 2-D weight, got shape {w.shape}")
    n, k = w.shape
    if x.shape[-1] != k:
        raise ValueError(f"contraction mismatch: x {x.shape} @ w.T {w.shape[::-1]}")
    xm = x.reshape(-1, k)
    if _fused_applies(xm, w):
        absmax = w.absmax
        if isinstance(absmax, QuantizedArray):
            assert absmax.code is not None and absmax.offset is not None
            assert not isinstance(absmax.absmax, QuantizedArray)
            out = gemv_nf4(xm, w.data, absmax.data, n, w.blocksize,
                           nested=(absmax.absmax, absmax.code, absmax.offset, absmax.blocksize))
        else:
            out = gemv_nf4(xm, w.data, absmax, n, w.blocksize)
    else:
        out = xm @ w.dequantize(x.dtype).T
    return out.reshape(*x.shape[:-1], n)


def _fused_applies(xm: jax.Array, w: QuantizedArray) -> bool:
    return (w.qtype == "nf4"
            and jnp.dtype(xm.dtype) in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16))
            and 0 < xm.shape[0] <= GEMV_MAX_M
            and xm.shape[1] % _CHUNK == 0
            and w.blocksize >= _CHUNK)
