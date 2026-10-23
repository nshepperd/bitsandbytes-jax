"""Instruction helpers that reproduce bitsandbytes' fast-math arithmetic.

bitsandbytes is compiled with ``--use_fast_math``, so its kernels use
flush-to-zero multiplies and maxes and an approximate reciprocal
(``1.0f / absmax`` is a bare ``MUFU.RCP``). The CuTe DSL's float operators are
IEEE, so to match bitwise these few operations are spelled out in PTX.
"""
from __future__ import annotations

import cutlass
import numpy as np
from cutlass.cute.arch import inline_ptx


def mul_ftz(a, b):
    return inline_ptx("mul.ftz.f32 {$w0}, {$r0}, {$r1};",
                      write_only_types=[cutlass.Float32], read_only_args=[a, b])


def rcp_approx_ftz(a):
    return inline_ptx("rcp.approx.ftz.f32 {$w0}, {$r0};",
                      write_only_types=[cutlass.Float32], read_only_args=[a])


def max_abs_ftz(m, x):
    """fmaxf(m, fabsf(x)), flushing denormals as bitsandbytes does."""
    return inline_ptx("{ .reg .f32 t; abs.ftz.f32 t, {$r1}; max.ftz.f32 {$w0}, {$r0}, t; }",
                      write_only_types=[cutlass.Float32], read_only_args=[m, x])


def max_ftz(a, b):
    return inline_ptx("max.ftz.f32 {$w0}, {$r0}, {$r1};",
                      write_only_types=[cutlass.Float32], read_only_args=[a, b])


def clamp_hi(i, hi):
    """min(i, hi) for Int32 (no integer min in the DSL's operator set)."""
    d = i - hi
    return i - (d & ~(d >> 31))


def f32_bits(bits: int) -> float:
    """The float32 with these bits, as a Python float (exactly)."""
    return float(np.uint32(bits).view(np.float32))
