"""bitsandbytes NF4 / 8-bit blockwise quantization as a CuTe DSL kernel.

Per block of ``blocksize`` elements, bitsandbytes' ``kQuantizeBlockwise``:

    absmax = max |x|                       (denormals flushed)
    r      = rcp.approx(absmax)            (fast-math ``1.0f / absmax``)
    q[i]   = quantize(x[i] * r)            (flush-to-zero multiply)

where ``quantize`` is the NF4 decision tree (two codes per byte, high nibble
first) or a binary search of the 256-entry codebook with round-to-nearest
between neighbours. Reproducing the fast-math arithmetic makes the codes and
absmax bitwise equal to bitsandbytes'.

Grid mapping: a group of ``L = min(32, blocksize // 8)`` lanes owns one block,
each lane taking 8 consecutive elements per pass (one vector load, one 32-bit
store of NF4 codes or one 64-bit store of 8-bit codes). The block's absmax is
a butterfly max within the group.
"""
from __future__ import annotations

import cutejax
import cutlass
import jax
import jax.numpy as jnp
import numpy as np
from cutlass import cute

from ._ptx import clamp_hi, f32_bits, max_abs_ftz, max_ftz, mul_ftz, rcp_approx_ftz
from .dequantize import _log2_blocksize

_CTA = 256
_WARPS = _CTA // 32
_UNIT = 8               # elements per lane per pass
_MAX_HELD = 32          # values a lane keeps in registers between the two passes
_DTYPES = (jnp.float16, jnp.bfloat16, jnp.float32)

# The NF4 decision tree's 15 thresholds in ascending order, as the float32
# bits nvcc put in the bitsandbytes binary. The tree is a binary search over
# them with strict ``>``, so the code is just how many ``x`` exceeds.
#
# These are bits rather than the decimal literals in kernels.cu because the
# literals are float32 midpoints written to ~17 digits: nvcc rounds each one
# to float32 once, while Python's decimal -> float64 -> float32 lands on the
# tie and rounds four of them (0.5016634..., 0.8614784..., 0.1202552...,
# -0.8480964...) to the other neighbour.
_NF4_THRESHOLDS = [f32_bits(b) for b in (
    0xbf591cd9, 0xbf1c5270, 0xbeeb8480, 0xbeadea76, 0xbe703cec,
    0xbe0d38bc, 0xbd3a7871, 0x3d22faff, 0x3df64863, 0x3e5067e0,
    0x3e9582d4, 0x3ec753f9, 0x3f006d03, 0x3f248daf, 0x3f5c89d9,
)]


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


def _quantize_nf4(x):
    q = cutlass.Int32(0)
    for t in _NF4_THRESHOLDS:
        q = q + cutlass.Int32(x > t)
    return q


def _select(cond, a, b):
    return type(a)(cutlass.select_(cond, a, b))


def _quantize_8bit(sC, x):
    """bitsandbytes' dQuantize<0>: binary search of the sorted codebook, then
    pick the nearer of the two neighbours by comparing against the midpoint."""
    pivot = cutlass.Int32(127)
    upper_pivot = cutlass.Int32(255)
    lower_pivot = cutlass.Int32(0)
    lower = cutlass.Float32(-1.0)
    upper = cutlass.Float32(1.0)
    val = sC[pivot]
    for i in (64, 32, 16, 8, 4, 2, 1):
        gt = x > val
        lower_pivot = _select(gt, pivot, lower_pivot)
        lower = _select(gt, val, lower)
        upper_pivot = _select(gt, upper_pivot, pivot)
        upper = _select(gt, upper, val)
        pivot = _select(gt, pivot + i, pivot - i)
        val = sC[pivot]
    upper = _select(upper_pivot == 255, sC[255], upper)
    lower = _select(lower_pivot == 0, sC[0], lower)

    gt = x > val
    up = _select(x > (upper + val) * 0.5, upper_pivot, pivot)
    down = _select(x < (lower + val) * 0.5, lower_pivot, pivot)
    return _select(gt, up, down)


@cute.kernel
def _quant_kernel(gX: cute.Tensor, gC: cute.Tensor, gQ: cute.Tensor, gA: cute.Tensor,
                  n_blocks: cutlass.Int32,
                  NF4: cutlass.Constexpr[bool], BS: cutlass.Constexpr[int]):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    L = min(32, BS // _UNIT)          # lanes per block
    P = BS // (_UNIT * L)             # passes per lane
    hold = P * _UNIT <= _MAX_HELD     # keep the values for pass 2, or reload them

    sC = None
    if cutlass.const_expr(not NF4):
        smem = cutlass.memory.SmemAllocator()
        sC = smem.allocate_tensor(cutlass.Float32, cute.make_layout(256), byte_alignment=16)
        sC[tid] = gC[tid]
        cute.arch.sync_threads()

    lane = tid & 31
    li = lane % L
    b = (bid * _WARPS + (tid >> 5)) * (32 // L) + lane // L
    # lanes past the last block load a valid one and keep in step for the
    # shuffles; their results are dropped
    bl = clamp_hi(b, n_blocks - 1)

    held = cute.make_rmem_tensor(P * _UNIT if hold else 1, cutlass.Float32)
    m = cutlass.Float32(-3.4028234663852886e38)
    for p in cutlass.range_constexpr(P):
        v = cute.make_rmem_tensor(_UNIT, gX.element_type)
        cute.autovec_copy(gX[bl * (BS // _UNIT) + p * L + li, None], v)
        for j in cutlass.range_constexpr(_UNIT):
            x = cutlass.Float32(v[j])
            if cutlass.const_expr(hold):
                held[p * _UNIT + j] = x
            m = max_abs_ftz(m, x)
    # offsets L/2 .. 1 stay within each aligned group of L lanes
    for s in cutlass.range_constexpr(L.bit_length() - 1):
        m = max_ftz(m, cute.arch.shuffle_sync_bfly(m, L >> (s + 1)))
    r = rcp_approx_ftz(m)

    if b < n_blocks:
        if li == 0:
            gA[b] = m
        for p in cutlass.range_constexpr(P):
            u = b * (BS // _UNIT) + p * L + li
            v = cute.make_rmem_tensor(_UNIT, gX.element_type)
            if cutlass.const_expr(not hold):
                cute.autovec_copy(gX[u, None], v)
            codes = cute.make_rmem_tensor(_UNIT, cutlass.Int32)
            for j in cutlass.range_constexpr(_UNIT):
                x = held[p * _UNIT + j] if hold else cutlass.Float32(v[j])
                x = mul_ftz(x, r)
                if cutlass.const_expr(NF4):
                    codes[j] = _quantize_nf4(x)
                else:
                    codes[j] = _quantize_8bit(sC, x)
            words = cute.make_rmem_tensor(_UNIT // 4 // (2 if NF4 else 1), cutlass.Int32)
            if cutlass.const_expr(NF4):
                words[0] = ((codes[0] << 4) | codes[1]
                            | (((codes[2] << 4) | codes[3]) << 8)
                            | (((codes[4] << 4) | codes[5]) << 16)
                            | (((codes[6] << 4) | codes[7]) << 24))
            else:
                for k in cutlass.range_constexpr(2):
                    words[k] = (codes[4 * k] | (codes[4 * k + 1] << 8)
                                | (codes[4 * k + 2] << 16) | (codes[4 * k + 3] << 24))
            cute.autovec_copy(words, gQ[u, None])


@cute.jit
def _quant_launch(stream, gX: cute.Tensor, gC: cute.Tensor, gQ: cute.Tensor, gA: cute.Tensor,
                  nf4: cutlass.Constexpr[bool], blocksize: cutlass.Constexpr[int]):
    n_units = gX.shape[0] // _UNIT
    n_blocks = gA.shape[0]
    words = 1 if nf4 else 2
    # input as 8-element rows and codes as 32-bit words, so each lane's
    # accesses are single vector loads/stores
    gX8 = cute.make_tensor(gX.iterator, cute.make_layout((n_units, _UNIT), stride=(_UNIT, 1)))
    gW = cute.make_tensor(cute.recast_ptr(gQ.iterator, dtype=cutlass.Int32),
                          cute.make_layout((n_units, words), stride=(words, 1)))
    blocks_per_cta = _WARPS * (32 // min(32, blocksize // _UNIT))
    n_cta = (n_blocks + blocks_per_cta - 1) // blocks_per_cta
    _quant_kernel(gX8, gC, gW, gA, n_blocks, nf4, blocksize).launch(
        grid=[n_cta, 1, 1], block=[_CTA, 1, 1], stream=stream)


def _quantize(input, code, blocksize, nf4):
    if jnp.dtype(input.dtype) not in map(jnp.dtype, _DTYPES):
        raise NotImplementedError(f"quantization of {input.dtype} is not implemented")
    _log2_blocksize(blocksize)
    x = input.ravel()
    n = x.size
    assert n < 2**31, "too many elements for int32 indexing"
    n_blocks = -(-n // blocksize)
    n_pad = n_blocks * blocksize
    if n_pad != n:
        # zeros change no block's absmax, and bitsandbytes pads its loads the
        # same way, so the codes of the real elements are unaffected
        x = jnp.pad(x, (0, n_pad - n))
    q, absmax = cutejax.call(
        _quant_launch,
        (jax.ShapeDtypeStruct((n_pad // 2 if nf4 else n_pad,), jnp.uint8),
         jax.ShapeDtypeStruct((n_blocks,), jnp.float32)),
        x, code,
        nf4=nf4, blocksize=blocksize,
        vmap_method="sequential",
    )
    n_bytes = -(-n // 2) if nf4 else n
    return q[:n_bytes], absmax


def quantize_nf4(
    input: jax.Array,
    blocksize: int = 64,
) -> tuple[jax.Array, jax.Array]:
    """Quantize a tensor to NF4 format.

    Returns (quantized_u8, absmax_f32): ``ceil(n / 2)`` bytes, each holding
    two 4-bit codes (high nibble first), and one absmax per block.
    """
    # the code input is unused for NF4; cutejax wants an array in its place
    return _quantize(input, jnp.zeros((256,), jnp.float32), blocksize, nf4=True)


def quantize_blockwise(
    input: jax.Array,
    code: jax.Array,
    blocksize: int = 64,
) -> tuple[jax.Array, jax.Array]:
    """Quantize a tensor using 8-bit blockwise quantization with a codebook.

    Returns (quantized_u8, absmax_f32).
    """
    code = jnp.asarray(code, jnp.float32).ravel()
    assert code.size == 256, f"codebook must have 256 entries, got {code.size}"
    return _quantize(input, code, blocksize, nf4=False)
