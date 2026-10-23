"""Tests for bnb_jax.matmul and the fused NF4 GEMV."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import bnb_jax
from bnb_jax.kernels.gemv import GEMV_MAX_M


def weight(n, k, blocksize=64, double_quant=True, qtype="nf4", seed=0):
    w = jax.random.normal(jax.random.key(seed), (n, k), jnp.bfloat16)
    return bnb_jax.quantize(w, qtype, blocksize, double_quant=double_quant)


@pytest.mark.parametrize("double_quant", [False, True])
@pytest.mark.parametrize("blocksize", [32, 64, 256])
def test_gemv_exact_weights(blocksize, double_quant):
    """With a one-hot x the GEMV returns the dequantized weights, bitwise:
    the per-element products are exact and each output is rounded once."""
    w = weight(1000, 1056, blocksize, double_quant)
    for m in [1, 3, GEMV_MAX_M]:
        x = jnp.eye(1056, dtype=jnp.bfloat16)[:m]
        np.testing.assert_array_equal(
            bnb_jax.matmul(x, w), w.dequantize(jnp.bfloat16)[:, :m].T)


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
@pytest.mark.parametrize("m", [1, 2, 3, 8, GEMV_MAX_M])
def test_gemv_accuracy(m, dtype):
    """Closer to the exact product than dequantize-then-matmul, which rounds
    every weight to x's dtype first."""
    w = weight(1000, 4096)
    x = jax.random.normal(jax.random.key(m), (m, 4096), dtype)
    # in float64 on the host: a float32 matmul on the GPU may use TF32
    exact = np.asarray(x, np.float64) @ np.asarray(w.dequantize(jnp.float32), np.float64).T
    fused = bnb_jax.matmul(x, w)
    unfused = x @ w.dequantize(dtype).T
    assert fused.dtype == dtype and fused.shape == (m, 1000)
    err_fused = np.abs(np.asarray(fused, np.float64) - exact).mean()
    err_unfused = np.abs(np.asarray(unfused, np.float64) - exact).mean()
    assert err_fused < err_unfused, (err_fused, err_unfused)


def test_matmul_batch_dims_jit():
    w = weight(512, 1024)
    x = jax.random.normal(jax.random.key(1), (2, 3, 1024), jnp.bfloat16)
    out = jax.jit(bnb_jax.matmul)(x, w)
    assert out.shape == (2, 3, 512)
    np.testing.assert_array_equal(out, bnb_jax.matmul(x.reshape(6, 1024), w).reshape(2, 3, 512))


@pytest.mark.parametrize("case", ["large_m", "8bit", "float32_x"])
def test_matmul_fallback(case):
    """Where the GEMV does not apply, x @ w.dequantize(x.dtype).T exactly."""
    m, dtype, qtype = 4, jnp.bfloat16, "nf4"
    if case == "large_m":
        m = GEMV_MAX_M + 1
    elif case == "8bit":
        qtype = "8bit"
    else:
        dtype = jnp.float32
    w = weight(256, 512, qtype=qtype)
    x = jax.random.normal(jax.random.key(2), (m, 512), dtype)
    np.testing.assert_array_equal(bnb_jax.matmul(x, w), x @ w.dequantize(dtype).T)


def test_matmul_shape_errors():
    w = weight(256, 512)
    with pytest.raises(ValueError, match="contraction"):
        bnb_jax.matmul(jnp.zeros((1, 256), jnp.bfloat16), w)
    w1 = bnb_jax.quantize(jnp.zeros((512,), jnp.bfloat16))
    with pytest.raises(ValueError, match="2-D"):
        bnb_jax.matmul(jnp.zeros((1, 512), jnp.bfloat16), w1)
