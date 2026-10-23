"""Tests for bitsandbytes-jax quantization: bitwise against bitsandbytes."""

import bitsandbytes.functional as bnb
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

import bnb_jax
from bnb_jax.quantize import _NF4_THRESHOLDS

DTYPES = {
    jnp.float16: torch.float16,
    jnp.bfloat16: torch.bfloat16,
    jnp.float32: torch.float32,
}
NF4_BLOCKSIZES = [32, 64, 128, 256, 512, 1024, 2048, 4096]
BLOCKWISE_BLOCKSIZES = [64, 128, 256, 512, 1024, 2048, 4096]


def to_jax(tensor: torch.Tensor) -> jax.Array:
    """Convert a PyTorch tensor to a JAX array."""
    if tensor.dtype == torch.bfloat16:
        return jnp.array(tensor.cpu().to(torch.float32).numpy(), dtype=jnp.bfloat16)
    return jnp.array(tensor.cpu().numpy())

def to_torch(array: jax.Array) -> torch.Tensor:
    """Convert a JAX array to a PyTorch tensor."""
    if array.dtype == jnp.bfloat16:
        return torch.tensor(np.array(array, dtype=np.float32), dtype=torch.bfloat16, device="cuda")
    return torch.tensor(np.array(array), device="cuda")


def sizes(blocksize):
    # one block, a partial (and odd-length) last block, and enough blocks
    # that the approximate reciprocal sees many absmax values
    return [blocksize, 3 * blocksize + 5, 1 << 18]


def check_nf4(torch_tensor, blocksize):
    jax_tensor = to_jax(torch_tensor)
    torch_quantized, torch_state = bnb.quantize_4bit(
        torch_tensor, blocksize=blocksize, quant_type="nf4"
    )
    jax_qa = bnb_jax.QuantizedArray.quantize(jax_tensor, blocksize=blocksize, quant_type="nf4")

    # The quantized bytes and absmax should match exactly
    np.testing.assert_array_equal(np.array(jax_qa.array), to_jax(torch_quantized).ravel())
    np.testing.assert_array_equal(np.array(jax_qa.absmax), np.array(to_jax(torch_state.absmax)).ravel())
    return jax_qa, torch_quantized, torch_state


def check_blockwise(torch_tensor, blocksize):
    jax_tensor = to_jax(torch_tensor)
    torch_quantized, torch_state = bnb.quantize_blockwise(torch_tensor, blocksize=blocksize)
    # with bitsandbytes' own codebook: see test_create_dynamic_map
    jax_qa = bnb_jax.QuantizedArray.quantize(
        jax_tensor, blocksize=blocksize, quant_type=None, code=to_jax(torch_state.code))

    np.testing.assert_array_equal(np.array(jax_qa.array), to_jax(torch_quantized).ravel())
    np.testing.assert_array_equal(np.array(jax_qa.absmax), np.array(to_jax(torch_state.absmax)).ravel())
    return jax_qa, torch_quantized, torch_state


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("blocksize", NF4_BLOCKSIZES)
def test_quantize_nf4(dtype, blocksize):
    """Test NF4 quantization by comparing JAX quantization with bitsandbytes."""
    for n in sizes(blocksize):
        check_nf4(torch.randn(n, dtype=DTYPES[dtype], device="cuda"), blocksize)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("blocksize", BLOCKWISE_BLOCKSIZES)
def test_quantize_blockwise(dtype, blocksize):
    """Test 8-bit blockwise quantization by comparing with bitsandbytes."""
    for n in sizes(blocksize):
        check_blockwise(torch.randn(n, dtype=DTYPES[dtype], device="cuda"), blocksize)


def test_create_dynamic_map():
    """Within a few ulp of bitsandbytes' codebook, not bitwise: bitsandbytes
    builds it with torch.linspace on the CPU, whose float32 rounding depends
    on which SIMD kernel torch dispatches to."""
    np.testing.assert_array_max_ulp(
        np.array(bnb_jax.create_dynamic_map()), bnb.create_dynamic_map().numpy(), maxulp=2)


def test_quantize_nf4_thresholds():
    """Values exactly on, and one ulp either side of, each NF4 decision
    threshold. Each block also holds 1.0, so absmax is 1 and the scaled value
    is the value itself."""
    t = np.array(_NF4_THRESHOLDS, np.float32)
    vals = np.concatenate([t, np.nextafter(t, np.inf), np.nextafter(t, -np.inf)])
    x = np.zeros(128, np.float32)
    x[:len(vals)] = vals
    x[63] = x[127] = 1.0
    check_nf4(torch.tensor(x, device="cuda"), 64)


def test_quantize_blockwise_midpoints():
    """Values exactly on, and one ulp either side of, the midpoint between
    each pair of neighbouring codebook entries."""
    code = np.array(bnb_jax.create_dynamic_map())
    mid = ((code[1:] + code[:-1]) * np.float32(0.5)).astype(np.float32)
    vals = np.concatenate([code, mid, np.nextafter(mid, np.inf), np.nextafter(mid, -np.inf)])
    blocksize = 256
    x = np.zeros(-(-len(vals) // (blocksize - 1)) * blocksize, np.float32)
    for i, start in enumerate(range(0, len(vals), blocksize - 1)):
        chunk = vals[start:start + blocksize - 1]
        x[i * blocksize:i * blocksize + len(chunk)] = chunk
        x[(i + 1) * blocksize - 1] = 1.0
    check_blockwise(torch.tensor(x, device="cuda"), blocksize)


@pytest.mark.parametrize("quant_type", ["nf4", "blockwise"])
def test_quantize_special_values(quant_type):
    """All-zero blocks, denormal-only blocks, and huge values."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(64 * 6).astype(np.float32)
    x[0:64] = 0.0
    x[64:128] = rng.standard_normal(64) * 1e-39      # float32 denormals
    x[128:192] *= 1e30
    x[192:256] = -x[192:256] * 1e-30
    x[256] = -0.0
    t = torch.tensor(x, device="cuda")
    if quant_type == "nf4":
        check_nf4(t, 64)
    else:
        check_blockwise(t, 64)


@pytest.mark.parametrize("dtype", DTYPES)
def test_quantize_nf4_roundtrip(dtype):
    """Test that quantize -> dequantize round-trip matches bitsandbytes."""
    torch_tensor = torch.randn(4096 + 64, dtype=DTYPES[dtype], device="cuda")
    jax_qa, torch_quantized, torch_state = check_nf4(torch_tensor, 64)

    # bitsandbytes round-trip
    torch_dequant = bnb.dequantize_4bit(torch_quantized, torch_state, quant_type="nf4")
    # JAX round-trip
    jax_dequant = jax_qa.dequantize()

    assert jax_dequant.dtype == dtype
    np.testing.assert_array_equal(
        np.array(jax_dequant.astype(jnp.float32)),
        torch_dequant.to(torch.float32).cpu().numpy(),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_quantize_blockwise_roundtrip(dtype):
    """Test 8-bit blockwise quantize -> dequantize round-trip."""
    torch_tensor = torch.randn(4096 + 64, dtype=DTYPES[dtype], device="cuda")
    jax_qa, torch_quantized, torch_state = check_blockwise(torch_tensor, 64)

    # bitsandbytes round-trip
    torch_dequant = bnb.dequantize_blockwise(torch_quantized, torch_state)
    # JAX round-trip (uses default codebook)
    jax_dequant = jax_qa.dequantize()

    assert jax_dequant.dtype == dtype
    np.testing.assert_array_equal(
        np.array(jax_dequant.astype(jnp.float32)),
        torch_dequant.to(torch.float32).cpu().numpy(),
    )


def test_quantize_jit_shape():
    """Quantize under jit, of a multi-dimensional array."""
    x = jax.random.normal(jax.random.key(0), (64, 96), jnp.bfloat16)
    qa = jax.jit(bnb_jax.QuantizedArray.quantize)(x)
    assert qa.shape == (64, 96)
    expected = bnb_jax.QuantizedArray.quantize(x.ravel())
    np.testing.assert_array_equal(qa.array, expected.array)
    assert qa.dequantize().shape == (64, 96)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
