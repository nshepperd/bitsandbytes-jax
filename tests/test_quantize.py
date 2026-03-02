"""Tests for bitsandbytes-jax quantization."""

import bitsandbytes.functional as bnb
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

import bnb_jax


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


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.float32])
def test_quantize_nf4(dtype):
    """Test NF4 quantization by comparing JAX quantization with bitsandbytes."""
    torch_dtype = {
        jnp.float16: torch.float16,
        jnp.bfloat16: torch.bfloat16,
        jnp.float32: torch.float32,
    }[dtype]

    n = 256
    blocksize = 64

    # Create random test tensor
    torch_tensor = torch.randn(n, dtype=torch_dtype, device="cuda")
    jax_tensor = to_jax(torch_tensor)

    # Quantize with bitsandbytes (PyTorch)
    torch_quantized, torch_state = bnb.quantize_4bit(
        torch_tensor, blocksize=blocksize, quant_type="nf4"
    )

    # Quantize with bnb_jax
    jax_qa = bnb_jax.QuantizedArray.quantize(jax_tensor, blocksize=blocksize, quant_type="nf4")

    # The quantized bytes should match exactly
    np.testing.assert_array_equal(
        np.array(jax_qa.array),
        to_jax(torch_quantized).ravel(),
    )

    # The absmax values should match exactly
    np.testing.assert_allclose(
        np.array(jax_qa.absmax),
        np.array(to_jax(torch_state.absmax)).ravel(),
    )


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.float32])
def test_quantize_nf4_roundtrip(dtype):
    """Test that quantize -> dequantize round-trip matches bitsandbytes."""
    torch_dtype = {
        jnp.float16: torch.float16,
        jnp.bfloat16: torch.bfloat16,
        jnp.float32: torch.float32,
    }[dtype]

    n = 256
    blocksize = 64

    torch_tensor = torch.randn(n, dtype=torch_dtype, device="cuda")
    jax_tensor = to_jax(torch_tensor)

    # bitsandbytes round-trip
    torch_quantized, torch_state = bnb.quantize_4bit(
        torch_tensor, blocksize=blocksize, quant_type="nf4"
    )
    torch_dequant = bnb.dequantize_4bit(torch_quantized, torch_state, quant_type="nf4")

    # JAX round-trip
    jax_qa = bnb_jax.QuantizedArray.quantize(jax_tensor, blocksize=blocksize, quant_type="nf4")
    jax_dequant = jax_qa.dequantize()

    # Compare dequantized results
    np.testing.assert_allclose(
        np.array(jax_dequant.astype(jnp.float32)),
        torch_dequant.to(torch.float32).cpu().numpy(),
    )


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.float32])
def test_quantize_blockwise(dtype):
    """Test 8-bit blockwise quantization by comparing with bitsandbytes."""
    torch_dtype = {
        jnp.float16: torch.float16,
        jnp.bfloat16: torch.bfloat16,
        jnp.float32: torch.float32,
    }[dtype]

    n = 256
    blocksize = 64

    torch_tensor = torch.randn(n, dtype=torch_dtype, device="cuda")
    jax_tensor = to_jax(torch_tensor)

    # Quantize with bitsandbytes
    torch_quantized, torch_state = bnb.quantize_blockwise(
        torch_tensor, blocksize=blocksize
    )

    # Quantize with bnb_jax (uses default codebook)
    jax_qa = bnb_jax.QuantizedArray.quantize(
        jax_tensor, blocksize=blocksize, quant_type=None
    )

    # The quantized bytes should match exactly
    np.testing.assert_array_equal(
        np.array(jax_qa.array),
        to_jax(torch_quantized).ravel(),
    )

    # The absmax values should match
    np.testing.assert_allclose(
        np.array(jax_qa.absmax),
        np.array(to_jax(torch_state.absmax)).ravel(),
    )


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.float32])
def test_quantize_blockwise_roundtrip(dtype):
    """Test 8-bit blockwise quantize -> dequantize round-trip."""
    torch_dtype = {
        jnp.float16: torch.float16,
        jnp.bfloat16: torch.bfloat16,
        jnp.float32: torch.float32,
    }[dtype]

    n = 256
    blocksize = 64

    torch_tensor = torch.randn(n, dtype=torch_dtype, device="cuda")
    jax_tensor = to_jax(torch_tensor)

    # bitsandbytes round-trip
    torch_quantized, torch_state = bnb.quantize_blockwise(
        torch_tensor, blocksize=blocksize
    )
    torch_dequant = bnb.dequantize_blockwise(torch_quantized, torch_state)

    # JAX round-trip (uses default codebook)
    jax_qa = bnb_jax.QuantizedArray.quantize(
        jax_tensor, blocksize=blocksize, quant_type=None
    )
    jax_dequant = jax_qa.dequantize()

    # Compare dequantized results (small atol for float32 codebook rounding)
    np.testing.assert_allclose(
        np.array(jax_dequant.astype(jnp.float32)),
        torch_dequant.to(torch.float32).cpu().numpy(),
        atol=3e-7,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
