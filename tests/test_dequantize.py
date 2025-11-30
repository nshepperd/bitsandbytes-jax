"""Tests for bitsandbytes-jax dequantization."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import bnb_jax
import bitsandbytes.functional as bnb
import torch

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
def test_dequantize_nf4_basic(dtype):
    """Test basic NF4 dequantization."""

    torch_dtype = {
        jnp.float16: torch.float16,
        jnp.bfloat16: torch.bfloat16
    }.get(dtype)

    # Create test data using bitsandbytes
    n = 256
    blocksize = 64

    # Create random tensor and quantize with bitsandbytes
    torch_tensor = torch.randn(n, dtype=torch_dtype, device="cuda")

    # Quantize using bitsandbytes
    quantized, state = bnb.quantize_4bit(
        torch_tensor,
        blocksize=blocksize,
        quant_type="nf4"
    )

    # Dequantize with bitsandbytes for reference
    torch_dequant = bnb.dequantize_4bit(
        quantized, state, quant_type="nf4"
    )

    # Convert to JAX arrays
    jax_quantized = to_jax(quantized)
    jax_absmax = to_jax(state.absmax)

    # Dequantize with bnb_jax
    jax_dequant = bnb_jax.dequantize_nf4(jax_quantized, jax_absmax, blocksize=blocksize, dtype=dtype)

    # Compare results
    np_torch = torch_dequant.to(torch.float32).cpu().numpy()
    np_jax = np.array(jax_dequant.astype(jnp.float32))

    np.testing.assert_allclose(np_jax, np_torch)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
