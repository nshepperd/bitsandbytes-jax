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
@pytest.mark.parametrize("quant_type", ["nf4", "blockwise"])
def test_dequantize_basic(dtype, quant_type):
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
    if quant_type in ("nf4",):
        quantized, state = bnb.quantize_4bit(
            torch_tensor,
            blocksize=blocksize,
            quant_type=quant_type
        )
        torch_dequant = bnb.dequantize_4bit(
            quantized, state, quant_type=quant_type
        )
    elif quant_type == "blockwise":
        quantized, state = bnb.quantize_blockwise(
            torch_tensor,
            blocksize=blocksize,
        )
        torch_dequant = bnb.dequantize_blockwise(
            quantized, state
        )

    # Convert to JAX arrays
    jax_quantized = bnb_jax.QuantizedArray(
        jnp.ravel(to_jax(quantized)),
        absmax=jnp.ravel(to_jax(state.absmax)),
        code=jnp.ravel(to_jax(state.code)) if state.code is not None else None,
        offset=None,
        quantize_type=quant_type if quant_type in ("nf4",) else None,
        blocksize=blocksize,
        shape=torch_tensor.shape,
        dtype=dtype,
    )

    # Dequantize with bnb_jax
    jax_dequant = jnp.asarray(jax_quantized)

    # Compare results
    np_torch = torch_dequant.to(torch.float32).cpu().numpy()
    np_jax = np.array(jax_dequant.astype(jnp.float32))

    np.testing.assert_allclose(np_jax, np_torch)

def test_fromdict():
    import safetensors as st
    path = "/data/models/Llama-3.2-11B-Vision-bnb-4bit/model-00001-of-00002.safetensors"
    state_dict = {}
    with st.safe_open(path, framework='flax') as f:
        for key in f.keys():
            prefix = 'vision_model.transformer.layers.25.mlp.fc1.weight'
            if key.startswith(prefix):
                state_dict[key[len(prefix):]] = f.get_tensor(key)
    weight = state_dict.pop('')
    q = bnb_jax.QuantizedArray.from_dict(weight, state_dict)
    assert jnp.all(jnp.isfinite(q.dequantize()))
    jax_dequant = jnp.asarray(q)
    torch_weight = to_torch(weight)
    torch_state_dict = jax.tree_util.tree_map(to_torch, state_dict)
    torch_qstate = bnb.QuantState.from_dict(torch_state_dict, device='cpu')
    torch_dequant = bnb.dequantize_4bit(
        torch_weight, torch_qstate
    )
    np.testing.assert_allclose(
        jax_dequant,
        to_jax(torch_dequant),
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
