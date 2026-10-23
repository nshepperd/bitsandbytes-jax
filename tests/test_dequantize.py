"""Tests for bitsandbytes-jax dequantization: bitwise against bitsandbytes."""

import os

import bitsandbytes.functional as bnb
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

import bnb_jax

DTYPES = {
    jnp.float16: torch.float16,
    jnp.bfloat16: torch.bfloat16,
    jnp.float32: torch.float32,
}


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


def bnb_quantized(quant_type, n, blocksize, torch_dtype=torch.float32):
    """Quantize random data with bitsandbytes; return (bnb_jax QuantizedArray,
    bitsandbytes' own dequantization) at ``torch_dtype``."""
    torch_tensor = torch.randn(n, dtype=torch_dtype, device="cuda")
    if quant_type == "nf4":
        quantized, state = bnb.quantize_4bit(torch_tensor, blocksize=blocksize, quant_type="nf4")
        torch_dequant = bnb.dequantize_4bit(quantized, state, quant_type="nf4")
    else:
        quantized, state = bnb.quantize_blockwise(torch_tensor, blocksize=blocksize)
        torch_dequant = bnb.dequantize_blockwise(quantized, state)

    jax_quantized = bnb_jax.QuantizedArray(
        data=jnp.ravel(to_jax(quantized)),
        absmax=jnp.ravel(to_jax(state.absmax)),
        code=jnp.ravel(to_jax(state.code)) if quant_type == "8bit" else None,
        offset=None,
        qtype=quant_type,
        blocksize=blocksize,
        shape=tuple(torch_tensor.shape),
        dtype=jnp.dtype(jnp.float32),
    )
    return jax_quantized, torch_dequant


def assert_bitwise(jax_array, torch_tensor):
    np.testing.assert_array_equal(
        np.array(jax_array.astype(jnp.float32)),
        torch_tensor.to(torch.float32).cpu().numpy(),
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("quant_type", ["nf4", "8bit"])
def test_dequantize_basic(dtype, quant_type):
    """Dequantize bitsandbytes-quantized data at the output dtype."""
    jax_quantized, torch_dequant = bnb_quantized(quant_type, 256, 64, DTYPES[dtype])
    jax_dequant = jax_quantized.dequantize(dtype)
    assert jax_dequant.dtype == dtype
    assert_bitwise(jax_dequant, torch_dequant)


@pytest.mark.parametrize("blocksize", [32, 64, 128, 256, 512, 1024, 2048, 4096])
@pytest.mark.parametrize("quant_type", ["nf4", "8bit"])
def test_dequantize_blocksizes(blocksize, quant_type):
    """Every blocksize, with a partial (odd-length) last block."""
    if quant_type == "8bit" and blocksize < 64:
        pytest.skip("bitsandbytes' 8-bit quantization needs blocksize >= 64")
    for n in [3 * blocksize + 5, 1 << 18]:
        jax_quantized, torch_dequant = bnb_quantized(quant_type, n, blocksize)
        assert_bitwise(jax_quantized.dequantize(), torch_dequant)


def test_dequantize_jit_vmap():
    """Dequantize under jit, and under vmap over stacked weights."""
    xs = jax.random.normal(jax.random.key(0), (3, 4096), jnp.float32)
    qs = [bnb_jax.quantize(x) for x in xs]
    stacked = jax.tree.map(lambda *a: jnp.stack(a), *qs)
    out = jax.jit(jax.vmap(lambda q: q.dequantize()))(stacked)
    for q, o in zip(qs, out):
        np.testing.assert_array_equal(o, q.dequantize())


@pytest.mark.parametrize("packed", [True, False])
@pytest.mark.parametrize("compress_statistics", [True, False])
def test_fromdict_quant_state(packed, compress_statistics):
    """from_dict on bitsandbytes' own QuantState.as_dict, in both the packed
    form saved to safetensors and the unpacked one."""
    torch_tensor = torch.randn(96, 128, dtype=torch.bfloat16, device="cuda")
    quantized, state = bnb.quantize_4bit(
        torch_tensor, blocksize=64, quant_type="nf4", compress_statistics=compress_statistics)
    torch_dequant = bnb.dequantize_4bit(quantized, state)
    qs_dict = {k: to_jax(v) if isinstance(v, torch.Tensor) else v
               for k, v in state.as_dict(packed=packed).items()}

    q = bnb_jax.QuantizedArray.from_dict(to_jax(quantized), qs_dict)
    assert q.shape == (96, 128) and q.dtype == jnp.bfloat16
    assert q.double_quantized == compress_statistics
    jax_dequant = q.dequantize()
    assert jax_dequant.dtype == jnp.bfloat16
    assert_bitwise(jax_dequant, torch_dequant)


def test_legacy_import():
    """Downstream code imports QuantizedArray from its old home."""
    from bnb_jax.dequantize import QuantizedArray
    assert QuantizedArray is bnb_jax.QuantizedArray


FROMDICT_PATH = "/data/models/Llama-3.2-11B-Vision-bnb-4bit/model-00001-of-00002.safetensors"

@pytest.mark.skipif(not os.path.exists(FROMDICT_PATH), reason="needs the Llama-3.2-11B-Vision bnb-4bit checkpoint")
def test_fromdict():
    import safetensors as st
    state_dict = {}
    with st.safe_open(FROMDICT_PATH, framework='flax') as f:
        for key in f.keys():
            prefix = 'vision_model.transformer.layers.25.mlp.fc1.weight'
            if key.startswith(prefix):
                state_dict[key[len(prefix):]] = f.get_tensor(key)
    weight = state_dict.pop('')
    q = bnb_jax.QuantizedArray.from_dict(weight, state_dict)
    assert jnp.all(jnp.isfinite(q.dequantize()))
    jax_dequant = q.dequantize()
    torch_weight = to_torch(weight)
    torch_state_dict = jax.tree_util.tree_map(to_torch, state_dict)
    torch_qstate = bnb.QuantState.from_dict(torch_state_dict, device=torch.device('cuda'))
    torch_dequant = bnb.dequantize_4bit(
        torch_weight, torch_qstate
    )

    absmax_jax = q.absmax.dequantize()
    absmax_torch = bnb.dequantize_blockwise(torch_qstate.absmax, torch_qstate.state2) + torch_qstate.offset
    np.testing.assert_array_equal(
        absmax_jax,
        to_jax(absmax_torch),
    )

    np.testing.assert_array_equal(
        jax_dequant,
        to_jax(torch_dequant),
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
