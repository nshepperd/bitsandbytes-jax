# bitsandbytes-jax

bitsandbytes NF4 and 8-bit blockwise quantization for JAX, written as CuTe DSL
kernels (via [cutejax](https://github.com/nshepperd/cutejax)). Pure Python:
there is no C++ extension to build, and bitsandbytes itself is not a runtime
dependency.

```python
import jax.numpy as jnp
import bnb_jax

# load a bnb-4bit checkpoint's weight + quant state (prefixes stripped)
w = bnb_jax.QuantizedArray.from_dict(packed, quant_state)
y = x @ w.dequantize(jnp.bfloat16).T     # inside your (jitted) forward pass

# or quantize yourself
q = bnb_jax.QuantizedArray.quantize(x, blocksize=64, quant_type="nf4")   # 4-bit NF4
q = bnb_jax.QuantizedArray.quantize(x, blocksize=256, quant_type=None)   # 8-bit blockwise
```

## Features

- **Bitwise-exact against bitsandbytes.** Quantized codes, absmax and
  dequantized values are bit-for-bit equal to bitsandbytes' CUDA kernels, for
  float16 / bfloat16 / float32 and every blocksize bitsandbytes supports
  (32–4096 for NF4, 64–4096 for 8-bit). bitsandbytes is built with
  `--use_fast_math`, so the kernels reproduce its approximate reciprocal and
  flush-to-zero arithmetic in PTX.
- **A pytree.** `QuantizedArray` is a registered dataclass (quant type,
  blocksize, shape and dtype are static), so it composes with `jax.jit`,
  `vmap`, `tree_map`, etc. Double-quantized checkpoints (`nested_absmax`) are
  an absmax that is itself a `QuantizedArray`.
- **Any length.** Tensors need not be a multiple of the blocksize; the last
  block is partial, as in bitsandbytes.

One known difference: `create_dynamic_map()` (the default 8-bit codebook) can
differ from bitsandbytes' by an ulp or two in some entries, because
bitsandbytes builds it with `torch.linspace` on the CPU, whose float32
rounding depends on torch's SIMD dispatch. Quantizing with bitsandbytes'
codebook (`code=...`) gives bitwise-identical codes.

## API

- `QuantizedArray.from_dict(weight, qs_dict)` — wrap a weight from a
  HuggingFace bnb-4bit state dict.
- `QuantizedArray.quantize(array, blocksize=64, quant_type="nf4", code=None)`
  — `quant_type=None` for 8-bit blockwise with codebook `code` (default
  `create_dynamic_map()`).
- `QuantizedArray.dequantize(dtype=None)` — decode to a dense array.
- `quantize_nf4`, `quantize_blockwise`, `dequantize_nf4`,
  `dequantize_blockwise` — the functional forms, on flat arrays.

## Development

```bash
uv sync
XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 uv run pytest
```

The test group pulls in `jax[cuda13]`, plus torch and bitsandbytes as the
reference the tests compare against. Install `bitsandbytes-jax[cu13]` to get
the CuTe DSL's CUDA 13 runtime libraries alongside `jax[cuda13]`.
