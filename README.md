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
# QuantizedArray(nf4, shape=(4096, 4096), dtype=bfloat16, blocksize=64, double_quantized)
y = x @ w.dequantize().T                 # inside your (jitted) forward pass

# or quantize yourself
q = bnb_jax.quantize(x, "nf4", 64)                      # 4-bit NF4
q = bnb_jax.quantize(x, "nf4", 64, double_quant=True)   # with quantized absmax, like HF checkpoints
q = bnb_jax.quantize(x, "8bit", 256)                    # 8-bit blockwise
```

## Features

- **Bitwise-exact against bitsandbytes.** Quantized codes, absmax and
  dequantized values are bit-for-bit equal to bitsandbytes' CUDA kernels, for
  float16 / bfloat16 / float32 and every blocksize bitsandbytes supports
  (32–4096 for NF4, 64–4096 for 8-bit). bitsandbytes is built with
  `--use_fast_math`, so the kernels reproduce its approximate reciprocal and
  flush-to-zero arithmetic in PTX.
- **A pytree.** `QuantizedArray` is a registered dataclass (`qtype`,
  `blocksize`, `shape` and `dtype` are static), so it composes with `jax.jit`,
  `vmap`, `tree_map`, etc. A double-quantized array's absmax is itself an
  `"8bit"` `QuantizedArray`.
- **Any length.** Tensors need not be a multiple of the blocksize; the last
  block is partial, as in bitsandbytes.

One known difference: `create_dynamic_map()` (the default 8-bit codebook) can
differ from bitsandbytes' by an ulp or two in some entries, because
bitsandbytes builds it with `torch.linspace` on the CPU, whose float32
rounding depends on torch's SIMD dispatch. Quantizing with bitsandbytes'
codebook (`code=...`) gives bitwise-identical codes. And with
`double_quant=True` the absmax offset is a float32 mean, whose rounding depends
on summation order, so large tensors can differ from bitsandbytes there by an
ulp.

## API

- `quantize(array, qtype="nf4", blocksize=64, *, code=None, double_quant=False, dtype=None)`
  — `qtype` is `"nf4"` (bitsandbytes' `quantize_4bit`) or `"8bit"`
  (`quantize_blockwise`, with codebook `code`, default `create_dynamic_map()`).
- `QuantizedArray.from_dict(weight, qs_dict)` — wrap a weight from a
  HuggingFace bnb-4bit state dict, packed or unpacked quant state.
- `QuantizedArray.dequantize(dtype=None)` — decode to a dense array (default
  `self.dtype`); `.astype(dtype)` changes that default.
- `QuantizedArray.shape` / `ndim` / `size` / `dtype` / `nbytes` /
  `double_quantized`.
- `bnb_jax.kernels` — the kernels themselves on flat arrays:
  `quantize_nf4`, `quantize_blockwise`, `dequantize_nf4`,
  `dequantize_blockwise`.

## Development

```bash
uv sync
XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 uv run pytest
```

The test group pulls in `jax[cuda13]`, plus torch and bitsandbytes as the
reference the tests compare against. Install `bitsandbytes-jax[cu13]` to get
the CuTe DSL's CUDA 13 runtime libraries alongside `jax[cuda13]`.
