"""QuantizedArray: a JAX-compatible dataclass for bitsandbytes quantized weights."""
from __future__ import annotations
from typing import Any

import jax
import jax.numpy as jnp
from dataclasses import dataclass, field, replace
import json

import numpy as np

from .dequantize import dequantize_nf4, dequantize_blockwise
from .quantize import quantize_nf4, quantize_blockwise, create_dynamic_map

valid_qs_type_keys = [f"bitsandbytes__{x}" for x in ("nf4",)]
valid_qs_keys = [
    "absmax",
    "quant_map",
    "nested_absmax",
    "nested_quant_map",
    "quant_state",
    "quant_type",
    "blocksize",
    "dtype",
    "shape",
    "nested_blocksize",
    "nested_dtype",
    "nested_offset",
]

def unpack_tensor_to_dict(tensor: jax.Array) -> dict[str, Any]:
    bs = bytes(np.array(tensor))
    json_str = bs.decode("utf-8")
    unpacked_dict = json.loads(json_str)
    return unpacked_dict

@jax.tree_util.register_dataclass
@dataclass
class QuantizedArray:
    array: jax.Array
    absmax: jax.Array | QuantizedArray # can be double-quantized
    code: jax.Array | None
    offset: jax.Array | None
    quantize_type: str | None = field(metadata=dict(static=True))
    blocksize: int = field(metadata=dict(static=True))
    shape: tuple[int, ...] = field(metadata=dict(static=True))
    dtype: jnp.dtype = field(metadata=dict(static=True))

    def astype(self, dtype: jnp.dtype) -> QuantizedArray:
        return replace(self, dtype=dtype)
    def dequantize(self, dtype: jnp.dtype | None = None) -> jax.Array:
        return dequantize(self, dtype=dtype)
    # def __jax_array__(self) -> jax.Array:
    #     return dequantize(self)

    @classmethod
    def quantize(cls, array: jax.Array, blocksize: int = 64, quant_type: str | None = "nf4", code: jax.Array | None = None) -> QuantizedArray:
        """Quantize a JAX array into a QuantizedArray.

        Args:
            array: Input tensor to quantize.
            blocksize: Block size for quantization.
            quant_type: "nf4" for 4-bit NF4, or None for 8-bit blockwise.
            code: Codebook for 8-bit blockwise quantization (required when quant_type is None).
        """
        flat = jnp.ravel(array)
        if quant_type == "nf4":
            quantized, absmax = quantize_nf4(flat, blocksize=blocksize)
            return cls(
                array=quantized,
                absmax=absmax,
                code=None,
                offset=None,
                quantize_type="nf4",
                blocksize=blocksize,
                shape=array.shape,
                dtype=array.dtype,
            )
        elif quant_type is None:
            if code is None:
                code = create_dynamic_map()
            quantized, absmax = quantize_blockwise(flat, code, blocksize=blocksize)
            return cls(
                array=quantized,
                absmax=absmax,
                code=code,
                offset=None,
                quantize_type=None,
                blocksize=blocksize,
                shape=array.shape,
                dtype=array.dtype,
            )
        else:
            raise NotImplementedError(f"Quantization not implemented for type {quant_type}")

    @classmethod
    def from_dict(cls, weight: jax.Array, qs_dict: dict[str, Any]) -> QuantizedArray:
        """
        Unpack the state_dict format used in HF models into a QuantizedArray.
        qs_dict: based on state_dict, with only relevant keys, striped of prefixes.
        item with key `quant_state.bitsandbytes__[nf4/fp4]` contains json metadata
        """
        qs_dict = dict(qs_dict)  # make a copy to avoid mutating input
        # unpacking tensor with non-tensor components
        qs_key = [k for k, v in qs_dict.items() if "quant_state" in k and isinstance(v, jax.Array)]
        if not len(qs_key) and "quant_type" not in qs_dict:
            raise ValueError("Expected packed or unpacked quant_state items, found neither")
        elif len(qs_key) != 1 or qs_key[0].split(".")[-1] not in valid_qs_type_keys:
            raise ValueError(
                f"There should be exactly one `quant_state` item with ending from {valid_qs_type_keys}.\nDetected {qs_key}.",
            )

        # unpacking minor and non-tensor quant state items if necessary
        if len(qs_key) == 1:
            first_qs_key = qs_key[0]
            qs_dict.update(unpack_tensor_to_dict(qs_dict.pop(first_qs_key)))

        qs_dict = {k.split(".")[-1]: v for k, v in qs_dict.items()}  # strip prefixes
        assert set(qs_dict.keys()).issubset(valid_qs_keys)

        if "nested_absmax" in qs_dict:
            offset = jnp.array(float(qs_dict["nested_offset"]))
            assert qs_dict["nested_dtype"] == 'float32'
            absmax = QuantizedArray(
                array=qs_dict["absmax"],
                absmax=qs_dict["nested_absmax"],
                offset=offset,
                blocksize=qs_dict["nested_blocksize"],
                code=qs_dict["nested_quant_map"],
                dtype=jnp.float32,
                quantize_type=None,
                shape=qs_dict["absmax"].shape,
            )
        else:
            absmax = qs_dict["absmax"]

        jax_dtype = {'float16': jnp.float16, 'bfloat16': jnp.bfloat16, 'float32': jnp.float32}[qs_dict["dtype"]]

        return QuantizedArray(
            array=weight,
            absmax=absmax,
            blocksize=qs_dict["blocksize"],
            code=qs_dict["quant_map"],
            dtype=jax_dtype,
            shape=tuple(qs_dict["shape"]),
            offset=None,
            quantize_type=qs_dict["quant_type"],
        )

def dequantize(array: QuantizedArray, dtype: jnp.dtype | None = None) -> jax.Array:
    """Dequantize a QuantizedArray to the specified dtype."""
    if isinstance(array.absmax, QuantizedArray):
        absmax = dequantize(array.absmax, dtype=jnp.float32)
    else:
        absmax = array.absmax
    if array.quantize_type == "nf4":
        out = dequantize_nf4(
            array.array,
            absmax,
            blocksize=array.blocksize,
            dtype=dtype or array.dtype,
        ).reshape(array.shape)
    elif array.quantize_type is None:
        assert array.code is not None, "Codebook must be provided for 8-bit dequantization."
        out = dequantize_blockwise(
            array.array,
            absmax,
            array.code,
            blocksize=array.blocksize,
            dtype=dtype or array.dtype,
        ).reshape(array.shape)
    else:
        raise NotImplementedError(f"Dequantization not implemented for type {array.quantize_type}")
    if array.offset is not None:
        out = out + array.offset.astype(out.dtype)
    return out
