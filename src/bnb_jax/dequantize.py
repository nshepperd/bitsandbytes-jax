"""Compatibility: QuantizedArray used to live here, and downstream code still
imports it as ``from bnb_jax.dequantize import QuantizedArray``."""

from .array import QuantizedArray

__all__ = ["QuantizedArray"]
