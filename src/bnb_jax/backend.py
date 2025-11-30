from bnb_jax_lib import _core
import bitsandbytes.cextension
import jax

_initialized = False

def ensure_initialized():
    global _initialized
    if _initialized:
        return
    handle = bitsandbytes.cextension.lib._lib._handle
    for name, fn in _core.ffi_registrations(handle).items():
        jax.ffi.register_ffi_target(name, fn, platform="CUDA")
    _initialized = True