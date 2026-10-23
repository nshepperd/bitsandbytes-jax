"""Benchmark bnb_jax.matmul (the fused NF4 GEMV) against dequantize-then-matmul.

    XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
        uv run python bench/bench_gemv.py

y = x @ W.T with W (N, K) NF4 (blocksize 64, double-quantized like HF
checkpoints), x (M, K) bf16. Each timed call cycles through ROTATE distinct
weights, more bytes than L2 holds, so the weights come from DRAM as they do in
a real decode step. Also times bitsandbytes' own GEMV (matmul_4bit at M=1).
"""
import time

import bitsandbytes.functional as bnbf
import jax
import jax.numpy as jnp
import numpy as np
import torch
from bitsandbytes import matmul_4bit

import bnb_jax

SHAPES = [(4096, 14336), (14336, 4096), (4096, 4096)]
ROTATE = 6


def bench_jax(fn, *args, iters=50, warmup=3):
    for _ in range(warmup):
        out = fn(*args)
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / iters / ROTATE


def bench_torch(fn, iters=50, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters / ROTATE


def main():
    print(f"device: {jax.devices()[0].device_kind}")
    for n, k in SHAPES:
        w0 = jax.random.normal(jax.random.key(0), (n, k), jnp.bfloat16)
        ws = [bnb_jax.quantize(w0 * (1 + i / 8), "nf4", 64, double_quant=True) for i in range(ROTATE)]
        dense = [w.dequantize(jnp.bfloat16) for w in ws]
        q_bytes = ws[0].nbytes

        t_w = torch.from_numpy(np.asarray(w0.astype(jnp.float32))).cuda().bfloat16()
        bnb_ws = [bnbf.quantize_4bit(t_w * (1 + i / 8), blocksize=64, quant_type="nf4",
                                     compress_statistics=True) for i in range(ROTATE)]

        run_fused = jax.jit(lambda x, ws: [bnb_jax.matmul(x, w) for w in ws])
        run_unfused = jax.jit(lambda x, ws: [x @ w.dequantize(jnp.bfloat16).T for w in ws])
        run_dense = jax.jit(lambda x, ds: [x @ d.T for d in ds])

        print(f"\nW: {n}x{k} NF4 ({q_bytes / 1e6:.1f}MB quantized, {n * k * 2 / 1e6:.0f}MB as bf16)")
        print(f"{'M':>3} {'matmul':>14} {'unfused':>8} {'dense bf16':>10} {'bnb gemv':>9}   vs unfused")
        for m in [1, 2, 4, 8, 12, 16, 20, 24]:
            x = jax.random.normal(jax.random.key(m), (m, k), jnp.bfloat16)
            t_f = bench_jax(run_fused, x, ws)
            t_u = bench_jax(run_unfused, x, ws)
            t_d = bench_jax(run_dense, x, dense)
            t_b = ""
            if m == 1:
                tx = torch.randn(1, k, dtype=torch.bfloat16, device="cuda")
                t_b = bench_torch(lambda tx=tx, bnb_ws=bnb_ws:
                                  [matmul_4bit(tx, q.t(), st) for q, st in bnb_ws])
                t_b = f"{t_b * 1e6:7.0f}us"
            print(f"{m:>3} {t_f * 1e6:5.0f}us {q_bytes / t_f / 1e9:4.0f}GB/s {t_u * 1e6:6.0f}us "
                  f"{t_d * 1e6:8.0f}us {t_b:>9}   {t_u / t_f:5.1f}x")


if __name__ == "__main__":
    main()
