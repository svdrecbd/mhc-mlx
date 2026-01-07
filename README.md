# mhc-mlx

An MLX implementation of mHC (Manifold-Constrained Hyper-Connections) with:

- a pure-MLX reference implementation (correctness baseline)
- an optional Metal fast path for Sinkhorn + fused RMSNorm/mix/add

Forward pass (matching the CUDA reference implementation semantics):

- M = sinkhorn_knopp(I + H_res)
- y_agg[b, c] = sum_i H_pre[i] * x_expanded[b, i, c]
- y_norm[b, c] = rms_norm(y_agg[b, :], weight, eps)
- y_dist[b, i, c] = H_post[i] * y_norm[b, c]
- x_mixed[b, i, c] = sum_j M[i, j] * x_expanded[b, j, c]
- out = x_mixed + y_dist

Quickstart

1) Create an environment (uv recommended)

    uv venv .venv
    source .venv/bin/activate
    uv pip install -e .

2) Run correctness checks

    python test_correctness.py

3) Run a simple benchmark

    python benchmark.py

    # Example: tune for your shape
    python benchmark.py --B 64 --n 8 --C 2048

Usage

```python
import mlx.core as mx
from mhc import MHCLayer

B, n, C = 2, 4, 256
x = mx.random.normal((B, n, C)).astype(mx.bfloat16)

layer = MHCLayer(n=n, C=C, use_metal=True)
y = layer(x)
mx.eval(y)
print(y.shape)  # (B, n, C)
```

Notes

- MHCLayer defaults to identity-friendly initialization. Pass identity_init=False if you want legacy ones/eye initialization.
- The Metal kernels default to n <= 64 (see `_MAX_N_ALLOWED` in `mhc_mlx/metal.py`). If you need larger n, raise that limit and re-run tests.
- Use `python benchmark.py` to tune `threads_per_group` for your target shape; the best value depends on C and n. You can also pass `threads_per_group=None` to use a simple heuristic.
- The Metal path is intended for inference and benchmarking. For training, start with the reference path so gradients are well-behaved and easy to debug.
- The first run will include Metal JIT compilation overhead.

Files

- mhc_mlx/reference.py: pure-MLX implementation of sinkhorn, aggregate, rmsnorm, distribute, and a reference stream mix
- mhc_mlx/metal.py: kernel builder and Metal Sinkhorn + fused forward wrappers
- mhc_mlx/layer.py: MHCLayer module (reference or Metal path)
- kernels/sinkhorn_knopp.metal: Sinkhorn-Knopp projection kernel body
- kernels/mhc_fused.metal: fused aggregate + RMSNorm + mix + add kernel body
- kernels/stream_mix_add.metal: legacy fused stream_mix + add(y_dist) kernel body
- test_correctness.py: reference vs Metal comparisons
- benchmark.py: tiny microbenchmark
