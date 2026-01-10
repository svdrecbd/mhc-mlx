# mhc-mlx

**High-performance MLX implementation of Manifold-Constrained Hyper-Connections (mHC)** for Apple Silicon.

This library provides a drop-in `MHCLayer` that fuses multiple operations into optimized Metal kernels, achieving **>1.5x speedup** over standard implementations for low-latency inference.

## Installation

Install from PyPI:

```bash
pip install mhc-mlx
```

## Quick Start

```python
import mlx.core as mx
from mhc_mlx import MHCLayer

# Create some dummy data
B, n, C = 1, 32, 2048
x = mx.random.normal((B, n, C)).astype(mx.bfloat16)

# Initialize layer (uses Metal kernels by default)
layer = MHCLayer(n=n, C=C)

# Forward pass
y = layer(x)
mx.eval(y)
print(y.shape)  # (1, 32, 2048)
```

**Note:** You can also import as `mlx_mhc` if you prefer the style of other community packages:
```python
from mlx_mhc import MHCLayer
```

## Performance

We benchmarked on an Apple M4 Pro (macOS 15.6). `mhc-mlx` automatically selects the best kernel strategy based on workload size.

| Channels (C) | Kernel Strategy | Speedup vs Compiled MLX |
|---|---|---|
| 256 | Fully Fused | **2.27x** |
| 1024 | Fully Fused | **1.57x** |
| 2048 | Fully Fused | **1.58x** |
| 4096 | Column Parallel | **1.41x** |
| 8192 | Column Parallel | **2.18x** |

*(Benchmarks run with Batch Size=1, Sequence Length=32, bfloat16)*

To reproduce:
```bash
mhc-mlx-bench --mode latency
```

## Key Optimizations

- **Fully Fused Kernel:** Single kernel for Aggregate + RMS + Mix + Add. Ideal for $C \le 2048$.
- **Column-Parallel Mixing:** Vectorized kernel maximizing throughput for $C > 2048$.
- **Adaptive Dispatch:** Runtime heuristic selects the fastest kernel.
- **Super-Fused Backward:** Fused gradients for maximum training efficiency.

## Troubleshooting

**Kernel Compilation Errors:**
If you see Metal build errors, ensure you are on macOS with Apple Silicon.
Run diagnostics to check your environment:
```bash
mhc-mlx-info
```

**"No module named mhc_mlx"**
Ensure you installed the package `mhc-mlx` (dash), not `mlx-mhc` (which is a different package).

## License

MIT
