# mhc-mlx

**High-performance MLX implementation of Manifold-Constrained Hyper-Connections (mHC)** for Apple Silicon.

This library provides a drop-in `MHCLayer` that fuses multiple operations into optimized Metal kernels, achieving massive speedups over compiled reference layers.

**Original Paper:** [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880) (DeepSeek-AI)

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

### Latency Floor ($B=1$, Sequence Length=32)

Optimized for ultra-low latency response times.

| Channels (C) | Kernel Strategy | Layer Speedup |
|---|---|---|
| 256 | Fully Fused | **2.27x** |
| 1024 | Fully Fused | **1.57x** |
| 2048 | Fully Fused | **1.58x** |
| 4096 | Column Parallel | **1.41x** |
| 8192 | Column Parallel | **2.18x** |

### High Throughput ($B=32$, Sequence Length=32)

Maximum speedups for heavy data processing.

| Operation | Scale (n, C) | Peak Speedup |
|---|---|---|
| **Sinkhorn-Knopp** | n=4 | **26.99x** |
| **Mix + Add (Fused)** | n=32, C=2048 | **14.92x** |
| **Full MHCLayer** | n=4, C=4096 | **17.33x** |

*(Benchmarks run with bfloat16)*

To reproduce:
```bash
mhc-mlx-bench --mode latency
mhc-mlx-bench --mode throughput --B 32
```

## Key Optimizations

- **Fully Fused Kernel:** Single kernel for Aggregate + RMS + Mix + Add. Ideal for $B 	imes C \le 2048$.
- **Column-Parallel Mixing:** Vectorized kernel maximizing throughput for larger workloads.
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

## Development & Publishing

**Workflow Name:** For PyPI Trusted Publishing, the workflow filename is `publish.yml`.

## License

MIT
