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

### Option 1: Drop-in Layer
Use `MHCLayer` as a replacement for standard residual blocks.

```python
import mlx.core as mx
from mhc_mlx import MHCLayer

# Initialize layer
layer = MHCLayer(n=32, C=64) # 32 streams, 64 channels each

# Forward pass
x = mx.random.normal((1, 32, 64))
y = layer(x)
```

### Option 2: Universal Wrapper (MHCRewire)
Enhance **any** existing MLX module with manifold-constrained stability.

```python
import mlx.nn as nn
from mhc_mlx import MHCRewire

# Wrap a standard Linear layer (or a whole Transformer block)
layer = MHCRewire(nn.Linear(512, 512), n=16)

x = mx.random.normal((1, 512))
y = layer(x) # Computes: H_post * (Linear(H_pre * x) + M * H_pre * x)
```

**Note:** You can also import as `mlx_mhc` if you prefer the style of other community packages:
```python
from mlx_mhc import MHCLayer
```

## Performance

We benchmarked on an Apple M4 Pro (macOS 15.6). `mhc-mlx` automatically selects the best kernel strategy based on workload size.

### Head-to-Head: mhc-mlx vs mlx-mhc

| Scenario | mhc-mlx | mlx-mhc | Speedup |
|---|---|---|---|
| **Latency** ($B=1, C=512$) | **456.67 us** | 966.17 us | **2.12x** |
| **Throughput** ($B=1, C=512$) | **85.56 us** | 804.49 us | **9.40x** |
| **Latency** ($B=32, C=2048$) | **575.46 us** | 1278.92 us | **2.22x** |
| **Throughput** ($B=32, C=2048$) | **249.43 us** | 1104.45 us | **4.43x** |

### Why We're Faster

| Implementation | Characteristics | Performance Impact |
|---|---|---|
| **Python / JIT** | Many small kernel launches | Higher overhead, low occupancy |
| **Fused Metal** | 1-3 highly optimized kernels | Minimal overhead, maximum bandwidth |

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

*(Benchmarks run with bfloat16. Reproduction: `PYTHONPATH=. python compare_mhc.py`)*

## Key Optimizations

- **Fully Fused Kernel:** Single kernel for Aggregate + RMS + Mix + Add. Ideal for $B \times C \le 2048$.
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

## Development & Publishing

**Workflow Name:** For PyPI Trusted Publishing, the workflow filename is `publish.yml`.

## License

MIT
