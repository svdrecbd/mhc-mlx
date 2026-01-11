# mhc-mlx

**High-performance MLX implementation of Manifold-Constrained Hyper-Connections (mHC)** for Apple Silicon.

This library provides optimized Metal kernels for mHC, achieving massive speedups over compiled reference layers and standard Python-based implementations.

**Original Paper:** [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880) (DeepSeek-AI)

## Installation

Install from PyPI:

```bash
pip install mhc-mlx
```

## Quick Start

### Option 1: Drop-in Layer
Use `MHCLayer` for maximum performance.

```python
import mlx.core as mx
from mhc_mlx import MHCLayer

layer = MHCLayer(n=32, C=64) # 32 streams, 64 channels each
x = mx.random.normal((1, 32, 64))
y = layer(x)
```

### Option 2: Universal Wrapper (MHCRewire)
Enhance **any** existing MLX module with manifold-constrained stability.

```python
import mlx.nn as nn
from mhc_mlx import MHCRewire

# Zero-Cost Folding: automatically optimizes Linear weights
layer = MHCRewire(nn.Linear(512, 512), dims=512, n=16)

x = mx.random.normal((1, 512))
y = layer(x) 
```

## Performance

We benchmarked on an Apple M4 Pro (macOS 15.6). `mhc-mlx` outperforms standard implementations across all scales.

### Head-to-Head: mhc-mlx vs mlx-mhc (Competitor)

| Scenario | mhc-mlx (ours) | mlx-mhc (them) | Speedup |
|---|---|---|---|
| **Latency** ($B=1, C=2048$) | **552.67 us** | 975.08 us | **1.76x** |
| **Throughput** ($B=1, C=2048$) | **148.05 us** | 802.47 us | **5.42x** |
| **Latency** ($B=32, C=2048$) | **581.67 us** | 1310.63 us | **2.25x** |
| **Throughput** ($B=32, C=2048$) | **243.65 us** | 1122.41 us | **4.61x** |

### Why We're Faster

| Implementation | Characteristics | Performance Impact |
|---|---|---|
| **Python / JIT** | Many small kernel launches | Higher overhead, low occupancy |
| **Fused Metal** | 1-3 highly optimized kernels | Minimal overhead, maximum bandwidth |

### Latency Floor ($B=1$, Sequence Length=32)

| Channels (C) | Kernel Strategy | Layer Speedup (vs Compiled MLX) |
|---|---|---|
| 256 | Fully Fused | **2.27x** |
| 1024 | Fully Fused | **1.57x** |
| 2048 | Fully Fused | **1.58x** |
| 4096 | Column Parallel | **1.41x** |
| 8192 | Column Parallel | **2.18x** |

## Key Optimizations

- **"Zero-Cost" Weight Folding:** `MHCRewire` folds scaling directly into `nn.Linear` weights, eliminating pre-scaling overhead.
- **Fully Fused Kernel:** Single-pass kernel for Aggregate + RMS + Mix + Add.
- **Column-Parallel Mixing:** Vectorized kernel maximizing throughput for larger workloads.
- **Adaptive Dispatch:** Runtime heuristic selects the fastest kernel strategy.

## Advanced Usage

### Custom Blocks: Fused Residual Add + Aggregate

If you are building custom Transformer blocks, you can use `residual_add_agg` to fuse the residual connection with the mHC aggregation step. This saves a full memory read/write round-trip (~1.4x speedup).

```python
from mhc_mlx import residual_add_agg

# Standard: x = x + res; y_agg = aggregate(x)
# Fused:
x, y_agg = residual_add_agg(x, res, H_pre)
```

## Troubleshooting

Run diagnostics to check your environment:
```bash
mhc-mlx-info
```

## Development & Publishing

**Workflow Name:** For PyPI Trusted Publishing, use `publish.yml`.

## License

MIT