# mhc-mlx

**High-performance MLX implementation of Manifold-Constrained Hyper-Connections (mHC)** for Apple Silicon.

This library provides a drop-in `MHCLayer` that fuses multiple operations into optimized Metal kernels, achieving massive speedups over compiled reference layers and standard Python-based implementations.

**Original Paper:** [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880) (DeepSeek-AI)

## Installation

Install from PyPI:

```bash
pip install mhc-mlx
```

## Quick Start

### Option 1: Drop-in Layer (Recommended)
Use `MHCLayer` for maximum performance.

```python
import mlx.core as mx
from mhc_mlx import MHCLayer

layer = MHCLayer(n=32, C=64) # 32 streams, 64 channels each
x = mx.random.normal((1, 32, 64))
y = layer(x)
```

### Option 2: Universal Wrapper (MHCRewire)
Enhance **any** existing MLX module (Linear, Conv2d, Transformers) with manifold-constrained stability. *Note: optimizing arbitrary modules incurs some overhead compared to the fused MHCLayer.*

```python
import mlx.nn as nn
from mhc_mlx import MHCRewire

# Wrap a standard Linear layer
layer = MHCRewire(nn.Linear(512, 512), dims=512, n=16)
```

## Performance

We benchmarked on an Apple M4 Pro (macOS 15.6). `mhc-mlx` outperforms standard implementations across all scales.

### Head-to-Head: mhc-mlx vs mlx-mhc (Competitor)

| Scenario | mhc-mlx (ours) | mlx-mhc (them) | Speedup |
|---|---|---|---|
| **Latency** ($B=1, C=512$) | **392 us** | 1120 us | **2.86x** |
| **Throughput** ($B=32, C=512$) | **105 us** | 866 us | **8.25x** |

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

- **Fully Fused Kernel:** Single kernel for Aggregate + RMS + Mix + Add.
- **Column-Parallel Mixing:** Vectorized kernel maximizing throughput for larger workloads.
- **Adaptive Dispatch:** Runtime heuristic selects the fastest kernel strategy.
- **Super-Fused Backward:** Fused gradients for maximum training efficiency.

## Troubleshooting

Run diagnostics to check your environment:
```bash
mhc-mlx-info
```

## License

MIT
