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

### Option 1: The "Easy Button" (AutoPatcher)
Automatically inject mHC into any existing MLX model (e.g., Llama, Mistral) by detecting compatible layers.

```python
import mlx.nn as nn
from mhc_mlx import AutoPatcher

# Load your model (e.g. from mlx-lm)
model = ... 

# One line to upgrade the entire network
# Automatically wraps all shape-preserving Linear/QuantizedLinear layers
AutoPatcher.patch(model, n_streams=32)
```

### Option 2: Drop-in Layer
Use `MHCLayer` for maximum performance when building custom models.

```python
import mlx.core as mx
from mhc_mlx import MHCLayer

layer = MHCLayer(n=32, C=64) 
x = mx.random.normal((1, 32, 64))
y = layer(x)
```

### Option 3: Universal Wrapper (MHCRewire)
Manually wrap specific modules.

```python
from mhc_mlx import MHCRewire
layer = MHCRewire(nn.Linear(512, 512), dims=512, n=16)
```

## Performance

We benchmarked on an Apple M4 Pro (macOS 15.6). `mhc-mlx` outperforms standard implementations across all scales.

### Head-to-Head: mhc-mlx vs mlx-mhc (Competitor)

| Scenario | mhc-mlx (ours) | mlx-mhc (them) | Speedup |
|---|---|---|---|
| **Latency** ($B=1, C=512$) | **435 us** | 1031 us | **2.37x** |
| **Throughput** ($B=32, C=512$) | **89 us/iter** | 940 us/iter | **10.53x** |
| **Throughput** ($B=32, C=2048$) | **243 us/iter** | 1122 us/iter | **4.61x** |

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

### High Throughput ($B=32$, Sequence Length=32)

Maximum speedups for heavy data processing.

| Operation | Scale (n, C) | Peak Speedup |
|---|---|---|
| **Sinkhorn-Knopp** | n=4 | **26.99x** |
| **Mix + Add (Fused)** | n=32, C=2048 | **14.92x** |
| **Full MHCLayer** | n=4, C=4096 | **17.33x** |

### Training / Backward Pass

Optimized gradients ensure training is as fast as inference.

| Batch Size | Channels (C) | Speedup vs Compiled MLX |
|---|---|---|
| 1 | 2048 | **4.18x** |
| 1 | 4096 | **2.12x** |
| 32 | 2048 | **3.10x** |

*(Benchmarks run with bfloat16)*

## Key Optimizations

- **"Zero-Cost" Weight Folding:** `MHCRewire` folds scaling directly into `nn.Linear` weights, eliminating pre-scaling overhead.
- **Quantized Layer Support:** Seamlessly wraps `nn.QuantizedLinear` (4-bit/8-bit) for efficient local LLM inference.
- **Fully Fused Kernel:** Single kernel for Aggregate + RMS + Mix + Add.
- **Column-Parallel Mixing:** Vectorized kernel maximizing throughput for larger workloads.
- **Adaptive Dispatch:** Runtime heuristic selects the fastest kernel strategy.
- **Super-Fused Backward:** Fused gradients for maximum training efficiency.

## Advanced Usage

### Auto-Tuning for Your Hardware

`mhc-mlx` comes with a JIT kernel auto-tuner that finds the optimal threadgroup sizes for your specific Mac.

```bash
# Run the tuner (takes ~30s)
python scripts/tune.py
```
This generates a `mhc_tuning.json` file. The library will automatically load this config to squeeze out the last 5-10% of performance.

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

## License

MIT
