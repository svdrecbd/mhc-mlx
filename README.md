# mhc-mlx

**High-performance MLX implementation of Manifold-Constrained Hyper-Connections (mHC).**

mHC improves training stability and performance in deep architectures by constraining residual connections to the Birkhoff polytope (doubly stochastic matrices). This library provides optimized Metal kernels for Apple Silicon and a fast compiled fallback for other platforms.

**Original Paper:** [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880) (DeepSeek-AI)

## Installation

```bash
pip install mhc-mlx
```

## Compatibility
- **Primary:** macOS + Apple Silicon for peak performance.
- **Support:** Linux (CPU/CUDA) and Intel Macs via automatic pure-MLX compiled fallback.
- **Software:** MLX >= 0.30.0.

## Quick Start (30-second Demo)

```python
import mlx.core as mx
import mlx.nn as nn
from mhc_mlx import MHCRewire

# 1. Take any standard MLX layer
layer = nn.Linear(2048, 2048)

# 2. Wrap it with mHC stability (automatically uses optimized Metal kernels)
model = MHCRewire(layer, dims=2048, n=32)

# 3. Run forward pass
x = mx.random.normal((1, 2048))
y = model(x)
mx.eval(y)

# 4. Run backward pass (fully vectorized)
loss_fn = lambda m, x: mx.sum(m(x))
grads = mx.grad(loss_fn)(model, x)
mx.eval(grads)

print(f"Output shape: {y.shape}") # (1, 2048)
```

*Note: You can also use `from mlx_mhc import MHCRewire` for a community-friendly alias.*

## Performance

`mhc-mlx` utilizes fused Metal kernels to minimize memory bandwidth bottlenecks. We benchmarked on an Apple M4 Pro (macOS 15.6).

### Comparative Benchmarks

Comparison with a standard MLX implementation of mHC ($C=512$):

| Metric | mhc-mlx | Baseline Impl | Speedup |
|---|---|---|---|
| **Inference Latency** ($B=1$) | **392 us** | 1120 us | **2.86x** |
| **Training Throughput** ($B=32$) | **105 us** | 866 us | **8.25x** |

### Why It's Faster

| Approach | Architecture | Impact |
|---|---|---|
| **Baseline** | Multiple kernel launches | High memory overhead, low GPU occupancy |
| **mhc-mlx** | Fused Metal Kernels | Minimal memory round-trips, maximal bandwidth |

### Reproduce Benchmarks
Run the standardized benchmark suite on your own hardware:
```bash
mhc-mlx-bench --mode latency --C 512,2048,4096
```

## Key Optimizations

- **Universal Rewiring:** `MHCRewire` wraps any existing `nn.Module` (Linear, Conv2d) to apply mHC dynamics.
- **Quantized Layer Support:** Seamlessly wraps `nn.QuantizedLinear` (4-bit/8-bit).
- **Fully Fused Kernel:** Single-pass kernel for Aggregate + RMS + Mix + Add.
- **Adaptive Dispatch:** Runtime heuristic selects the fastest kernel strategy for your workload.

## Diagnostics
If you encounter issues, run the diagnostic utility:
```bash
mhc-mlx-info
```

Set `MHC_MLX_DISABLE_METAL=1` in your environment to force the pure-MLX reference path (useful for debugging or non-Metal hardware).

## Support Policy
- **Tested:** macOS (Apple Silicon) + Linux (CPU/CUDA) using MLX 0.30.0+.
- **Best Effort:** Intel Macs, older macOS versions, and older MLX versions.
- **Reporting:** Please include OS, MLX version, and `mhc-mlx-info` output in bug reports.

## License
MIT
