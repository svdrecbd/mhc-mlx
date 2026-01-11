# mhc-mlx

**High-performance MLX implementation of Manifold-Constrained Hyper-Connections (mHC)** for Apple Silicon.

mHC improves training stability and performance in deep architectures by constraining residual connections to the Birkhoff polytope (doubly stochastic matrices). This library provides optimized Metal kernels that achieve significant speedups over standard baseline implementations.

**Original Paper:** [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880) (DeepSeek-AI)

## Installation

```bash
pip install mhc-mlx
```

## Compatibility
- **Hardware:** Apple Silicon (M1, M2, M3, M4).
- **Software:** macOS, MLX >= 0.30.0.
- **Fallback:** Automatically falls back to a compiled pure-MLX path on other platforms.

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

- **"Zero-Cost" Weight Folding:** `MHCRewire` folds scaling directly into `nn.Linear` weights where possible.
- **Quantized Layer Support:** Seamlessly wraps `nn.QuantizedLinear` (4-bit/8-bit).
- **Fully Fused Kernel:** Single-pass kernel for Aggregate + RMS + Mix + Add.
- **Adaptive Dispatch:** Runtime heuristic selects the fastest kernel strategy for your workload.

## Diagnostics
If you encounter issues, run the diagnostic utility:
```bash
mhc-mlx-info
```

## License
MIT
