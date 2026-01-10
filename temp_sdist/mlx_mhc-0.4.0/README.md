# mlx-mhc

First MLX implementation of DeepSeek's **Manifold-Constrained Hyper-Connections (mHC)** for Apple Silicon.

Based on: [arXiv:2512.24880](https://arxiv.org/abs/2512.24880)

## Installation

```bash
pip install mlx-mhc
```

## Quick Start

```python
import mlx.core as mx
import mlx_mhc as mhc

# Sinkhorn-Knopp projection to doubly stochastic matrix
matrix = mx.random.normal((8, 8))
doubly_stochastic = mhc.sinkhorn_knopp(matrix)

# Manifold Hyper-Connection module
connection = mhc.ManifoldHyperConnection(dims=512, expansion=2)
output = connection(x, layer_output)
```

## What is mHC?

mHC (Manifold-Constrained Hyper-Connections) improves training stability for large language models by constraining residual connection mixing matrices to the Birkhoff polytope (doubly stochastic matrices).

Key benefits:
- Prevents gradient explosion in deep networks
- Maintains identity mapping property
- 2.1% improvement on benchmarks with only 6.7% overhead

## API

### `sinkhorn_knopp(matrix, max_iterations=100, epsilon=1e-6, log_space=True)`

Project a matrix onto the Birkhoff polytope (set of doubly stochastic matrices).

### `ManifoldHyperConnection(dims, expansion=2, sinkhorn_iterations=10)`

MLX module implementing mHC for transformer residual connections.

## License

MIT
