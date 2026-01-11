# Contributing to mhc-mlx

We welcome contributions to improve speed, stability, and compatibility!

## Development Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/svdrecbd/mhc-mlx.git
   cd mhc-mlx
   ```

2. Install dependencies:
   ```bash
   pip install -e ".[dev,bench]"
   ```

## Running Tests

Run the full suite:
```bash
python -m pytest
```

Note: Some tests require Apple Silicon to execute the Metal paths.

## Benchmarking

Run the main benchmark:
```bash
PYTHONPATH=. python mhc_mlx/benchmark.py --mode latency
```

## Auto-Tuning

If you've modified kernels, run the tuner:
```bash
PYTHONPATH=. python scripts/tune.py
```
