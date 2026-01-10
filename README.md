# mhc-mlx

Unofficial MLX + Metal implementation of mHC (Manifold-Constrained Hyper-Connections) by DeepSeek-AI.

## What is mHC?

mHC adds manifold-constrained residual mixing across multiple streams. This repo provides a faithful MLX reference path plus Metal kernels for fast Apple-silicon execution.

## Installation

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# or
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```python
import mlx.core as mx
from mhc import MHCLayer

B, n, C = 2, 4, 256
x = mx.random.normal((B, n, C)).astype(mx.bfloat16)

layer = MHCLayer(n=n, C=C, use_metal=True)
y = layer(x)
mx.eval(y)
print(y.shape)  # (B, n, C)
```

## Correctness

Run the default correctness suite (also warms Metal JIT caches):

```bash
python run_correctness.py
```

Run the full pytest suite:

```bash
python -m pytest -m "not stress"
```

Stress tests (eps extremes, large C, seed sweep):

```bash
MHC_MLX_RUN_STRESS=1 python -m pytest -m stress
```

## Dispatch policies

- `dispatch_policy="auto"`: always use fused Metal.
- `dispatch_policy="latency"`: avoid fused Metal in latency corner cases (B=1 and/or n=32 small C).
- `dispatch_policy="throughput"`: allow fused Metal for n=32 only when it wins.
- `--metal-dispatch force`: benchmark flag to always use fused Metal.
- `hybrid_latency=True`: opt-in hybrid path for the latency corner when C is large.

## Benchmark

### Run

```bash
# End-to-end layer (auto-dispatch)
python benchmark.py --metal-dispatch auto

# Include backward
python benchmark.py --with-backward --metal-dispatch auto

# Latency policy (guardrails) + hybrid option
python benchmark.py --mode latency --dispatch-policy latency --hybrid-latency --hybrid-min-C 4096

# Safe throughput policy for n=32
python benchmark.py --mode throughput --dispatch-policy throughput

# Optional: reduce sync overhead variance in latency mode
MLX_METAL_FAST_SYNCH=1 python benchmark.py --mode latency

# Summarize and plot
python scripts/summarize_benchmarks.py --in results.jsonl
python scripts/plot_benchmark_speedup.py --summary summary_by_C.csv
```

### Results (Apple M4 Pro)

Speedup = reference / Metal (higher is faster), reported as median (p10-p90):

- Chip: Apple M4 Pro, macOS 15.6.1, MLX 0.30.0, device gpu
- Sweep: B={1,8,32}, n={4,8,16,32}, C={256,512,1024,2048,4096}, dtype=bfloat16,float16,float32
- Settings: iters=200, warmup=10, repeats=3, queue_guard=50, dispatch_policy=auto, hybrid_latency=off, output_dtype=none
- Backward compiled: on

Overall summary (median [p10-p90]):

| Benchmark      | Throughput | Latency |
|---------------|------------|---------|
| sinkhorn      | 19.16x (9.05-25.18) | 3.81x (2.14-5.01) |
| fused         | 1.83x (1.61-5.18) | 1.48x (1.24-4.42) |
| layer         | 10.58x (5.39-11.80) | 3.82x (2.58-4.44) |
| layer_backward| 11.93x (4.63-13.37) | 3.81x (2.60-4.95) |

Layer speedup by n (median [p10-p90]):

| n  | Throughput forward | Throughput backward | Latency forward | Latency backward |
|----|--------------------|---------------------|----------------|-----------------|
| 4  | 10.86x (9.89-11.61) | 12.71x (12.00-13.56) | 4.32x (3.99-4.58) | 4.60x (4.06-5.39) |
| 8  | 11.25x (10.01-12.14) | 12.88x (7.75-13.64) | 4.00x (3.74-4.30) | 4.22x (3.60-4.85) |
| 16 | 10.66x (5.61-11.76) | 10.42x (4.65-12.44) | 3.37x (3.14-3.88) | 3.55x (3.26-3.87) |
| 32 | 5.68x (4.90-8.81) | 5.06x (4.46-6.75) | 2.69x (2.36-4.30) | 2.70x (2.10-3.80) |

Layer speedup by dtype (median [p10-p90]):

| dtype    | Throughput forward | Throughput backward | Latency forward | Latency backward |
|----------|--------------------|---------------------|----------------|-----------------|
| float16  | 10.80x (5.43-11.97) | 12.02x (4.69-13.16) | 3.97x (2.58-4.55) | 3.72x (2.58-4.88) |
| bfloat16 | 9.97x (5.49-11.35) | 11.77x (4.62-13.40) | 3.80x (2.63-4.33) | 3.79x (2.65-4.75) |
| float32  | 10.79x (5.15-12.16) | 12.06x (4.68-13.37) | 3.78x (2.59-4.39) | 3.86x (2.65-5.29) |

![Speedup by C](./benchmark_speedup_by_C.png)

## Semantics

Forward equations used throughout this repo:

```
H_pre_act = sigmoid(H_pre_raw)
H_post_act = 2 * sigmoid(H_post_raw)
M = sinkhorn_knopp(exp(H_res_raw))
y_agg[b, c] = sum_i H_pre_act[i] * x_expanded[b, i, c]
y_norm[b, c] = rms_norm(y_agg[b, :], weight, eps)
y_dist[b, i, c] = H_post_act[i] * y_norm[b, c]
x_mixed[b, i, c] = sum_j M[i, j] * x_expanded[b, j, c]
out = x_mixed + y_dist
```

## Kernel cache keying

Metal kernel names include a hash of the rendered source plus a per-machine cache key:

- Set `MHC_MLX_KERNEL_CACHE_KEY` to override the cache key.
- Set `MHC_MLX_KERNEL_CACHE_INCLUDE_DEVICE=0` to omit the GPU name from the key.

## Paper

**mHC: Manifold-Constrained Hyper-Connections**  
https://arxiv.org/abs/2512.24880

DeepSeek-AI

## Citation

```bibtex
@article{xie2025mhc,
  title={mHC: Manifold-Constrained Hyper-Connections},
  author={Xie, Zhenda and Wei, Yixuan and Cao, Huanqi and Zhao, Chenggang and Deng, Chengqi and Li, Jiashi and Dai, Damai and Gao, Huazuo and Chang, Jiang and Zhao, Liang and Zhou, Shangyan and Xu, Zhean and Zhang, Zhengyan and Zeng, Wangding and Hu, Shengding and Wang, Yuqing and Yuan, Jingyang and Wang, Lean and Liang, Wenfeng},
  journal={arXiv preprint arXiv:2512.24880},
  year={2025}
}
```
