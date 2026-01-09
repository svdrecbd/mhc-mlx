# mhc-mlx

Unofficial MLX + Metal implementation of mHC: Manifold-Constrained Hyper-Connections by DeepSeek-AI.

## Installation

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .

# or
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Build

No manual build step. Metal kernels JIT-compile on first use. To pre-warm:

```bash
python test_correctness.py
```

## Test

```bash
python test_correctness.py
```

## Benchmark

```bash
# End-to-end layer (auto-dispatch)
python benchmark.py --metal-dispatch auto

# Include backward
python benchmark.py --with-backward --metal-dispatch auto

# Disable fused backward (use separate kernels)
python benchmark.py --with-backward --no-fused-backward

# Optional: reduce sync overhead variance for latency mode
MLX_METAL_FAST_SYNCH=1 python benchmark.py --mode latency

# Summarize and plot
python scripts/summarize_benchmarks.py --in results.jsonl
python scripts/plot_benchmark_speedup.py --summary summary_by_C.csv
```

### MLX Benchmark Results (Apple M4 Pro)

Auto-dispatch benchmark (speedup = reference / Metal, >1 is faster):

- Chip: Apple M4 Pro, macOS 15.6.1, MLX 0.30.0, device gpu
- Sweep: B={1,8}, n={4,8,16,32}, C={512,1024,2048,4096}, dtype=float16,float32
- Settings: iters=100, warmup=10, repeats=3, queue_guard=50, hybrid_latency=on, fused_backward=on, with_backward=on
- Backward compiled: off (benchmark disables mx.compile for backward when fused_backward=on)
- Latency corner (B=1, n=32, C>=1024): reference fallback (by construction)
- Results are hardware-specific; rerun on your machine for final numbers.

End-to-end MHCLayer (auto-dispatch, median speedup with p10-p90):

| Mode       | Forward | Backward |
|------------|---------|----------|
| Throughput | 2.44x (0.94-7.08) | 3.95x (1.05-11.29) |
| Latency    | 0.94x (0.53-1.86) | 1.45x (0.74-3.55) |

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

## Notes

- Auto-dispatch uses Metal for n <= 16 and falls back to the compiled reference path for n == 32, B == 1 (latency-sensitive). Set `hybrid_latency=False` to force the fused Metal path.
- Backward uses Metal kernels (no reference VJPs). Use `--no-fused-backward` if you want backward compatible with `mx.compile`.
- MHCLayer defaults to identity-friendly initialization under exp-parameterization (off-diagonal logits ~ -12). Pass identity_init=False for zero-init logits.
- Metal kernels default to n <= 64 (see `_MAX_N_ALLOWED` in `mhc_mlx/metal.py`). Raise the limit and rerun tests if needed.
- The first run includes Metal JIT compilation overhead.

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
