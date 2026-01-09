# mhc-mlx

MLX implementation of mHC (Manifold-Constrained Hyper-Connections) with:

- a pure-MLX reference path (source of truth)
- Metal kernels for Sinkhorn projection and fused RMSNorm + mix + add

## Forward Pass

Forward semantics used throughout this repo:

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

## Quickstart

1. Create an environment (uv recommended)

```
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

2. Run correctness checks

```
python test_correctness.py
```

3. Run a benchmark

```
python benchmark.py
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

## Benchmarking

```
# Sweep shapes and dtypes, write JSONL results (throughput + latency by default)
python benchmark.py --B 1,8 --n 4,8,16 --C 512,1024 --dtypes bfloat16,float32 --out results.jsonl

# Throughput (async) vs latency (sync) modes
python benchmark.py --modes throughput,latency
python benchmark.py --mode throughput
python benchmark.py --mode latency

# Override threads_per_group (default uses a heuristic)
python benchmark.py --threads-per-group 128

# More stable timings (median + p10/p90 with queue guard)
python benchmark.py --repeats 5 --queue-guard 50

# Measure default auto-dispatch behavior (includes latency fallback)
python benchmark.py --metal-dispatch auto

# Include backward benchmarks (gradients)
python benchmark.py --with-backward --metal-dispatch auto

# Optional: reduce sync overhead variance for latency mode
MLX_METAL_FAST_SYNCH=1 python benchmark.py --mode latency

# Summarize and plot results
python scripts/summarize_benchmarks.py --in results.jsonl
# If you have separate files, pass a comma list
python scripts/summarize_benchmarks.py --in results.jsonl,results_latency.jsonl
# Speedup is reference / metal (higher is faster)
python scripts/plot_benchmark_speedup.py --summary summary_by_C.csv
```

## Latest Results

Auto-dispatch benchmark (speedup = reference / Metal, >1 is faster):

- Chip: Apple M4 Pro, macOS 15.6.1, MLX 0.30.0, device gpu
- Sweep: B={1,8}, n={4,8,16,32}, C={512,1024,2048}, dtype=bfloat16
- Settings: iters=100, warmup=10, repeats=3, queue_guard=50, hybrid_latency=on, with_backward=on
- Latency corner (B=1, n=32): reference fallback, ~0.92-1.03x (by construction)
- Results are hardware-specific; rerun on your machine for final numbers.

| Mode       | Sinkhorn speedup | Fused speedup | Layer speedup | Layer backward speedup |
|------------|------------------|---------------|---------------|------------------------|
| Throughput | 2.62 (0.88-6.74) | 2.80 (1.12-3.35) | 2.62 (0.90-7.06) | 1.99 (0.90-5.37) |
| Latency    | 0.69 (0.28-1.53) | 1.39 (0.80-1.57) | 0.94 (0.50-1.70) | 1.01 (0.51-1.76) |

## Notes

- MHCLayer defaults to identity-friendly initialization under exp-parameterization (off-diagonal logits ~ -12). Pass identity_init=False for zero-init logits.
- Metal kernels default to n <= 64 (see `_MAX_N_ALLOWED` in `mhc_mlx/metal.py`). Raise the limit and rerun tests if needed.
- `benchmark.py` writes one JSON dict per line to results.jsonl with median/p10/p90 timings; include it in your reports.
- Auto-dispatch uses Metal for n <= 16 and falls back to the compiled reference path for n == 32, B == 1 (latency-sensitive). Set `hybrid_latency=False` to force the fused Metal path.
- Backward uses Metal kernels (no reference VJPs).
- Training is supported with Metal forward/backward; validate with the reference path first for new research.
- The first run includes Metal JIT compilation overhead.

## Files

- `mhc_mlx/reference.py`: pure-MLX implementation of Sinkhorn, aggregate, RMSNorm, distribute, and a reference stream mix
- `mhc_mlx/metal.py`: kernel builder and Metal Sinkhorn + fused forward wrappers
- `mhc_mlx/layer.py`: MHCLayer module (reference or Metal path)
- `kernels/sinkhorn_knopp.metal`: Sinkhorn-Knopp projection kernel body
- `kernels/sinkhorn_knopp_backward.metal`: Sinkhorn-Knopp backward kernel body
- `kernels/mhc_fused.metal`: fused aggregate + RMSNorm + mix + add kernel body
- `kernels/stream_mix_add.metal`: stream mix + add(y_dist) kernel body (optional hybrid experiments)
- `kernels/mhc_backward_*.metal`: fused backward kernel bodies (prep, dx, dM, dH_pre, dH_post, d_rms_weight)
- `kernels/stream_mix_backward_dx.metal`: stream-mix backward (dx) kernel body
- `test_correctness.py`: reference vs Metal comparisons
- `benchmark.py`: benchmark suite with correctness checks and JSONL output

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
