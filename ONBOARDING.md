# ONBOARDING

This repo exists for one reason: build a clean, correct, and testable mHC port on Apple silicon.

## Working Contract

- The pure-MLX reference path is the source of truth.
- The Metal path must match the reference numerically (within a tight tolerance) for the same inputs.
- If the Metal path diverges, fix the Metal code first. If the reference is wrong, fix both and add a regression test.

## Semantics

We follow the forward semantics used in this repo:

- The input to this layer is already expanded into streams.
- The stream mixing matrix is constructed by applying Sinkhorn-Knopp normalization to exp(H_res_raw).
- There is an aggregate -> RMSNorm -> distribute branch that produces a per-stream additive term.
- The output is stream_mix(x_expanded, M) + y_dist.
- Identity-friendly init uses off-diagonal H_res_raw logits around -12 and H_pre_raw/H_post_raw around -12.

Equations with x_expanded in R^{B x n x C}:

```
H_pre_act = sigmoid(H_pre_raw)
H_post_act = 2 * sigmoid(H_post_raw)
M = sinkhorn_knopp(exp(H_res_raw)) where H_res_raw in R^{n x n}
y_agg[b, c] = sum_i H_pre_act[i] * x_expanded[b, i, c]
y_norm[b, c] = rms_norm(y_agg[b, :], weight, eps)
y_dist[b, i, c] = H_post_act[i] * y_norm[b, c]
x_mixed[b, i, c] = sum_j M[i, j] * x_expanded[b, j, c]
out = x_mixed + y_dist
```

## Repo Layout

- `mhc_mlx/reference.py`
  - sinkhorn_knopp
  - stream_aggregate
  - rms_norm
  - stream_distribute
  - stream_mix_ref

- `kernels/sinkhorn_knopp.metal`
  - Metal kernel body that projects exp(H_res_raw) onto the Birkhoff polytope

- `kernels/sinkhorn_knopp_backward.metal`
  - Metal kernel body for Sinkhorn-Knopp backward (dH_res from dM)

- `kernels/mhc_fused.metal`
  - Metal kernel body that fuses:
    - stream aggregate + RMSNorm
    - stream mix: out[b,i,c] = sum_j M[i,j] * x[b,j,c]
    - add: out += H_post_act[i] * y_norm[b,c]
  - Includes an unrolled fast path for n=4

- `kernels/stream_mix_add.metal`
  - Metal kernel body that fuses stream mix + add(y_dist) for optional hybrid experiments

- `kernels/mhc_backward_*.metal`
  - Metal kernel bodies for fused backward (prep, dx, dM, dH_pre, dH_post, d_rms_weight)

- `kernels/stream_mix_backward_dx.metal`
  - Metal kernel body for stream-mix backward (dx)

- `mhc_mlx/metal.py`
  - Builds and calls custom Metal kernels using mlx.core.fast.metal_kernel

- `mhc_mlx/layer.py`
  - MHCLayer module that can run either:
    - reference path (use_metal=False)
    - metal path (use_metal=True)

- `test_correctness.py`
  - Runs reference vs Metal checks and prints max error

- `benchmark.py`
  - Benchmark suite with correctness checks and JSONL output

## Installation

We assume you are on Apple silicon and have a recent macOS.

Using uv:

```
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

Using pip:

```
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Correctness Workflow

1. Start with reference only
   - Run `python test_correctness.py` with use_metal=False inside the script.
   - Confirm shapes and dtypes match what you expect.

2. Turn on Metal
   - Run `python test_correctness.py` with use_metal=True.
   - If it fails, reduce the test to the smallest case:
     - B=1, n=4, C=8
     - random but deterministic inputs (seeded)

3. Lock in regression
   - Whenever you fix a bug, add a small regression case to test_correctness.py.

## Performance Workflow

These Metal kernels are not trying to beat a vendor-tuned GEMM.
They are trying to remove Python and intermediate tensor overhead for the RMSNorm + mixing step, and to keep memory traffic down by fusing the final add.

To measure something meaningful:

- Keep n small (4, 8, or 16 are tested).
- Benchmark with C in the same ballpark as your real model hidden size.
- Use large enough B to amortize launch overhead.
- Use throughput mode for steady-state speed and latency mode for per-call cost.
- Override threads_per_group if you want a fixed launch size; otherwise a heuristic is used.
- Use repeats + p10/p90 to avoid outliers, and queue-guard to avoid enqueue-only timing.
- Use `--metal-dispatch auto` to benchmark the default auto-dispatch behavior.
- Use `--with-backward` to time gradient computation.

Run:

```
python benchmark.py
```

By default this runs both throughput and latency modes and writes one JSON dict per line to results.jsonl.

Summarize and plot:

```
python scripts/summarize_benchmarks.py --in results.jsonl
python scripts/plot_benchmark_speedup.py --summary summary_by_C.csv
```

Tip: for latency mode, `MLX_METAL_FAST_SYNCH=1` can reduce sync overhead variance.

If you want to see the generated Metal source for debugging:

- Pass verbose=True when calling the kernel (see mhc_mlx/metal.py). This prints the kernel body source used for compilation.

## Training vs Inference

- For training, use the reference path first to validate numerics.
  The Metal path exposes gradients via Metal backward kernels (no reference VJPs).

- For inference, use_metal=True is fine and is the intended use. Auto-dispatch defaults to Metal for n <= 16 and falls back to the compiled reference path for n == 32, B == 1 (latency-sensitive). Set hybrid_latency=False to force the fused Metal path.

## Extending This Repo

Safe extensions that keep the repo coherent:

1) Keep the same semantics and fuse more

- Consider fusing additional per-stream logic if you add new branches.
- Keep n small unless you are willing to rework the kernels and threadgroup memory strategy.

2) Generalize n cleanly

- Today the Metal kernels default to n <= 64 (see `_MAX_N_ALLOWED` in `mhc_mlx/metal.py`).
- To increase it, bump `_MAX_N_ALLOWED` and re-run correctness tests.

3) Add a transformer-friendly wrapper

If you want to plug into a [B, S, D] transformer:

- reshape x to [B*S, n, C] where D = n*C
- call MHCLayer
- reshape back

## Ground Rules for PRs

- Never remove the reference implementation.
- Never change the Metal kernel without updating tests.
- Keep kernel interfaces simple: row-contiguous inputs, explicit shapes, no shape tricks.
