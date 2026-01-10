import argparse
import math
import time

import mlx.core as mx

from mhc_mlx.layer import MHCLayer
from mhc_mlx.reference import activate_pre_post, mixing_matrix_from_logits

from mlx_mhc import ManifoldHyperConnection


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _summarize(times: list[float]) -> dict:
    values = sorted(times)
    return {
        "time_s_median": _percentile(values, 50.0),
        "time_s_p10": _percentile(values, 10.0),
        "time_s_p90": _percentile(values, 90.0),
    }


def _bench_loop(f, x, iters: int, mode: str, queue_guard: int) -> float:
    t0 = time.perf_counter()
    if mode == "latency":
        for _ in range(iters):
            y = f(x)
            mx.eval(y)
        mx.synchronize()
    else:
        for i in range(iters):
            y = f(x)
            mx.async_eval(y)
            if queue_guard and (i + 1) % queue_guard == 0:
                mx.synchronize()
        mx.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def bench_repeat(
    fn,
    x,
    iters: int,
    warmup: int,
    repeats: int,
    mode: str,
    queue_guard: int,
) -> list[float]:
    mx.eval(x)
    mx.synchronize()

    for _ in range(warmup):
        y = fn(x)
        mx.eval(y)
    mx.synchronize()

    return [_bench_loop(fn, x, iters, mode, queue_guard) for _ in range(repeats)]


def _max_abs(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a - b)).item())


def _max_abs_val(a: mx.array) -> float:
    return float(mx.max(mx.abs(a)).item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare mhc_mlx vs mlx-mhc.")
    parser.add_argument("--B", type=int, default=8)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--C", type=int, default=2048)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--mode", type=str, choices=["throughput", "latency"], default="throughput")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--queue-guard", type=int, default=50)
    parser.add_argument("--sinkhorn-iters", type=int, default=20)
    parser.add_argument("--mix-kernel", type=str, default="auto")
    parser.add_argument("--use-metal", action="store_true")
    parser.add_argument("--layer-output", type=str, choices=["zeros", "input"], default="zeros")
    args = parser.parse_args()

    dtype_map = {
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
        "float32": mx.float32,
    }
    if args.dtype not in dtype_map:
        raise ValueError("dtype must be float16, bfloat16, or float32")
    dtype = dtype_map[args.dtype]

    B = int(args.B)
    n = int(args.n)
    C = int(args.C)
    dims = n * C

    x = mx.random.normal((B, n, C)).astype(dtype)
    x_flat = x.reshape(B, 1, dims)

    H_pre_raw = mx.random.normal((n,)).astype(mx.float32)
    H_post_raw = mx.random.normal((n,)).astype(mx.float32)
    H_res_raw = mx.random.normal((n, n)).astype(mx.float32)
    rms_weight = mx.ones((C,), dtype=mx.float32)

    ours = MHCLayer(
        n=n,
        C=C,
        use_metal=bool(args.use_metal),
        threads_per_group=256,
        dispatch_policy="auto",
        mix_kernel=args.mix_kernel,
    )
    ours.H_pre_raw = H_pre_raw
    ours.H_post_raw = H_post_raw
    ours.H_res_raw = H_res_raw
    ours.rms_weight = rms_weight

    theirs = ManifoldHyperConnection(dims=dims, expansion=n, sinkhorn_iterations=args.sinkhorn_iters)
    theirs.h_pre_raw = H_pre_raw
    theirs.h_pre_bias = mx.zeros_like(H_pre_raw)
    theirs.h_post_raw = H_post_raw
    theirs.h_post_bias = mx.zeros_like(H_post_raw)
    theirs.h_res_raw = H_res_raw

    # Component comparisons
    H_pre_act, H_post_act = activate_pre_post(H_pre_raw, H_post_raw)
    H_pre_ref = theirs._project_h_pre()
    H_post_ref = theirs._project_h_post()

    M_ours = mixing_matrix_from_logits(H_res_raw, iters=args.sinkhorn_iters, eps=1e-5)
    M_ref = theirs._project_h_res()

    mx.eval(H_pre_act, H_post_act, H_pre_ref, H_post_ref, M_ours, M_ref)
    mx.synchronize()

    print("== Component diffs ==")
    print(f"H_pre max_abs={_max_abs(H_pre_act, H_pre_ref):.3e}")
    print(f"H_post max_abs={_max_abs(H_post_act, H_post_ref):.3e}")
    print(f"M max_abs={_max_abs(M_ours, M_ref):.3e}")

    # Forward outputs (note: architectures differ, so expect mismatch)
    if args.layer_output == "input":
        layer_out = x_flat
    else:
        layer_out = mx.zeros_like(x_flat)

    out_ours = ours(x)
    out_theirs = theirs(x_flat, layer_out)
    out_ours_flat = out_ours.reshape(B, 1, dims)

    mx.eval(out_ours, out_theirs, out_ours_flat)
    mx.synchronize()

    print("== Output diff (not expected to match) ==")
    max_abs = _max_abs(out_ours_flat, out_theirs)
    denom = _max_abs_val(out_theirs)
    rel = max_abs / denom if denom else float("inf")
    print(f"out max_abs={max_abs:.3e} rel={rel:.3e}")

    # Forward timings
    ours_times = bench_repeat(
        lambda _x: ours(_x),
        x,
        iters=args.iters,
        warmup=args.warmup,
        repeats=args.repeats,
        mode=args.mode,
        queue_guard=args.queue_guard,
    )
    theirs_times = bench_repeat(
        lambda _x: theirs(_x.reshape(B, 1, dims), layer_out),
        x,
        iters=args.iters,
        warmup=args.warmup,
        repeats=args.repeats,
        mode=args.mode,
        queue_guard=args.queue_guard,
    )

    ours_summary = _summarize(ours_times)
    theirs_summary = _summarize(theirs_times)
    speedup = theirs_summary["time_s_median"] / ours_summary["time_s_median"]

    print("== Forward timing ==")
    print(f"ours median={ours_summary['time_s_median']:.6e}s")
    print(f"theirs median={theirs_summary['time_s_median']:.6e}s")
    print(f"speedup (theirs/ours)={speedup:.3f}x")


if __name__ == "__main__":
    main()
