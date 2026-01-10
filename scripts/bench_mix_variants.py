import argparse
import json
import math
import os
import platform
import subprocess
import time
from importlib import metadata as importlib_metadata

import mlx.core as mx

from mhc_mlx.metal import mhc_forward_fused_metal, mix_add_rms_threadgroup_size, suggest_threads_per_group
from mhc_mlx.reference import activate_pre_post


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


def _make_runner(fn, compiled: bool):
    compile_fn = getattr(mx, "compile", None)
    if compiled and callable(compile_fn):
        return compile_fn(lambda _x: fn(_x))
    return lambda _x: fn(_x)


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
    compiled: bool,
    mode: str,
    queue_guard: int,
) -> list[float]:
    mx.eval(x)
    mx.synchronize()

    f = _make_runner(fn, compiled)

    for _ in range(warmup):
        y = f(x)
        mx.eval(y)
    mx.synchronize()

    return [_bench_loop(f, x, iters, mode, queue_guard) for _ in range(repeats)]


def _parse_int_list(value: str) -> list[int]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise ValueError("list must be a comma-separated list of ints")
    return [int(p) for p in parts]


def _parse_dtype_list(value: str) -> list[mx.Dtype]:
    lookup = {
        "float16": mx.float16,
        "fp16": mx.float16,
        "half": mx.float16,
        "bfloat16": mx.bfloat16,
        "bf16": mx.bfloat16,
        "float32": mx.float32,
        "fp32": mx.float32,
    }
    parts = [p.strip().lower() for p in value.split(",") if p.strip()]
    if not parts:
        raise ValueError("dtypes must be a comma-separated list")
    dtypes = []
    for part in parts:
        if part not in lookup:
            raise ValueError(f"unsupported dtype: {part}")
        dtypes.append(lookup[part])
    return dtypes


def _dtype_name(dtype: mx.Dtype) -> str:
    if dtype == mx.float16:
        return "float16"
    if dtype == mx.bfloat16:
        return "bfloat16"
    if dtype == mx.float32:
        return "float32"
    return str(dtype)


def _sysctl_value(name: str) -> str | None:
    try:
        out = subprocess.check_output(["sysctl", "-n", name], text=True).strip()
        return out if out else None
    except Exception:
        return None


def _get_chip_name() -> str:
    return (
        _sysctl_value("machdep.cpu.brand_string")
        or _sysctl_value("hw.model")
        or platform.processor()
        or platform.machine()
        or "unknown"
    )


def _get_device() -> str:
    try:
        return str(mx.default_device())
    except Exception:
        return "unknown"


def _get_mlx_version() -> str:
    try:
        return importlib_metadata.version("mlx")
    except Exception:
        return getattr(mx, "__version__", "unknown")


def _write_jsonl(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare mix/add kernel variants.")
    parser.add_argument("--B", type=int, default=8)
    parser.add_argument("--n", type=str, default="16,32")
    parser.add_argument("--C", type=str, default="2048,4096")
    parser.add_argument("--dtypes", type=str, default="float16,bfloat16")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mode", type=str, choices=["throughput", "latency"], default="throughput")
    parser.add_argument("--queue-guard", type=int, default=50)
    parser.add_argument("--threads-per-group", type=int, default=None)
    parser.add_argument("--compiled", dest="compiled", action="store_true")
    parser.add_argument("--no-compiled", dest="compiled", action="store_false")
    parser.set_defaults(compiled=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="results_mix_variants.jsonl")
    args = parser.parse_args()

    B = int(args.B)
    ns = _parse_int_list(args.n)
    Cs = _parse_int_list(args.C)
    dtypes = _parse_dtype_list(args.dtypes)

    device = _get_device()
    chip = _get_chip_name()
    machine = platform.machine() or "unknown"
    macos = platform.mac_ver()[0] or platform.platform()
    mlx_version = _get_mlx_version()

    variants = [
        ("1d_fp32", "1d", None),
        ("2d_fp32", "2d", None),
        ("2d_half", "2d", "half"),
        ("1d_half", "1d", "half"),
    ]

    case_id = 0
    for n in ns:
        for C in Cs:
            for dtype in dtypes:
                mx.random.seed(args.seed + case_id)
                case_id += 1

                x = mx.random.normal((B, n, C)).astype(dtype)
                H_pre_raw = mx.random.normal((n,)).astype(mx.float32)
                H_post_raw = mx.random.normal((n,)).astype(mx.float32)
                M = mx.random.normal((n, n)).astype(mx.float32)
                rms_weight = mx.ones((C,), dtype=mx.float32)

                H_pre_act, H_post_act = activate_pre_post(H_pre_raw, H_post_raw)
                mx.eval(M, H_pre_act, H_post_act, rms_weight)
                mx.synchronize()

                tpg = args.threads_per_group
                if tpg is None:
                    tpg = suggest_threads_per_group(C)

                baseline = None
                baseline_name = "1d_fp32"
                results = {}
                for name, mix_kernel, out_kind in variants:
                    if out_kind == "half":
                        output_dtype = dtype
                    else:
                        output_dtype = None

                    mix_tpg = mix_add_rms_threadgroup_size(n, output_dtype, tpg, mix_kernel)

                    def _fn(_x):
                        return mhc_forward_fused_metal(
                            _x,
                            M,
                            H_pre_act,
                            H_post_act,
                            rms_weight,
                            eps=1e-5,
                            threads_per_group=tpg,
                            output_dtype=output_dtype,
                            mix_kernel=mix_kernel,
                            verbose=False,
                        )

                    times = bench_repeat(
                        _fn,
                        x,
                        iters=args.iters,
                        warmup=args.warmup,
                        repeats=args.repeats,
                        compiled=args.compiled,
                        mode=args.mode,
                        queue_guard=args.queue_guard,
                    )
                    summary = _summarize(times)
                    results[name] = summary["time_s_median"]
                    if name == baseline_name:
                        baseline = summary["time_s_median"]

                    record = {
                        "benchmark": "fused_metal",
                        "variant": name,
                        "mix_kernel": mix_kernel,
                        "output_dtype": _dtype_name(output_dtype) if output_dtype is not None else "none",
                        "B": B,
                        "n": n,
                        "C": C,
                        "dtype": _dtype_name(dtype),
                        "threads_per_group": tpg,
                        "mix_threads_per_group": mix_tpg,
                        "mode": args.mode,
                        "compiled": args.compiled,
                        "iters": args.iters,
                        "warmup": args.warmup,
                        "repeats": args.repeats,
                        "queue_guard": args.queue_guard,
                        "device": device,
                        "chip": chip,
                        "machine": machine,
                        "macos_version": macos,
                        "mlx_version": mlx_version,
                        **summary,
                    }
                    _write_jsonl(args.out, record)

                baseline = baseline or 1.0
                print(
                    f"[{args.mode}] B={B} n={n} C={C} dtype={_dtype_name(dtype)} "
                    f"tpg={tpg} baseline={baseline_name}"
                )
                for name, _, _ in variants:
                    t = results.get(name, float("nan"))
                    speedup = baseline / t if t and t > 0 else float("nan")
                    print(f"  {name:8s} time={t:.6e}s speedup={speedup:.3f}x")


if __name__ == "__main__":
    main()
