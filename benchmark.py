import argparse
import json
import os
import platform
import subprocess
import time
from importlib import metadata as importlib_metadata

import mlx.core as mx

from mhc_mlx.layer import MHCLayer
from mhc_mlx.metal import mhc_forward_fused_metal, sinkhorn_knopp_metal, suggest_threads_per_group
from mhc_mlx.reference import (
    activate_pre_post,
    mixing_matrix_from_logits,
    rms_norm,
    stream_aggregate,
    stream_distribute,
    stream_mix_ref,
)


def bench_once(layer, x, iters=200, warmup=10, compiled=True, mode="throughput"):
    # mode:
    # - "throughput": async_eval each iter + single synchronize at end
    # - "latency": eval each iter (includes sync cost)

    # Pre-eval inputs so they exist on device before timing
    mx.eval(x)
    mx.synchronize()

    compile_fn = getattr(mx, "compile", None)
    if compiled and callable(compile_fn):
        f = compile_fn(lambda _x: layer(_x))
    else:
        f = lambda _x: layer(_x)

    # Warmup (also triggers compilation/JIT for compiled funcs)
    for _ in range(warmup):
        y = f(x)
        mx.eval(y)
    mx.synchronize()

    t0 = time.perf_counter()
    if mode == "latency":
        for _ in range(iters):
            y = f(x)
            mx.eval(y)
        mx.synchronize()
    else:  # throughput
        for _ in range(iters):
            y = f(x)
            mx.async_eval(y)
        mx.synchronize()
    t1 = time.perf_counter()

    return (t1 - t0) / iters


def fused_reference(x, M, H_pre_act, H_post_act, rms_weight, eps):
    y_agg = stream_aggregate(x, H_pre_act)
    y_norm = rms_norm(y_agg, rms_weight, eps=eps)
    y_dist = stream_distribute(y_norm, H_post_act)
    x_mixed = stream_mix_ref(x, M)
    return x_mixed + y_dist


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


def _get_mlx_version() -> str:
    try:
        return importlib_metadata.version("mlx")
    except Exception:
        return getattr(mx, "__version__", "unknown")


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
        dev = mx.default_device()
        return str(dev)
    except Exception:
        try:
            return str(mx.default_device)
        except Exception:
            return "unknown"


def _write_jsonl(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _base_record(args, device: str, chip: str, machine: str, macos: str, mlx_version: str) -> dict:
    return {
        "compiled": args.compiled,
        "device": device,
        "mode": args.mode,
        "mlx_version": mlx_version,
        "macos_version": macos,
        "machine": machine,
        "chip": chip,
        "sinkhorn_iters": args.sinkhorn_iters,
        "threads_per_group": args.threads_per_group,
    }


def _max_abs(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a - b)).item())


def _max_abs_val(a: mx.array) -> float:
    return float(mx.max(mx.abs(a)).item())


def _run_case(
    args,
    case_id: int,
    B: int,
    n: int,
    C: int,
    dtype: mx.Dtype,
    base: dict,
    out_path: str,
) -> None:
    mx.random.seed(args.seed + case_id)

    x = mx.random.normal((B, n, C)).astype(dtype)

    H_pre_raw = mx.random.normal((n,)).astype(mx.float32)
    H_post_raw = mx.random.normal((n,)).astype(mx.float32)
    H_res_raw = mx.random.normal((n, n)).astype(mx.float32)
    rms_weight = mx.ones((C,), dtype=mx.float32)

    tpg = args.threads_per_group
    if tpg is None:
        tpg = suggest_threads_per_group(C)

    ref = MHCLayer(n=n, C=C, use_metal=False, threads_per_group=tpg)
    metal = MHCLayer(n=n, C=C, use_metal=True, threads_per_group=tpg)

    ref.H_pre_raw = H_pre_raw
    ref.H_post_raw = H_post_raw
    ref.H_res_raw = H_res_raw
    ref.rms_weight = rms_weight

    metal.H_pre_raw = H_pre_raw
    metal.H_post_raw = H_post_raw
    metal.H_res_raw = H_res_raw
    metal.rms_weight = rms_weight

    H_pre_act, H_post_act = activate_pre_post(H_pre_raw, H_post_raw)
    M = mixing_matrix_from_logits(H_res_raw, iters=args.sinkhorn_iters, eps=args.eps)
    mx.eval(M)
    mx.eval(H_pre_act)
    mx.eval(H_post_act)
    mx.eval(rms_weight)

    common = {
        **base,
        "B": B,
        "n": n,
        "C": C,
        "dtype": _dtype_name(dtype),
        "threads_per_group": tpg,
        "seed": args.seed + case_id,
    }

    if not args.no_correctness:
        y_ref = ref(x)
        y_metal = metal(x)
        mx.eval(y_ref)
        mx.eval(y_metal)
        mx.synchronize()

        max_abs = _max_abs(y_ref, y_metal)
        denom = max(_max_abs_val(y_ref), 1e-8)
        rel_err = max_abs / denom

        record = {
            **common,
            "benchmark": "correctness",
            "max_abs_error": max_abs,
            "rel_error": rel_err,
        }
        _write_jsonl(out_path, record)

    # Sinkhorn only
    sinkhorn_ref = bench_once(
        lambda h: mixing_matrix_from_logits(h, iters=args.sinkhorn_iters, eps=args.eps),
        H_res_raw,
        iters=args.iters,
        warmup=args.warmup,
        compiled=args.compiled,
        mode=args.mode,
    )
    _write_jsonl(
        out_path,
        {
            **common,
            "benchmark": "sinkhorn_ref",
            "time_s": sinkhorn_ref,
        },
    )

    sinkhorn_metal = bench_once(
        lambda h: sinkhorn_knopp_metal(
            h,
            iters=args.sinkhorn_iters,
            eps=args.eps,
            threads_per_group=tpg,
            verbose=False,
        ),
        H_res_raw,
        iters=args.iters,
        warmup=args.warmup,
        compiled=args.compiled,
        mode=args.mode,
    )
    _write_jsonl(
        out_path,
        {
            **common,
            "benchmark": "sinkhorn_metal",
            "time_s": sinkhorn_metal,
        },
    )

    # Fused forward only (precomputed M and activations)
    fused_ref = bench_once(
        lambda _x: fused_reference(_x, M, H_pre_act, H_post_act, rms_weight, args.eps),
        x,
        iters=args.iters,
        warmup=args.warmup,
        compiled=args.compiled,
        mode=args.mode,
    )
    _write_jsonl(
        out_path,
        {
            **common,
            "benchmark": "fused_ref",
            "time_s": fused_ref,
        },
    )

    fused_metal = bench_once(
        lambda _x: mhc_forward_fused_metal(
            _x,
            M,
            H_pre_act,
            H_post_act,
            rms_weight,
            eps=args.eps,
            threads_per_group=tpg,
            verbose=False,
        ),
        x,
        iters=args.iters,
        warmup=args.warmup,
        compiled=args.compiled,
        mode=args.mode,
    )
    _write_jsonl(
        out_path,
        {
            **common,
            "benchmark": "fused_metal",
            "time_s": fused_metal,
        },
    )

    # End-to-end layer
    layer_ref = bench_once(
        ref,
        x,
        iters=args.iters,
        warmup=args.warmup,
        compiled=args.compiled,
        mode=args.mode,
    )
    _write_jsonl(
        out_path,
        {
            **common,
            "benchmark": "layer_ref",
            "time_s": layer_ref,
        },
    )

    layer_metal = bench_once(
        metal,
        x,
        iters=args.iters,
        warmup=args.warmup,
        compiled=args.compiled,
        mode=args.mode,
    )
    _write_jsonl(
        out_path,
        {
            **common,
            "benchmark": "layer_metal",
            "time_s": layer_metal,
        },
    )

    print(
        f"case {case_id}: B={B} n={n} C={C} dtype={_dtype_name(dtype)} tpg={tpg}"
    )


def main():
    parser = argparse.ArgumentParser(description="MHCLayer benchmark suite.")
    parser.add_argument("--B", type=str, default="1,8,32")
    parser.add_argument("--n", type=str, default="4,8,16,32")
    parser.add_argument("--C", type=str, default="256,512,1024,2048,4096")
    parser.add_argument("--dtypes", type=str, default="bfloat16,float16,float32")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--sinkhorn-iters", type=int, default=20)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--threads-per-group", type=int, default=None)
    parser.add_argument("--mode", type=str, choices=["throughput", "latency"], default="throughput")
    parser.add_argument("--compiled", dest="compiled", action="store_true")
    parser.add_argument("--no-compiled", dest="compiled", action="store_false")
    parser.set_defaults(compiled=True)
    parser.add_argument("--no-correctness", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="results.jsonl")
    args = parser.parse_args()

    device = _get_device()
    chip = _get_chip_name()
    machine = platform.machine() or "unknown"
    macos = platform.mac_ver()[0] or platform.platform()
    mlx_version = _get_mlx_version()

    base = _base_record(args, device, chip, machine, macos, mlx_version)

    Bs = _parse_int_list(args.B)
    ns = _parse_int_list(args.n)
    Cs = _parse_int_list(args.C)
    dtypes = _parse_dtype_list(args.dtypes)

    case_id = 0
    for B in Bs:
        for n in ns:
            for C in Cs:
                for dtype in dtypes:
                    case_id += 1
                    _run_case(args, case_id, B, n, C, dtype, base, args.out)

    print(f"Done. Results written to {args.out}")


if __name__ == "__main__":
    main()
