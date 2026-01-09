import argparse
import json
import math
import os
import platform
import subprocess
import time
from importlib import metadata as importlib_metadata

import mlx.core as mx

from mhc_mlx.layer import MHCLayer
from mhc_mlx.metal import (
    mhc_forward_fused_metal,
    mhc_forward_fused_metal_autograd,
    sinkhorn_knopp_metal,
    sinkhorn_knopp_metal_autograd,
    suggest_threads_per_group,
    stream_mix_add_metal_autograd,
)
from mhc_mlx.reference import (
    activate_pre_post,
    mhc_forward_fused_reference,
    mhc_forward_reference,
    mixing_matrix_from_logits,
    rms_norm,
    stream_aggregate,
    stream_distribute,
)


def _make_runner(layer, compiled: bool):
    compile_fn = getattr(mx, "compile", None)
    if compiled and callable(compile_fn):
        return compile_fn(lambda _x: layer(_x))
    return lambda _x: layer(_x)


def _bench_loop(f, x, iters: int, mode: str, queue_guard: int) -> float:
    t0 = time.perf_counter()
    if mode == "latency":
        for _ in range(iters):
            y = f(x)
            mx.eval(y)
        mx.synchronize()
    else:  # throughput
        for i in range(iters):
            y = f(x)
            mx.async_eval(y)
            if queue_guard and (i + 1) % queue_guard == 0:
                mx.synchronize()
        mx.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters


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


def bench_repeat(
    layer,
    x,
    iters: int = 200,
    warmup: int = 10,
    repeats: int = 3,
    compiled: bool = True,
    mode: str = "throughput",
    queue_guard: int = 0,
) -> list[float]:
    # Pre-eval inputs so they exist on device before timing
    mx.eval(x)
    mx.synchronize()

    f = _make_runner(layer, compiled)

    # Warmup (also triggers compilation/JIT for compiled funcs)
    for _ in range(warmup):
        y = f(x)
        mx.eval(y)
    mx.synchronize()

    times = []
    for _ in range(repeats):
        times.append(_bench_loop(f, x, iters, mode, queue_guard))
    return times


def _flatten_grads(grads) -> list[mx.array]:
    if grads is None:
        return []
    if isinstance(grads, (list, tuple)):
        return list(grads)
    return [grads]


def bench_backward_repeat(
    loss_fn,
    inputs: list[mx.array],
    iters: int = 200,
    warmup: int = 10,
    repeats: int = 3,
    compiled: bool = True,
    mode: str = "throughput",
    queue_guard: int = 0,
) -> list[float]:
    # Pre-eval inputs so they exist on device before timing
    mx.eval(*inputs)
    mx.synchronize()

    argnums = list(range(len(inputs)))

    def _loss(*args):
        return loss_fn(*args)

    grad_fn = mx.value_and_grad(_loss, argnums=argnums)

    if compiled and callable(getattr(mx, "compile", None)):
        grad_fn = mx.compile(grad_fn)

    # Warmup
    for _ in range(warmup):
        loss, grads = grad_fn(*inputs)
        mx.eval(loss, *_flatten_grads(grads))
    mx.synchronize()

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        if mode == "latency":
            for _ in range(iters):
                loss, grads = grad_fn(*inputs)
                mx.eval(loss, *_flatten_grads(grads))
            mx.synchronize()
        else:
            for i in range(iters):
                loss, grads = grad_fn(*inputs)
                mx.async_eval(loss, *_flatten_grads(grads))
                if queue_guard and (i + 1) % queue_guard == 0:
                    mx.synchronize()
            mx.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) / iters)
    return times


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


def _parse_mode_list(value: str) -> list[str]:
    parts = [p.strip().lower() for p in value.split(",") if p.strip()]
    if not parts:
        raise ValueError("modes must be a comma-separated list")
    allowed = {"throughput", "latency"}
    for part in parts:
        if part not in allowed:
            raise ValueError(f"unsupported mode: {part}")
    return parts


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
        "metal_dispatch": args.metal_dispatch,
        "hybrid_latency": args.hybrid_latency,
        "fused_backward": args.fused_backward,
        "with_backward": args.with_backward,
        "threads_per_group": args.threads_per_group,
    }


def _max_abs(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a - b)).item())


def _max_abs_val(a: mx.array) -> float:
    return float(mx.max(mx.abs(a)).item())


def _summarize_times(times: list[float]) -> dict:
    values = sorted(times)
    return {
        "time_s_median": _percentile(values, 50.0),
        "time_s_p10": _percentile(values, 10.0),
        "time_s_p90": _percentile(values, 90.0),
    }


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

    ref = MHCLayer(
        n=n,
        C=C,
        use_metal=False,
        threads_per_group=tpg,
        compile_reference=False,
    )
    metal = MHCLayer(
        n=n,
        C=C,
        use_metal=True,
        threads_per_group=tpg,
        auto_dispatch=(args.metal_dispatch == "auto"),
        compile_reference=False,
        hybrid_latency=args.hybrid_latency,
        fused_backward=args.fused_backward,
    )

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
        "iters": args.iters,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "queue_guard": args.queue_guard,
        "threads_per_group": tpg,
        "seed": args.seed + case_id,
    }

    use_auto_dispatch = args.metal_dispatch == "auto"
    use_hybrid = use_auto_dispatch and args.hybrid_latency and n == 32 and B == 1
    use_ref_fallback = use_auto_dispatch and (not args.hybrid_latency) and n == 32 and B == 1
    use_fused_metal = not use_hybrid and not use_ref_fallback
    fused_backward_effective = (
        args.fused_backward and (not use_auto_dispatch or B >= 8) and use_fused_metal
    )

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
            "input_dtype": _dtype_name(dtype),
            "accum_dtype": "float32",
            "output_dtype": "float32",
            "compare_dtype": "float32",
            "max_abs_error": max_abs,
            "rel_error": rel_err,
        }
        _write_jsonl(out_path, record)

    # Sinkhorn only
    sinkhorn_ref_times = bench_repeat(
        lambda h: mixing_matrix_from_logits(h, iters=args.sinkhorn_iters, eps=args.eps),
        H_res_raw,
        iters=args.iters,
        warmup=args.warmup,
        repeats=args.repeats,
        compiled=args.compiled,
        mode=args.mode,
        queue_guard=args.queue_guard,
    )
    _write_jsonl(
        out_path,
        {
            **common,
            "benchmark": "sinkhorn_ref",
            **_summarize_times(sinkhorn_ref_times),
        },
    )

    sinkhorn_metal_times = bench_repeat(
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
        repeats=args.repeats,
        compiled=args.compiled,
        mode=args.mode,
        queue_guard=args.queue_guard,
    )
    _write_jsonl(
        out_path,
        {
            **common,
            "benchmark": "sinkhorn_metal",
            **_summarize_times(sinkhorn_metal_times),
        },
    )

    # Fused forward only (precomputed M and activations)
    fused_ref_times = bench_repeat(
        lambda _x: mhc_forward_fused_reference(_x, M, H_pre_act, H_post_act, rms_weight, args.eps),
        x,
        iters=args.iters,
        warmup=args.warmup,
        repeats=args.repeats,
        compiled=args.compiled,
        mode=args.mode,
        queue_guard=args.queue_guard,
    )
    _write_jsonl(
        out_path,
        {
            **common,
            "benchmark": "fused_ref",
            **_summarize_times(fused_ref_times),
        },
    )

    fused_metal_times = bench_repeat(
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
        repeats=args.repeats,
        compiled=args.compiled,
        mode=args.mode,
        queue_guard=args.queue_guard,
    )
    _write_jsonl(
        out_path,
        {
            **common,
            "benchmark": "fused_metal",
            **_summarize_times(fused_metal_times),
        },
    )

    # End-to-end layer
    layer_ref_times = bench_repeat(
        ref,
        x,
        iters=args.iters,
        warmup=args.warmup,
        repeats=args.repeats,
        compiled=args.compiled,
        mode=args.mode,
        queue_guard=args.queue_guard,
    )
    _write_jsonl(
        out_path,
        {
            **common,
            "benchmark": "layer_ref",
            **_summarize_times(layer_ref_times),
        },
    )

    layer_metal_times = bench_repeat(
        metal,
        x,
        iters=args.iters,
        warmup=args.warmup,
        repeats=args.repeats,
        compiled=args.compiled,
        mode=args.mode,
        queue_guard=args.queue_guard,
    )
    _write_jsonl(
        out_path,
        {
            **common,
            "benchmark": "layer_metal",
            **_summarize_times(layer_metal_times),
        },
    )

    if args.with_backward:
        inputs = [x, H_pre_raw, H_post_raw, H_res_raw, rms_weight]
        backward_compiled = args.compiled
        if fused_backward_effective and backward_compiled:
            if not getattr(args, "_warned_backward_compile", False):
                print("Note: disabling mx.compile for backward when fused backward is enabled.")
                args._warned_backward_compile = True
            backward_compiled = False

        def ref_loss(x, H_pre_raw, H_post_raw, H_res_raw, rms_weight):
            out = mhc_forward_reference(
                x_expanded=x,
                H_pre_raw=H_pre_raw,
                H_post_raw=H_post_raw,
                H_res_raw=H_res_raw,
                rms_weight=rms_weight,
                sinkhorn_iters=args.sinkhorn_iters,
                eps=args.eps,
            )
            return mx.sum(out)

        def metal_loss(x, H_pre_raw, H_post_raw, H_res_raw, rms_weight):
            if use_ref_fallback:
                return ref_loss(x, H_pre_raw, H_post_raw, H_res_raw, rms_weight)

            H_pre_act, H_post_act = activate_pre_post(H_pre_raw, H_post_raw)
            if use_hybrid:
                M = sinkhorn_knopp_metal_autograd(
                    H_res_raw,
                    iters=args.sinkhorn_iters,
                    eps=args.eps,
                    threads_per_group=tpg,
                    verbose=False,
                )
                y_agg = stream_aggregate(x, H_pre_act)
                y_norm = rms_norm(y_agg, rms_weight, eps=args.eps)
                y_dist = stream_distribute(y_norm, H_post_act)
                out = stream_mix_add_metal_autograd(
                    x,
                    M,
                    y_dist,
                    threads_per_group=tpg,
                    verbose=False,
                )
                return mx.sum(out)

            M = sinkhorn_knopp_metal_autograd(
                H_res_raw,
                iters=args.sinkhorn_iters,
                eps=args.eps,
                threads_per_group=tpg,
                verbose=False,
            )
            out = mhc_forward_fused_metal_autograd(
                x,
                M,
                H_pre_act,
                H_post_act,
                rms_weight,
                eps=args.eps,
                threads_per_group=tpg,
                fused_backward=fused_backward_effective,
                verbose=False,
            )
            return mx.sum(out)

        ref_back_times = bench_backward_repeat(
            ref_loss,
            inputs,
            iters=args.iters,
            warmup=args.warmup,
            repeats=args.repeats,
            compiled=backward_compiled,
            mode=args.mode,
            queue_guard=args.queue_guard,
        )
        _write_jsonl(
            out_path,
            {
                **common,
                "benchmark": "layer_backward_ref",
                "backward_compiled": backward_compiled,
                **_summarize_times(ref_back_times),
            },
        )

        metal_back_times = bench_backward_repeat(
            metal_loss,
            inputs,
            iters=args.iters,
            warmup=args.warmup,
            repeats=args.repeats,
            compiled=backward_compiled,
            mode=args.mode,
            queue_guard=args.queue_guard,
        )
        _write_jsonl(
            out_path,
            {
                **common,
                "benchmark": "layer_backward_metal",
                "backward_compiled": backward_compiled,
                **_summarize_times(metal_back_times),
            },
        )

    print(
        f"[{args.mode}] case {case_id}: B={B} n={n} C={C} dtype={_dtype_name(dtype)} tpg={tpg}"
    )


def main():
    parser = argparse.ArgumentParser(description="MHCLayer benchmark suite.")
    parser.add_argument("--B", type=str, default="1,8,32")
    parser.add_argument("--n", type=str, default="4,8,16,32")
    parser.add_argument("--C", type=str, default="256,512,1024,2048,4096")
    parser.add_argument("--dtypes", type=str, default="bfloat16,float16,float32")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sinkhorn-iters", type=int, default=20)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--threads-per-group", type=int, default=None)
    parser.add_argument("--metal-dispatch", type=str, choices=["auto", "force"], default="auto")
    parser.add_argument("--hybrid-latency", dest="hybrid_latency", action="store_true")
    parser.add_argument("--no-hybrid-latency", dest="hybrid_latency", action="store_false")
    parser.set_defaults(hybrid_latency=True)
    parser.add_argument("--queue-guard", type=int, default=50)
    parser.add_argument("--modes", type=str, default="throughput,latency")
    parser.add_argument("--mode", type=str, choices=["throughput", "latency"], default=None)
    parser.add_argument("--compiled", dest="compiled", action="store_true")
    parser.add_argument("--no-compiled", dest="compiled", action="store_false")
    parser.set_defaults(compiled=True)
    parser.add_argument("--no-correctness", action="store_true")
    parser.add_argument("--with-backward", action="store_true")
    parser.add_argument("--fused-backward", dest="fused_backward", action="store_true")
    parser.add_argument("--no-fused-backward", dest="fused_backward", action="store_false")
    parser.set_defaults(fused_backward=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="results.jsonl")
    args = parser.parse_args()

    device = _get_device()
    chip = _get_chip_name()
    machine = platform.machine() or "unknown"
    macos = platform.mac_ver()[0] or platform.platform()
    mlx_version = _get_mlx_version()

    Bs = _parse_int_list(args.B)
    ns = _parse_int_list(args.n)
    Cs = _parse_int_list(args.C)
    dtypes = _parse_dtype_list(args.dtypes)
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.queue_guard < 0:
        raise ValueError("--queue-guard must be >= 0")

    if args.mode is not None:
        modes = [args.mode]
    else:
        modes = _parse_mode_list(args.modes)

    for mode in modes:
        args.mode = mode
        base = _base_record(args, device, chip, machine, macos, mlx_version)

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
