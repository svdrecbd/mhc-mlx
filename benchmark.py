import argparse
import time
import mlx.core as mx

from mhc_mlx.layer import MHCLayer


def bench_once(layer: MHCLayer, x: mx.array, iters: int = 100) -> float:
    # Warm-up
    y = layer(x)
    mx.eval(y)

    t0 = time.perf_counter()
    for _ in range(iters):
        y = layer(x)
    mx.eval(y)
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def tune_threads_per_group(
    layer: MHCLayer, x: mx.array, candidates: list[int], iters: int = 50
) -> int:
    best_tpg = candidates[0]
    best_dt = float("inf")
    for tpg in candidates:
        layer.threads_per_group = int(tpg)
        dt = bench_once(layer, x, iters=iters)
        print(f"threads_per_group={tpg:>3d} avg seconds/iter: {dt:.6e}")
        if dt < best_dt:
            best_dt = dt
            best_tpg = tpg
    return best_tpg


def _parse_tpgs(value: str) -> list[int]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise ValueError("tpgs must be a comma-separated list of ints")
    return [int(p) for p in parts]


def main():
    parser = argparse.ArgumentParser(description="Benchmark MHCLayer Metal vs reference.")
    parser.add_argument("--B", type=int, default=32)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--C", type=int, default=1280)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--tune-iters", type=int, default=50)
    parser.add_argument("--tpgs", type=str, default="32,64,128,256")
    parser.add_argument("--no-tune", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    mx.random.seed(args.seed)

    # Tune these to resemble your real hidden size.
    B = args.B
    n = args.n
    C = args.C

    x = mx.random.normal((B, n, C)).astype(mx.bfloat16)

    ref = MHCLayer(n=n, C=C, use_metal=False)
    metal = MHCLayer(n=n, C=C, use_metal=True, threads_per_group=None)

    # Make params identical
    ref.H_pre = mx.ones((n,), dtype=mx.float32)
    ref.H_post = mx.ones((n,), dtype=mx.float32)
    ref.H_res = mx.eye(n, dtype=mx.float32)
    ref.rms_weight = mx.ones((C,), dtype=mx.float32)

    metal.H_pre = ref.H_pre
    metal.H_post = ref.H_post
    metal.H_res = ref.H_res
    metal.rms_weight = ref.rms_weight

    print(f"Benchmark shape: B={B}, n={n}, C={C}")

    dt_ref = bench_once(ref, x, iters=args.iters)
    if args.no_tune:
        print(f"threads_per_group (heuristic): {metal.threads_per_group}")
    else:
        candidates = _parse_tpgs(args.tpgs)
        print("Tuning threads_per_group...")
        best_tpg = tune_threads_per_group(metal, x, candidates=candidates, iters=args.tune_iters)
        print(f"best threads_per_group: {best_tpg}")
        metal.threads_per_group = best_tpg

    dt_metal = bench_once(metal, x, iters=args.iters)

    print(f"reference avg seconds/iter: {dt_ref:.6e}")
    print(f"metal     avg seconds/iter: {dt_metal:.6e}")
    if dt_metal > 0:
        print(f"speedup (ref/metal): {dt_ref / dt_metal:.2f}x")


if __name__ == "__main__":
    main()
