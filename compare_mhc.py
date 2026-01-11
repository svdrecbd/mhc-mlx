import time
import mlx.core as mx
import mlx.nn as nn
from mhc_mlx.layer import MHCLayer, MHCRewire
import mlx_mhc as their_mhc

def benchmark_fn(fn, x, iters=100, warmup=20, mode="latency"):
    # Warmup
    for _ in range(warmup):
        y = fn(x)
        mx.eval(y)
    mx.synchronize()
    
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        y = fn(x)
        if mode == "latency":
            mx.eval(y)
            mx.synchronize()
        else:
            mx.async_eval(y)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    
    if mode == "throughput":
        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            y = fn(x)
            mx.async_eval(y)
        mx.synchronize()
        t1 = time.perf_counter()
        return (t1 - t0) / iters

    return sorted(times)[len(times)//2]

def run_comparison(B, n, C, iters=100):
    dims = n * C
    print(f"\n--- Head-to-Head: B={B}, n={n}, C={C} (dims={dims}) ---")
    x = mx.random.normal((B, 1, dims)).astype(mx.bfloat16)
    
    # 1. mhc-mlx: MHCLayer (Direct)
    base_layer = MHCLayer(n=n, C=C, use_metal=True, identity_init=False)
    base_layer.train(True)
    def base_bench(x_in):
        x_mhc = x_in.reshape(B, n, C)
        return base_layer(x_mhc)

    # 2. mlx-mhc
    their_layer = their_mhc.ManifoldHyperConnection(dims=dims, expansion=n, sinkhorn_iterations=20)
    their_layer.train(True)
    def their_bench(x_in):
        x_pre = their_layer.pre_scale(x_in)
        return their_layer.post_combine(x_in, x_pre)

    # Benchmarks
    results = {}
    for mode in ["latency", "throughput"]:
        results[mode] = {
            "base": benchmark_fn(base_bench, x, iters=iters, mode=mode),
            "their": benchmark_fn(their_bench, x, iters=iters, mode=mode),
        }

    for mode in ["latency", "throughput"]:
        label = "Latency (median)" if mode == "latency" else "Throughput (avg)"
        unit = "us" if mode == "latency" else "us/iter"
        print(f"{label}:")
        print(f"  mhc-mlx (MHCLayer):  {results[mode]['base']*1e6:.2f} {unit}")
        print(f"  mlx-mhc (Them):       {results[mode]['their']*1e6:.2f} {unit}")
        
        speedup_vs_them = results[mode]['their'] / results[mode]['base']
        print(f"  Speedup vs Competitor: {speedup_vs_them:.2f}x")

if __name__ == "__main__":
    mx.set_default_device(mx.gpu)
    # Larger channels where Metal shines
    run_comparison(B=1, n=32, C=2048)
    run_comparison(B=32, n=32, C=2048)