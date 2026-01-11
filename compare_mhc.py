import time
import mlx.core as mx
import mlx.nn as nn
from mhc_mlx.layer import MHCLayer
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
    print(f"\n--- Head-to-Head: B={B}, n={n}, C={C} ---")
    x = mx.random.normal((B, n, C)).astype(mx.bfloat16)
    
    # mhc-mlx (our package)
    # We use identity_init=False to match their random init for a fair fight
    our_layer = MHCLayer(n=n, C=C, use_metal=True, identity_init=False)
    # They use 20 iters by default in my script, let's keep it same
    our_layer.sinkhorn_iters = 20
    our_layer.train(True) # Force Sinkhorn computation every pass
    
    # mlx-mhc (slop)
    their_layer = their_mhc.ManifoldHyperConnection(dims=n*C, expansion=n, sinkhorn_iterations=20)
    their_layer.train(True) # Force Sinkhorn computation every pass
    
    def our_bench(x_in):
        return our_layer(x_in)
        
    def their_bench(x_in):
        # Full mHC graph: 
        # x_pre = H_pre * x
        # x_next = H_post * (layer(x_pre) + M * x_pre)
        # We assume layer(x_pre) is identity for pure overhead test
        x_pre = their_layer.pre_scale(x_in)
        return their_layer.post_combine(x_in, x_pre)

    # Latency
    our_lat = benchmark_fn(our_bench, x, iters=iters, mode="latency")
    their_lat = benchmark_fn(their_bench, x, iters=iters, mode="latency")
    print(f"Latency (median):")
    print(f"  mhc-mlx (ours): {our_lat*1e6:.2f} us")
    print(f"  mlx-mhc (them): {their_lat*1e6:.2f} us")
    print(f"  Speedup: {their_lat/our_lat:.2f}x")

    # Throughput
    our_thr = benchmark_fn(our_bench, x, iters=iters, mode="throughput")
    their_thr = benchmark_fn(their_bench, x, iters=iters, mode="throughput")
    print(f"Throughput (avg):")
    print(f"  mhc-mlx (ours): {our_thr*1e6:.2f} us/iter")
    print(f"  mlx-mhc (them): {their_thr*1e6:.2f} us/iter")
    print(f"  Speedup: {their_thr/our_thr:.2f}x")

if __name__ == "__main__":
    mx.set_default_device(mx.gpu)
    # Small case
    run_comparison(B=1, n=32, C=512)
    # Large case
    run_comparison(B=32, n=32, C=2048)