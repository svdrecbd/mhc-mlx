import time
import mlx.core as mx
from mhc_mlx.metal import residual_add_agg_metal
from mhc_mlx.reference import stream_aggregate

def benchmark_fn(fn, x, res, H, iters=100, warmup=20):
    for _ in range(warmup):
        out = fn(x, res, H)
        mx.eval(out)
    mx.synchronize()
    
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(x, res, H)
        mx.eval(out)
    mx.synchronize()
    t1 = time.perf_counter()
    
    return (t1 - t0) / iters

def run_benchmark(B, n, C, iters=100):
    print(f"\n--- Benchmarking Super-Block: B={B}, n={n}, C={C} ---")
    x = mx.random.normal((B, n, C)).astype(mx.float32)
    res = mx.random.normal((B, n, C)).astype(mx.float32)
    H = mx.random.normal((n,)).astype(mx.float32)
    
    def standard_impl(x, res, H):
        out = x + res
        # y_agg = sum(out * H)
        # We use stream_aggregate from reference which is sum(x * H)
        y_agg = stream_aggregate(out, H)
        return out, y_agg
        
    def fused_impl(x, res, H):
        return residual_add_agg_metal(x, res, H, threads_per_group=256)
        
    # Correctness check
    out_std, agg_std = standard_impl(x, res, H)
    out_fused, agg_fused = fused_impl(x, res, H)
    
    mx.eval(out_std, agg_std, out_fused, agg_fused)
    diff_out = mx.max(mx.abs(out_std - out_fused)).item()
    diff_agg = mx.max(mx.abs(agg_std - agg_fused)).item()
    
    print(f"Max diff out: {diff_out}")
    print(f"Max diff agg: {diff_agg}")
    
    if diff_out > 1e-4 or diff_agg > 1e-4:
        print("WARNING: Output mismatch!")
        
    t_std = benchmark_fn(standard_impl, x, res, H, iters=iters)
    t_fused = benchmark_fn(fused_impl, x, res, H, iters=iters)
    
    print(f"Standard: {t_std*1e6:.2f} us")
    print(f"Fused:    {t_fused*1e6:.2f} us")
    print(f"Speedup:  {t_std/t_fused:.2f}x")

if __name__ == "__main__":
    mx.set_default_device(mx.gpu)
    # Typical large layer sizes
    run_benchmark(B=1, n=32, C=4096)
    run_benchmark(B=32, n=32, C=4096)
