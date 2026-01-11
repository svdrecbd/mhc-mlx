import json
import time
import os
import mlx.core as mx
from mhc_mlx.layer import MHCLayer
from mhc_mlx.metal import (
    stream_mix_add_metal, 
    sinkhorn_knopp_metal, 
    residual_add_agg_metal,
    _MAX_TPG_ALLOWED
)

CONFIG_FILE = "mhc_tuning.json"

def benchmark_fn(fn, warmup=10, iters=50):
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    mx.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters

def tune_kernel(name, candidates, runner_fn):
    print(f"\n--- Tuning {name} ---")
    best_tpg = 256
    best_time = float("inf")
    
    for tpg in candidates:
        if tpg > _MAX_TPG_ALLOWED:
            continue
            
        try:
            # Run benchmark
            avg_time = benchmark_fn(lambda: runner_fn(tpg))
            print(f"  TPG={tpg:4d}: {avg_time*1e6:.2f} us")
            
            if avg_time < best_time:
                best_time = avg_time
                best_tpg = tpg
        except Exception as e:
            print(f"  TPG={tpg:4d}: Failed ({e})")
            
    print(f"  -> Winner: TPG={best_tpg} ({best_time*1e6:.2f} us)")
    return best_tpg

def main():
    print(f"Running Auto-Tuner on {mx.default_device()}")
    
    # Define workload (Medium/Large representative)
    B = 1
    n = 32
    C = 4096 
    
    x = mx.random.normal((B, n, C)).astype(mx.float32)
    res = mx.random.normal((B, n, C)).astype(mx.float32)
    M = mx.random.normal((n, n)).astype(mx.float32)
    H_pre = mx.random.normal((n,)).astype(mx.float32)
    y_dist = mx.random.normal((B, n, C)).astype(mx.float32)
    H_res = mx.random.normal((n, n)).astype(mx.float32)
    
    candidates = [32, 64, 128, 256, 512, 1024]
    
    config = {}
    
    # 1. Residual Add Agg (Bandwidth bound)
    config["residual_add_agg"] = tune_kernel(
        "residual_add_agg", 
        candidates,
        lambda tpg: residual_add_agg_metal(x, res, H_pre, threads_per_group=tpg)
    )
    
    # 2. Stream Mix Add (Compute/Bandwidth mix)
    # Note: Column kernel logic inside metal.py might override tpg for small C, 
    # but for C=4096 it uses the vectorized column path which respects tpg?
    # Wait, the vectorized column kernel uses grid=(C//4, B, 1).
    # The tpg controls how many columns a threadgroup handles?
    # No, grid size is fixed. tpg controls threads per threadgroup.
    # Higher TPG = fewer threadgroups.
    config["stream_mix_add"] = tune_kernel(
        "stream_mix_add", 
        candidates,
        lambda tpg: stream_mix_add_metal(x, M, y_dist, threads_per_group=tpg)
    )
    
    # 3. Sinkhorn (Compute/Shared Mem bound)
    # Sinkhorn is sensitive to N.
    # For N=32, we know 32 is best (SIMD).
    # Let's tune for N=128 (Large).
    H_large = mx.random.normal((128, 128)).astype(mx.float32)
    config["sinkhorn_128"] = tune_kernel(
        "sinkhorn_128", 
        candidates,
        lambda tpg: sinkhorn_knopp_metal(H_large, iters=10, eps=1e-5, threads_per_group=tpg)
    )
    
    # Save config
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)
    print(f"\nSaved best configuration to {CONFIG_FILE}")

if __name__ == "__main__":
    mx.set_default_device(mx.gpu)
    main()
