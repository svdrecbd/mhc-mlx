import time
import csv
import os
import datetime
import subprocess
import mlx.core as mx
import mlx.nn as nn
from mhc_mlx.layer import MHCLayer, MHCRewire

# Configuration
BENCHMARK_FILE = "BENCHMARKS.csv"
SCENARIOS = [
    # Latency Floor (Small Batch)
    {"B": 1, "n": 32, "C": 256, "mode": "latency"},
    {"B": 1, "n": 32, "C": 512, "mode": "latency"},
    {"B": 1, "n": 32, "C": 1024, "mode": "latency"},
    {"B": 1, "n": 32, "C": 2048, "mode": "latency"},
    {"B": 1, "n": 32, "C": 4096, "mode": "latency"},
    
    # High Throughput (Large Batch)
    {"B": 32, "n": 32, "C": 512, "mode": "throughput"},
    {"B": 32, "n": 32, "C": 2048, "mode": "throughput"},
    {"B": 32, "n": 32, "C": 4096, "mode": "throughput"},
]

def get_git_info():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode("utf-8")
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip().decode("utf-8")
        return commit, branch
    except:
        return "unknown", "unknown"

def benchmark_fn(fn, x, iters=50, warmup=10, mode="latency"):
    for _ in range(warmup):
        y = fn(x)
        mx.eval(y)
    mx.synchronize()
    
    times = []
    if mode == "latency":
        for _ in range(iters):
            t0 = time.perf_counter()
            y = fn(x)
            mx.eval(y)
            mx.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)
        return sorted(times)[len(times)//2] # Median
    else: # throughput
        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            y = fn(x)
            mx.async_eval(y)
        mx.synchronize()
        t1 = time.perf_counter()
        return (t1 - t0) / iters # Mean

def run_suite():
    commit, branch = get_git_info()
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Initialize CSV if needed
    file_exists = os.path.isfile(BENCHMARK_FILE)
    
    with open(BENCHMARK_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["date", "commit", "branch", "B", "n", "C", "mode", "impl", "time_us"])
            
        print(f"Running Benchmark Suite (Commit: {commit})...")
        
        for s in SCENARIOS:
            B, n, C, mode = s["B"], s["n"], s["C"], s["mode"]
            dims = n * C
            print(f"  Running {mode}: B={B}, n={n}, C={C} ...")
            
            x = mx.random.normal((B, 1, dims)).astype(mx.bfloat16)
            
            # 1. MHCLayer (Direct)
            layer = MHCLayer(n=n, C=C, use_metal=True, identity_init=False)
            layer.train(False)
            def bench_layer(x_in):
                return layer(x_in.reshape(B, n, C))
            
            t_layer = benchmark_fn(bench_layer, x, mode=mode)
            writer.writerow([date, commit, branch, B, n, C, mode, "MHCLayer", f"{t_layer*1e6:.2f}"])
            
            # 2. MHCRewire (Universal) - Only for smaller C to avoid memory explosion if not fully optimized yet
            if C <= 1024:
                linear = nn.Linear(dims, dims)
                # Use zeros to avoid huge allocation/init cost
                linear.weight = mx.zeros((dims, dims), dtype=mx.bfloat16) 
                linear.bias = None
                rewire = MHCRewire(linear, dims=dims, n=n, identity_init=False)
                rewire.mhc.train(False)
                
                t_rewire = benchmark_fn(rewire, x, mode=mode)
                writer.writerow([date, commit, branch, B, n, C, mode, "MHCRewire", f"{t_rewire*1e6:.2f}"])

    print(f"\nDone. Results appended to {BENCHMARK_FILE}")

if __name__ == "__main__":
    mx.set_default_device(mx.gpu)
    run_suite()
