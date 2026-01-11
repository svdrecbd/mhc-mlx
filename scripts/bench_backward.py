import time
import mlx.core as mx
import mlx.nn as nn
from mhc_mlx.layer import MHCLayer

def benchmark_fn(fn, x, iters=50, warmup=10):
    # Flatten helper for eval
    def eval_grads(grads):
        from mlx.utils import tree_flatten
        mx.eval(*[p for _, p in tree_flatten(grads)])

    for _ in range(warmup):
        eval_grads(fn(x))
    mx.synchronize()
    
    t0 = time.perf_counter()
    for _ in range(iters):
        # We must eval the gradients to trigger computation
        eval_grads(fn(x))
    mx.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters

def run_backward_bench(B=1, n=32, C=4096):
    print(f"\n--- Benchmarking Backward: B={B}, n={n}, C={C} ---")
    x = mx.random.normal((B, n, C)).astype(mx.bfloat16)
    
    # 1. Reference (Python VJP)
    layer_ref = MHCLayer(n=n, C=C, use_metal=False, identity_init=False)
    
    def loss_ref(params, x):
        layer_ref.update(params)
        return mx.sum(layer_ref(x))
    
    # Standard MLX pattern: compile the grad function
    # @mx.compile
    def step_ref(params, x):
        return mx.grad(loss_ref)(params, x)
    
    # 2. Metal (Fused Backward)
    layer_metal = MHCLayer(n=n, C=C, use_metal=True, identity_init=False, fused_backward=True)
    
    def loss_metal(params, x):
        layer_metal.update(params)
        return mx.sum(layer_metal(x))
        
    # We compile the metal step too to be fair (kernel dispatch overhead vs python overhead)
    # @mx.compile
    def step_metal(params, x):
        return mx.grad(loss_metal)(params, x)
    
    # Run
    # Must pass current parameters to step
    t_ref = benchmark_fn(lambda x: step_ref(layer_ref.parameters(), x), x)
    t_metal = benchmark_fn(lambda x: step_metal(layer_metal.parameters(), x), x)
    
    print(f"Reference Backward: {t_ref*1e6:.2f} us")
    print(f"Metal Backward:     {t_metal*1e6:.2f} us")
    print(f"Speedup:            {t_ref/t_metal:.2f}x")
if __name__ == "__main__":
    mx.set_default_device(mx.gpu)
    run_backward_bench(B=1, n=32, C=2048)
    run_backward_bench(B=1, n=32, C=4096)
    run_backward_bench(B=32, n=32, C=2048)
