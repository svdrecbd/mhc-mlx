import mlx.core as mx
import mlx.nn as nn
import multiprocessing as mp
import time
from mhc_mlx import MHCLayer

def stress_proc(proc_id, n_iters=50):
    print(f"Process {proc_id} starting...")
    # New process gets its own MLX state
    mx.set_default_device(mx.gpu)
    
    try:
        B, n, C = 2, 32, 512
        layer = MHCLayer(n=n, C=C, use_metal=True)
        
        for i in range(n_iters):
            x = mx.random.normal((B, n, C))
            
            def loss_fn(model, x):
                return mx.sum(model(x))
            
            # Use value_and_grad
            loss, grads = mx.value_and_grad(loss_fn)(layer, x)
            mx.eval(loss, grads)
            
            if i % 10 == 0:
                print(f"Process {proc_id} iteration {i}")
                
        print(f"Process {proc_id} finished successfully.")
    except Exception as e:
        print(f"Process {proc_id} FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise e

def test_metal_multiprocessing():
    """Stress test Metal kernels with multiple concurrent processes."""
    n_procs = 4 # Fewer than threads to be safe
    processes = []
    
    print(f"Launching {n_procs} concurrent stress processes...")
    
    for i in range(n_procs):
        p = mp.Process(target=stress_proc, args=(i,))
        processes.append(p)
        p.start()
        
    for p in processes:
        p.join()
        assert p.exitcode == 0, f"Process failed with exit code {p.exitcode}"
        
    print("\nAll processes completed. Metal multiprocessing test PASSED.")

if __name__ == "__main__":
    test_metal_multiprocessing()