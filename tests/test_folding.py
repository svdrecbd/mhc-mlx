import mlx.core as mx
import mlx.nn as nn
from mhc_mlx import MHCRewire
import numpy as np

def test_weight_folding_correctness():
    """Verify that Weight Folding produces identical results to explicit multiply."""
    dims = 256
    n = 16
    C = dims // n
    mx.random.seed(42)
    
    x = mx.random.normal((1, dims))
    
    # 1. With Folding (MHCRewire detects nn.Linear)
    linear = nn.Linear(dims, dims)
    rewire_folded = MHCRewire(linear, dims=dims, n=n)
    y_folded = rewire_folded(x)
    
    # 2. Without Folding (Force fallback by wrapping Linear in a simple Module)
    class NoFold(nn.Module):
        def __init__(self, l):
            super().__init__()
            self.l = l
        def __call__(self, x):
            return self.l(x)
            
    rewire_explicit = MHCRewire(NoFold(linear), dims=dims, n=n)
    # Copy parameters to match
    rewire_explicit.mhc.H_pre_raw = rewire_folded.mhc.H_pre_raw
    rewire_explicit.mhc.H_post_raw = rewire_folded.mhc.H_post_raw
    rewire_explicit.mhc.H_res_raw = rewire_folded.mhc.H_res_raw
    
    y_explicit = rewire_explicit(x)
    
    # Compare
    max_diff = mx.max(mx.abs(y_folded - y_explicit)).item()
    print(f"Max diff between folded and explicit: {max_diff}")
    assert max_diff < 1e-5

def test_folding_gradient_flow():
    """Ensure gradients flow back to H_pre even when folded into weights."""
    dims = 64
    n = 4
    mx.random.seed(0)
    
    model = MHCRewire(nn.Linear(dims, dims), dims=dims, n=n)
    x = mx.random.normal((1, dims))
    
    def loss_fn(m, x):
        return mx.sum(m(x))
        
    grad_fn = mx.grad(loss_fn)
    grads = grad_fn(model, x)
    
    # Check that H_pre_raw received gradients
    h_pre_grad = grads['mhc']['H_pre_raw']
    assert mx.abs(mx.sum(h_pre_grad)).item() > 0, "No gradient found for H_pre_raw"
    
    # Check that inner linear weight received gradients
    w_grad = grads['inner']['weight']
    assert mx.abs(mx.sum(w_grad)).item() > 0, "No gradient found for linear weights"

if __name__ == "__main__":
    test_weight_folding_correctness()
    test_folding_gradient_flow()
