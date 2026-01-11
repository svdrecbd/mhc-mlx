import mlx.core as mx
import mlx.nn as nn
from mhc_mlx import MHCRewire

def test_rewire_linear():
    """Test wrapping a Linear layer."""
    dims = 512
    n = 16
    inner = nn.Linear(dims, dims)
    model = MHCRewire(inner, n=n)
    
    x = mx.random.normal((2, dims))
    y = model(x)
    
    assert y.shape == (2, dims)
    assert model.mhc.n == n
    assert model.mhc.C == dims // n

def test_rewire_conv():
    """Test that it fails for non-preserving shapes (as expected)."""
    inner = nn.Linear(512, 256)
    model = MHCRewire(inner, n=32)
    
    x = mx.random.normal((1, 512))
    try:
        model(x)
        assert False, "Should have failed due to shape mismatch"
    except ValueError:
        pass

def test_rewire_transformer_block():
    """Test wrapping a complex block."""
    class Block(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.ln = nn.RMSNorm(d)
            self.lin = nn.Linear(d, d)
        def __call__(self, x):
            return self.lin(self.ln(x))
            
    d = 1024
    model = MHCRewire(Block(d), n=32)
    x = mx.random.normal((1, 16, d))
    y = model(x)
    assert y.shape == (1, 16, d)
