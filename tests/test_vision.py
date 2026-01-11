import mlx.core as mx
import mlx.nn as nn
from mhc_mlx import MHCRewire

def test_conv2d_rewire():
    """Test wrapping a Conv2d layer (4D input)."""
    B, H, W, C = 2, 32, 32, 64
    n = 16
    
    # Inner layer: Standard Conv2d
    inner = nn.Conv2d(in_channels=C, out_channels=C, kernel_size=3, padding=1)
    
    # Rewire it
    model = MHCRewire(inner, dims=C, n=n)
    
    x = mx.random.normal((B, H, W, C))
    y = model(x)
    
    assert y.shape == (B, H, W, C)
    
    # Verify values changed (mHC applied)
    y_inner = inner(x)
    assert not mx.array_equal(y, y_inner)

def test_vision_transformer_block_rewire():
    """Test wrapping a ViT-style block [B, L, D] where L = H*W."""
    B, L, D = 2, 196, 256
    n = 32
    
    class ViTBlock(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.norm = nn.LayerNorm(d)
            self.linear = nn.Linear(d, d)
        def __call__(self, x):
            return self.linear(self.norm(x))
            
    model = MHCRewire(ViTBlock(D), dims=D, n=n)
    
    x = mx.random.normal((B, L, D))
    y = model(x)
    
    assert y.shape == (B, L, D)

if __name__ == "__main__":
    test_conv2d_rewire()
    test_vision_transformer_block_rewire()
