import mlx.core as mx
import mlx.nn as nn
from mhc_mlx import AutoPatcher, MHCRewire

def test_autopatcher_simple():
    """Test patching a simple sequential model."""
    dims = 64
    n = 4
    
    # Valid candidate (64->64)
    l1 = nn.Linear(dims, dims)
    # Invalid candidate (64->32)
    l2 = nn.Linear(dims, 32)
    # Valid candidate (32->32) but requires n=4 (divisible)
    l3 = nn.Linear(32, 32)
    
    model = nn.Sequential(l1, l2, l3)
    
    # Patch
    AutoPatcher.patch(model, n_streams=n, verbose=True)
    
    # Check l1
    # nn.Sequential stores layers in 'layers' list usually, but let's check hierarchy
    # In MLX Sequential, layers are accessed via indexing or layers attribute
    # AutoPatcher uses named_modules traversal.
    
    # We need to verify if the module instance in the model structure changed to MHCRewire
    # Since Sequential holds a list, replacing attributes on the Sequential object 
    # (like "layers.0") might be tricky depending on how named_modules works.
    # MLX named_modules yields "layers.0", "layers.1".
    # setattr(model.layers, "0", ...) doesn't work on list.
    # setattr(model, "layers.0", ...) doesn't work.
    
def test_autopatcher_list_container():
    """Test patching modules inside a list (simulating nn.Sequential or layer lists)."""
    dims = 64
    n = 4
    
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            # A list of layers
            self.layers = [
                nn.Linear(dims, dims), # Should patch
                nn.Linear(dims, 32)    # Should not
            ]
            
    model = Model()
    AutoPatcher.patch(model, n_streams=n, verbose=True)
    
    assert isinstance(model.layers[0], MHCRewire)
    assert not isinstance(model.layers[1], MHCRewire)

if __name__ == "__main__":
    test_autopatcher_custom_block()
    test_autopatcher_list_container()
    """Test patching a custom block structure (like Llama)."""
    dims = 64
    n = 4
    
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(dims, dims) # Should patch
            self.linear2 = nn.Linear(dims, dims * 2) # Should NOT patch (dim change)
            self.linear3 = nn.Linear(dims * 2, dims) # Should NOT patch (dim change)
            self.linear4 = nn.Linear(dims, dims) # Should patch
            
    model = Block()
    
    print("\nBefore patch:")
    print(model)
    
    AutoPatcher.patch(model, n_streams=n, verbose=True)
    
    print("\nAfter patch:")
    print(model)
    
    assert isinstance(model.linear1, MHCRewire)
    assert not isinstance(model.linear2, MHCRewire)
    assert not isinstance(model.linear3, MHCRewire)
    assert isinstance(model.linear4, MHCRewire)
    
    # Verify forward pass
    x = mx.random.normal((1, dims))
    y = model.linear1(x)
    assert y.shape == (1, dims)

if __name__ == "__main__":
    test_autopatcher_custom_block()
