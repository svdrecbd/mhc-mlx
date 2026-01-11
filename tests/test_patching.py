import mlx.core as mx
import mlx.nn as nn
from mhc_mlx import AutoPatcher, MHCRewire

def test_autopatcher_custom_block():
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
    AutoPatcher.patch(model, n_streams=n, verbose=True)
    
    assert isinstance(model.linear1, MHCRewire)
    assert not isinstance(model.linear2, MHCRewire)
    assert not isinstance(model.linear3, MHCRewire)
    assert isinstance(model.linear4, MHCRewire)
    
    # Verify forward pass
    x = mx.random.normal((1, dims))
    y = model.linear1(x)
    assert y.shape == (1, dims)

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