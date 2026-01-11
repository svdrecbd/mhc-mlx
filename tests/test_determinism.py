import mlx.core as mx
import mlx.nn as nn
import pickle
import io
from mhc_mlx import MHCLayer

def test_initialization_determinism():
    """Verify that same seed produces identical layers."""
    n, C = 16, 64
    
    mx.random.seed(42)
    layer1 = MHCLayer(n=n, C=C)
    
    mx.random.seed(42)
    layer2 = MHCLayer(n=n, C=C)
    
    # Check parameters
    for (k1, v1), (k2, v2) in zip(layer1.parameters().items(), layer2.parameters().items()):
        assert k1 == k2
        assert mx.array_equal(v1, v2), f"Parameter mismatch for {k1}"
    
    # Check output
    x = mx.random.normal((1, n, C))
    y1 = layer1(x)
    y2 = layer2(x)
    assert mx.array_equal(y1, y2), "Output mismatch between identical seeds"
    print("Determinism test passed.")

def test_pickling():
    """Verify that MHCLayer can be pickled (needed for some distributed runners)."""
    n, C = 16, 64
    layer = MHCLayer(n=n, C=C)
    
    # Pickle to byte stream
    buf = io.BytesIO()
    pickle.dump(layer, buf)
    buf.seek(0)
    
    # Unpickle
    loaded_layer = pickle.load(buf)
    
    # Verify parameters match
    for (k1, v1), (k2, v2) in zip(layer.parameters().items(), loaded_layer.parameters().items()):
        assert k1 == k2
        assert mx.array_equal(v1, v2), f"Parameter mismatch after unpickling {k1}"
        
    print("Pickling test passed.")

if __name__ == "__main__":
    test_initialization_determinism()
    test_pickling()
