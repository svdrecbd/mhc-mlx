import mlx.core as mx
import mlx.nn as nn
from mhc_mlx import MHCRewire

def test_quantized_linear_rewire():
    """Test wrapping a QuantizedLinear layer."""
    dims = 512
    n = 16
    group_size = 64
    bits = 4
    
    # Simulate a quantized linear layer
    # Note: In MLX, we typically use nn.QuantizedLinear.from_linear
    linear = nn.Linear(dims, dims)
    qlinear = nn.QuantizedLinear.from_linear(linear, group_size=group_size, bits=bits)
    
    # Wrap it
    model = MHCRewire(qlinear, dims=dims, n=n)
    
    # Freeze the quantized layer (standard practice for quantized fine-tuning)
    # This tells MLX not to compute gradients for the inner weights.
    model.inner.freeze()
    
    x = mx.random.normal((1, dims))
    
    # Forward pass
    y = model(x)
    
    assert y.shape == (1, dims)
    print("Quantized forward pass successful.")

    # Verify only mHC parameters are trainable
    trainable = model.trainable_parameters()
    print("Trainable params:", trainable.keys())
    # Should contain 'mhc' with params
    assert 'mhc' in trainable
    assert len(trainable['mhc']) > 0
    
    # 'inner' might be present but should be empty if frozen
    if 'inner' in trainable:
        assert len(trainable['inner']) == 0, "Inner module should have no trainable params"

    # Check if we can compute gradients for the mHC parameters
    def loss_fn(params, x):
        # We need to update the model with the params that are being differentiated
        model.update(params)
        return mx.sum(model(x))
    
    # We explicitly pass only the trainable parameters to grad
    grad_fn = mx.grad(loss_fn)
    grads = grad_fn(trainable, x)
    
    assert "mhc" in grads
    assert "H_pre_raw" in grads["mhc"]
    print("Gradients for mHC parameters computed successfully.")

if __name__ == "__main__":
    test_quantized_linear_rewire()
