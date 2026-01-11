import mlx.core as mx
from mhc_mlx.layer import MHCLayer
import math

def test_sinkhorn_numerical_stability():
    """
    Test that Sinkhorn-Knopp survives large input values that would typically
    cause exp() overflow in standard-space implementations.
    """
    n = 32
    C = 64
    
    # Create a layer
    layer = MHCLayer(n=n, C=C, use_metal=True)
    
    # Inject "bomb" values: 
    # float32 exp(89) is approx 4e38, which overflows float32 max (3.4e38)
    # We use 100.0 to guarantee overflow if not handled correctly.
    print("Injecting large values (100.0) into H_res_raw...")
    layer.H_res_raw = mx.full((n, n), 100.0, dtype=mx.float32)
    
    # Trigger computation
    try:
        M = layer.mixing_matrix()
        mx.eval(M)
        
        # Check for NaNs
        if mx.any(mx.isnan(M)).item():
            print("FAILURE: NaNs detected in mixing matrix!")
            print(M)
            raise ValueError("Sinkhorn produced NaNs on large input")
            
        # Check if it's a valid doubly stochastic matrix (rows/cols sum to 1)
        row_sums = mx.sum(M, axis=1)
        col_sums = mx.sum(M, axis=0)
        
        row_err = mx.max(mx.abs(row_sums - 1.0)).item()
        col_err = mx.max(mx.abs(col_sums - 1.0)).item()
        
        print(f"Max Row Error: {row_err}")
        print(f"Max Col Error: {col_err}")
        
        if row_err > 1e-3 or col_err > 1e-3:
            print("FAILURE: Matrix is not doubly stochastic.")
            raise ValueError(f"Sinkhorn failed convergence. Row Err: {row_err}, Col Err: {col_err}")
            
        print("SUCCESS: Stability test passed.")
        
    except Exception as e:
        print(f"Test crashed or failed: {e}")
        # Re-raise to signal failure to pytest
        raise e

if __name__ == "__main__":
    test_sinkhorn_numerical_stability()
