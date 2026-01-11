import mlx.core as mx
from .metal import residual_add_agg_metal

def residual_add_agg(x: mx.array, res: mx.array, H_pre: mx.array) -> tuple[mx.array, mx.array]:
    """
    Fused Residual Add + mHC Aggregate.
    
    Computes:
        out = x + res
        y_agg = sum(out * H_pre)
        
    This kernel fuses the element-wise addition of the residual stream with the 
    weighted aggregation required for the mHC projection. It reduces memory 
    bandwidth usage by reading the inputs only once.
    
    Args:
        x: Input tensor [B, n, C]
        res: Residual tensor [B, n, C]
        H_pre: Pre-scaling factors [n]
        
    Returns:
        (out, y_agg) where out is [B, n, C] and y_agg is [B, C]
    """
    return residual_add_agg_metal(x, res, H_pre)
