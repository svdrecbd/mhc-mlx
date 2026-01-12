import importlib.util
import os
import pytest
from importlib import resources

def test_mlx_mhc_alias():
    """Test that the alias package mlx_mhc works."""
    import mlx_mhc
    from mlx_mhc import MHCLayer
    assert mlx_mhc.__version__ is not None
    assert MHCLayer is not None

def test_kernels_present():
    """Test that metal kernels are present in the installed package."""
    import mhc_mlx.kernels_embedded as K
    
    kernels = [k for k in dir(K) if k.endswith("_METAL")]
    
    expected = [
        "MHC_FUSED_METAL",
        "STREAM_MIX_ADD_RMS_COL_METAL"
    ]
    
    assert len(kernels) > 0, "No embedded kernels found"
    for k in expected:
        assert k in kernels, f"Missing kernel: {k}"
