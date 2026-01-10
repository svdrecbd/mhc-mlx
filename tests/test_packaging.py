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
    # This works for python 3.10+
    try:
        kernel_files = [
            p.name for p in resources.files("mhc_mlx.kernels").iterdir() 
            if p.name.endswith(".metal")
        ]
    except (AttributeError, TypeError):
        # Fallback or if not installed as package
        import mhc_mlx
        kernel_dir = os.path.join(os.path.dirname(mhc_mlx.__file__), "kernels")
        if os.path.exists(kernel_dir):
            kernel_files = [f for f in os.listdir(kernel_dir) if f.endswith(".metal")]
        else:
            kernel_files = []

    expected = [
        "mhc_fused.metal",
        "stream_mix_add_rms_col.metal"
    ]
    
    assert len(kernel_files) > 0, "No kernel files found"
    for k in expected:
        assert k in kernel_files, f"Missing kernel: {k}"
