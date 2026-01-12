import platform
import sys
import os
import mlx.core as mx
from importlib import metadata

def diagnostics():
    """Prints diagnostic information for mhc-mlx."""
    try:
        mlx_version = metadata.version("mlx")
    except metadata.PackageNotFoundError:
        mlx_version = "unknown"
    
    try:
        pkg_version = metadata.version("mhc-mlx")
    except metadata.PackageNotFoundError:
        pkg_version = "dev"

    print(f"mhc-mlx version: {pkg_version}")
    print(f"MLX version:     {mlx_version}")
    print(f"Python version:  {sys.version.split()[0]}")
    print(f"Platform:        {platform.platform()}")
    print(f"Machine:         {platform.machine()}")
    
    device = mx.default_device()
    print(f"Device:          {device}")
    
    # Check for Metal availability
    if device.type == mx.cpu:
        print("\nWARNING: Default device is CPU. Metal acceleration is NOT active.")
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            print("  Note: Metal should be available on this machine. Check MLX installation.")
    elif device.type == mx.gpu:
        print("Metal acceleration: Available")

    # Check for embedded kernels
    try:
        from . import kernels_embedded
        kernels = [k for k in dir(kernels_embedded) if k.endswith("_METAL")]
        print(f"\nEmbedded kernels found: {len(kernels)}")
    except ImportError:
        print("\nERROR: Embedded kernels module not found.")

def main():
    diagnostics()

if __name__ == "__main__":
    main()
