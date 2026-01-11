from .layer import MHCLayer, MHCRewire
from .diagnostics import diagnostics
from .utils import residual_add_agg
from .patching import AutoPatcher
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("mhc-mlx")
except PackageNotFoundError:
    __version__ = "dev"

__all__ = ["MHCLayer", "MHCRewire", "diagnostics", "residual_add_agg", "AutoPatcher", "__version__"]