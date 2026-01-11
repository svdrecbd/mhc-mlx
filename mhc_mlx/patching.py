import mlx.nn as nn
from .layer import MHCRewire

class AutoPatcher:
    """
    Automatic patching utility to inject mHC into existing MLX models.
    
    It traverses the model structure and wraps suitable Linear/QuantizedLinear 
    layers with MHCRewire to apply manifold-constrained stability.
    """
    
    @staticmethod
    def patch(model: nn.Module, n_streams: int = 32, verbose: bool = True):
        """
        Recursively patch the model in-place.
        
        Args:
            model: The MLX model (nn.Module) to patch.
            n_streams: Number of hyper-connection streams.
            verbose: If True, print which layers are being rewired.
        """
        count = 0
        
        # Convert to list first to ensure stable traversal while mutating
        modules = list(model.named_modules())
        
        for name, module in modules:
            # Skip the root model itself to avoid recursion issues if passed directly
            if name == "":
                continue
                
            # We look for the parent module to set the attribute
            parent_path, _, child_name = name.rpartition(".")
            
            if parent_path:
                # Retrieve parent module
                parent = model
                for segment in parent_path.split("."):
                    if segment.isdigit() and isinstance(parent, (list, tuple)):
                        parent = parent[int(segment)]
                    elif isinstance(parent, dict):
                        parent = parent[segment]
                    else:
                        parent = getattr(parent, segment)
            else:
                parent = model
                
            # Check if this module is a candidate for rewiring
            if AutoPatcher._is_patchable(module, n_streams):
                if verbose:
                    dims = AutoPatcher._get_dims(module)
                    print(f"[AutoPatcher] Rewiring {name} (dims={dims}, streams={n_streams})")
                
                # Wrap it
                dims = AutoPatcher._get_dims(module)
                wrapped = MHCRewire(module, dims=dims, n=n_streams)
                
                # Replace in parent
                if child_name.isdigit() and isinstance(parent, list):
                    parent[int(child_name)] = wrapped
                elif isinstance(parent, dict):
                    parent[child_name] = wrapped
                elif not isinstance(parent, tuple):
                    setattr(parent, child_name, wrapped)
                count += 1
                
        if verbose:
            print(f"[AutoPatcher] Patched {count} layers.")

    @staticmethod
    def _get_dims(module: nn.Module) -> int:
        if isinstance(module, nn.Linear):
            return module.weight.shape[1] # [out, in]
        elif isinstance(module, nn.QuantizedLinear):
            # Try to infer from scales: (out, in//group)
            if hasattr(module, "scales"):
                return module.scales.shape[1] * module.group_size
            # Fallback for some quantized implementations
            if hasattr(module, "input_dims"):
                return module.input_dims
        return 0

    @staticmethod
    def _is_patchable(module: nn.Module, n: int) -> bool:
        """
        Determine if a module is a valid candidate for mHC rewiring.
        Criteria:
        1. Must be Linear or QuantizedLinear.
        2. Input dimension must equal Output dimension (residual compatibility).
        3. Dimension must be divisible by n_streams.
        """
        is_linear = isinstance(module, nn.Linear)
        is_quant = isinstance(module, nn.QuantizedLinear)
        
        if not (is_linear or is_quant):
            return False
            
        in_dims = 0
        out_dims = 0
        
        if is_linear:
            in_dims = module.weight.shape[1]
            out_dims = module.weight.shape[0]
        elif is_quant:
            if hasattr(module, "scales"):
                in_dims = module.scales.shape[1] * module.group_size
                out_dims = module.scales.shape[0]
            elif hasattr(module, "input_dims"):
                in_dims = module.input_dims
                out_dims = module.output_dims
            else:
                return False

        if in_dims == 0 or out_dims == 0:
            return False
            
        # mHC fundamentally relies on the residual connection x + f(x).
        # This requires dimensions to be preserved.
        if in_dims != out_dims:
            return False
            
        # Streams must divide dimensions evenly
        if in_dims % n != 0:
            return False
            
        return True
