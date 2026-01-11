import mlx.core as mx
import mlx.nn as nn
from mhc_mlx import MHCRewire

class LlamaAttention(nn.Module):
    def __init__(self, dims: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.wq = nn.Linear(dims, dims, bias=False)
        self.wk = nn.Linear(dims, dims, bias=False)
        self.wv = nn.Linear(dims, dims, bias=False)
        self.wo = nn.Linear(dims, dims, bias=False)

    def __call__(self, x: mx.array, mask=None):
        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)
        # Simplified attention for mock purposes
        # In real Llama, there's RoPE, KV cache, etc.
        scale = 1 / (x.shape[-1] ** 0.5)
        scores = (queries * scale) @ keys.transpose(0, 2, 1)
        attn = mx.softmax(scores, axis=-1)
        output = attn @ values
        return self.wo(output)

class LlamaMLP(nn.Module):
    def __init__(self, dims: int, mlp_dims: int):
        super().__init__()
        self.w1 = nn.Linear(dims, mlp_dims, bias=False)
        self.w2 = nn.Linear(mlp_dims, dims, bias=False)
        self.w3 = nn.Linear(dims, mlp_dims, bias=False)

    def __call__(self, x):
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))

class LlamaBlock(nn.Module):
    def __init__(self, dims: int, num_heads: int, mlp_dims: int):
        super().__init__()
        self.n_heads = num_heads
        self.attention_norm = nn.RMSNorm(dims)
        self.attention = LlamaAttention(dims, num_heads)
        self.ffn_norm = nn.RMSNorm(dims)
        self.feed_forward = LlamaMLP(dims, mlp_dims)

    def __call__(self, x, mask=None):
        # Standard: r = attention(norm(x))
        # h = x + r
        # out = h + ffn(norm(h))
        
        # We will wrap the linear layers INSIDE these blocks using MHCRewire
        # This emulates "patching" a pretrained model
        h = x + self.attention(self.attention_norm(x), mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

def patch_model(model: nn.Module, n_streams: int = 32):
    """
    Recursively apply MHCRewire to all Linear layers in the model.
    This simulates the 'universal stability' upgrade.
    """
    # MLX Module.children() returns a dict of immediate submodules
    for name, module in model.children().items():
        if isinstance(module, nn.Linear):
            # Linear weights are [out, in]
            in_dims = module.weight.shape[1]
            out_dims = module.weight.shape[0]
            
            if in_dims == out_dims and in_dims % n_streams == 0:
                print(f"  Rewiring {name} ({in_dims}->{out_dims})")
                setattr(model, name, MHCRewire(module, dims=in_dims, n=n_streams))
        
        elif isinstance(module, nn.QuantizedLinear):
            # Infer dims from scales
            out_dims = module.scales.shape[0]
            in_dims = module.scales.shape[1] * module.group_size
            
            if in_dims == out_dims and in_dims % n_streams == 0:
                print(f"  Rewiring {name} (Quantized {in_dims}->{out_dims})")
                setattr(model, name, MHCRewire(module, dims=in_dims, n=n_streams))
                
        else:
            # Recurse
            patch_model(module, n_streams)

def main():
    print("Initializing Mock Llama Block...")
    dims = 4096
    model = LlamaBlock(dims=dims, num_heads=32, mlp_dims=11008)
    
    # Quantize one layer to prove mixed support
    print("Quantizing attention output projection (wo)...")
    model.attention.wo = nn.QuantizedLinear.from_linear(model.attention.wo, group_size=64, bits=4)
    
    x = mx.random.normal((1, 16, dims))
    
    print("\n--- Applying mHC Patch ---")
    patch_model(model, n_streams=32)
    
    print("\n--- Running Forward Pass ---")
    mx.eval(model.parameters())
    
    # Run
    y = model(x)
    mx.eval(y)
    
    print(f"Output shape: {y.shape}")
    print("Success! Llama block (mixed quantized/float) running with mHC.")

if __name__ == "__main__":
    main()
