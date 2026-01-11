import mlx.core as mx
import mlx.nn as nn
from mhc_mlx import MHCRewire

def debug_grad():
    dims = 64
    n = 4
    mx.random.seed(0)

    model = MHCRewire(nn.Linear(dims, dims), dims=dims, n=n)
    x = mx.random.normal((1, dims))

    def loss_fn(m, x):
        return mx.sum(m(x))

    grad_fn = mx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model, x)
    print("Loss:", loss)
    print("Grad keys:", grads.keys())

if __name__ == "__main__":
    try:
        debug_grad()
        print("Success!")
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()
