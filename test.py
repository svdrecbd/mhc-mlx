import time
import numpy as np

import mlx.core as mx

from mhc import ManifoldConstrainedMixing, ManifoldConstrainedMixingReference


def max_abs_diff(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def main():
    mx.random.seed(0)

    B, S = 2, 16
    C = 128
    N = 4

    x = mx.random.normal((B, S, N * C)).astype(mx.bfloat16)

    fused = ManifoldConstrainedMixing(dim=C, n_streams=N, threads_per_group=256)
    ref = ManifoldConstrainedMixingReference(dim=C, n_streams=N)

    # make sure both modules share identical parameters for a fair comparison
    ref.mix_logits_proj.weight = fused.mix_logits_proj.weight
    ref.mix_logits_proj.bias = fused.mix_logits_proj.bias

    # warmup (includes Metal JIT compile)
    t0 = time.time()
    y0 = fused(x)
    mx.eval(y0)
    t1 = time.time()

    # warm timing
    t2 = time.time()
    y_fused = fused(x)
    mx.eval(y_fused)
    t3 = time.time()

    # reference timing
    t4 = time.time()
    y_ref = ref(x)
    mx.eval(y_ref)
    t5 = time.time()

    d = max_abs_diff(y_fused, y_ref)

    print("sizes:")
    print(f"  x: {tuple(x.shape)} dtype={x.dtype}")
    print("")
    print("timing:")
    print(f"  fused warmup (compile+run): {t1 - t0:.4f}s")
    print(f"  fused warm run:            {t3 - t2:.4f}s")
    print(f"  reference run:             {t5 - t4:.4f}s")
    print("")
    print("correctness:")
    print(f"  max |fused - ref|: {d:.6f}")

    # tolerance: bfloat16 output + exp + iterative normalization
    tol = 5e-2
    if d <= tol:
        print("ok")
    else:
        print("mismatch")


if __name__ == "__main__":
    main()
