import mlx.core as mx

from mhc_mlx.layer import MHCLayer
from mhc_mlx.metal import mhc_forward_fused_metal, sinkhorn_knopp_metal
from mhc_mlx.reference import mhc_forward_reference, mixing_matrix_from_residual


def max_abs(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a - b)).item())


def test_sinkhorn_kernel():
    mx.random.seed(0)

    n = 4
    H_res = mx.random.normal((n, n)).astype(mx.float32)

    ref = mixing_matrix_from_residual(H_res, iters=20, eps=1e-5)
    metal = sinkhorn_knopp_metal(H_res, iters=20, eps=1e-5, threads_per_group=64, verbose=False)

    err = max_abs(ref, metal)
    print(f"sinkhorn_knopp kernel max abs error: {err:.6e}")
    assert err < 1e-5


def test_sinkhorn_kernel_n16():
    mx.random.seed(3)

    n = 16
    H_res = mx.random.normal((n, n)).astype(mx.float32)

    ref = mixing_matrix_from_residual(H_res, iters=10, eps=1e-5)
    metal = sinkhorn_knopp_metal(H_res, iters=10, eps=1e-5, threads_per_group=64, verbose=False)

    err = max_abs(ref, metal)
    print(f"sinkhorn_knopp kernel (n=16) max abs error: {err:.6e}")
    assert err < 1e-4


def test_fused_kernel():
    mx.random.seed(1)

    B, n, C = 2, 4, 128
    x = mx.random.normal((B, n, C)).astype(mx.bfloat16)

    H_pre = mx.ones((n,), dtype=mx.float32)
    H_post = mx.ones((n,), dtype=mx.float32)
    H_res = mx.random.normal((n, n)).astype(mx.float32)
    rms_weight = mx.ones((C,), dtype=mx.float32)

    ref = mhc_forward_reference(
        x_expanded=x,
        H_pre=H_pre,
        H_post=H_post,
        H_res=H_res,
        rms_weight=rms_weight,
        sinkhorn_iters=20,
        eps=1e-5,
    )

    M = mixing_matrix_from_residual(H_res, iters=20, eps=1e-5)
    metal = mhc_forward_fused_metal(
        x=x,
        M=M,
        H_pre=H_pre,
        H_post=H_post,
        rms_weight=rms_weight,
        eps=1e-5,
        threads_per_group=256,
        verbose=False,
    )

    err = max_abs(ref, metal)
    print(f"mhc_fused kernel max abs error: {err:.6e}")
    assert err < 1e-5


def test_fused_kernel_n16():
    mx.random.seed(2)

    B, n, C = 1, 16, 32
    x = mx.random.normal((B, n, C)).astype(mx.bfloat16)

    H_pre = mx.ones((n,), dtype=mx.float32)
    H_post = mx.ones((n,), dtype=mx.float32)
    H_res = mx.random.normal((n, n)).astype(mx.float32)
    rms_weight = mx.ones((C,), dtype=mx.float32)

    ref = mhc_forward_reference(
        x_expanded=x,
        H_pre=H_pre,
        H_post=H_post,
        H_res=H_res,
        rms_weight=rms_weight,
        sinkhorn_iters=10,
        eps=1e-5,
    )

    M = mixing_matrix_from_residual(H_res, iters=10, eps=1e-5)
    metal = mhc_forward_fused_metal(
        x=x,
        M=M,
        H_pre=H_pre,
        H_post=H_post,
        rms_weight=rms_weight,
        eps=1e-5,
        threads_per_group=128,
        verbose=False,
    )

    err = max_abs(ref, metal)
    print(f"mhc_fused kernel (n=16) max abs error: {err:.6e}")
    assert err < 1e-5


def test_identity_mapping_n4():
    mx.random.seed(123)

    B, n, C = 2, 4, 64
    x = mx.random.normal((B, n, C)).astype(mx.bfloat16)

    H_pre = mx.zeros((n,), dtype=mx.float32)
    H_post = mx.zeros((n,), dtype=mx.float32)
    H_res = mx.zeros((n, n), dtype=mx.float32)
    rms_weight = mx.ones((C,), dtype=mx.float32)

    ref = mhc_forward_reference(
        x_expanded=x,
        H_pre=H_pre,
        H_post=H_post,
        H_res=H_res,
        rms_weight=rms_weight,
        sinkhorn_iters=20,
        eps=1e-5,
    )

    M = mixing_matrix_from_residual(H_res, iters=20, eps=1e-5)
    metal = mhc_forward_fused_metal(
        x=x,
        M=M,
        H_pre=H_pre,
        H_post=H_post,
        rms_weight=rms_weight,
        eps=1e-5,
        threads_per_group=256,
        verbose=False,
    )

    x_f = x.astype(mx.float32)
    err_ref = max_abs(ref, x_f)
    err_metal = max_abs(metal, x_f)
    print(f"identity mapping error (reference): {err_ref:.6e}")
    print(f"identity mapping error (metal): {err_metal:.6e}")
    assert err_ref < 1e-5
    assert err_metal < 1e-5


def test_layer_forward(use_metal: bool):
    mx.random.seed(42)

    B, n, C = 2, 4, 256
    layer = MHCLayer(n=n, C=C, use_metal=use_metal)

    # Make parameters deterministic and non-degenerate.
    layer.H_pre = mx.ones((n,), dtype=mx.float32)
    layer.H_post = mx.ones((n,), dtype=mx.float32)
    layer.H_res = mx.eye(n, dtype=mx.float32)
    layer.rms_weight = mx.ones((C,), dtype=mx.float32)

    x = mx.random.normal((B, n, C)).astype(mx.bfloat16)
    y = layer(x)
    mx.eval(y)
    return y, layer, x


def main():
    print("--- Testing Sinkhorn kernel ---")
    test_sinkhorn_kernel()

    print("\n--- Testing Sinkhorn kernel (n=16) ---")
    test_sinkhorn_kernel_n16()

    print("\n--- Testing fused mhc kernel ---")
    test_fused_kernel()

    print("\n--- Testing fused mhc kernel (n=16) ---")
    test_fused_kernel_n16()

    print("\n--- Testing identity mapping (n=4) ---")
    test_identity_mapping_n4()

    print("\n--- Testing full layer: reference vs metal ---")
    y_ref, _, _ = test_layer_forward(use_metal=False)
    y_metal, _, _ = test_layer_forward(use_metal=True)

    # Evaluate before comparing
    mx.eval(y_ref)
    mx.eval(y_metal)

    err = max_abs(y_ref, y_metal)
    print(f"MHCLayer max abs error (reference vs metal): {err:.6e}")

    # Tolerance is tight because both routes use float32 for the mix.
    # If you change dtypes or add fusions, widen this only if justified.
    assert err < 1e-5

    print("\nAll correctness tests passed.")


if __name__ == "__main__":
    main()
