import mlx.core as mx

from mhc_mlx.layer import MHCLayer
from mhc_mlx.metal import (
    mhc_forward_fused_metal,
    mhc_forward_fused_metal_autograd,
    sinkhorn_knopp_metal,
    sinkhorn_knopp_metal_autograd,
)
from mhc_mlx.reference import (
    activate_pre_post,
    mhc_forward_reference,
    mixing_matrix_from_logits,
)


def max_abs(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a - b)).item())


def max_abs_val(a: mx.array) -> float:
    return float(mx.max(mx.abs(a)).item())


def identity_residual(n: int, off_diag: float = 12.0) -> mx.array:
    return mx.full((n, n), -off_diag, dtype=mx.float32) + mx.eye(n, dtype=mx.float32) * off_diag


def test_sinkhorn_kernel():
    mx.random.seed(0)

    n = 4
    H_res_raw = mx.random.normal((n, n)).astype(mx.float32)

    ref = mixing_matrix_from_logits(H_res_raw, iters=20, eps=1e-5)
    metal = sinkhorn_knopp_metal(H_res_raw, iters=20, eps=1e-5, threads_per_group=64, verbose=False)

    err = max_abs(ref, metal)
    print(f"sinkhorn_knopp kernel max abs error: {err:.6e}")
    assert err < 1e-5


def test_sinkhorn_kernel_n16():
    mx.random.seed(3)

    n = 16
    H_res_raw = mx.random.normal((n, n)).astype(mx.float32)

    ref = mixing_matrix_from_logits(H_res_raw, iters=10, eps=1e-5)
    metal = sinkhorn_knopp_metal(H_res_raw, iters=10, eps=1e-5, threads_per_group=64, verbose=False)

    err = max_abs(ref, metal)
    print(f"sinkhorn_knopp kernel (n=16) max abs error: {err:.6e}")
    assert err < 1e-4


def test_fused_kernel():
    mx.random.seed(1)

    B, n, C = 2, 4, 128
    x = mx.random.normal((B, n, C)).astype(mx.bfloat16)

    H_pre_raw = mx.random.normal((n,)).astype(mx.float32)
    H_post_raw = mx.random.normal((n,)).astype(mx.float32)
    H_res_raw = mx.random.normal((n, n)).astype(mx.float32)
    rms_weight = mx.ones((C,), dtype=mx.float32)

    ref = mhc_forward_reference(
        x_expanded=x,
        H_pre_raw=H_pre_raw,
        H_post_raw=H_post_raw,
        H_res_raw=H_res_raw,
        rms_weight=rms_weight,
        sinkhorn_iters=20,
        eps=1e-5,
    )

    H_pre_act, H_post_act = activate_pre_post(H_pre_raw, H_post_raw)
    M = mixing_matrix_from_logits(H_res_raw, iters=20, eps=1e-5)
    metal = mhc_forward_fused_metal(
        x=x,
        M=M,
        H_pre=H_pre_act,
        H_post=H_post_act,
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

    H_pre_raw = mx.random.normal((n,)).astype(mx.float32)
    H_post_raw = mx.random.normal((n,)).astype(mx.float32)
    H_res_raw = mx.random.normal((n, n)).astype(mx.float32)
    rms_weight = mx.ones((C,), dtype=mx.float32)

    ref = mhc_forward_reference(
        x_expanded=x,
        H_pre_raw=H_pre_raw,
        H_post_raw=H_post_raw,
        H_res_raw=H_res_raw,
        rms_weight=rms_weight,
        sinkhorn_iters=10,
        eps=1e-5,
    )

    H_pre_act, H_post_act = activate_pre_post(H_pre_raw, H_post_raw)
    M = mixing_matrix_from_logits(H_res_raw, iters=10, eps=1e-5)
    metal = mhc_forward_fused_metal(
        x=x,
        M=M,
        H_pre=H_pre_act,
        H_post=H_post_act,
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

    H_pre_raw = mx.full((n,), -12.0, dtype=mx.float32)
    H_post_raw = mx.full((n,), -12.0, dtype=mx.float32)
    H_res_raw = identity_residual(n, off_diag=12.0)
    rms_weight = mx.ones((C,), dtype=mx.float32)

    ref = mhc_forward_reference(
        x_expanded=x,
        H_pre_raw=H_pre_raw,
        H_post_raw=H_post_raw,
        H_res_raw=H_res_raw,
        rms_weight=rms_weight,
        sinkhorn_iters=20,
        eps=1e-5,
    )

    H_pre_act, H_post_act = activate_pre_post(H_pre_raw, H_post_raw)
    M = mixing_matrix_from_logits(H_res_raw, iters=20, eps=1e-5)
    metal = mhc_forward_fused_metal(
        x=x,
        M=M,
        H_pre=H_pre_act,
        H_post=H_post_act,
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
    assert err_ref < 1e-4
    assert err_metal < 1e-4


def test_layer_forward(use_metal: bool):
    mx.random.seed(42)

    B, n, C = 2, 4, 256
    layer = MHCLayer(
        n=n,
        C=C,
        use_metal=use_metal,
        auto_dispatch=not use_metal,
        compile_reference=False,
    )

    # Make parameters deterministic and non-degenerate.
    layer.H_pre_raw = mx.ones((n,), dtype=mx.float32)
    layer.H_post_raw = mx.ones((n,), dtype=mx.float32)
    layer.H_res_raw = mx.eye(n, dtype=mx.float32)
    layer.rms_weight = mx.ones((C,), dtype=mx.float32)

    x = mx.random.normal((B, n, C)).astype(mx.bfloat16)
    y = layer(x)
    mx.eval(y)
    return y, layer, x


def test_guardrail_shapes():
    cases = [
        (1, 4, 64),
        (2, 8, 128),
        (2, 16, 256),
        (1, 32, 256),
    ]
    tol_abs = 1e-4
    tol_rel = 1e-4

    for idx, (B, n, C) in enumerate(cases, start=1):
        mx.random.seed(100 + idx)
        x = mx.random.normal((B, n, C)).astype(mx.bfloat16)

        H_pre_raw = mx.random.normal((n,)).astype(mx.float32)
        H_post_raw = mx.random.normal((n,)).astype(mx.float32)
        H_res_raw = mx.random.normal((n, n)).astype(mx.float32)
        rms_weight = mx.ones((C,), dtype=mx.float32)

        ref = MHCLayer(
            n=n,
            C=C,
            use_metal=False,
            compile_reference=False,
        )
        metal = MHCLayer(
            n=n,
            C=C,
            use_metal=True,
            auto_dispatch=False,
            compile_reference=False,
        )

        ref.H_pre_raw = H_pre_raw
        ref.H_post_raw = H_post_raw
        ref.H_res_raw = H_res_raw
        ref.rms_weight = rms_weight

        metal.H_pre_raw = H_pre_raw
        metal.H_post_raw = H_post_raw
        metal.H_res_raw = H_res_raw
        metal.rms_weight = rms_weight

        y_ref = ref(x)
        y_metal = metal(x)
        mx.eval(y_ref)
        mx.eval(y_metal)
        mx.synchronize()

        max_err = max_abs(y_ref, y_metal)
        rel_err = max_err / max(max_abs_val(y_ref), 1e-8)
        print(f"guardrail case {idx}: B={B} n={n} C={C} max_abs={max_err:.6e} rel={rel_err:.6e}")
        assert max_err < tol_abs
        assert rel_err < tol_rel


def test_latency_fallback_corner():
    mx.random.seed(7)

    B, n, C = 1, 32, 1024
    x = mx.random.normal((B, n, C)).astype(mx.bfloat16)

    H_pre_raw = mx.random.normal((n,)).astype(mx.float32)
    H_post_raw = mx.random.normal((n,)).astype(mx.float32)
    H_res_raw = mx.random.normal((n, n)).astype(mx.float32)
    rms_weight = mx.ones((C,), dtype=mx.float32)

    ref = MHCLayer(
        n=n,
        C=C,
        use_metal=False,
        compile_reference=False,
    )
    auto = MHCLayer(
        n=n,
        C=C,
        use_metal=True,
        auto_dispatch=True,
        compile_reference=False,
        hybrid_latency=True,
    )

    ref.H_pre_raw = H_pre_raw
    ref.H_post_raw = H_post_raw
    ref.H_res_raw = H_res_raw
    ref.rms_weight = rms_weight

    auto.H_pre_raw = H_pre_raw
    auto.H_post_raw = H_post_raw
    auto.H_res_raw = H_res_raw
    auto.rms_weight = rms_weight

    y_ref = ref(x)
    y_auto = auto(x)
    mx.eval(y_ref)
    mx.eval(y_auto)
    mx.synchronize()

    max_err = max_abs(y_ref, y_auto)
    rel_err = max_err / max(max_abs_val(y_ref), 1e-8)
    print(f"latency fallback corner max_abs={max_err:.6e} rel={rel_err:.6e}")
    assert max_err < 1e-4
    assert rel_err < 1e-4


def test_backward_gradients():
    mx.random.seed(9)

    B, n, C = 1, 4, 16
    x = mx.random.normal((B, n, C)).astype(mx.float32)

    H_pre_raw = mx.random.normal((n,)).astype(mx.float32)
    H_post_raw = mx.random.normal((n,)).astype(mx.float32)
    H_res_raw = mx.random.normal((n, n)).astype(mx.float32)
    rms_weight = mx.ones((C,), dtype=mx.float32)

    def ref_loss(x, H_pre_raw, H_post_raw, H_res_raw, rms_weight):
        out = mhc_forward_reference(
            x_expanded=x,
            H_pre_raw=H_pre_raw,
            H_post_raw=H_post_raw,
            H_res_raw=H_res_raw,
            rms_weight=rms_weight,
            sinkhorn_iters=5,
            eps=1e-5,
        )
        return mx.sum(out)

    def metal_loss(x, H_pre_raw, H_post_raw, H_res_raw, rms_weight):
        M = sinkhorn_knopp_metal_autograd(H_res_raw, iters=5, eps=1e-5, threads_per_group=64)
        H_pre_act, H_post_act = activate_pre_post(H_pre_raw, H_post_raw)
        out = mhc_forward_fused_metal_autograd(
            x,
            M,
            H_pre_act,
            H_post_act,
            rms_weight,
            eps=1e-5,
            threads_per_group=64,
        )
        return mx.sum(out)

    argnums = [0, 1, 2, 3, 4]
    ref_loss_val, ref_grads = mx.value_and_grad(ref_loss, argnums=argnums)(
        x, H_pre_raw, H_post_raw, H_res_raw, rms_weight
    )
    metal_loss_val, metal_grads = mx.value_and_grad(metal_loss, argnums=argnums)(
        x, H_pre_raw, H_post_raw, H_res_raw, rms_weight
    )

    mx.eval(ref_loss_val, metal_loss_val, *ref_grads, *metal_grads)
    mx.synchronize()

    loss_err = float(mx.abs(ref_loss_val - metal_loss_val).item())
    print(f"backward loss diff: {loss_err:.6e}")
    assert loss_err < 1e-4

    tol = 1e-4
    for idx, (g_ref, g_metal) in enumerate(zip(ref_grads, metal_grads), start=1):
        err = max_abs(g_ref, g_metal)
        print(f"backward grad {idx} max abs error: {err:.6e}")
        assert err < tol


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

    print("\n--- Testing guardrail shape sweep ---")
    test_guardrail_shapes()

    print("\n--- Testing latency fallback corner ---")
    test_latency_fallback_corner()

    print("\n--- Testing backward gradients ---")
    test_backward_gradients()

    print("\nAll correctness tests passed.")


if __name__ == "__main__":
    main()
