import mlx.core as mx
import pytest

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


@pytest.mark.parametrize(
    "n,iters,tol",
    [
        (4, 20, 1e-5),
        (16, 10, 1e-4),
    ],
)
def test_sinkhorn_matches_reference(n: int, iters: int, tol: float) -> None:
    mx.random.seed(0)
    H_res_raw = mx.random.normal((n, n)).astype(mx.float32)

    ref = mixing_matrix_from_logits(H_res_raw, iters=iters, eps=1e-5)
    metal = sinkhorn_knopp_metal(H_res_raw, iters=iters, eps=1e-5, threads_per_group=64, verbose=False)

    err = max_abs(ref, metal)
    assert err < tol


@pytest.mark.parametrize(
    "dtype,tol",
    [
        (mx.float16, 3e-2),
        (mx.bfloat16, 2e-1),
        (mx.float32, 1e-5),
    ],
)
def test_fused_forward_matches_reference(dtype: mx.Dtype, tol: float) -> None:
    mx.random.seed(1)

    B, n, C = 2, 4, 128
    x = mx.random.normal((B, n, C)).astype(dtype)

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
        threads_per_group=256,
        verbose=False,
    )

    err = max_abs(ref.astype(mx.float32), metal.astype(mx.float32))
    assert err < tol


def test_fused_forward_matches_reference_n16() -> None:
    mx.random.seed(2)

    B, n, C = 1, 16, 64
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

    err = max_abs(ref.astype(mx.float32), metal.astype(mx.float32))
    assert err < 1e-4


@pytest.mark.parametrize(
    "dtype,tol",
    [
        (mx.float16, 3e-2),
        (mx.bfloat16, 2e-1),
    ],
)
def test_fused_forward_half_output(dtype: mx.Dtype, tol: float) -> None:
    B, n, C = 2, 4, 129
    H_pre_raw = (mx.arange(n, dtype=mx.float32) * 0.1) - 0.2
    H_post_raw = (mx.arange(n, dtype=mx.float32) * 0.07) - 0.1
    H_res_raw = (mx.arange(n * n, dtype=mx.float32).reshape(n, n) * 0.01) - 0.05
    rms_weight = mx.ones((C,), dtype=mx.float32)

    H_pre_act, H_post_act = activate_pre_post(H_pre_raw, H_post_raw)
    M = mixing_matrix_from_logits(H_res_raw, iters=10, eps=1e-5)

    x = (mx.arange(B * n * C, dtype=mx.float32).reshape(B, n, C) * 0.001).astype(dtype)
    ref = mhc_forward_reference(
        x_expanded=x,
        H_pre_raw=H_pre_raw,
        H_post_raw=H_post_raw,
        H_res_raw=H_res_raw,
        rms_weight=rms_weight,
        sinkhorn_iters=10,
        eps=1e-5,
    )
    metal = mhc_forward_fused_metal(
        x=x,
        M=M,
        H_pre=H_pre_act,
        H_post=H_post_act,
        rms_weight=rms_weight,
        eps=1e-5,
        threads_per_group=128,
        output_dtype=dtype,
        verbose=False,
    )

    err = max_abs(ref.astype(mx.float32), metal.astype(mx.float32))
    assert err < tol


def test_identity_mapping_n4() -> None:
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
    err_ref = max_abs(ref.astype(mx.float32), x_f)
    err_metal = max_abs(metal.astype(mx.float32), x_f)
    assert err_ref < 1e-4
    assert err_metal < 1e-4


@pytest.mark.parametrize("B,n,C", [(1, 4, 64), (2, 8, 128), (2, 16, 256), (1, 32, 256)])
def test_layer_forward_matches_reference(B: int, n: int, C: int) -> None:
    mx.random.seed(100 + n)

    x = mx.random.normal((B, n, C)).astype(mx.bfloat16)
    H_pre_raw = mx.random.normal((n,)).astype(mx.float32)
    H_post_raw = mx.random.normal((n,)).astype(mx.float32)
    H_res_raw = mx.random.normal((n, n)).astype(mx.float32)
    rms_weight = mx.ones((C,), dtype=mx.float32)

    ref = MHCLayer(n=n, C=C, use_metal=False, compile_reference=False)
    metal = MHCLayer(n=n, C=C, use_metal=True, auto_dispatch=False, compile_reference=False)

    for layer in (ref, metal):
        layer.H_pre_raw = H_pre_raw
        layer.H_post_raw = H_post_raw
        layer.H_res_raw = H_res_raw
        layer.rms_weight = rms_weight

    y_ref = ref(x)
    y_metal = metal(x)
    mx.eval(y_ref)
    mx.eval(y_metal)
    mx.synchronize()

    max_err = max_abs(y_ref.astype(mx.float32), y_metal.astype(mx.float32))
    rel_err = max_err / max(max_abs_val(y_ref.astype(mx.float32)), 1e-8)
    assert max_err < 1e-4
    assert rel_err < 1e-4


def test_latency_fallback_corner() -> None:
    mx.random.seed(7)

    B, n, C = 1, 32, 1024
    x = mx.random.normal((B, n, C)).astype(mx.bfloat16)

    H_pre_raw = mx.random.normal((n,)).astype(mx.float32)
    H_post_raw = mx.random.normal((n,)).astype(mx.float32)
    H_res_raw = mx.random.normal((n, n)).astype(mx.float32)
    rms_weight = mx.ones((C,), dtype=mx.float32)

    ref = MHCLayer(n=n, C=C, use_metal=False, compile_reference=False)
    auto = MHCLayer(
        n=n,
        C=C,
        use_metal=True,
        auto_dispatch=True,
        compile_reference=False,
        dispatch_policy="latency",
        hybrid_latency=True,
    )

    for layer in (ref, auto):
        layer.H_pre_raw = H_pre_raw
        layer.H_post_raw = H_post_raw
        layer.H_res_raw = H_res_raw
        layer.rms_weight = rms_weight

    y_ref = ref(x)
    y_auto = auto(x)
    mx.eval(y_ref)
    mx.eval(y_auto)
    mx.synchronize()

    max_err = max_abs(y_ref.astype(mx.float32), y_auto.astype(mx.float32))
    rel_err = max_err / max(max_abs_val(y_ref.astype(mx.float32)), 1e-8)
    assert max_err < 1e-4
    assert rel_err < 1e-4


def test_backward_matches_reference() -> None:
    mx.random.seed(9)

    B, n, C = 1, 4, 16
    x = mx.random.normal((B, n, C)).astype(mx.float32)

    H_pre_raw = mx.random.normal((n,)).astype(mx.float32)
    H_post_raw = mx.random.normal((n,)).astype(mx.float32)
    H_res_raw = mx.random.normal((n, n)).astype(mx.float32)
    rms_weight = mx.ones((C,), dtype=mx.float32)

    def ref_loss(x_in, H_pre, H_post, H_res, rms_w):
        out = mhc_forward_reference(
            x_expanded=x_in,
            H_pre_raw=H_pre,
            H_post_raw=H_post,
            H_res_raw=H_res,
            rms_weight=rms_w,
            sinkhorn_iters=5,
            eps=1e-5,
        )
        return mx.sum(out)

    def metal_loss(x_in, H_pre, H_post, H_res, rms_w):
        M = sinkhorn_knopp_metal_autograd(H_res, iters=5, eps=1e-5, threads_per_group=64)
        H_pre_act, H_post_act = activate_pre_post(H_pre, H_post)
        out = mhc_forward_fused_metal_autograd(
            x_in,
            M,
            H_pre_act,
            H_post_act,
            rms_w,
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
    assert loss_err < 1e-4

def test_reference_fallback():
    """Verify that MHC_MLX_DISABLE_METAL=1 forces the reference path."""
    import os
    os.environ["MHC_MLX_DISABLE_METAL"] = "1"
    try:
        layer = MHCLayer(n=4, C=16, use_metal=True)
        # Should be false now
        assert layer._should_use_metal(1, 4, 16) is False
        
        x = mx.random.normal((1, 4, 16))
        y = layer(x)
        mx.eval(y)
        assert y.shape == (1, 4, 16)
    finally:
        del os.environ["MHC_MLX_DISABLE_METAL"]

    

    

    
