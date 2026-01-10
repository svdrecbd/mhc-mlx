import os

import mlx.core as mx
import pytest

from mhc_mlx.metal import mhc_forward_fused_metal, sinkhorn_knopp_metal
from mhc_mlx.reference import activate_pre_post, mhc_forward_reference, mixing_matrix_from_logits


STRESS_ENABLED = os.getenv("MHC_MLX_RUN_STRESS", "0") == "1"


def _finite(a: mx.array) -> bool:
    return bool(mx.all(mx.isfinite(a)).item())


@pytest.mark.stress
@pytest.mark.skipif(not STRESS_ENABLED, reason="set MHC_MLX_RUN_STRESS=1 to enable")
def test_sinkhorn_eps_extremes() -> None:
    mx.random.seed(123)

    n = 8
    H_res_raw = mx.random.normal((n, n)).astype(mx.float32)

    for eps in (1e-3, 1e-6):
        ref = mixing_matrix_from_logits(H_res_raw, iters=20, eps=eps)
        metal = sinkhorn_knopp_metal(H_res_raw, iters=20, eps=eps, threads_per_group=64, verbose=False)

        assert _finite(metal)
        err = float(mx.max(mx.abs(ref - metal)).item())
        assert err < 5e-3


@pytest.mark.stress
@pytest.mark.skipif(not STRESS_ENABLED, reason="set MHC_MLX_RUN_STRESS=1 to enable")
def test_large_c_forward() -> None:
    mx.random.seed(321)

    B, n, C = 1, 8, 4096
    x = mx.random.normal((B, n, C)).astype(mx.float16)

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
        output_dtype=mx.float16,
        verbose=False,
    )

    assert _finite(metal)
    err = float(mx.max(mx.abs(ref.astype(mx.float32) - metal.astype(mx.float32))).item())
    assert err < 5e-2


@pytest.mark.stress
@pytest.mark.skipif(not STRESS_ENABLED, reason="set MHC_MLX_RUN_STRESS=1 to enable")
def test_seed_sweep() -> None:
    B, n, C = 2, 4, 256
    for seed in (1, 7, 13):
        mx.random.seed(seed)
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
            threads_per_group=256,
            verbose=False,
        )

        assert _finite(ref)
        assert _finite(metal)
