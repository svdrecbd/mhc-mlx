"""MHCLayer: MLX module with optional Metal fast path.

The forward semantics match the reference implementation in this repo:

- H_pre_act = sigmoid(H_pre_raw)
- H_post_act = 2 * sigmoid(H_post_raw)
- M = sinkhorn_knopp(exp(H_res_raw))
- y_agg = stream_aggregate(x_expanded, H_pre_act)
- y_norm = rms_norm(y_agg, rms_weight, eps)
- y_dist = stream_distribute(y_norm, H_post_act)
- out = stream_mix(x_expanded, M) + y_dist

The Metal fast path uses custom kernels for Sinkhorn and a fused RMSNorm/mix/add pass.
Backward uses custom VJPs that fall back to MLX reference ops for gradients.
Auto-dispatch defaults to Metal for n <= 16 and uses a latency-friendly reference
fallback for n == 32, B == 1.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .metal import (
    mhc_forward_fused_metal_autograd,
    sinkhorn_knopp_metal_autograd,
    stream_mix_add_metal_autograd,
    suggest_threads_per_group,
)
from .reference import (
    activate_pre_post,
    mhc_forward_reference,
    mixing_matrix_from_logits,
    rms_norm,
    stream_aggregate,
    stream_distribute,
)


class MHCLayer(nn.Module):
    def __init__(
        self,
        n: int,
        C: int,
        sinkhorn_iters: int = 20,
        eps: float = 1e-5,
        use_metal: bool = True,
        threads_per_group: int | None = 256,
        identity_init: bool = True,
        auto_dispatch: bool = True,
        compile_reference: bool | None = None,
        hybrid_latency: bool = True,
    ):
        super().__init__()

        if n <= 0:
            raise ValueError("n must be positive")
        if C <= 0:
            raise ValueError("C must be positive")

        self.n = int(n)
        self.C = int(C)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.eps = float(eps)
        self.use_metal = bool(use_metal)
        self.auto_dispatch = bool(auto_dispatch)
        self.hybrid_latency = bool(hybrid_latency)
        if threads_per_group is None:
            self.threads_per_group = suggest_threads_per_group(self.C)
        else:
            self.threads_per_group = int(threads_per_group)
        self.identity_init = bool(identity_init)
        if compile_reference is None:
            self.compile_reference = self.use_metal and self.auto_dispatch
        else:
            self.compile_reference = bool(compile_reference)
        self._compiled_reference = None

        # Trainable parameters (MLX treats public mx.array attributes as parameters)
        # Identity-friendly initialization keeps signal propagation well-conditioned.
        if self.identity_init:
            off_diag = 12.0
            self.H_pre_raw = mx.full((self.n,), -off_diag, dtype=mx.float32)
            self.H_post_raw = mx.full((self.n,), -off_diag, dtype=mx.float32)
            self.H_res_raw = (
                mx.full((self.n, self.n), -off_diag, dtype=mx.float32)
                + mx.eye(self.n, dtype=mx.float32) * off_diag
            )
        else:
            self.H_pre_raw = mx.zeros((self.n,), dtype=mx.float32)
            self.H_post_raw = mx.zeros((self.n,), dtype=mx.float32)
            self.H_res_raw = mx.zeros((self.n, self.n), dtype=mx.float32)
        self.rms_weight = mx.ones((self.C,), dtype=mx.float32)

    def mixing_matrix(self) -> mx.array:
        """Return the current mixing matrix M (after Sinkhorn)."""
        return mixing_matrix_from_logits(self.H_res_raw, iters=self.sinkhorn_iters, eps=self.eps)

    def _should_use_metal(self, B: int, n: int, C: int) -> bool:
        if not self.use_metal:
            return False
        if not self.auto_dispatch:
            return True
        return True

    def _should_use_hybrid(self, B: int, n: int, C: int) -> bool:
        if not self.use_metal or not self.auto_dispatch or not self.hybrid_latency:
            return False
        return n == 32 and B == 1

    def _should_use_reference_fallback(self, B: int, n: int, C: int) -> bool:
        if not self.use_metal or not self.auto_dispatch or not self.hybrid_latency:
            return False
        return n == 32 and B == 1

    def _reference_forward(
        self,
        x_expanded: mx.array,
        H_pre_raw: mx.array,
        H_post_raw: mx.array,
        H_res_raw: mx.array,
        rms_weight: mx.array,
    ) -> mx.array:
        return mhc_forward_reference(
            x_expanded=x_expanded,
            H_pre_raw=H_pre_raw,
            H_post_raw=H_post_raw,
            H_res_raw=H_res_raw,
            rms_weight=rms_weight,
            sinkhorn_iters=self.sinkhorn_iters,
            eps=self.eps,
        )

    def _hybrid_forward(
        self,
        x_expanded: mx.array,
        H_pre_raw: mx.array,
        H_post_raw: mx.array,
        H_res_raw: mx.array,
        rms_weight: mx.array,
    ) -> mx.array:
        H_pre_act, H_post_act = activate_pre_post(H_pre_raw, H_post_raw)
        M = sinkhorn_knopp_metal_autograd(
            H_res_raw,
            iters=self.sinkhorn_iters,
            eps=self.eps,
            threads_per_group=self.threads_per_group,
            verbose=False,
        )
        y_agg = stream_aggregate(x_expanded, H_pre_act)
        y_norm = rms_norm(y_agg, rms_weight, eps=self.eps)
        y_dist = stream_distribute(y_norm, H_post_act)
        return stream_mix_add_metal_autograd(
            x_expanded,
            M,
            y_dist,
            threads_per_group=self.threads_per_group,
            verbose=False,
        )

    def _get_reference_runner(self):
        if not self.compile_reference:
            return self._reference_forward
        if self._compiled_reference is None:
            compile_fn = getattr(mx, "compile", None)
            if callable(compile_fn):
                self._compiled_reference = compile_fn(self._reference_forward)
            else:
                self._compiled_reference = self._reference_forward
        return self._compiled_reference

    def __call__(self, x_expanded: mx.array, *, return_dtype: mx.Dtype | None = None) -> mx.array:
        """Run the mHC layer.

        Args:
            x_expanded: [B, n, C] input
            return_dtype: if set, cast the output to this dtype

        Returns:
            out: [B, n, C]
        """
        if x_expanded.ndim != 3:
            raise ValueError(f"x_expanded must be [B, n, C], got shape {x_expanded.shape}")
        B, n, C = x_expanded.shape
        if n != self.n:
            raise ValueError(f"x_expanded n={n} does not match layer n={self.n}")
        if C != self.C:
            raise ValueError(f"x_expanded C={C} does not match layer C={self.C}")

        if self._should_use_reference_fallback(B, n, C):
            ref_fn = self._get_reference_runner()
            out = ref_fn(
                x_expanded,
                self.H_pre_raw,
                self.H_post_raw,
                self.H_res_raw,
                self.rms_weight,
            )
        elif self._should_use_hybrid(B, n, C):
            out = self._hybrid_forward(
                x_expanded,
                self.H_pre_raw,
                self.H_post_raw,
                self.H_res_raw,
                self.rms_weight,
            )
        elif self._should_use_metal(B, n, C):
            # Use Metal kernels for Sinkhorn + fused RMSNorm/mix/add.
            M = sinkhorn_knopp_metal_autograd(
                self.H_res_raw,
                iters=self.sinkhorn_iters,
                eps=self.eps,
                threads_per_group=self.threads_per_group,
                verbose=False,
            )
            H_pre_act, H_post_act = activate_pre_post(self.H_pre_raw, self.H_post_raw)
            out = mhc_forward_fused_metal_autograd(
                x=x_expanded,
                M=M,
                H_pre=H_pre_act,
                H_post=H_post_act,
                rms_weight=self.rms_weight,
                eps=self.eps,
                threads_per_group=self.threads_per_group,
                verbose=False,
            )
        else:
            ref_fn = self._get_reference_runner()
            out = ref_fn(
                x_expanded,
                self.H_pre_raw,
                self.H_post_raw,
                self.H_res_raw,
                self.rms_weight,
            )

        if return_dtype is not None:
            out = out.astype(return_dtype)
        return out
