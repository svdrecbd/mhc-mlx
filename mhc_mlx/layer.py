"""MHCLayer: MLX module with optional Metal fast path.

The forward semantics match the CUDA reference implementation:

- M = sinkhorn_knopp(H_res)
- y_agg = stream_aggregate(x_expanded, H_pre)
- y_norm = rms_norm(y_agg, rms_weight, eps)
- y_dist = stream_distribute(y_norm, H_post)
- out = stream_mix(x_expanded, M) + y_dist

The Metal fast path uses custom kernels for Sinkhorn and a fused RMSNorm/mix/add pass.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .metal import mhc_forward_fused_metal, sinkhorn_knopp_metal, suggest_threads_per_group
from .reference import (
    mhc_forward_reference,
    mixing_matrix_from_residual,
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
        if threads_per_group is None:
            self.threads_per_group = suggest_threads_per_group(self.C)
        else:
            self.threads_per_group = int(threads_per_group)
        self.identity_init = bool(identity_init)

        # Trainable parameters (MLX treats public mx.array attributes as parameters)
        # Identity-friendly initialization keeps signal propagation well-conditioned.
        if self.identity_init:
            self.H_pre = mx.zeros((self.n,), dtype=mx.float32)
            self.H_post = mx.zeros((self.n,), dtype=mx.float32)
            self.H_res = mx.zeros((self.n, self.n), dtype=mx.float32)
        else:
            self.H_pre = mx.ones((self.n,), dtype=mx.float32)
            self.H_post = mx.ones((self.n,), dtype=mx.float32)
            self.H_res = mx.eye(self.n, dtype=mx.float32)
        self.rms_weight = mx.ones((self.C,), dtype=mx.float32)

    def mixing_matrix(self) -> mx.array:
        """Return the current mixing matrix M (after Sinkhorn)."""
        return mixing_matrix_from_residual(self.H_res, iters=self.sinkhorn_iters, eps=self.eps)

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

        if not self.use_metal:
            out = mhc_forward_reference(
                x_expanded=x_expanded,
                H_pre=self.H_pre,
                H_post=self.H_post,
                H_res=self.H_res,
                rms_weight=self.rms_weight,
                sinkhorn_iters=self.sinkhorn_iters,
                eps=self.eps,
            )
        else:
            # Use Metal kernels for Sinkhorn + fused RMSNorm/mix/add.
            M = sinkhorn_knopp_metal(
                self.H_res,
                iters=self.sinkhorn_iters,
                eps=self.eps,
                threads_per_group=self.threads_per_group,
                verbose=False,
            )
            out = mhc_forward_fused_metal(
                x=x_expanded,
                M=M,
                H_pre=self.H_pre,
                H_post=self.H_post,
                rms_weight=self.rms_weight,
                eps=self.eps,
                threads_per_group=self.threads_per_group,
                verbose=False,
            )

        if return_dtype is not None:
            out = out.astype(return_dtype)
        return out
