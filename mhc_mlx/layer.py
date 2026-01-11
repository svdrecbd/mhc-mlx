"""MHCLayer: MLX module with optional Metal fast path.

The forward semantics match the reference implementation in this repo:

- H_pre_act = sigmoid(H_pre_raw)
- H_post_act = 2 * sigmoid(H_post_raw)
- M = sinkhorn_knopp(exp(H_res_raw))
- y_agg = stream_aggregate(x_expanded, H_pre_act)
- y_norm = rms_norm(y_agg, rms_weight, eps)
- y_dist = stream_distribute(y_norm, H_post_act)
- out = stream_mix(x_expanded, M) + y_dist

The Metal fast path uses custom kernels for Sinkhorn plus a token-parallel
aggregate + RMS reduce + mix/add path.
Backward uses Metal kernels (no reference VJPs).
Auto-dispatch defaults to Metal with no guardrails. Guardrails only apply when
dispatch_policy is explicitly set to "latency" or "throughput". The hybrid path
is opt-in and gated on C.
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


def _dtype_eq(a, b) -> bool:
    if a is b:
        return True
    if a is None or b is None:
        return False
    try:
        return a == b
    except TypeError:
        return str(a) == str(b)


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
        dispatch_policy: str = "auto",
        hybrid_latency: bool = False,
        hybrid_min_C: int = 8192,
        latency_avoid_fused_n32_max_C: int = 2048,
        latency_avoid_fused_B1_min_n: int = 16,
        throughput_allow_fused_n32_min_B: int = 8,
        throughput_allow_fused_n32_min_C: int = 4096,
        throughput_allow_fused_n32_small_C: int = 512,
        fused_backward: bool = False,
        mix_kernel: str = "auto",
    ):
        super().__init__()

        if n <= 0:
            raise ValueError("n must be positive")
        if C <= 0:
            raise ValueError("C must be positive")
        if dispatch_policy not in {"auto", "throughput", "latency"}:
            raise ValueError("dispatch_policy must be one of: auto, throughput, latency")
        if hybrid_min_C <= 0:
            raise ValueError("hybrid_min_C must be positive")
        if latency_avoid_fused_n32_max_C <= 0:
            raise ValueError("latency_avoid_fused_n32_max_C must be positive")
        if latency_avoid_fused_B1_min_n <= 0:
            raise ValueError("latency_avoid_fused_B1_min_n must be positive")
        if throughput_allow_fused_n32_min_B <= 0:
            raise ValueError("throughput_allow_fused_n32_min_B must be positive")
        if throughput_allow_fused_n32_min_C <= 0:
            raise ValueError("throughput_allow_fused_n32_min_C must be positive")
        if throughput_allow_fused_n32_small_C < 0:
            raise ValueError("throughput_allow_fused_n32_small_C must be >= 0")
        if mix_kernel not in {"auto", "1d", "2d"}:
            raise ValueError("mix_kernel must be one of: auto, 1d, 2d")

        self.n = int(n)
        self.C = int(C)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.eps = float(eps)
        self.use_metal = bool(use_metal)
        self.auto_dispatch = bool(auto_dispatch)
        self.dispatch_policy = str(dispatch_policy)
        self.hybrid_latency = bool(hybrid_latency)
        self.hybrid_min_C = int(hybrid_min_C)
        self.latency_avoid_fused_n32_max_C = int(latency_avoid_fused_n32_max_C)
        self.latency_avoid_fused_B1_min_n = int(latency_avoid_fused_B1_min_n)
        self.throughput_allow_fused_n32_min_B = int(throughput_allow_fused_n32_min_B)
        self.throughput_allow_fused_n32_min_C = int(throughput_allow_fused_n32_min_C)
        self.throughput_allow_fused_n32_small_C = int(throughput_allow_fused_n32_small_C)
        self.fused_backward = bool(fused_backward)
        self.mix_kernel = str(mix_kernel)
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

    def _latency_policy_enabled(self) -> bool:
        return self.dispatch_policy == "latency"

    def _throughput_policy_enabled(self) -> bool:
        return self.dispatch_policy == "throughput"

    def _should_avoid_fused_latency(self, B: int, n: int, C: int) -> bool:
        if n == 32 and C <= self.latency_avoid_fused_n32_max_C:
            return True
        if B == 1 and n >= self.latency_avoid_fused_B1_min_n:
            return True
        return False

    def _should_avoid_fused_throughput(self, B: int, n: int, C: int) -> bool:
        if n != 32:
            return False
        if B < self.throughput_allow_fused_n32_min_B:
            return True
        allow_small = (
            self.throughput_allow_fused_n32_small_C > 0
            and C == self.throughput_allow_fused_n32_small_C
        )
        allow_large = C >= self.throughput_allow_fused_n32_min_C
        if allow_small or allow_large:
            return False
        return True

    def _should_avoid_fused(self, B: int, n: int, C: int) -> bool:
        if not self.use_metal or not self.auto_dispatch:
            return False
        latency_policy = self._latency_policy_enabled()
        throughput_policy = self._throughput_policy_enabled()
        if not (latency_policy or throughput_policy):
            return False
        if latency_policy and self._should_avoid_fused_latency(B, n, C):
            return True
        if throughput_policy and self._should_avoid_fused_throughput(B, n, C):
            return True
        return False

    def _should_use_hybrid(self, B: int, n: int, C: int) -> bool:
        if not self.use_metal or not self.auto_dispatch or not self.hybrid_latency:
            return False
        if not self._latency_policy_enabled():
            return False
        return n == 32 and B == 1 and C >= self.hybrid_min_C

    def _should_use_reference_fallback(self, B: int, n: int, C: int) -> bool:
        if not self.use_metal or not self.auto_dispatch:
            return False
        if not self._should_avoid_fused(B, n, C):
            return False
        return not self._should_use_hybrid(B, n, C)

    def _should_use_fused_backward(self, B: int, n: int, C: int) -> bool:
        if not self.fused_backward:
            return False
        # Token-parallel backward prep is always used in the Metal path.
        return False

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

        Note:
            When return_dtype is float16/bfloat16 and matches x_expanded.dtype,
            the Metal path emits that dtype directly.

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

        output_dtype = None
        if return_dtype is not None and _dtype_eq(return_dtype, x_expanded.dtype):
            if _dtype_eq(return_dtype, mx.float16) or _dtype_eq(return_dtype, mx.bfloat16):
                output_dtype = return_dtype

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
                fused_backward=self._should_use_fused_backward(B, n, C),
                output_dtype=output_dtype,
                mix_kernel=self.mix_kernel,
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

        if return_dtype is not None and not _dtype_eq(out.dtype, return_dtype):
            out = out.astype(return_dtype)
        return out


class MHCRewire(nn.Module):
    """Universal wrapper to add mHC manifold-constrained residuals to any module.

    Usage:
        import mlx.nn as nn
        from mhc_mlx import MHCRewire
        
        # Wrap a standard Linear layer with 16 streams
        layer = MHCRewire(nn.Linear(512, 512), dims=512, n=16)
        
        x = mx.random.normal((1, 512))
        y = layer(x) # Returns H_post * (Linear(H_pre * x) + M * H_pre * x)
    """

    def __init__(
        self,
        inner: nn.Module,
        dims: int,
        n: int = 32,
        **mhc_kwargs,
    ):
        super().__init__()
        self.inner = inner
        self.n = n
        self.dims = dims
        
        if dims % n != 0:
            raise ValueError(f"dims {dims} must be divisible by n={n}")
        
        self.mhc = MHCLayer(n=n, C=dims // n, **mhc_kwargs)

    def __call__(self, x: mx.array) -> mx.array:
        dims = x.shape[-1]
        if dims != self.dims:
            raise ValueError(f"Input dims {dims} != expected {self.dims}")
        C = dims // self.n

        # Get learnable scales and mixing matrix
        H_pre_act, H_post_act = activate_pre_post(self.mhc.H_pre_raw, self.mhc.H_post_raw)
        M = self.mhc.mixing_matrix()

        # OPTIMIZATION: Weight Folding
        if isinstance(self.inner, nn.Linear):
            h_pre_expanded = mx.repeat(H_pre_act, C)
            folded_weight = self.inner.weight * h_pre_expanded
            y_inner = x @ folded_weight.T
            if self.inner.bias is not None:
                y_inner = y_inner + self.inner.bias
        else:
            # Fallback: explicit multiply
            x_reshaped = x.reshape(-1, self.n, C)
            x_pre = x_reshaped * H_pre_act.reshape(1, self.n, 1)
            y_inner = self.inner(x_pre.reshape(x.shape))

        # OPTIMIZATION: Residual Folding
        M_folded = M * H_pre_act.reshape(1, -1)
        
        x_reshaped = x.reshape(-1, self.n, C)
        y_inner_reshaped = y_inner.reshape(-1, self.n, C)
        
        x_mixed = mx.einsum('ij,...jd->...id', M_folded, x_reshaped)
        out = (y_inner_reshaped + x_mixed) * H_post_act.reshape(1, self.n, 1)
        
        return out.reshape(x.shape)
