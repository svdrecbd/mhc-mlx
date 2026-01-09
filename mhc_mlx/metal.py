"""Metal fast path.

This file contains the pieces that call mlx.core.fast.metal_kernel.
The Metal kernels live in kernels/ and are kernel bodies, not full .metal libraries.

Forward uses Metal kernels. Backward uses MLX reference ops via custom VJPs.
"""

from __future__ import annotations

import os
from functools import lru_cache

import mlx.core as mx

from .reference import mhc_forward_fused_reference, mixing_matrix_from_logits, stream_mix_ref

_KERNEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "kernels")
_STREAM_MIX_ADD_PATH = os.path.join(_KERNEL_DIR, "stream_mix_add.metal")
_SINKHORN_PATH = os.path.join(_KERNEL_DIR, "sinkhorn_knopp.metal")
_MHC_FUSED_PATH = os.path.join(_KERNEL_DIR, "mhc_fused.metal")

_MAX_N_ALLOWED = 64
_MAX_TPG_ALLOWED = 256


def _read_source(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _format_float_literal(value: float) -> str:
    literal = f"{float(value):.8g}"
    if "e" not in literal and "." not in literal:
        literal += ".0"
    return f"{literal}f"


def _render_source(path: str, **replacements: str) -> str:
    source = _read_source(path)
    for key, value in replacements.items():
        source = source.replace(f"{{{{{key}}}}}", value)
    return source


def _maybe_print_source(source: str, name: str, verbose: bool) -> None:
    if verbose:
        print(f"--- {name} kernel source ---")
        print(source)


def _as_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _match_structure(primals, grads):
    if isinstance(primals, tuple):
        return tuple(grads)
    if isinstance(primals, list):
        return grads
    return grads[0] if grads else None


@lru_cache(maxsize=8)
def _stream_mix_add_kernel(max_n: int) -> object:
    source = _stream_mix_add_source(max_n)
    return mx.fast.metal_kernel(
        name="stream_mix_add",
        input_names=["x", "M", "y_dist"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


def _stream_mix_add_source(max_n: int) -> str:
    return _render_source(
        _STREAM_MIX_ADD_PATH,
        MAX_N=str(int(max_n)),
    )


@lru_cache(maxsize=32)
def _sinkhorn_kernel(max_n: int, iters: int, eps: float) -> object:
    source = _sinkhorn_source(max_n, iters, eps)
    return mx.fast.metal_kernel(
        name="sinkhorn_knopp",
        input_names=["H_res"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


def _sinkhorn_source(max_n: int, iters: int, eps: float) -> str:
    return _render_source(
        _SINKHORN_PATH,
        MAX_N=str(int(max_n)),
        MAX_TPG=str(int(_MAX_TPG_ALLOWED)),
        ITERS=str(int(iters)),
        EPS=_format_float_literal(eps),
    )


@lru_cache(maxsize=16)
def _mhc_fused_kernel(max_n: int, eps: float) -> object:
    source = _mhc_fused_source(max_n, eps)
    return mx.fast.metal_kernel(
        name="mhc_fused",
        input_names=["x", "M", "H_pre", "H_post", "rms_weight"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


def _mhc_fused_source(max_n: int, eps: float) -> str:
    return _render_source(
        _MHC_FUSED_PATH,
        MAX_N=str(int(max_n)),
        MAX_TPG=str(int(_MAX_TPG_ALLOWED)),
        EPS=_format_float_literal(eps),
    )


def _validate_n(n: int) -> int:
    if n <= 0:
        raise ValueError("n must be positive")
    if n > _MAX_N_ALLOWED:
        raise ValueError(
            f"n={n} exceeds MAX_N_ALLOWED={_MAX_N_ALLOWED}. Increase _MAX_N_ALLOWED if needed."
        )
    return int(n)


def suggest_threads_per_group(C: int, max_tpg: int = _MAX_TPG_ALLOWED) -> int:
    """Heuristic threadgroup size based on channel count."""
    if C <= 0:
        raise ValueError("C must be positive")
    if max_tpg <= 0:
        raise ValueError("max_tpg must be positive")
    tpg = 1 << (int(C) - 1).bit_length()
    if tpg > max_tpg:
        tpg = int(max_tpg)
    return int(tpg)


def stream_mix_add_metal(
    x: mx.array,
    M: mx.array,
    y_dist: mx.array,
    threads_per_group: int = 256,
    verbose: bool = False,
) -> mx.array:
    """Compute out = stream_mix(x, M) + y_dist using a Metal kernel.

    Args:
        x: [B, n, C] float32, row contiguous
        M: [n, n] float32
        y_dist: [B, n, C] float32
        threads_per_group: threadgroup size along x
        verbose: if True, print the kernel body source

    Returns:
        out: [B, n, C] float32
    """
    if x.ndim != 3:
        raise ValueError(f"x must be [B, n, C], got shape {x.shape}")
    if M.ndim != 2:
        raise ValueError(f"M must be [n, n], got shape {M.shape}")
    if y_dist.shape != x.shape:
        raise ValueError(f"y_dist must match x shape, got {y_dist.shape} vs {x.shape}")

    B, n, C = x.shape
    if M.shape != (n, n):
        raise ValueError(f"M must be shape (n,n)=( {n},{n} ), got {M.shape}")

    max_n = _validate_n(n)
    if threads_per_group <= 0:
        raise ValueError("threads_per_group must be positive")
    if threads_per_group > _MAX_TPG_ALLOWED:
        raise ValueError(f"threads_per_group must be <= {_MAX_TPG_ALLOWED}")

    if verbose:
        _maybe_print_source(_stream_mix_add_source(max_n), "stream_mix_add", verbose=True)

    # Force float32 for the kernel.
    x_f = x.astype(mx.float32)
    M_f = M.astype(mx.float32)
    y_f = y_dist.astype(mx.float32)

    kernel = _stream_mix_add_kernel(max_n)

    # Grid:
    # - x dimension: channels C
    # - y dimension: B*n (each y index corresponds to one (b, i) pair)
    out = kernel(
        inputs=[x_f, M_f, y_f],
        grid=(C, B * n, 1),
        threadgroup=(threads_per_group, 1, 1),
        output_shapes=[x_f.shape],
        output_dtypes=[mx.float32],
    )[0]

    return out


def sinkhorn_knopp_metal(
    H_res_raw: mx.array,
    iters: int = 20,
    eps: float = 1e-5,
    threads_per_group: int = 64,
    verbose: bool = False,
) -> mx.array:
    """Compute M = sinkhorn_knopp(exp(H_res_raw)) using a Metal kernel.

    Args:
        H_res_raw: [n, n] residual logits
        iters: number of Sinkhorn iterations
        eps: stability epsilon
        threads_per_group: threadgroup size along x
        verbose: if True, print the kernel body source

    Returns:
        M: [n, n] float32
    """
    if H_res_raw.ndim != 2 or H_res_raw.shape[0] != H_res_raw.shape[1]:
        raise ValueError(f"H_res_raw must be square [n, n], got shape {H_res_raw.shape}")

    n = _validate_n(H_res_raw.shape[0])
    if threads_per_group <= 0:
        raise ValueError("threads_per_group must be positive")
    if threads_per_group > _MAX_TPG_ALLOWED:
        raise ValueError(f"threads_per_group must be <= {_MAX_TPG_ALLOWED}")

    if verbose:
        _maybe_print_source(_sinkhorn_source(n, iters, eps), "sinkhorn_knopp", verbose=True)

    H_f = H_res_raw.astype(mx.float32)

    kernel = _sinkhorn_kernel(n, int(iters), float(eps))
    out = kernel(
        inputs=[H_f],
        grid=(threads_per_group, 1, 1),
        threadgroup=(threads_per_group, 1, 1),
        output_shapes=[H_f.shape],
        output_dtypes=[mx.float32],
    )[0]

    return out


def mhc_forward_fused_metal(
    x: mx.array,
    M: mx.array,
    H_pre: mx.array,
    H_post: mx.array,
    rms_weight: mx.array,
    eps: float = 1e-5,
    threads_per_group: int = 256,
    verbose: bool = False,
) -> mx.array:
    """Fused Metal forward: RMSNorm + stream mix + add(y_dist).

    Args:
        x: [B, n, C] float32, row contiguous
        M: [n, n] float32
        H_pre: [n] float32 (activated)
        H_post: [n] float32 (activated)
        rms_weight: [C] float32
        eps: RMSNorm epsilon
        threads_per_group: threadgroup size along x
        verbose: if True, print the kernel body source

    Returns:
        out: [B, n, C] float32
    """
    if x.ndim != 3:
        raise ValueError(f"x must be [B, n, C], got shape {x.shape}")
    if M.ndim != 2:
        raise ValueError(f"M must be [n, n], got shape {M.shape}")
    if H_pre.ndim != 1:
        raise ValueError(f"H_pre must be [n], got shape {H_pre.shape}")
    if H_post.ndim != 1:
        raise ValueError(f"H_post must be [n], got shape {H_post.shape}")
    if rms_weight.ndim != 1:
        raise ValueError(f"rms_weight must be [C], got shape {rms_weight.shape}")

    B, n, C = x.shape
    if M.shape != (n, n):
        raise ValueError(f"M must be shape (n,n)=( {n},{n} ), got {M.shape}")
    if H_pre.shape != (n,):
        raise ValueError(f"H_pre must be shape (n,)=( {n}, ), got {H_pre.shape}")
    if H_post.shape != (n,):
        raise ValueError(f"H_post must be shape (n,)=( {n}, ), got {H_post.shape}")
    if rms_weight.shape != (C,):
        raise ValueError(f"rms_weight must be shape (C,)=( {C}, ), got {rms_weight.shape}")

    max_n = _validate_n(n)
    if threads_per_group <= 0:
        raise ValueError("threads_per_group must be positive")
    if threads_per_group > _MAX_TPG_ALLOWED:
        raise ValueError(f"threads_per_group must be <= {_MAX_TPG_ALLOWED} for the fused kernel")

    if verbose:
        _maybe_print_source(_mhc_fused_source(max_n, eps), "mhc_fused", verbose=True)

    x_f = x.astype(mx.float32)
    M_f = M.astype(mx.float32)
    H_pre_f = H_pre.astype(mx.float32)
    H_post_f = H_post.astype(mx.float32)
    rms_weight_f = rms_weight.astype(mx.float32)

    kernel = _mhc_fused_kernel(max_n, float(eps))

    out = kernel(
        inputs=[x_f, M_f, H_pre_f, H_post_f, rms_weight_f],
        grid=(threads_per_group, B, 1),
        threadgroup=(threads_per_group, 1, 1),
        output_shapes=[x_f.shape],
        output_dtypes=[mx.float32],
    )[0]

    return out


@lru_cache(maxsize=32)
def _stream_mix_add_autograd_fn(threads_per_group: int):
    @mx.custom_function
    def _f(x, M, y_dist):
        return stream_mix_add_metal(
            x,
            M,
            y_dist,
            threads_per_group=threads_per_group,
            verbose=False,
        )

    @_f.vjp
    def _f_vjp(primals, cotangents, _):
        x, M, y_dist = primals
        dout = _as_list(cotangents)[0]

        def ref_fn(x, M, y_dist):
            return stream_mix_ref(x, M) + y_dist

        _, grads = mx.vjp(ref_fn, [x, M, y_dist], [dout])
        return _match_structure(primals, grads)

    return _f


def stream_mix_add_metal_autograd(
    x: mx.array,
    M: mx.array,
    y_dist: mx.array,
    threads_per_group: int = 256,
    verbose: bool = False,
) -> mx.array:
    """Stream mix + add with a custom VJP for gradients."""
    if verbose:
        _maybe_print_source(_stream_mix_add_source(_validate_n(M.shape[0])), "stream_mix_add", verbose=True)
    return _stream_mix_add_autograd_fn(int(threads_per_group))(x, M, y_dist)


@lru_cache(maxsize=32)
def _sinkhorn_autograd_fn(iters: int, eps: float, threads_per_group: int):
    @mx.custom_function
    def _f(H_res_raw):
        return sinkhorn_knopp_metal(
            H_res_raw,
            iters=iters,
            eps=eps,
            threads_per_group=threads_per_group,
            verbose=False,
        )

    @_f.vjp
    def _f_vjp(primals, cotangents, _):
        if isinstance(primals, (list, tuple)):
            H_res_raw = primals[0]
        else:
            H_res_raw = primals
        dM = _as_list(cotangents)[0]

        def ref_fn(h):
            return mixing_matrix_from_logits(h, iters=iters, eps=eps)

        _, grads = mx.vjp(ref_fn, [H_res_raw], [dM])
        return _match_structure(primals, grads)

    return _f


def sinkhorn_knopp_metal_autograd(
    H_res_raw: mx.array,
    iters: int = 20,
    eps: float = 1e-5,
    threads_per_group: int = 64,
    verbose: bool = False,
) -> mx.array:
    """Sinkhorn with a custom VJP for gradients."""
    if verbose:
        _maybe_print_source(_sinkhorn_source(_validate_n(H_res_raw.shape[0]), iters, eps), "sinkhorn_knopp", True)
    return _sinkhorn_autograd_fn(int(iters), float(eps), int(threads_per_group))(H_res_raw)


@lru_cache(maxsize=32)
def _mhc_fused_autograd_fn(eps: float, threads_per_group: int):
    @mx.custom_function
    def _f(x, M, H_pre, H_post, rms_weight):
        return mhc_forward_fused_metal(
            x,
            M,
            H_pre,
            H_post,
            rms_weight,
            eps=eps,
            threads_per_group=threads_per_group,
            verbose=False,
        )

    @_f.vjp
    def _f_vjp(primals, cotangents, _):
        x, M, H_pre, H_post, rms_weight = primals
        dout = _as_list(cotangents)[0]

        def ref_fn(x, M, H_pre, H_post, rms_weight):
            return mhc_forward_fused_reference(x, M, H_pre, H_post, rms_weight, eps=eps)

        _, grads = mx.vjp(ref_fn, [x, M, H_pre, H_post, rms_weight], [dout])
        return _match_structure(primals, grads)

    return _f


def mhc_forward_fused_metal_autograd(
    x: mx.array,
    M: mx.array,
    H_pre: mx.array,
    H_post: mx.array,
    rms_weight: mx.array,
    eps: float = 1e-5,
    threads_per_group: int = 256,
    verbose: bool = False,
) -> mx.array:
    """Fused forward with a custom VJP for gradients."""
    if verbose:
        _maybe_print_source(_mhc_fused_source(_validate_n(M.shape[0]), eps), "mhc_fused", True)
    return _mhc_fused_autograd_fn(float(eps), int(threads_per_group))(
        x, M, H_pre, H_post, rms_weight
    )
