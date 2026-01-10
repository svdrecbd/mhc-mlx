"""Metal fast path.

This file contains the pieces that call mlx.core.fast.metal_kernel.
The Metal kernels live in mhc_mlx/kernels/ and are kernel bodies, not full .metal libraries.

Forward and backward use Metal kernels.
"""

from __future__ import annotations

import hashlib
import os
import platform
from functools import lru_cache

import mlx.core as mx

_KERNEL_DIR = os.path.join(os.path.dirname(__file__), "kernels")
_STREAM_MIX_ADD_PATH = os.path.join(_KERNEL_DIR, "stream_mix_add.metal")
_STREAM_MIX_ADD_RMS_PATH = os.path.join(_KERNEL_DIR, "stream_mix_add_rms.metal")
_STREAM_MIX_ADD_RMS_FP16_PATH = os.path.join(_KERNEL_DIR, "stream_mix_add_rms_fp16.metal")
_STREAM_MIX_ADD_RMS_BF16_PATH = os.path.join(_KERNEL_DIR, "stream_mix_add_rms_bf16.metal")
_STREAM_MIX_ADD_RMS_TILE_PATH = os.path.join(_KERNEL_DIR, "stream_mix_add_rms_tile.metal")
_STREAM_MIX_ADD_RMS_TILE2D_FP16_PATH = os.path.join(_KERNEL_DIR, "stream_mix_add_rms_tile2d_fp16.metal")
_STREAM_MIX_ADD_RMS_TILE2D_BF16_PATH = os.path.join(_KERNEL_DIR, "stream_mix_add_rms_tile2d_bf16.metal")
_STREAM_MIX_ADD_RMS_TILE_F32_PATH = os.path.join(_KERNEL_DIR, "stream_mix_add_rms_tile_f32.metal")
_STREAM_MIX_ADD_RMS_TILE_FP16_PATH = os.path.join(_KERNEL_DIR, "stream_mix_add_rms_tile_fp16.metal")
_STREAM_MIX_ADD_RMS_TILE_BF16_PATH = os.path.join(_KERNEL_DIR, "stream_mix_add_rms_tile_bf16.metal")
_STREAM_MIX_ADD_RMS_COL_PATH = os.path.join(_KERNEL_DIR, "stream_mix_add_rms_col.metal")
_STREAM_MIX_ADD_RMS_COL_BF16_PATH = os.path.join(_KERNEL_DIR, "stream_mix_add_rms_col_bf16.metal")
_SINKHORN_PATH = os.path.join(_KERNEL_DIR, "sinkhorn_knopp.metal")
_SINKHORN_BACKWARD_PATH = os.path.join(_KERNEL_DIR, "sinkhorn_knopp_backward.metal")
_MHC_FUSED_PATH = os.path.join(_KERNEL_DIR, "mhc_fused.metal")
_MHC_FORWARD_AGG_PATH = os.path.join(_KERNEL_DIR, "mhc_forward_agg.metal")
_MHC_FORWARD_AGG_BF16_PATH = os.path.join(_KERNEL_DIR, "mhc_forward_agg_bf16.metal")
_MHC_FORWARD_RMS_REDUCE_PATH = os.path.join(_KERNEL_DIR, "mhc_forward_rms_reduce.metal")
_MHC_BACKWARD_PREP_PATH = os.path.join(_KERNEL_DIR, "mhc_backward_prep.metal")
_MHC_BACKWARD_PREP_TILE_PATH = os.path.join(_KERNEL_DIR, "mhc_backward_prep_tile.metal")
_MHC_BACKWARD_DX_PATH = os.path.join(_KERNEL_DIR, "mhc_backward_dx.metal")
_MHC_BACKWARD_DX_COL_PATH = os.path.join(_KERNEL_DIR, "mhc_backward_dx_col.metal")
_MHC_BACKWARD_GRADS_FUSED_PATH = os.path.join(_KERNEL_DIR, "mhc_backward_grads_fused.metal")
_MHC_BACKWARD_FUSED_DX_PATH = os.path.join(_KERNEL_DIR, "mhc_backward_fused_dx.metal")
_MHC_BACKWARD_DM_PATH = os.path.join(_KERNEL_DIR, "mhc_backward_dM.metal")
_MHC_BACKWARD_DH_PRE_PATH = os.path.join(_KERNEL_DIR, "mhc_backward_dH_pre.metal")
_MHC_BACKWARD_DH_POST_PATH = os.path.join(_KERNEL_DIR, "mhc_backward_dH_post.metal")
_MHC_BACKWARD_DH_PRE_POST_PATH = os.path.join(_KERNEL_DIR, "mhc_backward_dH_pre_post.metal")
_MHC_BACKWARD_DRMS_PATH = os.path.join(_KERNEL_DIR, "mhc_backward_d_rms_weight.metal")
_MHC_BACKWARD_RMS_REDUCE_PATH = os.path.join(_KERNEL_DIR, "mhc_backward_rms_reduce.metal")
_STREAM_MIX_BACKWARD_DX_PATH = os.path.join(_KERNEL_DIR, "stream_mix_backward_dx.metal")

_MAX_N_ALLOWED = 64
_MAX_TPG_ALLOWED = 1024


def _max_threads_per_threadgroup() -> int:
    return 1024


def _tile_tpg_x(n: int, requested: int) -> int:
    if n <= 0:
        raise ValueError("n must be positive")
    if requested <= 0:
        raise ValueError("requested must be positive")
    cap = min(int(_MAX_TPG_ALLOWED), int(_max_threads_per_threadgroup() // int(n)))
    if cap <= 0:
        return 1
    target = min(int(requested), int(cap))
    tpg = 1 << (target.bit_length() - 1)
    return max(1, int(tpg))


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


def _metal_device_name() -> str | None:
    try:
        metal = getattr(mx, "metal", None)
        if metal is None:
            return None
        device_fn = getattr(metal, "device", None)
        if device_fn is None:
            return None
        device = device_fn()
        name = getattr(device, "name", None)
        if name:
            return str(name)
    except Exception:
        return None
    return None


@lru_cache(maxsize=1)
def _kernel_cache_key() -> str:
    override = os.getenv("MHC_MLX_KERNEL_CACHE_KEY")
    if override:
        return override

    parts = []
    mlx_version = getattr(mx, "__version__", "unknown")
    parts.append(f"mlx={mlx_version}")
    macos_version = platform.mac_ver()[0] or platform.platform()
    parts.append(f"macos={macos_version}")
    parts.append(f"machine={platform.machine()}")

    if os.getenv("MHC_MLX_KERNEL_CACHE_INCLUDE_DEVICE", "1") == "1":
        device = _metal_device_name()
        if device:
            parts.append(f"gpu={device}")

    return "|".join(parts)


def _kernel_name(base: str, source: str) -> str:
    cache_key = _kernel_cache_key()
    src_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()[:8]
    key_hash = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:8]
    return f"{base}_{src_hash}_{key_hash}"


def _make_kernel(*, name: str, source: str, input_names, output_names, **kwargs) -> object:
    kwargs.setdefault("ensure_row_contiguous", True)
    return mx.fast.metal_kernel(
        name=_kernel_name(name, source),
        input_names=input_names,
        output_names=output_names,
        source=source,
        **kwargs,
    )


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
    return _make_kernel(
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


@lru_cache(maxsize=8)
def _stream_mix_add_rms_kernel(max_n: int) -> object:
    source = _stream_mix_add_rms_source(max_n)
    return _make_kernel(
        name="stream_mix_add_rms",
        input_names=["x", "M", "H_post", "y_agg", "inv_rms", "rms_weight"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


def _stream_mix_add_rms_source(max_n: int) -> str:
    return _render_source(
        _STREAM_MIX_ADD_RMS_PATH,
        MAX_N=str(int(max_n)),
    )


@lru_cache(maxsize=8)
def _stream_mix_add_rms_fp16_kernel(max_n: int) -> object:
    source = _stream_mix_add_rms_fp16_source(max_n)
    return _make_kernel(
        name="stream_mix_add_rms_fp16",
        input_names=["x", "M", "H_post", "y_agg", "inv_rms", "rms_weight"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


def _stream_mix_add_rms_fp16_source(max_n: int) -> str:
    return _render_source(
        _STREAM_MIX_ADD_RMS_FP16_PATH,
        MAX_N=str(int(max_n)),
    )


@lru_cache(maxsize=8)
def _stream_mix_add_rms_bf16_kernel(max_n: int) -> object:
    source = _stream_mix_add_rms_bf16_source(max_n)
    return _make_kernel(
        name="stream_mix_add_rms_bf16",
        input_names=["x", "M", "H_post", "y_agg", "inv_rms", "rms_weight"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


def _stream_mix_add_rms_bf16_source(max_n: int) -> str:
    return _render_source(
        _STREAM_MIX_ADD_RMS_BF16_PATH,
        MAX_N=str(int(max_n)),
    )


@lru_cache(maxsize=8)
def _stream_mix_add_rms_tile_kernel(max_n: int) -> object:
    source = _stream_mix_add_rms_tile_source(max_n)
    return _make_kernel(
        name="stream_mix_add_rms_tile",
        input_names=["x", "M", "H_post", "y_agg", "inv_rms", "rms_weight"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


def _stream_mix_add_rms_tile_source(max_n: int) -> str:
    return _render_source(
        _STREAM_MIX_ADD_RMS_TILE_PATH,
        MAX_N=str(int(max_n)),
    )


@lru_cache(maxsize=8)
def _stream_mix_add_rms_tile2d_fp16_kernel(max_n: int) -> object:
    source = _stream_mix_add_rms_tile2d_fp16_source(max_n)
    return _make_kernel(
        name="stream_mix_add_rms_tile2d_fp16",
        input_names=["x", "M", "H_post", "y_agg", "inv_rms", "rms_weight"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


def _stream_mix_add_rms_tile2d_fp16_source(max_n: int) -> str:
    return _render_source(
        _STREAM_MIX_ADD_RMS_TILE2D_FP16_PATH,
        MAX_N=str(int(max_n)),
    )


@lru_cache(maxsize=8)
def _stream_mix_add_rms_tile2d_bf16_kernel(max_n: int) -> object:
    source = _stream_mix_add_rms_tile2d_bf16_source(max_n)
    return _make_kernel(
        name="stream_mix_add_rms_tile2d_bf16",
        input_names=["x", "M", "H_post", "y_agg", "inv_rms", "rms_weight"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


def _stream_mix_add_rms_tile2d_bf16_source(max_n: int) -> str:
    return _render_source(
        _STREAM_MIX_ADD_RMS_TILE2D_BF16_PATH,
        MAX_N=str(int(max_n)),
    )


@lru_cache(maxsize=8)
def _stream_mix_add_rms_col_kernel(max_n: int) -> object:
    source = _stream_mix_add_rms_col_source(max_n)
    return _make_kernel(
        name="stream_mix_add_rms_col",
        input_names=["x", "M", "H_post", "y_agg", "inv_rms", "rms_weight"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


def _stream_mix_add_rms_col_source(max_n: int) -> str:
    return _render_source(
        _STREAM_MIX_ADD_RMS_COL_PATH,
        MAX_N=str(int(max_n)),
    )


@lru_cache(maxsize=8)
def _stream_mix_add_rms_col_bf16_kernel(max_n: int) -> object:
    source = _stream_mix_add_rms_col_bf16_source(max_n)
    return _make_kernel(
        name="stream_mix_add_rms_col_bf16",
        input_names=["x", "M", "H_post", "y_agg", "inv_rms", "rms_weight"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


def _stream_mix_add_rms_col_bf16_source(max_n: int) -> str:
    return _render_source(
        _STREAM_MIX_ADD_RMS_COL_BF16_PATH,
        MAX_N=str(int(max_n)),
    )


@lru_cache(maxsize=8)
def _stream_mix_add_rms_tile_f32_kernel(tile_n: int, tile_c: int) -> object:
    source = _stream_mix_add_rms_tile_f32_source(tile_n, tile_c)
    return _make_kernel(
        name="stream_mix_add_rms_tile_f32",
        input_names=["x", "M", "H_post", "y_agg", "inv_rms", "rms_weight"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


def _stream_mix_add_rms_tile_f32_source(tile_n: int, tile_c: int) -> str:
    return _render_source(
        _STREAM_MIX_ADD_RMS_TILE_F32_PATH,
        TILE_N=str(int(tile_n)),
        TILE_C=str(int(tile_c)),
    )


@lru_cache(maxsize=8)
def _stream_mix_add_rms_tile_fp16_kernel(tile_n: int, tile_c: int) -> object:
    source = _stream_mix_add_rms_tile_fp16_source(tile_n, tile_c)
    return _make_kernel(
        name="stream_mix_add_rms_tile_fp16",
        input_names=["x", "M", "H_post", "y_agg", "inv_rms", "rms_weight"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


def _stream_mix_add_rms_tile_fp16_source(tile_n: int, tile_c: int) -> str:
    return _render_source(
        _STREAM_MIX_ADD_RMS_TILE_FP16_PATH,
        TILE_N=str(int(tile_n)),
        TILE_C=str(int(tile_c)),
    )


@lru_cache(maxsize=8)
def _stream_mix_add_rms_tile_bf16_kernel(tile_n: int, tile_c: int) -> object:
    source = _stream_mix_add_rms_tile_bf16_source(tile_n, tile_c)
    return _make_kernel(
        name="stream_mix_add_rms_tile_bf16",
        input_names=["x", "M", "H_post", "y_agg", "inv_rms", "rms_weight"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


def _stream_mix_add_rms_tile_bf16_source(tile_n: int, tile_c: int) -> str:
    return _render_source(
        _STREAM_MIX_ADD_RMS_TILE_BF16_PATH,
        TILE_N=str(int(tile_n)),
        TILE_C=str(int(tile_c)),
    )


@lru_cache(maxsize=32)
def _sinkhorn_kernel(max_n: int, iters: int, eps: float) -> object:
    source = _sinkhorn_source(max_n, iters, eps)
    return _make_kernel(
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
def _mhc_fused_kernel(max_n: int, eps: float, output_dtype: mx.Dtype | None = None) -> object:
    source = _mhc_fused_source(max_n, eps, output_dtype)
    return _make_kernel(
        name="mhc_fused",
        input_names=["x", "M", "H_pre", "H_post", "rms_weight"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


def _mhc_fused_source(max_n: int, eps: float, output_dtype: mx.Dtype | None = None) -> str:
    out_t = "float"
    if output_dtype is not None:
        if _dtype_eq(output_dtype, mx.float16):
            out_t = "half"
        elif _dtype_eq(output_dtype, mx.bfloat16):
            out_t = "bfloat"

    return _render_source(
        _MHC_FUSED_PATH,
        MAX_N=str(int(max_n)),
        MAX_TPG=str(int(_MAX_TPG_ALLOWED)),
        EPS=_format_float_literal(eps),
        OUT_T=out_t,
    )


@lru_cache(maxsize=16)
def _mhc_forward_agg_kernel(max_n: int) -> object:
    source = _mhc_forward_agg_source(max_n)
    return _make_kernel(
        name="mhc_forward_agg",
        input_names=["x", "H_pre"],
        output_names=["y_agg", "partial_sq"],
        source=source,
        ensure_row_contiguous=True,
    )


def _mhc_forward_agg_source(max_n: int) -> str:
    return _render_source(
        _MHC_FORWARD_AGG_PATH,
        MAX_N=str(int(max_n)),
        MAX_TPG=str(int(_MAX_TPG_ALLOWED)),
    )


@lru_cache(maxsize=16)
def _mhc_forward_agg_bf16_kernel(max_n: int) -> object:
    source = _mhc_forward_agg_bf16_source(max_n)
    return _make_kernel(
        name="mhc_forward_agg_bf16",
        input_names=["x", "H_pre"],
        output_names=["y_agg", "partial_sq"],
        source=source,
        ensure_row_contiguous=True,
    )


def _mhc_forward_agg_bf16_source(max_n: int) -> str:
    return _render_source(
        _MHC_FORWARD_AGG_BF16_PATH,
        MAX_N=str(int(max_n)),
        MAX_TPG=str(int(_MAX_TPG_ALLOWED)),
    )


@lru_cache(maxsize=16)
def _mhc_forward_rms_reduce_kernel(eps: float) -> object:
    source = _mhc_forward_rms_reduce_source(eps)
    return _make_kernel(
        name="mhc_forward_rms_reduce",
        input_names=["y_agg", "partial_sq"],
        output_names=["inv_rms"],
        source=source,
        ensure_row_contiguous=True,
    )


def _mhc_forward_rms_reduce_source(eps: float) -> str:
    return _render_source(
        _MHC_FORWARD_RMS_REDUCE_PATH,
        MAX_TPG=str(int(_MAX_TPG_ALLOWED)),
        EPS=_format_float_literal(eps),
    )


@lru_cache(maxsize=32)
def _sinkhorn_backward_kernel(max_n: int, iters: int, eps: float) -> object:
    source = _sinkhorn_backward_source(max_n, iters, eps)
    return _make_kernel(
        name="sinkhorn_knopp_backward",
        input_names=["H_res", "dM"],
        output_names=["dH_res"],
        source=source,
        ensure_row_contiguous=True,
    )


def _sinkhorn_backward_source(max_n: int, iters: int, eps: float) -> str:
    return _render_source(
        _SINKHORN_BACKWARD_PATH,
        MAX_N=str(int(max_n)),
        MAX_TPG=str(int(_MAX_TPG_ALLOWED)),
        ITERS=str(int(iters)),
        EPS=_format_float_literal(eps),
    )


@lru_cache(maxsize=16)
def _mhc_backward_prep_kernel(max_n: int, eps: float) -> object:
    source = _mhc_backward_prep_source(max_n, eps)
    return _make_kernel(
        name="mhc_backward_prep",
        input_names=["x", "H_pre", "H_post", "rms_weight", "d_out"],
        output_names=["y_agg", "d_y_norm", "inv_rms", "d_r"],
        source=source,
        ensure_row_contiguous=True,
    )


def _mhc_backward_prep_source(max_n: int, eps: float) -> str:
    return _render_source(
        _MHC_BACKWARD_PREP_PATH,
        MAX_N=str(int(max_n)),
        MAX_TPG=str(int(_MAX_TPG_ALLOWED)),
        EPS=_format_float_literal(eps),
    )


@lru_cache(maxsize=16)
def _mhc_backward_prep_tile_kernel(max_n: int) -> object:
    source = _mhc_backward_prep_tile_source(max_n)
    return _make_kernel(
        name="mhc_backward_prep_tile",
        input_names=["x", "H_pre", "H_post", "rms_weight", "d_out"],
        output_names=["y_agg", "d_y_norm", "partial_sq", "partial_dr"],
        source=source,
        ensure_row_contiguous=True,
    )


def _mhc_backward_prep_tile_source(max_n: int) -> str:
    return _render_source(
        _MHC_BACKWARD_PREP_TILE_PATH,
        MAX_N=str(int(max_n)),
        MAX_TPG=str(int(_MAX_TPG_ALLOWED)),
    )


@lru_cache(maxsize=16)
def _mhc_backward_rms_reduce_kernel(eps: float) -> object:
    source = _mhc_backward_rms_reduce_source(eps)
    return _make_kernel(
        name="mhc_backward_rms_reduce",
        input_names=["y_agg", "partial_sq", "partial_dr"],
        output_names=["inv_rms", "d_r"],
        source=source,
        ensure_row_contiguous=True,
    )


def _mhc_backward_rms_reduce_source(eps: float) -> str:
    return _render_source(
        _MHC_BACKWARD_RMS_REDUCE_PATH,
        MAX_TPG=str(int(_MAX_TPG_ALLOWED)),
        EPS=_format_float_literal(eps),
    )


@lru_cache(maxsize=16)
def _mhc_backward_dx_kernel(max_n: int) -> object:
    source = _mhc_backward_dx_source(max_n)
    return _make_kernel(
        name="mhc_backward_dx",
        input_names=[
            "x",
            "M",
            "H_pre",
            "rms_weight",
            "d_out",
            "y_agg",
            "d_y_norm",
            "inv_rms",
            "d_r",
        ],
        output_names=["dx"],
        source=source,
        ensure_row_contiguous=True,
    )


def _mhc_backward_dx_source(max_n: int) -> str:
    return _render_source(
        _MHC_BACKWARD_DX_PATH,
        MAX_N=str(int(max_n)),
    )


@lru_cache(maxsize=16)
def _mhc_backward_dx_col_kernel(max_n: int) -> object:
    source = _mhc_backward_dx_col_source(max_n)
    return _make_kernel(
        name="mhc_backward_dx_col",
        input_names=[
            "x",
            "M",
            "H_pre",
            "rms_weight",
            "d_out",
            "y_agg",
            "d_y_norm",
            "inv_rms",
            "d_r",
        ],
        output_names=["dx"],
        source=source,
        ensure_row_contiguous=True,
    )


def _mhc_backward_dx_col_source(max_n: int) -> str:
    return _render_source(
        _MHC_BACKWARD_DX_COL_PATH,
        MAX_N=str(int(max_n)),
    )


@lru_cache(maxsize=16)
def _mhc_backward_grads_fused_kernel(max_n: int) -> object:
    source = _mhc_backward_grads_fused_source(max_n)
    return _make_kernel(
        name="mhc_backward_grads_fused",
        input_names=[
            "x",
            "d_out",
            "y_agg",
            "d_y_norm",
            "inv_rms",
            "d_r",
            "rms_weight",
            "dM",
            "dH_pre",
            "dH_post",
            "d_rms_weight",
        ],
        output_names=["dummy_out"],
        source=source,
        ensure_row_contiguous=True,
    )


def _mhc_backward_grads_fused_source(max_n: int) -> str:
    return _render_source(
        _MHC_BACKWARD_GRADS_FUSED_PATH,
        MAX_N=str(int(max_n)),
    )


@lru_cache(maxsize=16)
def _mhc_backward_fused_dx_kernel(max_n: int, eps: float) -> object:
    source = _mhc_backward_fused_dx_source(max_n, eps)
    return _make_kernel(
        name="mhc_backward_fused_dx",
        input_names=["x", "M", "H_pre", "H_post", "rms_weight", "d_out"],
        output_names=["dx", "y_agg", "d_y_norm", "inv_rms", "d_r"],
        source=source,
        ensure_row_contiguous=True,
    )


def _mhc_backward_fused_dx_source(max_n: int, eps: float) -> str:
    return _render_source(
        _MHC_BACKWARD_FUSED_DX_PATH,
        MAX_N=str(int(max_n)),
        MAX_TPG=str(int(_MAX_TPG_ALLOWED)),
        EPS=_format_float_literal(eps),
    )


@lru_cache(maxsize=16)
def _mhc_backward_dM_kernel() -> object:
    source = _mhc_backward_dM_source()
    return _make_kernel(
        name="mhc_backward_dM",
        input_names=["x", "d_out"],
        output_names=["dM"],
        source=source,
        ensure_row_contiguous=True,
    )


def _mhc_backward_dM_source() -> str:
    return _render_source(
        _MHC_BACKWARD_DM_PATH,
        MAX_TPG=str(int(_MAX_TPG_ALLOWED)),
    )


@lru_cache(maxsize=16)
def _mhc_backward_dH_pre_kernel() -> object:
    source = _mhc_backward_dH_pre_source()
    return _make_kernel(
        name="mhc_backward_dH_pre",
        input_names=["x", "y_agg", "d_y_norm", "inv_rms", "d_r", "rms_weight"],
        output_names=["dH_pre"],
        source=source,
        ensure_row_contiguous=True,
    )


def _mhc_backward_dH_pre_source() -> str:
    return _render_source(
        _MHC_BACKWARD_DH_PRE_PATH,
        MAX_TPG=str(int(_MAX_TPG_ALLOWED)),
    )


@lru_cache(maxsize=16)
def _mhc_backward_dH_post_kernel() -> object:
    source = _mhc_backward_dH_post_source()
    return _make_kernel(
        name="mhc_backward_dH_post",
        input_names=["d_out", "y_agg", "inv_rms", "rms_weight"],
        output_names=["dH_post"],
        source=source,
        ensure_row_contiguous=True,
    )


def _mhc_backward_dH_post_source() -> str:
    return _render_source(
        _MHC_BACKWARD_DH_POST_PATH,
        MAX_TPG=str(int(_MAX_TPG_ALLOWED)),
    )


@lru_cache(maxsize=16)
def _mhc_backward_dH_pre_post_kernel() -> object:
    source = _mhc_backward_dH_pre_post_source()
    return _make_kernel(
        name="mhc_backward_dH_pre_post",
        input_names=["x", "d_out", "y_agg", "d_y_norm", "inv_rms", "d_r", "rms_weight"],
        output_names=["dH_pre", "dH_post"],
        source=source,
        ensure_row_contiguous=True,
    )


def _mhc_backward_dH_pre_post_source() -> str:
    return _render_source(
        _MHC_BACKWARD_DH_PRE_POST_PATH,
        MAX_TPG=str(int(_MAX_TPG_ALLOWED)),
    )


@lru_cache(maxsize=16)
def _mhc_backward_drms_kernel() -> object:
    source = _mhc_backward_drms_source()
    return _make_kernel(
        name="mhc_backward_drms",
        input_names=["y_agg", "d_y_norm", "inv_rms"],
        output_names=["d_rms_weight"],
        source=source,
        ensure_row_contiguous=True,
    )


def _mhc_backward_drms_source() -> str:
    return _render_source(_MHC_BACKWARD_DRMS_PATH)


@lru_cache(maxsize=8)
def _stream_mix_backward_dx_kernel(max_n: int) -> object:
    source = _stream_mix_backward_dx_source(max_n)
    return _make_kernel(
        name="stream_mix_backward_dx",
        input_names=["M", "d_out"],
        output_names=["dx"],
        source=source,
        ensure_row_contiguous=True,
    )


def _stream_mix_backward_dx_source(max_n: int) -> str:
    return _render_source(
        _STREAM_MIX_BACKWARD_DX_PATH,
        MAX_N=str(int(max_n)),
    )


def _validate_n(n: int) -> int:
    if n <= 0:
        raise ValueError("n must be positive")
    if n > _MAX_N_ALLOWED:
        raise ValueError(
            f"n={n} exceeds MAX_N_ALLOWED={_MAX_N_ALLOWED}. Increase _MAX_N_ALLOWED if needed."
        )
    return int(n)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _dtype_eq(a, b) -> bool:
    if a is b:
        return True
    if a is None or b is None:
        return False
    try:
        return a == b
    except TypeError:
        return str(a) == str(b)


def _dtype_in(a, options) -> bool:
    return any(_dtype_eq(a, opt) for opt in options)


def _maybe_cast_float32(x: mx.array) -> mx.array:
    if _dtype_in(x.dtype, (mx.float16, mx.float32)):
        return x
    return x.astype(mx.float32)


def _normalize_output_dtype(output_dtype: mx.Dtype | None) -> mx.Dtype | None:
    if output_dtype is None:
        return None
    if _dtype_in(output_dtype, (mx.float32, mx.float16, mx.bfloat16)):
        return output_dtype
    raise ValueError(f"unsupported output_dtype: {output_dtype}")


def _normalize_mix_kernel(mix_kernel: str) -> str:
    key = (mix_kernel or "auto").strip().lower()
    if key in {"auto", "1d", "2d", "col"}:
        return key
    raise ValueError(f"unsupported mix_kernel: {mix_kernel}")


def _output_dtype_key(output_dtype: mx.Dtype | None) -> str:
    if output_dtype is None or _dtype_eq(output_dtype, mx.float32):
        return "float32"
    if _dtype_eq(output_dtype, mx.float16):
        return "float16"
    if _dtype_eq(output_dtype, mx.bfloat16):
        return "bfloat16"
    raise ValueError(f"unsupported output_dtype: {output_dtype}")


def _output_dtype_from_key(key: str) -> mx.Dtype | None:
    if key == "float32":
        return None
    if key == "float16":
        return mx.float16
    if key == "bfloat16":
        return mx.bfloat16
    raise ValueError(f"unsupported output_dtype key: {key}")


def _mix_add_rms_tile_config(
    n: int, output_dtype: mx.Dtype | None
) -> tuple[int, int, int, int] | None:
    if n not in (16, 32):
        return None
    if output_dtype is not None and (
        _dtype_eq(output_dtype, mx.float16) or _dtype_eq(output_dtype, mx.bfloat16)
    ):
        tile_c = 8 if n == 16 else 4
        vec = 2
    else:
        tile_c = 8 if n == 16 else 4
        vec = 1
    tpg = int(n * tile_c)
    tile_channels = int(tile_c * vec)
    return tile_c, tile_channels, tpg, vec


def mix_add_rms_threadgroup_size(
    n: int, output_dtype: mx.Dtype | None, threads_per_group: int, mix_kernel: str = "auto"
) -> int:
    output_dtype = _normalize_output_dtype(output_dtype)
    mix_kernel = _normalize_mix_kernel(mix_kernel)
    if mix_kernel == "2d":
        tpg_x = _tile_tpg_x(n, threads_per_group)
        return int(tpg_x * n)
    if mix_kernel == "1d":
        tile_cfg = _mix_add_rms_tile_config(n, output_dtype)
        if tile_cfg is None:
            return int(threads_per_group)
        _, _, tpg, _ = tile_cfg
        return int(tpg)
    if output_dtype is None and n >= 16:
        tpg_x = _tile_tpg_x(n, threads_per_group)
        return int(tpg_x * n)
    if _dtype_in(output_dtype, (mx.float16, mx.bfloat16)) and n >= 16:
        tpg_x = _tile_tpg_x(n, threads_per_group)
        return int(tpg_x * n)
    tile_cfg = _mix_add_rms_tile_config(n, output_dtype)
    if tile_cfg is None:
        return int(threads_per_group)
    _, _, tpg, _ = tile_cfg
    return int(tpg)


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

    x_in = _maybe_cast_float32(x)
    M_f = M.astype(mx.float32)
    y_f = y_dist.astype(mx.float32)

    kernel = _stream_mix_add_kernel(max_n)

    # Grid:
    # - x dimension: channels C
    # - y dimension: B*n (each y index corresponds to one (b, i) pair)
    out = kernel(
        inputs=[x_in, M_f, y_f],
        grid=(C, B * n, 1),
        threadgroup=(threads_per_group, 1, 1),
        output_shapes=[x_in.shape],
        output_dtypes=[mx.float32],
    )[0]

    return out


def stream_mix_add_rms_metal(
    x: mx.array,
    M: mx.array,
    H_post: mx.array,
    y_agg: mx.array,
    inv_rms: mx.array,
    rms_weight: mx.array,
    threads_per_group: int = 256,
    output_dtype: mx.Dtype | None = None,
    mix_kernel: str = "auto",
    verbose: bool = False,
) -> mx.array:
    """Compute out = stream_mix(x, M) + H_post * (y_agg * inv_rms * rms_weight).

    When output_dtype is float16 or bfloat16, x must have the same dtype.
    mix_kernel selects the implementation ("auto", "1d", "2d").
    """
    if x.ndim != 3:
        raise ValueError(f"x must be [B, n, C], got shape {x.shape}")
    if M.ndim != 2:
        raise ValueError(f"M must be [n, n], got shape {M.shape}")
    if H_post.ndim != 1:
        raise ValueError(f"H_post must be [n], got shape {H_post.shape}")
    if y_agg.ndim != 2:
        raise ValueError(f"y_agg must be [B, C], got shape {y_agg.shape}")
    if inv_rms.ndim != 1:
        raise ValueError(f"inv_rms must be [B], got shape {inv_rms.shape}")
    if rms_weight.ndim != 1:
        raise ValueError(f"rms_weight must be [C], got shape {rms_weight.shape}")

    B, n, C = x.shape
    if M.shape != (n, n):
        raise ValueError(f"M must be shape (n,n)=( {n},{n} ), got {M.shape}")
    if H_post.shape != (n,):
        raise ValueError(f"H_post must be shape (n,)=( {n}, ), got {H_post.shape}")
    if y_agg.shape != (B, C):
        raise ValueError(f"y_agg must be shape (B,C)=( {B},{C} ), got {y_agg.shape}")
    if inv_rms.shape != (B,):
        raise ValueError(f"inv_rms must be shape (B,)=( {B}, ), got {inv_rms.shape}")
    if rms_weight.shape != (C,):
        raise ValueError(f"rms_weight must be shape (C,)=( {C}, ), got {rms_weight.shape}")

    max_n = _validate_n(n)
    if threads_per_group <= 0:
        raise ValueError("threads_per_group must be positive")
    output_dtype = _normalize_output_dtype(output_dtype)
    mix_kernel = _normalize_mix_kernel(mix_kernel)
    if mix_kernel == "2d":
        if output_dtype is None:
            return stream_mix_add_rms_tile_metal(
                x,
                M,
                H_post,
                y_agg,
                inv_rms,
                rms_weight,
                threads_per_group=threads_per_group,
                verbose=verbose,
            )
        if _dtype_eq(output_dtype, mx.float16):
            return stream_mix_add_rms_tile2d_fp16_metal(
                x,
                M,
                H_post,
                y_agg,
                inv_rms,
                rms_weight,
                threads_per_group=threads_per_group,
                verbose=verbose,
            )
        if _dtype_eq(output_dtype, mx.bfloat16):
            return stream_mix_add_rms_tile2d_bf16_metal(
                x,
                M,
                H_post,
                y_agg,
                inv_rms,
                rms_weight,
                threads_per_group=threads_per_group,
                verbose=verbose,
            )
    if mix_kernel == "auto" and output_dtype is None and n >= 16:
        return stream_mix_add_rms_tile_metal(
            x,
            M,
            H_post,
            y_agg,
            inv_rms,
            rms_weight,
            threads_per_group=threads_per_group,
            verbose=verbose,
        )
    if mix_kernel == "auto" and _dtype_eq(output_dtype, mx.float16) and n >= 16:
        return stream_mix_add_rms_tile2d_fp16_metal(
            x,
            M,
            H_post,
            y_agg,
            inv_rms,
            rms_weight,
            threads_per_group=threads_per_group,
            verbose=verbose,
        )
    if mix_kernel == "auto" and _dtype_eq(output_dtype, mx.bfloat16) and n >= 16:
        return stream_mix_add_rms_tile2d_bf16_metal(
            x,
            M,
            H_post,
            y_agg,
            inv_rms,
            rms_weight,
            threads_per_group=threads_per_group,
            verbose=verbose,
        )

    tile_cfg = _mix_add_rms_tile_config(n, output_dtype)
    if tile_cfg is None:
        effective_tpg = int(threads_per_group)
    else:
        _, _, effective_tpg, _ = tile_cfg
    if effective_tpg > _MAX_TPG_ALLOWED:
        raise ValueError(f"threads_per_group must be <= {_MAX_TPG_ALLOWED}")

    if verbose:
        if mix_kernel == "2d":
            if _dtype_eq(output_dtype, mx.float16):
                _maybe_print_source(
                    _stream_mix_add_rms_tile2d_fp16_source(max_n),
                    "stream_mix_add_rms_tile2d_fp16",
                    True,
                )
            elif _dtype_eq(output_dtype, mx.bfloat16):
                _maybe_print_source(
                    _stream_mix_add_rms_tile2d_bf16_source(max_n),
                    "stream_mix_add_rms_tile2d_bf16",
                    True,
                )
            else:
                _maybe_print_source(_stream_mix_add_rms_tile_source(max_n), "stream_mix_add_rms_tile", True)
        elif tile_cfg is not None:
            tile_c, _, _, _ = tile_cfg
            if _dtype_eq(output_dtype, mx.float16):
                _maybe_print_source(
                    _stream_mix_add_rms_tile_fp16_source(n, tile_c),
                    "stream_mix_add_rms_tile_fp16",
                    True,
                )
            elif _dtype_eq(output_dtype, mx.bfloat16):
                _maybe_print_source(
                    _stream_mix_add_rms_tile_bf16_source(n, tile_c),
                    "stream_mix_add_rms_tile_bf16",
                    True,
                )
            else:
                _maybe_print_source(
                    _stream_mix_add_rms_tile_f32_source(n, tile_c),
                    "stream_mix_add_rms_tile_f32",
                    True,
                )
        elif _dtype_eq(output_dtype, mx.float16):
            _maybe_print_source(_stream_mix_add_rms_fp16_source(max_n), "stream_mix_add_rms_fp16", True)
        elif _dtype_eq(output_dtype, mx.bfloat16):
            _maybe_print_source(_stream_mix_add_rms_bf16_source(max_n), "stream_mix_add_rms_bf16", True)
        else:
            _maybe_print_source(_stream_mix_add_rms_source(max_n), "stream_mix_add_rms", True)
    if output_dtype is None or _dtype_eq(output_dtype, mx.float32):
        x_in = _maybe_cast_float32(x)
        if tile_cfg is None:
            kernel = _stream_mix_add_rms_kernel(max_n)
            grid_x = C
            tpg = effective_tpg
        else:
            tile_c, tile_channels, tpg, _ = tile_cfg
            kernel = _stream_mix_add_rms_tile_f32_kernel(n, tile_c)
            grid_x = _ceil_div(C, tile_channels) * tpg
        out_dtype = mx.float32
    elif _dtype_eq(output_dtype, mx.float16):
        if not _dtype_eq(x.dtype, mx.float16):
            raise ValueError("x must be float16 when output_dtype is float16")
        x_in = x
        if tile_cfg is None:
            kernel = _stream_mix_add_rms_fp16_kernel(max_n)
            grid_x = _ceil_div(C, 2)
            tpg = effective_tpg
        else:
            tile_c, tile_channels, tpg, _ = tile_cfg
            kernel = _stream_mix_add_rms_tile_fp16_kernel(n, tile_c)
            grid_x = _ceil_div(C, tile_channels) * tpg
        out_dtype = mx.float16
    elif _dtype_eq(output_dtype, mx.bfloat16):
        if not _dtype_eq(x.dtype, mx.bfloat16):
            raise ValueError("x must be bfloat16 when output_dtype is bfloat16")
        x_in = x
        if tile_cfg is None:
            kernel = _stream_mix_add_rms_bf16_kernel(max_n)
            grid_x = _ceil_div(C, 2)
            tpg = effective_tpg
        else:
            tile_c, tile_channels, tpg, _ = tile_cfg
            kernel = _stream_mix_add_rms_tile_bf16_kernel(n, tile_c)
            grid_x = _ceil_div(C, tile_channels) * tpg
        out_dtype = mx.bfloat16
    else:
        raise ValueError(f"unsupported output_dtype: {output_dtype}")

    M_f = M.astype(mx.float32)
    H_post_f = H_post.astype(mx.float32)
    y_agg_f = y_agg.astype(mx.float32)
    inv_rms_f = inv_rms.astype(mx.float32)
    rms_weight_f = rms_weight.astype(mx.float32)

    out = kernel(
        inputs=[x_in, M_f, H_post_f, y_agg_f, inv_rms_f, rms_weight_f],
        grid=(grid_x, B * n, 1),
        threadgroup=(tpg, 1, 1),
        output_shapes=[x_in.shape],
        output_dtypes=[out_dtype],
    )[0]

    return out


def stream_mix_add_rms_col_metal(
    x: mx.array,
    M: mx.array,
    H_post: mx.array,
    y_agg: mx.array,
    inv_rms: mx.array,
    rms_weight: mx.array,
    threads_per_group: int = 256,
    verbose: bool = False,
) -> mx.array:
    """Column-parallel stream mix add + RMS (thread per column)."""
    if x.ndim != 3:
        raise ValueError(f"x must be [B, n, C], got shape {x.shape}")
    B, n, C = x.shape
    max_n = _validate_n(n)
    
    if threads_per_group <= 0:
        raise ValueError("threads_per_group must be positive")
    tpg = int(threads_per_group)

    if verbose:
        _maybe_print_source(
            _stream_mix_add_rms_col_source(max_n),
            "stream_mix_add_rms_col",
            verbose=True,
        )

    x_in = _maybe_cast_float32(x)
    M_f = M.astype(mx.float32)
    H_post_f = H_post.astype(mx.float32)
    y_agg_f = y_agg.astype(mx.float32)
    inv_rms_f = inv_rms.astype(mx.float32)
    rms_weight_f = rms_weight.astype(mx.float32)

    kernel = _stream_mix_add_rms_col_kernel(max_n)
    out = kernel(
        inputs=[x_in, M_f, H_post_f, y_agg_f, inv_rms_f, rms_weight_f],
        grid=(C, B, 1),
        threadgroup=(tpg, 1, 1),
        output_shapes=[(B, n, C)],
        output_dtypes=[mx.float32],
    )[0]
    return out


def stream_mix_add_rms_col_bf16_metal(
    x: mx.array,
    M: mx.array,
    H_post: mx.array,
    y_agg: mx.array,
    inv_rms: mx.array,
    rms_weight: mx.array,
    threads_per_group: int = 256,
    verbose: bool = False,
) -> mx.array:
    """Column-parallel stream mix add + RMS for BF16 (thread per 2 columns)."""
    if x.ndim != 3:
        raise ValueError(f"x must be [B, n, C], got shape {x.shape}")
    B, n, C = x.shape
    max_n = _validate_n(n)
    
    if threads_per_group <= 0:
        raise ValueError("threads_per_group must be positive")
    tpg = int(threads_per_group)

    if verbose:
        _maybe_print_source(
            _stream_mix_add_rms_col_bf16_source(max_n),
            "stream_mix_add_rms_col_bf16",
            verbose=True,
        )

    # x is bfloat16, pass as is
    M_f = M.astype(mx.float32)
    H_post_f = H_post.astype(mx.float32)
    y_agg_f = y_agg.astype(mx.float32)
    inv_rms_f = inv_rms.astype(mx.float32)
    rms_weight_f = rms_weight.astype(mx.float32)

    kernel = _stream_mix_add_rms_col_bf16_kernel(max_n)
    
    # Grid: ceil(C/2) threads
    grid_x = (C + 1) // 2
    
    out = kernel(
        inputs=[x, M_f, H_post_f, y_agg_f, inv_rms_f, rms_weight_f],
        grid=(grid_x, B, 1),
        threadgroup=(tpg, 1, 1),
        output_shapes=[(B, n, C)],
        output_dtypes=[mx.bfloat16],
    )[0]
    return out


def stream_mix_add_rms_tile_metal(
    x: mx.array,
    M: mx.array,
    H_post: mx.array,
    y_agg: mx.array,
    inv_rms: mx.array,
    rms_weight: mx.array,
    threads_per_group: int = 256,
    verbose: bool = False,
) -> mx.array:
    """Tile-parallel stream mix add + RMS in float32 using a 2D threadgroup."""
    if x.ndim != 3:
        raise ValueError(f"x must be [B, n, C], got shape {x.shape}")
    if M.ndim != 2:
        raise ValueError(f"M must be [n, n], got shape {M.shape}")
    if H_post.ndim != 1:
        raise ValueError(f"H_post must be [n], got shape {H_post.shape}")
    if y_agg.ndim != 2:
        raise ValueError(f"y_agg must be [B, C], got shape {y_agg.shape}")
    if inv_rms.ndim != 1:
        raise ValueError(f"inv_rms must be [B], got shape {inv_rms.shape}")
    if rms_weight.ndim != 1:
        raise ValueError(f"rms_weight must be [C], got shape {rms_weight.shape}")

    B, n, C = x.shape
    if M.shape != (n, n):
        raise ValueError(f"M must be shape (n,n)=( {n},{n} ), got {M.shape}")
    if H_post.shape != (n,):
        raise ValueError(f"H_post must be shape (n,)=( {n}, ), got {H_post.shape}")
    if y_agg.shape != (B, C):
        raise ValueError(f"y_agg must be shape (B,C)=( {B},{C} ), got {y_agg.shape}")
    if inv_rms.shape != (B,):
        raise ValueError(f"inv_rms must be shape (B,)=( {B}, ), got {inv_rms.shape}")
    if rms_weight.shape != (C,):
        raise ValueError(f"rms_weight must be shape (C,)=( {C}, ), got {rms_weight.shape}")

    max_n = _validate_n(n)
    if threads_per_group <= 0:
        raise ValueError("threads_per_group must be positive")
    tpg_x = _tile_tpg_x(n, int(threads_per_group))
    tpg_y = int(n)
    if tpg_x * tpg_y > _max_threads_per_threadgroup():
        raise ValueError("invalid threadgroup size")

    if verbose:
        _maybe_print_source(
            _stream_mix_add_rms_tile_source(max_n),
            "stream_mix_add_rms_tile",
            verbose=True,
        )

    x_in = _maybe_cast_float32(x)
    M_f = M.astype(mx.float32)
    H_post_f = H_post.astype(mx.float32)
    y_agg_f = y_agg.astype(mx.float32)
    inv_rms_f = inv_rms.astype(mx.float32)
    rms_weight_f = rms_weight.astype(mx.float32)

    kernel = _stream_mix_add_rms_tile_kernel(max_n)
    out = kernel(
        inputs=[x_in, M_f, H_post_f, y_agg_f, inv_rms_f, rms_weight_f],
        grid=(C, B * n, 1),
        threadgroup=(tpg_x, tpg_y, 1),
        output_shapes=[(B, n, C)],
        output_dtypes=[mx.float32],
    )[0]
    return out


def stream_mix_add_rms_tile2d_fp16_metal(
    x: mx.array,
    M: mx.array,
    H_post: mx.array,
    y_agg: mx.array,
    inv_rms: mx.array,
    rms_weight: mx.array,
    threads_per_group: int = 256,
    verbose: bool = False,
) -> mx.array:
    """2D threadgroup stream mix add + RMS with float16 output."""
    if x.ndim != 3:
        raise ValueError(f"x must be [B, n, C], got shape {x.shape}")
    if M.ndim != 2:
        raise ValueError(f"M must be [n, n], got shape {M.shape}")
    if H_post.ndim != 1:
        raise ValueError(f"H_post must be [n], got shape {H_post.shape}")
    if y_agg.ndim != 2:
        raise ValueError(f"y_agg must be [B, C], got shape {y_agg.shape}")
    if inv_rms.ndim != 1:
        raise ValueError(f"inv_rms must be [B], got shape {inv_rms.shape}")
    if rms_weight.ndim != 1:
        raise ValueError(f"rms_weight must be [C], got shape {rms_weight.shape}")

    B, n, C = x.shape
    if M.shape != (n, n):
        raise ValueError(f"M must be shape (n,n)=( {n},{n} ), got {M.shape}")
    if H_post.shape != (n,):
        raise ValueError(f"H_post must be shape (n,)=( {n}, ), got {H_post.shape}")
    if y_agg.shape != (B, C):
        raise ValueError(f"y_agg must be shape (B,C)=( {B},{C} ), got {y_agg.shape}")
    if inv_rms.shape != (B,):
        raise ValueError(f"inv_rms must be shape (B,)=( {B}, ), got {inv_rms.shape}")
    if rms_weight.shape != (C,):
        raise ValueError(f"rms_weight must be shape (C,)=( {C}, ), got {rms_weight.shape}")
    if not _dtype_eq(x.dtype, mx.float16):
        raise ValueError("x must be float16 for float16 output")

    max_n = _validate_n(n)
    if threads_per_group <= 0:
        raise ValueError("threads_per_group must be positive")
    tpg_x = _tile_tpg_x(n, int(threads_per_group))
    tpg_y = int(n)
    if tpg_x * tpg_y > _max_threads_per_threadgroup():
        raise ValueError("invalid threadgroup size")

    if verbose:
        _maybe_print_source(
            _stream_mix_add_rms_tile2d_fp16_source(max_n),
            "stream_mix_add_rms_tile2d_fp16",
            verbose=True,
        )

    M_f = M.astype(mx.float32)
    H_post_f = H_post.astype(mx.float32)
    y_agg_f = y_agg.astype(mx.float32)
    inv_rms_f = inv_rms.astype(mx.float32)
    rms_weight_f = rms_weight.astype(mx.float32)

    kernel = _stream_mix_add_rms_tile2d_fp16_kernel(max_n)
    out = kernel(
        inputs=[x, M_f, H_post_f, y_agg_f, inv_rms_f, rms_weight_f],
        grid=(_ceil_div(C, 2), B * n, 1),
        threadgroup=(tpg_x, tpg_y, 1),
        output_shapes=[(B, n, C)],
        output_dtypes=[mx.float16],
    )[0]
    return out


def stream_mix_add_rms_tile2d_bf16_metal(
    x: mx.array,
    M: mx.array,
    H_post: mx.array,
    y_agg: mx.array,
    inv_rms: mx.array,
    rms_weight: mx.array,
    threads_per_group: int = 256,
    verbose: bool = False,
) -> mx.array:
    """2D threadgroup stream mix add + RMS with bfloat16 output."""
    if x.ndim != 3:
        raise ValueError(f"x must be [B, n, C], got shape {x.shape}")
    if M.ndim != 2:
        raise ValueError(f"M must be [n, n], got shape {M.shape}")
    if H_post.ndim != 1:
        raise ValueError(f"H_post must be [n], got shape {H_post.shape}")
    if y_agg.ndim != 2:
        raise ValueError(f"y_agg must be [B, C], got shape {y_agg.shape}")
    if inv_rms.ndim != 1:
        raise ValueError(f"inv_rms must be [B], got shape {inv_rms.shape}")
    if rms_weight.ndim != 1:
        raise ValueError(f"rms_weight must be [C], got shape {rms_weight.shape}")

    B, n, C = x.shape
    if M.shape != (n, n):
        raise ValueError(f"M must be shape (n,n)=( {n},{n} ), got {M.shape}")
    if H_post.shape != (n,):
        raise ValueError(f"H_post must be shape (n,)=( {n}, ), got {H_post.shape}")
    if y_agg.shape != (B, C):
        raise ValueError(f"y_agg must be shape (B,C)=( {B},{C} ), got {y_agg.shape}")
    if inv_rms.shape != (B,):
        raise ValueError(f"inv_rms must be shape (B,)=( {B}, ), got {inv_rms.shape}")
    if rms_weight.shape != (C,):
        raise ValueError(f"rms_weight must be shape (C,)=( {C}, ), got {rms_weight.shape}")
    if not _dtype_eq(x.dtype, mx.bfloat16):
        raise ValueError("x must be bfloat16 for bfloat16 output")

    max_n = _validate_n(n)
    if threads_per_group <= 0:
        raise ValueError("threads_per_group must be positive")
    tpg_x = _tile_tpg_x(n, int(threads_per_group))
    tpg_y = int(n)
    if tpg_x * tpg_y > _max_threads_per_threadgroup():
        raise ValueError("invalid threadgroup size")

    if verbose:
        _maybe_print_source(
            _stream_mix_add_rms_tile2d_bf16_source(max_n),
            "stream_mix_add_rms_tile2d_bf16",
            verbose=True,
        )

    M_f = M.astype(mx.float32)
    H_post_f = H_post.astype(mx.float32)
    y_agg_f = y_agg.astype(mx.float32)
    inv_rms_f = inv_rms.astype(mx.float32)
    rms_weight_f = rms_weight.astype(mx.float32)

    kernel = _stream_mix_add_rms_tile2d_bf16_kernel(max_n)
    out = kernel(
        inputs=[x, M_f, H_post_f, y_agg_f, inv_rms_f, rms_weight_f],
        grid=(_ceil_div(C, 2), B * n, 1),
        threadgroup=(tpg_x, tpg_y, 1),
        output_shapes=[(B, n, C)],
        output_dtypes=[mx.bfloat16],
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
    tpg = int(threads_per_group)
    if n <= 32 and tpg > n:
        tpg = int(n)
    tpg = int(threads_per_group)
    if n <= 32 and tpg > n:
        tpg = int(n)
    tpg = int(threads_per_group)
    if n <= 32 and tpg > n:
        tpg = int(n)

    if verbose:
        _maybe_print_source(_sinkhorn_source(n, iters, eps), "sinkhorn_knopp", verbose=True)

    H_f = H_res_raw.astype(mx.float32)

    kernel = _sinkhorn_kernel(n, int(iters), float(eps))
    out = kernel(
        inputs=[H_f],
        grid=(tpg, 1, 1),
        threadgroup=(tpg, 1, 1),
        output_shapes=[H_f.shape],
        output_dtypes=[mx.float32],
    )[0]

    return out


def mhc_fully_fused_metal(
    x: mx.array,
    M: mx.array,
    H_pre: mx.array,
    H_post: mx.array,
    rms_weight: mx.array,
    eps: float = 1e-5,
    threads_per_group: int = 256,
    output_dtype: mx.Dtype | None = None,
    verbose: bool = False,
) -> mx.array:
    """Fully fused forward (Aggregate + RMS + Mix + Add) in one kernel.

    Best for small workloads (e.g. B=1, small C) where kernel launch overhead
    and occupancy dominates bandwidth. This kernel uses one threadgroup per batch item.
    """
    if x.ndim != 3:
        raise ValueError(f"x must be [B, n, C], got shape {x.shape}")
    B, n, C = x.shape
    max_n = _validate_n(n)
    
    if threads_per_group <= 0:
        raise ValueError("threads_per_group must be positive")
    if threads_per_group > _MAX_TPG_ALLOWED:
        raise ValueError(f"threads_per_group must be <= {_MAX_TPG_ALLOWED}")

    output_dtype = _normalize_output_dtype(output_dtype)
    out_dtype = mx.float32
    if output_dtype is not None:
        if _dtype_eq(output_dtype, mx.float16) or _dtype_eq(output_dtype, mx.bfloat16):
            if not _dtype_eq(x.dtype, output_dtype):
                raise ValueError(f"x must be {output_dtype} when output_dtype is {output_dtype}")
            out_dtype = output_dtype

    if verbose:
        _maybe_print_source(
            _mhc_fused_source(max_n, eps, output_dtype),
            "mhc_fused",
            verbose=True,
        )

    x_in = _maybe_cast_float32(x)
    M_f = M.astype(mx.float32)
    H_pre_f = H_pre.astype(mx.float32)
    H_post_f = H_post.astype(mx.float32)
    rms_weight_f = rms_weight.astype(mx.float32)

    kernel = _mhc_fused_kernel(max_n, float(eps), output_dtype)
    
    # Grid: (threads_per_group, B, 1)
    # Each threadgroup handles one batch item.
    out = kernel(
        inputs=[x_in, M_f, H_pre_f, H_post_f, rms_weight_f],
        grid=(threads_per_group, B, 1),
        threadgroup=(threads_per_group, 1, 1),
        output_shapes=[x_in.shape],
        output_dtypes=[out_dtype],
    )[0]

    return out


def mhc_forward_agg_metal(
    x: mx.array,
    H_pre: mx.array,
    threads_per_group: int = 256,
    verbose: bool = False,
) -> tuple[mx.array, mx.array]:
    """Compute y_agg and partial sums for RMS in a tile-parallel kernel."""
    if x.ndim != 3:
        raise ValueError(f"x must be [B, n, C], got shape {x.shape}")
    if H_pre.ndim != 1:
        raise ValueError(f"H_pre must be [n], got shape {H_pre.shape}")

    B, n, C = x.shape
    if H_pre.shape != (n,):
        raise ValueError(f"H_pre must be shape (n,)=( {n}, ), got {H_pre.shape}")

    max_n = _validate_n(n)
    if threads_per_group <= 0:
        raise ValueError("threads_per_group must be positive")
    if threads_per_group > _MAX_TPG_ALLOWED:
        raise ValueError(f"threads_per_group must be <= {_MAX_TPG_ALLOWED}")

    x_in: mx.array
    kernel: object
    if _dtype_eq(x.dtype, mx.bfloat16):
        if verbose:
            _maybe_print_source(_mhc_forward_agg_bf16_source(max_n), "mhc_forward_agg_bf16", True)
        x_in = x
        kernel = _mhc_forward_agg_bf16_kernel(max_n)
    else:
        if verbose:
            _maybe_print_source(_mhc_forward_agg_source(max_n), "mhc_forward_agg", True)
        x_in = _maybe_cast_float32(x)
        kernel = _mhc_forward_agg_kernel(max_n)

    H_pre_f = H_pre.astype(mx.float32)

    tiles = _ceil_div(C, threads_per_group)
    y_agg, partial_sq = kernel(
        inputs=[x_in, H_pre_f],
        grid=(C, B, 1),
        threadgroup=(threads_per_group, 1, 1),
        output_shapes=[(B, C), (B, tiles)],
        output_dtypes=[mx.float32, mx.float32],
    )

    return y_agg, partial_sq


def mhc_forward_rms_reduce_metal(
    y_agg: mx.array,
    partial_sq: mx.array,
    eps: float = 1e-5,
    threads_per_group: int = 256,
    verbose: bool = False,
) -> mx.array:
    """Reduce partial sums into inv_rms for RMSNorm."""
    if y_agg.ndim != 2:
        raise ValueError(f"y_agg must be [B, C], got shape {y_agg.shape}")
    if partial_sq.ndim != 2:
        raise ValueError(f"partial_sq must be [B, T], got shape {partial_sq.shape}")

    B, _ = y_agg.shape
    if partial_sq.shape[0] != B:
        raise ValueError(f"partial_sq must have B rows, got {partial_sq.shape[0]} vs {B}")

    if threads_per_group <= 0:
        raise ValueError("threads_per_group must be positive")
    if threads_per_group > _MAX_TPG_ALLOWED:
        raise ValueError(f"threads_per_group must be <= {_MAX_TPG_ALLOWED}")

    if verbose:
        _maybe_print_source(_mhc_forward_rms_reduce_source(eps), "mhc_forward_rms_reduce", True)

    kernel = _mhc_forward_rms_reduce_kernel(float(eps))
    inv_rms = kernel(
        inputs=[y_agg.astype(mx.float32), partial_sq.astype(mx.float32)],
        grid=(threads_per_group, B, 1),
        threadgroup=(threads_per_group, 1, 1),
        output_shapes=[(B,)],
        output_dtypes=[mx.float32],
    )[0]

    return inv_rms


def mhc_forward_fused_metal(
    x: mx.array,
    M: mx.array,
    H_pre: mx.array,
    H_post: mx.array,
    rms_weight: mx.array,
    eps: float = 1e-5,
    threads_per_group: int = 256,
    output_dtype: mx.Dtype | None = None,
    mix_kernel: str = "auto",
    verbose: bool = False,
) -> mx.array:
    """Token-parallel Metal forward: aggregate + RMS + mix/add without y_dist.

    Args:
        x: [B, n, C] float32, row contiguous
        M: [n, n] float32
        H_pre: [n] float32 (activated)
        H_post: [n] float32 (activated)
        rms_weight: [C] float32
        eps: RMSNorm epsilon
        threads_per_group: threadgroup size along x
        output_dtype: optional output dtype for the mix/add kernel (float16/bfloat16 requires x dtype match)
        mix_kernel: mix/add kernel selection ("auto", "1d", "2d")
        verbose: if True, print the kernel body source

    Returns:
        out: [B, n, C] float32 unless output_dtype overrides
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
        _maybe_print_source(_mhc_forward_agg_source(max_n), "mhc_forward_agg", verbose=True)
        _maybe_print_source(_mhc_forward_rms_reduce_source(eps), "mhc_forward_rms_reduce", verbose=True)
        _maybe_print_source(_stream_mix_add_rms_source(max_n), "stream_mix_add_rms", verbose=True)

    y_agg, partial_sq = mhc_forward_agg_metal(
        x,
        H_pre,
        threads_per_group=threads_per_group,
        verbose=False,
    )
    inv_rms = mhc_forward_rms_reduce_metal(
        y_agg,
        partial_sq,
        eps=eps,
        threads_per_group=threads_per_group,
        verbose=False,
    )

    use_col = False
    use_col_bf16 = False
    normalized_mix = _normalize_mix_kernel(mix_kernel)
    if normalized_mix == "col":
        if _dtype_eq(output_dtype, mx.bfloat16):
            use_col_bf16 = True
        else:
            use_col = True
    elif normalized_mix == "auto":
        # Use col kernel if output_dtype is None/float32
        if output_dtype is None or _dtype_eq(output_dtype, mx.float32):
            use_col = True
        elif _dtype_eq(output_dtype, mx.bfloat16):
            use_col_bf16 = True

    # Occupancy heuristic: if B*C is small, the column kernel (grid B*C) might not
    # fill the GPU. Use the fully fused kernel (1 kernel vs 3) to reduce overhead.
    use_fully_fused = False
    if B * C <= 2048:
        use_fully_fused = True
        use_col = False
        use_col_bf16 = False

    if use_fully_fused:
        # Scale TPG with C for the fully fused kernel, up to 1024.
        # This kernel benefits significantly from higher occupancy within the single block.
        fused_tpg = suggest_threads_per_group(C, max_tpg=1024)
        out = mhc_fully_fused_metal(
            x,
            M,
            H_pre,
            H_post,
            rms_weight,
            eps=eps,
            threads_per_group=fused_tpg,
            output_dtype=output_dtype,
            verbose=False,
        )
    elif use_col:
        out = stream_mix_add_rms_col_metal(
            x,
            M,
            H_post,
            y_agg,
            inv_rms,
            rms_weight,
            threads_per_group=threads_per_group,
            verbose=False,
        )
    elif use_col_bf16:
        out = stream_mix_add_rms_col_bf16_metal(
            x,
            M,
            H_post,
            y_agg,
            inv_rms,
            rms_weight,
            threads_per_group=threads_per_group,
            verbose=False,
        )
    else:
        out = stream_mix_add_rms_metal(
            x,
            M,
            H_post,
            y_agg,
            inv_rms,
            rms_weight,
            threads_per_group=threads_per_group,
            output_dtype=output_dtype,
            mix_kernel=mix_kernel,
            verbose=False,
        )

    return out


def sinkhorn_knopp_backward_metal(
    H_res_raw: mx.array,
    dM: mx.array,
    iters: int = 20,
    eps: float = 1e-5,
    threads_per_group: int = 64,
    verbose: bool = False,
) -> mx.array:
    """Backward for Sinkhorn-Knopp: returns dH_res for M = sinkhorn(exp(H_res_raw))."""
    if H_res_raw.ndim != 2 or H_res_raw.shape[0] != H_res_raw.shape[1]:
        raise ValueError(f"H_res_raw must be square [n, n], got shape {H_res_raw.shape}")
    if dM.shape != H_res_raw.shape:
        raise ValueError(f"dM must match H_res_raw shape, got {dM.shape} vs {H_res_raw.shape}")

    n = _validate_n(H_res_raw.shape[0])
    if threads_per_group <= 0:
        raise ValueError("threads_per_group must be positive")
    if threads_per_group > _MAX_TPG_ALLOWED:
        raise ValueError(f"threads_per_group must be <= {_MAX_TPG_ALLOWED}")
    tpg = int(threads_per_group)
    if n <= 32 and tpg > n:
        tpg = int(n)

    if verbose:
        _maybe_print_source(_sinkhorn_backward_source(n, iters, eps), "sinkhorn_knopp_backward", True)

    H_f = H_res_raw.astype(mx.float32)
    dM_f = dM.astype(mx.float32)

    kernel = _sinkhorn_backward_kernel(n, int(iters), float(eps))
    out = kernel(
        inputs=[H_f, dM_f],
        grid=(tpg, 1, 1),
        threadgroup=(tpg, 1, 1),
        output_shapes=[H_f.shape],
        output_dtypes=[mx.float32],
    )[0]

    return out


def mhc_backward_prep_metal(
    x: mx.array,
    H_pre: mx.array,
    H_post: mx.array,
    rms_weight: mx.array,
    d_out: mx.array,
    eps: float = 1e-5,
    threads_per_group: int = 256,
    verbose: bool = False,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Compute y_agg, d_y_norm, inv_rms, d_r for backward."""
    if x.ndim != 3:
        raise ValueError(f"x must be [B, n, C], got shape {x.shape}")
    if d_out.shape != x.shape:
        raise ValueError(f"d_out must match x shape, got {d_out.shape} vs {x.shape}")
    if H_pre.ndim != 1:
        raise ValueError(f"H_pre must be [n], got shape {H_pre.shape}")
    if H_post.ndim != 1:
        raise ValueError(f"H_post must be [n], got shape {H_post.shape}")
    if rms_weight.ndim != 1:
        raise ValueError(f"rms_weight must be [C], got shape {rms_weight.shape}")

    B, n, C = x.shape
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
        raise ValueError(f"threads_per_group must be <= {_MAX_TPG_ALLOWED}")

    if verbose:
        _maybe_print_source(_mhc_backward_prep_source(max_n, eps), "mhc_backward_prep", True)

    x_f = x.astype(mx.float32)
    x_in = _maybe_cast_float32(x)
    d_out_in = _maybe_cast_float32(d_out)
    H_pre_f = H_pre.astype(mx.float32)
    H_post_f = H_post.astype(mx.float32)
    rms_weight_f = rms_weight.astype(mx.float32)
    d_out_f = d_out.astype(mx.float32)

    kernel = _mhc_backward_prep_kernel(max_n, float(eps))
    y_agg, d_y_norm, inv_rms, d_r = kernel(
        inputs=[x_f, H_pre_f, H_post_f, rms_weight_f, d_out_f],
        grid=(threads_per_group, B, 1),
        threadgroup=(threads_per_group, 1, 1),
        output_shapes=[(B, C), (B, C), (B,), (B,)],
        output_dtypes=[mx.float32, mx.float32, mx.float32, mx.float32],
    )

    return y_agg, d_y_norm, inv_rms, d_r


def mhc_backward_prep_tile_metal(
    x: mx.array,
    H_pre: mx.array,
    H_post: mx.array,
    rms_weight: mx.array,
    d_out: mx.array,
    threads_per_group: int = 256,
    verbose: bool = False,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Tile-parallel prep for backward (y_agg, d_y_norm, partial sums)."""
    if x.ndim != 3:
        raise ValueError(f"x must be [B, n, C], got shape {x.shape}")
    if d_out.shape != x.shape:
        raise ValueError(f"d_out must match x shape, got {d_out.shape} vs {x.shape}")
    if H_pre.ndim != 1:
        raise ValueError(f"H_pre must be [n], got shape {H_pre.shape}")
    if H_post.ndim != 1:
        raise ValueError(f"H_post must be [n], got shape {H_post.shape}")
    if rms_weight.ndim != 1:
        raise ValueError(f"rms_weight must be [C], got shape {rms_weight.shape}")

    B, n, C = x.shape
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
        raise ValueError(f"threads_per_group must be <= {_MAX_TPG_ALLOWED}")

    if verbose:
        _maybe_print_source(_mhc_backward_prep_tile_source(max_n), "mhc_backward_prep_tile", True)

    x_in = _maybe_cast_float32(x)
    d_out_in = _maybe_cast_float32(d_out)
    H_pre_f = H_pre.astype(mx.float32)
    H_post_f = H_post.astype(mx.float32)
    rms_weight_f = rms_weight.astype(mx.float32)

    tiles = _ceil_div(C, threads_per_group)
    kernel = _mhc_backward_prep_tile_kernel(max_n)
    y_agg, d_y_norm, partial_sq, partial_dr = kernel(
        inputs=[x_in, H_pre_f, H_post_f, rms_weight_f, d_out_in],
        grid=(C, B, 1),
        threadgroup=(threads_per_group, 1, 1),
        output_shapes=[(B, C), (B, C), (B, tiles), (B, tiles)],
        output_dtypes=[mx.float32, mx.float32, mx.float32, mx.float32],
    )

    return y_agg, d_y_norm, partial_sq, partial_dr


def mhc_backward_rms_reduce_metal(
    y_agg: mx.array,
    partial_sq: mx.array,
    partial_dr: mx.array,
    eps: float = 1e-5,
    threads_per_group: int = 256,
    verbose: bool = False,
) -> tuple[mx.array, mx.array]:
    """Reduce partial sums into inv_rms and d_r for backward."""
    if y_agg.ndim != 2:
        raise ValueError(f"y_agg must be [B, C], got shape {y_agg.shape}")
    if partial_sq.ndim != 2:
        raise ValueError(f"partial_sq must be [B, T], got shape {partial_sq.shape}")
    if partial_dr.ndim != 2:
        raise ValueError(f"partial_dr must be [B, T], got shape {partial_dr.shape}")

    B, _ = y_agg.shape
    if partial_sq.shape[0] != B or partial_dr.shape[0] != B:
        raise ValueError(
            "partial sums must have the same batch size as y_agg "
            f"(got {partial_sq.shape[0]} and {partial_dr.shape[0]} vs {B})"
        )

    if threads_per_group <= 0:
        raise ValueError("threads_per_group must be positive")
    if threads_per_group > _MAX_TPG_ALLOWED:
        raise ValueError(f"threads_per_group must be <= {_MAX_TPG_ALLOWED}")

    if verbose:
        _maybe_print_source(_mhc_backward_rms_reduce_source(eps), "mhc_backward_rms_reduce", True)

    kernel = _mhc_backward_rms_reduce_kernel(float(eps))
    inv_rms, d_r = kernel(
        inputs=[y_agg.astype(mx.float32), partial_sq.astype(mx.float32), partial_dr.astype(mx.float32)],
        grid=(threads_per_group, B, 1),
        threadgroup=(threads_per_group, 1, 1),
        output_shapes=[(B,), (B,)],
        output_dtypes=[mx.float32, mx.float32],
    )

    return inv_rms, d_r


def mhc_backward_dx_metal(
    x: mx.array,
    M: mx.array,
    H_pre: mx.array,
    rms_weight: mx.array,
    d_out: mx.array,
    y_agg: mx.array,
    d_y_norm: mx.array,
    inv_rms: mx.array,
    d_r: mx.array,
    threads_per_group: int = 256,
    verbose: bool = False,
) -> mx.array:
    """Compute dx for the fused mHC forward."""
    if x.ndim != 3:
        raise ValueError(f"x must be [B, n, C], got shape {x.shape}")
    if M.ndim != 2:
        raise ValueError(f"M must be [n, n], got shape {M.shape}")
    if d_out.shape != x.shape:
        raise ValueError(f"d_out must match x shape, got {d_out.shape} vs {x.shape}")

    B, n, C = x.shape
    if M.shape != (n, n):
        raise ValueError(f"M must be shape (n,n)=( {n},{n} ), got {M.shape}")

    max_n = _validate_n(n)
    if threads_per_group <= 0:
        raise ValueError("threads_per_group must be positive")
    if threads_per_group > _MAX_TPG_ALLOWED:
        raise ValueError(f"threads_per_group must be <= {_MAX_TPG_ALLOWED}")

    if verbose:
        _maybe_print_source(_mhc_backward_dx_col_source(max_n), "mhc_backward_dx_col", True)

    x_in = _maybe_cast_float32(x)
    d_out_in = _maybe_cast_float32(d_out)
    M_f = M.astype(mx.float32)
    H_pre_f = H_pre.astype(mx.float32)
    rms_weight_f = rms_weight.astype(mx.float32)
    y_agg_f = y_agg.astype(mx.float32)
    d_y_norm_f = d_y_norm.astype(mx.float32)
    inv_rms_f = inv_rms.astype(mx.float32)
    d_r_f = d_r.astype(mx.float32)

    kernel = _mhc_backward_dx_col_kernel(max_n)
    out = kernel(
        inputs=[x_in, M_f, H_pre_f, rms_weight_f, d_out_in, y_agg_f, d_y_norm_f, inv_rms_f, d_r_f],
        grid=(C, B, 1),
        threadgroup=(threads_per_group, 1, 1),
        output_shapes=[x_in.shape],
        output_dtypes=[mx.float32],
    )[0]

    return out


def mhc_backward_fused_dx_metal(
    x: mx.array,
    M: mx.array,
    H_pre: mx.array,
    H_post: mx.array,
    rms_weight: mx.array,
    d_out: mx.array,
    eps: float = 1e-5,
    threads_per_group: int = 256,
    verbose: bool = False,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Compute dx and RMS intermediates in one fused kernel."""
    if x.ndim != 3:
        raise ValueError(f"x must be [B, n, C], got shape {x.shape}")
    if M.ndim != 2:
        raise ValueError(f"M must be [n, n], got shape {M.shape}")
    if d_out.shape != x.shape:
        raise ValueError(f"d_out must match x shape, got {d_out.shape} vs {x.shape}")
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
        raise ValueError(f"threads_per_group must be <= {_MAX_TPG_ALLOWED}")

    if verbose:
        _maybe_print_source(_mhc_backward_fused_dx_source(max_n, eps), "mhc_backward_fused_dx", True)

    x_f = x.astype(mx.float32)
    M_f = M.astype(mx.float32)
    H_pre_f = H_pre.astype(mx.float32)
    H_post_f = H_post.astype(mx.float32)
    rms_weight_f = rms_weight.astype(mx.float32)
    d_out_f = d_out.astype(mx.float32)

    kernel = _mhc_backward_fused_dx_kernel(max_n, float(eps))
    dx, y_agg, d_y_norm, inv_rms, d_r = kernel(
        inputs=[x_f, M_f, H_pre_f, H_post_f, rms_weight_f, d_out_f],
        grid=(threads_per_group, B, 1),
        threadgroup=(threads_per_group, 1, 1),
        output_shapes=[x_f.shape, (B, C), (B, C), (B,), (B,)],
        output_dtypes=[mx.float32, mx.float32, mx.float32, mx.float32, mx.float32],
    )

    return dx, y_agg, d_y_norm, inv_rms, d_r


def mhc_backward_dM_metal(
    x: mx.array,
    d_out: mx.array,
    threads_per_group: int = 256,
    verbose: bool = False,
) -> mx.array:
    """Compute dM for stream mixing."""
    if x.ndim != 3:
        raise ValueError(f"x must be [B, n, C], got shape {x.shape}")
    if d_out.shape != x.shape:
        raise ValueError(f"d_out must match x shape, got {d_out.shape} vs {x.shape}")

    B, n, C = x.shape
    _ = B, C
    _validate_n(n)

    if threads_per_group <= 0:
        raise ValueError("threads_per_group must be positive")
    if threads_per_group > _MAX_TPG_ALLOWED:
        raise ValueError(f"threads_per_group must be <= {_MAX_TPG_ALLOWED}")

    if verbose:
        _maybe_print_source(_mhc_backward_dM_source(), "mhc_backward_dM", True)

    x_f = x.astype(mx.float32)
    d_out_f = d_out.astype(mx.float32)

    kernel = _mhc_backward_dM_kernel()
    out = kernel(
        inputs=[x_f, d_out_f],
        grid=(threads_per_group, n * n, 1),
        threadgroup=(threads_per_group, 1, 1),
        output_shapes=[(n, n)],
        output_dtypes=[mx.float32],
    )[0]

    return out


def mhc_backward_dH_pre_metal(
    x: mx.array,
    y_agg: mx.array,
    d_y_norm: mx.array,
    inv_rms: mx.array,
    d_r: mx.array,
    rms_weight: mx.array,
    threads_per_group: int = 256,
    verbose: bool = False,
) -> mx.array:
    """Compute dH_pre for the aggregate branch."""
    if x.ndim != 3:
        raise ValueError(f"x must be [B, n, C], got shape {x.shape}")
    B, n, C = x.shape
    if y_agg.shape != (B, C):
        raise ValueError(f"y_agg must be [B, C], got {y_agg.shape}")
    if d_y_norm.shape != (B, C):
        raise ValueError(f"d_y_norm must be [B, C], got {d_y_norm.shape}")
    if inv_rms.shape != (B,):
        raise ValueError(f"inv_rms must be [B], got {inv_rms.shape}")
    if d_r.shape != (B,):
        raise ValueError(f"d_r must be [B], got {d_r.shape}")
    if rms_weight.shape != (C,):
        raise ValueError(f"rms_weight must be [C], got {rms_weight.shape}")

    _validate_n(n)
    if threads_per_group <= 0:
        raise ValueError("threads_per_group must be positive")
    if threads_per_group > _MAX_TPG_ALLOWED:
        raise ValueError(f"threads_per_group must be <= {_MAX_TPG_ALLOWED}")

    if verbose:
        _maybe_print_source(_mhc_backward_dH_pre_source(), "mhc_backward_dH_pre", True)

    kernel = _mhc_backward_dH_pre_kernel()
    x_in = _maybe_cast_float32(x)
    out = kernel(
        inputs=[
            x_in,
            y_agg.astype(mx.float32),
            d_y_norm.astype(mx.float32),
            inv_rms.astype(mx.float32),
            d_r.astype(mx.float32),
            rms_weight.astype(mx.float32),
        ],
        grid=(threads_per_group, n, 1),
        threadgroup=(threads_per_group, 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[mx.float32],
    )[0]

    return out


def mhc_backward_dH_post_metal(
    d_out: mx.array,
    y_agg: mx.array,
    inv_rms: mx.array,
    rms_weight: mx.array,
    threads_per_group: int = 256,
    verbose: bool = False,
) -> mx.array:
    """Compute dH_post for the distribute branch."""
    if d_out.ndim != 3:
        raise ValueError(f"d_out must be [B, n, C], got shape {d_out.shape}")
    B, n, C = d_out.shape
    if y_agg.shape != (B, C):
        raise ValueError(f"y_agg must be [B, C], got {y_agg.shape}")
    if inv_rms.shape != (B,):
        raise ValueError(f"inv_rms must be [B], got {inv_rms.shape}")
    if rms_weight.shape != (C,):
        raise ValueError(f"rms_weight must be [C], got {rms_weight.shape}")

    _validate_n(n)
    if threads_per_group <= 0:
        raise ValueError("threads_per_group must be positive")
    if threads_per_group > _MAX_TPG_ALLOWED:
        raise ValueError(f"threads_per_group must be <= {_MAX_TPG_ALLOWED}")

    if verbose:
        _maybe_print_source(_mhc_backward_dH_post_source(), "mhc_backward_dH_post", True)

    kernel = _mhc_backward_dH_post_kernel()
    d_out_in = _maybe_cast_float32(d_out)
    out = kernel(
        inputs=[
            d_out_in,
            y_agg.astype(mx.float32),
            inv_rms.astype(mx.float32),
            rms_weight.astype(mx.float32),
        ],
        grid=(threads_per_group, n, 1),
        threadgroup=(threads_per_group, 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[mx.float32],
    )[0]

    return out


def mhc_backward_dH_pre_post_metal(
    x: mx.array,
    d_out: mx.array,
    y_agg: mx.array,
    d_y_norm: mx.array,
    inv_rms: mx.array,
    d_r: mx.array,
    rms_weight: mx.array,
    threads_per_group: int = 256,
    verbose: bool = False,
) -> tuple[mx.array, mx.array]:
    """Compute dH_pre and dH_post in one pass."""
    if x.ndim != 3:
        raise ValueError(f"x must be [B, n, C], got shape {x.shape}")
    if d_out.shape != x.shape:
        raise ValueError(f"d_out must match x shape, got {d_out.shape} vs {x.shape}")

    B, n, C = x.shape
    if y_agg.shape != (B, C):
        raise ValueError(f"y_agg must be [B, C], got {y_agg.shape}")
    if d_y_norm.shape != (B, C):
        raise ValueError(f"d_y_norm must be [B, C], got {d_y_norm.shape}")
    if inv_rms.shape != (B,):
        raise ValueError(f"inv_rms must be [B], got {inv_rms.shape}")
    if d_r.shape != (B,):
        raise ValueError(f"d_r must be [B], got {d_r.shape}")
    if rms_weight.shape != (C,):
        raise ValueError(f"rms_weight must be [C], got {rms_weight.shape}")

    _validate_n(n)
    if threads_per_group <= 0:
        raise ValueError("threads_per_group must be positive")
    if threads_per_group > _MAX_TPG_ALLOWED:
        raise ValueError(f"threads_per_group must be <= {_MAX_TPG_ALLOWED}")

    if verbose:
        _maybe_print_source(_mhc_backward_dH_pre_post_source(), "mhc_backward_dH_pre_post", True)

    kernel = _mhc_backward_dH_pre_post_kernel()
    x_in = _maybe_cast_float32(x)
    d_out_in = _maybe_cast_float32(d_out)
    dH_pre, dH_post = kernel(
        inputs=[
            x_in,
            d_out_in,
            y_agg.astype(mx.float32),
            d_y_norm.astype(mx.float32),
            inv_rms.astype(mx.float32),
            d_r.astype(mx.float32),
            rms_weight.astype(mx.float32),
        ],
        grid=(threads_per_group, n, 1),
        threadgroup=(threads_per_group, 1, 1),
        output_shapes=[(n,), (n,)],
        output_dtypes=[mx.float32, mx.float32],
    )

    return dH_pre, dH_post


def mhc_backward_drms_weight_metal(
    y_agg: mx.array,
    d_y_norm: mx.array,
    inv_rms: mx.array,
    threads_per_group: int = 256,
    verbose: bool = False,
) -> mx.array:
    """Compute d_rms_weight for RMSNorm."""
    if y_agg.ndim != 2:
        raise ValueError(f"y_agg must be [B, C], got shape {y_agg.shape}")
    if d_y_norm.shape != y_agg.shape:
        raise ValueError(f"d_y_norm must match y_agg shape, got {d_y_norm.shape}")
    if inv_rms.ndim != 1:
        raise ValueError(f"inv_rms must be [B], got shape {inv_rms.shape}")

    B, C = y_agg.shape
    if inv_rms.shape != (B,):
        raise ValueError(f"inv_rms must be [B], got {inv_rms.shape}")

    if threads_per_group <= 0:
        raise ValueError("threads_per_group must be positive")
    if threads_per_group > _MAX_TPG_ALLOWED:
        raise ValueError(f"threads_per_group must be <= {_MAX_TPG_ALLOWED}")

    if verbose:
        _maybe_print_source(_mhc_backward_drms_source(), "mhc_backward_drms", True)

    kernel = _mhc_backward_drms_kernel()
    out = kernel(
        inputs=[y_agg.astype(mx.float32), d_y_norm.astype(mx.float32), inv_rms.astype(mx.float32)],
        grid=(C, 1, 1),
        threadgroup=(threads_per_group, 1, 1),
        output_shapes=[(C,)],
        output_dtypes=[mx.float32],
    )[0]

    return out


def mhc_backward_grads_fused_metal(
    x: mx.array,
    d_out: mx.array,
    y_agg: mx.array,
    d_y_norm: mx.array,
    inv_rms: mx.array,
    d_r: mx.array,
    rms_weight: mx.array,
    threads_per_group: int = 256,
    verbose: bool = False,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Compute dM, dH_pre, dH_post, d_rms_weight in one fused pass."""
    if x.ndim != 3:
        raise ValueError(f"x must be [B, n, C], got {x.shape}")
    B, n, C = x.shape
    
    # Fallback for n != 32 (only n=32 is optimized for now)
    if n != 32:
        dM = mhc_backward_dM_metal(x, d_out, threads_per_group, verbose)
        dH_pre, dH_post = mhc_backward_dH_pre_post_metal(
            x, d_out, y_agg, d_y_norm, inv_rms, d_r, rms_weight, threads_per_group, verbose
        )
        d_rms = mhc_backward_drms_weight_metal(y_agg, d_y_norm, inv_rms, threads_per_group, verbose)
        return dM, dH_pre, dH_post, d_rms

    if verbose:
        _maybe_print_source(
            _mhc_backward_grads_fused_source(n),
            "mhc_backward_grads_fused",
            verbose=True,
        )

    # Initialize accumulators
    dM = mx.zeros((n, n), dtype=mx.float32)
    dH_pre = mx.zeros((n,), dtype=mx.float32)
    dH_post = mx.zeros((n,), dtype=mx.float32)
    d_rms = mx.zeros((C,), dtype=mx.float32)

    x_in = _maybe_cast_float32(x)
    dout_in = _maybe_cast_float32(d_out)
    y_agg_f = y_agg.astype(mx.float32)
    dy_f = d_y_norm.astype(mx.float32)
    inv_f = inv_rms.astype(mx.float32)
    dr_f = d_r.astype(mx.float32)
    rw_f = rms_weight.astype(mx.float32)

    kernel = _mhc_backward_grads_fused_kernel(n)
    
    # Launch with 1024 threadgroups of size 32
    num_groups = 1024
    tpg = 32
    
    kernel(
        inputs=[
            x_in, dout_in, y_agg_f, dy_f, inv_f, dr_f, rw_f,
            dM, dH_pre, dH_post, d_rms
        ],
        grid=(num_groups * tpg, 1, 1),
        threadgroup=(tpg, 1, 1),
        output_shapes=[(1,)],
        output_dtypes=[mx.float32],
    )
    
    return dM, dH_pre, dH_post, d_rms


def stream_mix_backward_dx_metal(
    M: mx.array,
    d_out: mx.array,
    threads_per_group: int = 256,
    verbose: bool = False,
) -> mx.array:
    """Compute dx for stream mixing."""
    if d_out.ndim != 3:
        raise ValueError(f"d_out must be [B, n, C], got shape {d_out.shape}")
    if M.ndim != 2:
        raise ValueError(f"M must be [n, n], got shape {M.shape}")

    B, n, C = d_out.shape
    if M.shape != (n, n):
        raise ValueError(f"M must be shape (n,n)=( {n},{n} ), got {M.shape}")

    max_n = _validate_n(n)
    if threads_per_group <= 0:
        raise ValueError("threads_per_group must be positive")
    if threads_per_group > _MAX_TPG_ALLOWED:
        raise ValueError(f"threads_per_group must be <= {_MAX_TPG_ALLOWED}")

    if verbose:
        _maybe_print_source(_stream_mix_backward_dx_source(max_n), "stream_mix_backward_dx", True)

    M_f = M.astype(mx.float32)

    kernel = _stream_mix_backward_dx_kernel(max_n)
    out = kernel(
        inputs=[M_f, d_out_in],
        grid=(C, B * n, 1),
        threadgroup=(threads_per_group, 1, 1),
        output_shapes=[d_out_in.shape],
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
        dx = stream_mix_backward_dx_metal(
            M,
            dout,
            threads_per_group=threads_per_group,
            verbose=False,
        )
        dM = mhc_backward_dM_metal(
            x,
            dout,
            threads_per_group=threads_per_group,
            verbose=False,
        )
        return _match_structure(primals, [dx, dM, dout])

    return _f


def stream_mix_add_metal_autograd(
    x: mx.array,
    M: mx.array,
    y_dist: mx.array,
    threads_per_group: int = 256,
    verbose: bool = False,
) -> mx.array:
    """Stream mix + add with Metal backward kernels."""
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
        dH_res = sinkhorn_knopp_backward_metal(
            H_res_raw,
            dM,
            iters=iters,
            eps=eps,
            threads_per_group=threads_per_group,
            verbose=False,
        )
        return _match_structure(primals, [dH_res])

    return _f


def sinkhorn_knopp_metal_autograd(
    H_res_raw: mx.array,
    iters: int = 20,
    eps: float = 1e-5,
    threads_per_group: int = 64,
    verbose: bool = False,
) -> mx.array:
    """Sinkhorn with Metal backward kernels."""
    if verbose:
        _maybe_print_source(_sinkhorn_source(_validate_n(H_res_raw.shape[0]), iters, eps), "sinkhorn_knopp", True)
    return _sinkhorn_autograd_fn(int(iters), float(eps), int(threads_per_group))(H_res_raw)


@lru_cache(maxsize=64)
def _mhc_fused_autograd_fn(
    eps: float,
    threads_per_group: int,
    fused_backward: bool,
    output_dtype_key: str,
    mix_kernel_key: str,
):
    output_dtype = _output_dtype_from_key(output_dtype_key)
    mix_kernel = _normalize_mix_kernel(mix_kernel_key)

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
            output_dtype=output_dtype,
            mix_kernel=mix_kernel,
            verbose=False,
        )

    @_f.vjp
    def _f_vjp(primals, cotangents, _):
        x, M, H_pre, H_post, rms_weight = primals
        dout = _as_list(cotangents)[0]

        if fused_backward:
            dx, y_agg, d_y_norm, inv_rms, d_r = mhc_backward_fused_dx_metal(
                x,
                M,
                H_pre,
                H_post,
                rms_weight,
                dout,
                eps=eps,
                threads_per_group=threads_per_group,
                verbose=False,
            )
            dM, dH_pre, dH_post, d_rms_weight = mhc_backward_grads_fused_metal(
                x,
                dout,
                y_agg,
                d_y_norm,
                inv_rms,
                d_r,
                rms_weight,
                threads_per_group=threads_per_group,
                verbose=False,
            )
        else:
            y_agg, d_y_norm, partial_sq, partial_dr = mhc_backward_prep_tile_metal(
                x,
                H_pre,
                H_post,
                rms_weight,
                dout,
                threads_per_group=threads_per_group,
                verbose=False,
            )
            inv_rms, d_r = mhc_backward_rms_reduce_metal(
                y_agg,
                partial_sq,
                partial_dr,
                eps=eps,
                threads_per_group=threads_per_group,
                verbose=False,
            )
            dx = mhc_backward_dx_metal(
                x,
                M,
                H_pre,
                rms_weight,
                dout,
                y_agg,
                d_y_norm,
                inv_rms,
                d_r,
                threads_per_group=threads_per_group,
                verbose=False,
            )
            dM = mhc_backward_dM_metal(x, dout, threads_per_group=threads_per_group)
            dH_pre = mhc_backward_dH_pre_metal(x, dout, d_y_norm, inv_rms, threads_per_group=threads_per_group)
            dH_post = mhc_backward_dH_post_metal(dout, d_y_norm, threads_per_group=threads_per_group)
            d_rms_weight = mhc_backward_rms_weight_metal(y_agg, inv_rms, dout, threads_per_group=threads_per_group)

        return _match_structure(primals, [dx, dM, dH_pre, dH_post, d_rms_weight])

    return _f


def mhc_forward_fused_metal_autograd(
    x: mx.array,
    M: mx.array,
    H_pre: mx.array,
    H_post: mx.array,
    rms_weight: mx.array,
    eps: float = 1e-5,
    threads_per_group: int = 256,
    fused_backward: bool = True,
    output_dtype: mx.Dtype | None = None,
    mix_kernel: str = "auto",
    verbose: bool = False,
) -> mx.array:
    """Fused forward with Metal backward kernels.

    output_dtype can request float16/bfloat16 output when x matches that dtype.
    mix_kernel selects the mix/add implementation ("auto", "1d", "2d").
    """
    if verbose:
        max_n = _validate_n(M.shape[0])
        _maybe_print_source(_mhc_forward_agg_source(max_n), "mhc_forward_agg", True)
        _maybe_print_source(_mhc_forward_rms_reduce_source(eps), "mhc_forward_rms_reduce", True)
        _maybe_print_source(_stream_mix_add_rms_source(max_n), "stream_mix_add_rms", True)
    output_dtype = _normalize_output_dtype(output_dtype)
    mix_kernel = _normalize_mix_kernel(mix_kernel)
    return _mhc_fused_autograd_fn(
        float(eps),
        int(threads_per_group),
        bool(fused_backward),
        _output_dtype_key(output_dtype),
        mix_kernel,
    )(x, M, H_pre, H_post, rms_weight)
