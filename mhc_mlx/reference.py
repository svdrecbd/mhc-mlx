"""Pure-MLX reference implementations.

These functions are intentionally written in a simple, readable style.
They are the source of truth for correctness; the Metal kernel must match them.

Shapes

- x_expanded: [B, n, C] (row contiguous)
- H_pre_raw: [n]
- H_post_raw: [n]
- H_res_raw: [n, n] (logits)
- weight: [C]
"""

from __future__ import annotations

import mlx.core as mx


def sinkhorn_knopp(H: mx.array, iters: int = 20, eps: float = 1e-5) -> mx.array:
    """Sinkhorn-Knopp normalization.

    Matches the reference behavior used in this repo:
    - row normalization: if row_sum <= eps, leave the row unchanged
    - column normalization: if col_sum <= eps, zero that column

    Args:
        H: [n, n] matrix (typically positive)
        iters: number of Sinkhorn iterations
        eps: stability epsilon

    Returns:
        P: [n, n] approximately doubly-stochastic matrix
    """
    P = H.astype(mx.float32)

    for _ in range(iters):
        # Row normalization
        row_sum = mx.sum(P, axis=1)  # [n]
        row_scale = mx.where(row_sum > eps, 1.0 / row_sum, 1.0)
        P = P * row_scale[:, None]

        # Column normalization
        col_sum = mx.sum(P, axis=0)  # [n]
        col_scale = mx.where(col_sum > eps, 1.0 / col_sum, 0.0)
        P = P * col_scale[None, :]

    return P


def activate_pre_post(H_pre_raw: mx.array, H_post_raw: mx.array) -> tuple[mx.array, mx.array]:
    """Apply project activations for pre/post scalars."""
    H_pre_act = mx.sigmoid(H_pre_raw.astype(mx.float32))
    H_post_act = 2.0 * mx.sigmoid(H_post_raw.astype(mx.float32))
    return H_pre_act, H_post_act


def mixing_matrix_from_logits(H_res_raw: mx.array, iters: int = 20, eps: float = 1e-5) -> mx.array:
    """Project logits onto the Birkhoff polytope via Sinkhorn.

    The parameterization is:
    M = sinkhorn_knopp(exp(H_res_raw)).
    """
    if H_res_raw.ndim != 2 or H_res_raw.shape[0] != H_res_raw.shape[1]:
        raise ValueError(f"H_res_raw must be square [n, n], got shape {H_res_raw.shape}")
    H = mx.exp(H_res_raw.astype(mx.float32))
    return sinkhorn_knopp(H, iters=iters, eps=eps)


def stream_aggregate(x_expanded: mx.array, H_pre: mx.array) -> mx.array:
    """Aggregate n streams into one vector per batch.

    y_agg[b, c] = sum_i H_pre[i] * x_expanded[b, i, c]

    Args:
        x_expanded: [B, n, C]
        H_pre: [n] (activated)

    Returns:
        y_agg: [B, C]
    """
    x = x_expanded.astype(mx.float32)
    w = H_pre.astype(mx.float32)
    return mx.sum(x * w[None, :, None], axis=1)


def rms_norm(x: mx.array, weight: mx.array, eps: float = 1e-5) -> mx.array:
    """RMSNorm over the last axis.

    Args:
        x: [B, C]
        weight: [C]
        eps: stability epsilon

    Returns:
        y: [B, C]
    """
    x_f = x.astype(mx.float32)
    w_f = weight.astype(mx.float32)

    mean_sq = mx.mean(x_f * x_f, axis=-1, keepdims=True)
    inv_rms = 1.0 / mx.sqrt(mean_sq + eps)
    return (x_f * inv_rms) * w_f[None, :]


def stream_distribute(y_norm: mx.array, H_post: mx.array) -> mx.array:
    """Distribute a [B, C] vector back into n streams.

    y_dist[b, i, c] = H_post[i] * y_norm[b, c]

    Args:
        y_norm: [B, C]
        H_post: [n] (activated)

    Returns:
        y_dist: [B, n, C]
    """
    y = y_norm.astype(mx.float32)
    w = H_post.astype(mx.float32)
    return y[:, None, :] * w[None, :, None]


def stream_mix_ref(x_expanded: mx.array, M: mx.array) -> mx.array:
    """Reference stream mixing.

    x_mixed[b, i, c] = sum_j M[i, j] * x_expanded[b, j, c]

    Args:
        x_expanded: [B, n, C]
        M: [n, n]

    Returns:
        x_mixed: [B, n, C]
    """
    x = x_expanded.astype(mx.float32)
    M_f = M.astype(mx.float32)

    # Broadcast form that is easy to read and matches the math.
    # Shapes:
    # - M:      [n, n] -> [1, n, n, 1]
    # - x: [B, n, C] -> [B, 1, n, C]
    # Then sum over the "j" axis (axis=2).
    return mx.sum(M_f[None, :, :, None] * x[:, None, :, :], axis=2)


def mhc_forward_fused_reference(
    x_expanded: mx.array,
    M: mx.array,
    H_pre_act: mx.array,
    H_post_act: mx.array,
    rms_weight: mx.array,
    eps: float = 1e-5,
) -> mx.array:
    """Reference fused forward pass using activated pre/post and a mixing matrix."""
    y_agg = stream_aggregate(x_expanded, H_pre_act)
    y_norm = rms_norm(y_agg, rms_weight, eps=eps)
    y_dist = stream_distribute(y_norm, H_post_act)
    x_mixed = stream_mix_ref(x_expanded, M)
    return x_mixed + y_dist


def mhc_forward_reference(
    x_expanded: mx.array,
    H_pre_raw: mx.array,
    H_post_raw: mx.array,
    H_res_raw: mx.array,
    rms_weight: mx.array,
    sinkhorn_iters: int = 20,
    eps: float = 1e-5,
) -> mx.array:
    """Full reference forward pass.

    Returns:
        out: [B, n, C]
    """
    H_pre_act, H_post_act = activate_pre_post(H_pre_raw, H_post_raw)
    M = mixing_matrix_from_logits(H_res_raw, iters=sinkhorn_iters, eps=eps)
    y_agg = stream_aggregate(x_expanded, H_pre_act)
    y_norm = rms_norm(y_agg, rms_weight, eps=eps)
    y_dist = stream_distribute(y_norm, H_post_act)
    x_mixed = stream_mix_ref(x_expanded, M)
    return x_mixed + y_dist
