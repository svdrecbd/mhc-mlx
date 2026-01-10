"""Sinkhorn-Knopp algorithm for projecting matrices to doubly stochastic."""

import mlx.core as mx


def sinkhorn_knopp(
    matrix: mx.array,
    max_iterations: int = 100,
    epsilon: float = 1e-6,
    log_space: bool = True,
) -> mx.array:
    """
    Project a matrix onto the Birkhoff polytope (doubly stochastic matrices).

    Uses the Sinkhorn-Knopp algorithm to iteratively normalize rows and columns
    until both sum to 1.

    Args:
        matrix: Input matrix of shape (n, n). Will be exponentiated internally.
        max_iterations: Maximum number of alternating normalization steps.
        epsilon: Convergence threshold for row/column sums.
        log_space: If True, use log-space computation for numerical stability.

    Returns:
        Doubly stochastic matrix where all rows and columns sum to 1.

    Example:
        >>> matrix = mx.random.normal((4, 4))
        >>> ds = sinkhorn_knopp(matrix)
        >>> mx.sum(ds, axis=1)  # All close to 1
        >>> mx.sum(ds, axis=0)  # All close to 1
    """
    if log_space:
        return _sinkhorn_log_space(matrix, max_iterations, epsilon)
    else:
        return _sinkhorn_direct(matrix, max_iterations, epsilon)


def _sinkhorn_log_space(
    matrix: mx.array,
    max_iterations: int,
    epsilon: float,
) -> mx.array:
    """Log-space Sinkhorn for numerical stability."""
    # Initialize in log space
    log_P = matrix  # Input is treated as log-scores

    for _ in range(max_iterations):
        # Log-space row normalization: subtract logsumexp of each row
        log_P = log_P - mx.logsumexp(log_P, axis=1, keepdims=True)

        # Log-space column normalization: subtract logsumexp of each column
        log_P = log_P - mx.logsumexp(log_P, axis=0, keepdims=True)

        # Check convergence
        P = mx.exp(log_P)
        row_sums = mx.sum(P, axis=1)
        col_sums = mx.sum(P, axis=0)

        row_err = mx.max(mx.abs(row_sums - 1.0))
        col_err = mx.max(mx.abs(col_sums - 1.0))

        # MLX arrays need .item() for Python comparison
        if float(row_err) < epsilon and float(col_err) < epsilon:
            break

    return mx.exp(log_P)


def _sinkhorn_direct(
    matrix: mx.array,
    max_iterations: int,
    epsilon: float,
) -> mx.array:
    """Direct Sinkhorn (less stable but faster for small matrices)."""
    # Exponentiate and ensure positive
    P = mx.exp(matrix)
    P = mx.maximum(P, 1e-10)  # Avoid division by zero

    for _ in range(max_iterations):
        # Row normalization
        P = P / mx.sum(P, axis=1, keepdims=True)

        # Column normalization
        P = P / mx.sum(P, axis=0, keepdims=True)

        # Check convergence
        row_sums = mx.sum(P, axis=1)
        col_sums = mx.sum(P, axis=0)

        row_err = mx.max(mx.abs(row_sums - 1.0))
        col_err = mx.max(mx.abs(col_sums - 1.0))

        if float(row_err) < epsilon and float(col_err) < epsilon:
            break

    return P
