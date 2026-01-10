"""Tests for Sinkhorn-Knopp algorithm."""

import mlx.core as mx
import pytest


def test_sinkhorn_knopp_returns_doubly_stochastic():
    """Output rows and columns should sum to 1."""
    from mlx_mhc import sinkhorn_knopp

    # Random positive matrix
    matrix = mx.abs(mx.random.normal((4, 4))) + 0.1
    result = sinkhorn_knopp(matrix)

    # Check rows sum to 1
    row_sums = mx.sum(result, axis=1)
    assert mx.allclose(row_sums, mx.ones(4), atol=1e-5), f"Row sums: {row_sums}"

    # Check columns sum to 1
    col_sums = mx.sum(result, axis=0)
    assert mx.allclose(col_sums, mx.ones(4), atol=1e-5), f"Col sums: {col_sums}"


def test_sinkhorn_knopp_all_positive():
    """Output should have all non-negative entries."""
    from mlx_mhc import sinkhorn_knopp

    matrix = mx.abs(mx.random.normal((4, 4))) + 0.1
    result = sinkhorn_knopp(matrix)

    assert mx.all(result >= 0), "All entries should be non-negative"


def test_sinkhorn_knopp_preserves_shape():
    """Output shape should match input."""
    from mlx_mhc import sinkhorn_knopp

    for shape in [(4, 4), (8, 8), (16, 16)]:
        matrix = mx.abs(mx.random.normal(shape)) + 0.1
        result = sinkhorn_knopp(matrix)
        assert result.shape == shape


def test_sinkhorn_knopp_log_space_stability():
    """Log-space version should handle extreme values."""
    from mlx_mhc import sinkhorn_knopp

    # Large values that could cause overflow
    matrix = mx.random.normal((4, 4)) * 10
    result = sinkhorn_knopp(matrix, log_space=True)

    # Should not have NaN or Inf
    assert not mx.any(mx.isnan(result)), "Should not have NaN"
    assert not mx.any(mx.isinf(result)), "Should not have Inf"


def test_sinkhorn_knopp_direct_mode():
    """Direct (non-log) mode should also produce valid results."""
    from mlx_mhc import sinkhorn_knopp

    # Small values where direct mode is safe
    matrix = mx.random.normal((4, 4)) * 0.5
    result = sinkhorn_knopp(matrix, log_space=False)

    row_sums = mx.sum(result, axis=1)
    col_sums = mx.sum(result, axis=0)

    assert mx.allclose(row_sums, mx.ones(4), atol=1e-4)
    assert mx.allclose(col_sums, mx.ones(4), atol=1e-4)
