"""Tests for benchmark utilities - TDD: one test at a time."""

import mlx.core as mx
import pytest


def test_tracker_records_gradient_norms():
    """GradientTracker should record norm at each step."""
    from mlx_mhc.benchmark import GradientTracker

    tracker = GradientTracker()

    # Simulate 3 training steps with gradients
    for i in range(3):
        grads = {"layer1": mx.ones((10, 10)) * (i + 1)}
        tracker.record(grads)

    assert len(tracker.history) == 3
    assert tracker.history[0] < tracker.history[1] < tracker.history[2]


def test_tracker_computes_mean_std():
    """Should compute mean and std of gradient norms."""
    from mlx_mhc.benchmark import GradientTracker

    tracker = GradientTracker()
    for val in [1.0, 2.0, 3.0]:
        grads = {"w": mx.ones((10,)) * val}
        tracker.record(grads)

    stats = tracker.stats()
    assert "mean" in stats
    assert "std" in stats
    assert stats["mean"] > 0


def test_create_baseline_model():
    """Should create transformer block with standard residuals."""
    from mlx_mhc.benchmark import create_baseline_model

    model = create_baseline_model(dims=64, num_layers=2)

    x = mx.random.normal((2, 8, 64))
    out = model(x)
    mx.eval(out)

    assert out.shape == x.shape


def test_create_mhc_model():
    """Should create transformer block with mHC residuals."""
    from mlx_mhc.benchmark import create_mhc_model

    model = create_mhc_model(dims=64, num_layers=2, expansion=2)

    x = mx.random.normal((2, 8, 64))
    out = model(x)
    mx.eval(out)

    assert out.shape == x.shape


def test_mhc_has_more_params():
    """mHC model should have more parameters than baseline."""
    from mlx_mhc.benchmark import create_baseline_model, create_mhc_model, count_params

    baseline = create_baseline_model(dims=64, num_layers=2)
    mhc = create_mhc_model(dims=64, num_layers=2, expansion=2)

    baseline_params = count_params(baseline)
    mhc_params = count_params(mhc)

    assert mhc_params > baseline_params


def test_train_step_returns_loss_and_grads():
    """Single train step should return loss and gradient norms."""
    from mlx_mhc.benchmark import create_baseline_model, train_step

    model = create_baseline_model(dims=32, num_layers=1)
    x = mx.random.normal((2, 4, 32))
    y = mx.random.normal((2, 4, 32))

    loss, grad_norm = train_step(model, x, y)
    mx.eval(loss)

    assert float(loss) > 0
    assert grad_norm > 0


def test_compare_models_returns_results():
    """compare_models should return gradient stats for both models."""
    from mlx_mhc.benchmark import compare_models

    results = compare_models(
        dims=32,
        num_layers=1,
        num_steps=3,
        batch_size=2,
        seq_len=4
    )

    assert "baseline" in results
    assert "mhc" in results
    assert "grad_mean" in results["baseline"]
    assert "grad_mean" in results["mhc"]


def test_compare_models_returns_results():
    """Should return comparison results for baseline and mHC."""
    from mlx_mhc.benchmark import compare_models

    results = compare_models(
        dims=32,
        num_layers=1,
        num_steps=3,
        batch_size=2,
        seq_len=4
    )

    assert "baseline" in results
    assert "mhc" in results
    assert "grad_mean" in results["baseline"]
    assert "grad_std" in results["baseline"]
    assert "param_overhead_pct" in results
