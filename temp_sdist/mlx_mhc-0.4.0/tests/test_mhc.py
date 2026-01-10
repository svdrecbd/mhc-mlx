"""Tests for ManifoldHyperConnection module."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_mhc.mhc import ManifoldHyperConnection


class TestManifoldHyperConnection:
    """Test suite for ManifoldHyperConnection."""

    def test_output_shape_matches_input(self):
        """Output shape should match input shape."""
        dims = 64
        batch_size = 2
        seq_len = 16
        expansion = 2

        mhc = ManifoldHyperConnection(dims=dims, expansion=expansion)

        x = mx.random.normal((batch_size, seq_len, dims))
        layer_output = mx.random.normal((batch_size, seq_len, dims))

        output = mhc(x, layer_output)

        assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"

    def test_output_shape_with_different_expansions(self):
        """Output shape should match for various expansion factors."""
        dims = 128
        batch_size = 4
        seq_len = 32

        for expansion in [2, 4, 8]:
            mhc = ManifoldHyperConnection(dims=dims, expansion=expansion)

            x = mx.random.normal((batch_size, seq_len, dims))
            layer_output = mx.random.normal((batch_size, seq_len, dims))

            output = mhc(x, layer_output)

            assert output.shape == (batch_size, seq_len, dims), (
                f"Expansion {expansion}: Expected {(batch_size, seq_len, dims)}, got {output.shape}"
            )

    def test_inherits_from_nn_module(self):
        """ManifoldHyperConnection should inherit from nn.Module."""
        mhc = ManifoldHyperConnection(dims=64)

        assert isinstance(mhc, nn.Module), "Should inherit from nn.Module"

    def test_has_trainable_parameters(self):
        """Module should have trainable parameters."""
        mhc = ManifoldHyperConnection(dims=64, expansion=2)

        # Get parameters from the module
        params = mhc.trainable_parameters()

        assert len(params) > 0, "Should have trainable parameters"

        # Check specific parameter names exist (MLX uses dict-like access)
        param_names = set(params.keys())

        expected_params = {"h_res_raw", "h_pre_raw", "h_pre_bias", "h_post_raw", "h_post_bias"}
        assert expected_params.issubset(param_names), (
            f"Missing parameters. Expected {expected_params}, got {param_names}"
        )

    def test_h_res_is_doubly_stochastic(self):
        """H_res should be doubly stochastic (rows and columns sum to 1)."""
        mhc = ManifoldHyperConnection(dims=64, expansion=4, sinkhorn_iterations=20)

        h_res = mhc._project_h_res()
        mx.eval(h_res)

        # Check row sums
        row_sums = mx.sum(h_res, axis=1)
        mx.eval(row_sums)
        for i, s in enumerate(row_sums.tolist()):
            assert abs(s - 1.0) < 1e-4, f"Row {i} sum is {s}, expected 1.0"

        # Check column sums
        col_sums = mx.sum(h_res, axis=0)
        mx.eval(col_sums)
        for i, s in enumerate(col_sums.tolist()):
            assert abs(s - 1.0) < 1e-4, f"Column {i} sum is {s}, expected 1.0"

    def test_h_res_has_correct_shape(self):
        """H_res should have shape (expansion, expansion)."""
        for expansion in [2, 4, 8]:
            mhc = ManifoldHyperConnection(dims=64, expansion=expansion)
            h_res = mhc._project_h_res()

            assert h_res.shape == (expansion, expansion), (
                f"Expected ({expansion}, {expansion}), got {h_res.shape}"
            )

    def test_h_pre_is_non_negative(self):
        """H_pre should be non-negative (in [0, 1] via sigmoid)."""
        mhc = ManifoldHyperConnection(dims=64, expansion=4)

        # Test with various raw values
        mhc.h_pre_raw = mx.array([-10.0, -1.0, 0.0, 5.0])
        mhc.h_pre_bias = mx.zeros((4,))

        h_pre = mhc._project_h_pre()
        mx.eval(h_pre)

        for i, v in enumerate(h_pre.tolist()):
            assert v >= 0.0, f"H_pre[{i}] = {v} is negative"
            assert v <= 1.0, f"H_pre[{i}] = {v} exceeds 1.0"

    def test_h_post_is_non_negative(self):
        """H_post should be non-negative (in [0, 2] via scaled sigmoid)."""
        mhc = ManifoldHyperConnection(dims=64, expansion=4)

        # Test with various raw values
        mhc.h_post_raw = mx.array([-10.0, -1.0, 0.0, 10.0])
        mhc.h_post_bias = mx.zeros((4,))

        h_post = mhc._project_h_post()
        mx.eval(h_post)

        for i, v in enumerate(h_post.tolist()):
            assert v >= 0.0, f"H_post[{i}] = {v} is negative"
            assert v <= 2.0, f"H_post[{i}] = {v} exceeds 2.0"

    def test_h_post_range(self):
        """H_post should span [0, 2] range based on input."""
        mhc = ManifoldHyperConnection(dims=64, expansion=2)

        # Very negative input -> close to 0
        mhc.h_post_raw = mx.array([-100.0, -100.0])
        mhc.h_post_bias = mx.zeros((2,))
        h_post_low = mhc._project_h_post()
        mx.eval(h_post_low)

        # Very positive input -> close to 2
        mhc.h_post_raw = mx.array([100.0, 100.0])
        h_post_high = mhc._project_h_post()
        mx.eval(h_post_high)

        assert all(v < 0.01 for v in h_post_low.tolist()), "Large negative should give ~0"
        assert all(v > 1.99 for v in h_post_high.tolist()), "Large positive should give ~2"

    def test_forward_pass_is_differentiable(self):
        """Forward pass should be differentiable for gradient computation."""
        dims = 32
        batch_size = 2
        seq_len = 8
        expansion = 2

        mhc = ManifoldHyperConnection(dims=dims, expansion=expansion)

        x = mx.random.normal((batch_size, seq_len, dims))
        layer_output = mx.random.normal((batch_size, seq_len, dims))

        def loss_fn(model, x, layer_output):
            output = model(x, layer_output)
            return mx.mean(output ** 2)

        # Compute gradients
        loss, grads = mx.value_and_grad(loss_fn)(mhc, x, layer_output)
        mx.eval(loss, grads)

        # Verify loss is a scalar
        assert loss.shape == (), f"Loss should be scalar, got shape {loss.shape}"

        # Verify we got gradients for parameters
        assert grads is not None, "Should have gradients"

    def test_default_initialization(self):
        """Test default parameter initialization."""
        mhc = ManifoldHyperConnection(dims=64, expansion=2)

        # H_pre and H_post raw should start at 0
        assert mx.allclose(mhc.h_pre_raw, mx.zeros((2,))), "h_pre_raw should be zeros"
        assert mx.allclose(mhc.h_pre_bias, mx.zeros((2,))), "h_pre_bias should be zeros"
        assert mx.allclose(mhc.h_post_raw, mx.zeros((2,))), "h_post_raw should be zeros"
        assert mx.allclose(mhc.h_post_bias, mx.zeros((2,))), "h_post_bias should be zeros"

        # Initial H_pre should be 0.5 (sigmoid(0) = 0.5)
        h_pre = mhc._project_h_pre()
        mx.eval(h_pre)
        assert mx.allclose(h_pre, mx.full((2,), 0.5), atol=1e-6), "Initial H_pre should be 0.5"

        # Initial H_post should be 1.0 (2 * sigmoid(0) = 1.0)
        h_post = mhc._project_h_post()
        mx.eval(h_post)
        assert mx.allclose(h_post, mx.full((2,), 1.0), atol=1e-6), "Initial H_post should be 1.0"


class TestManifoldHyperConnectionEdgeCases:
    """Edge case tests for ManifoldHyperConnection."""

    def test_single_element_batch(self):
        """Should handle batch size of 1."""
        mhc = ManifoldHyperConnection(dims=32, expansion=2)

        x = mx.random.normal((1, 8, 32))
        layer_output = mx.random.normal((1, 8, 32))

        output = mhc(x, layer_output)

        assert output.shape == (1, 8, 32)

    def test_single_token_sequence(self):
        """Should handle sequence length of 1."""
        mhc = ManifoldHyperConnection(dims=32, expansion=2)

        x = mx.random.normal((4, 1, 32))
        layer_output = mx.random.normal((4, 1, 32))

        output = mhc(x, layer_output)

        assert output.shape == (4, 1, 32)

    def test_large_expansion(self):
        """Should handle large expansion factors."""
        dims = 256
        expansion = 16
        mhc = ManifoldHyperConnection(dims=dims, expansion=expansion)

        x = mx.random.normal((2, 4, dims))
        layer_output = mx.random.normal((2, 4, dims))

        output = mhc(x, layer_output)

        assert output.shape == (2, 4, dims)

        # Verify H_res is still doubly stochastic
        h_res = mhc._project_h_res()
        mx.eval(h_res)
        row_sums = mx.sum(h_res, axis=1)
        col_sums = mx.sum(h_res, axis=0)
        mx.eval(row_sums, col_sums)

        assert mx.allclose(row_sums, mx.ones((expansion,)), atol=1e-3)
        assert mx.allclose(col_sums, mx.ones((expansion,)), atol=1e-3)


class TestTwoStageAPI:
    """Tests for the two-stage pre_scale/post_combine API."""

    def test_pre_scale_exists(self):
        """pre_scale method should exist and be callable."""
        mhc = ManifoldHyperConnection(dims=64, expansion=2)
        assert hasattr(mhc, 'pre_scale'), "pre_scale method should exist"
        assert callable(mhc.pre_scale), "pre_scale should be callable"

    def test_pre_scale_output_shape(self):
        """pre_scale should preserve input shape."""
        dims = 64
        mhc = ManifoldHyperConnection(dims=dims, expansion=2)
        x = mx.random.normal((2, 16, dims))
        x_pre = mhc.pre_scale(x)
        mx.eval(x_pre)
        assert x_pre.shape == x.shape

    def test_post_combine_exists(self):
        """post_combine method should exist and be callable."""
        mhc = ManifoldHyperConnection(dims=64, expansion=2)
        assert hasattr(mhc, 'post_combine'), "post_combine method should exist"
        assert callable(mhc.post_combine), "post_combine should be callable"


class TestPaperEquation:
    """Tests verifying implementation matches the DeepSeek paper equation."""

    def test_two_stage_matches_paper_equation(self):
        """Verify: x_{l+1} = H_post * (F(H_pre * x) + H_res * H_pre * x)"""
        dims = 8
        expansion = 2
        mhc = ManifoldHyperConnection(dims=dims, expansion=expansion, sinkhorn_iterations=20)

        h_pre = mhc._project_h_pre()
        h_res = mhc._project_h_res()
        h_post = mhc._project_h_post()
        mx.eval(h_pre, h_res, h_post)

        x = mx.random.normal((1, 1, dims))
        mx.eval(x)

        # Manual computation
        x_expanded = x.reshape(1, 1, expansion, -1)
        h_pre_x = x_expanded * h_pre.reshape(1, 1, expansion, 1)

        def layer_f(inp):
            return inp * 2.0

        f_h_pre_x = layer_f(h_pre_x)
        h_res_h_pre_x = mx.einsum('ij,...jd->...id', h_res, h_pre_x)
        combined = f_h_pre_x + h_res_h_pre_x
        expected = combined * h_post.reshape(1, 1, expansion, 1)
        expected = expected.reshape(1, 1, dims)
        mx.eval(expected)

        # Two-stage API
        x_pre = mhc.pre_scale(x)
        layer_out = layer_f(x_pre)
        actual = mhc.post_combine(x, layer_out)
        mx.eval(actual)

        assert mx.allclose(actual, expected, atol=1e-5), "Two-stage API should match paper equation"

    def test_gradient_flow_through_full_equation(self):
        """Verify gradients flow correctly through the full paper equation."""
        dims = 16
        expansion = 2

        mhc = ManifoldHyperConnection(dims=dims, expansion=expansion)
        w = mx.random.normal((dims, dims)) * 0.1

        def forward(model, w, x):
            x_pre = model.pre_scale(x)
            layer_out = x_pre @ w
            output = model.post_combine(x, layer_out)
            return mx.mean(output ** 2)

        x = mx.random.normal((2, 4, dims))
        loss, grads = mx.value_and_grad(forward)(mhc, w, x)
        mx.eval(loss, grads)

        assert loss.shape == ()
        assert 'h_pre_raw' in grads
        assert 'h_post_raw' in grads
        assert 'h_res_raw' in grads


class TestEvalModeCaching:
    """Tests for eval-mode Sinkhorn caching optimization."""

    def test_eval_mode_caches_h_matrices(self):
        """H matrices should be cached after forward pass in eval mode."""
        mhc = ManifoldHyperConnection(dims=64, expansion=2)
        mhc.eval()
        
        x = mx.random.normal((2, 8, 64))
        layer_out = mx.random.normal((2, 8, 64))
        # Use post_combine which exercises all H matrices
        _ = mhc.post_combine(x, layer_out)
        mx.eval(_)
        
        # All caches should be populated
        assert mhc._cached_h_res is not None, "H_res should be cached in eval mode"
        assert mhc._cached_h_pre is not None, "H_pre should be cached in eval mode"
        assert mhc._cached_h_post is not None, "H_post should be cached in eval mode"

    def test_train_mode_clears_cache(self):
        """Switching to train mode should clear the cache."""
        mhc = ManifoldHyperConnection(dims=64, expansion=2)
        mhc.eval()
        
        x = mx.random.normal((2, 8, 64))
        layer_out = mx.random.normal((2, 8, 64))
        _ = mhc.post_combine(x, layer_out)
        mx.eval(_)
        
        # Verify cache is populated
        assert mhc._cached_h_res is not None
        
        # Switch to train mode
        mhc.train()
        
        # Cache should be cleared
        assert mhc._cached_h_res is None, "Cache should be cleared when entering train mode"


class TestPaperAlignment:
    """Tests verifying alignment with paper recommendations."""

    def test_default_sinkhorn_iterations_matches_paper(self):
        """Default sinkhorn_iterations should be 20 per paper tmax=20."""
        mhc = ManifoldHyperConnection(dims=64)
        assert mhc.sinkhorn_iterations == 20, "Paper recommends tmax=20"
