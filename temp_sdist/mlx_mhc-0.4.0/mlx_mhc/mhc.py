"""Manifold-Constrained Hyper-Connections module."""

import math
import mlx.core as mx
import mlx.nn as nn

from .sinkhorn import sinkhorn_knopp


class ManifoldHyperConnection(nn.Module):
    """
    Manifold-Constrained Hyper-Connection (mHC) module.

    Implements the mHC architecture from DeepSeek's paper (arXiv:2512.24880).
    
    Paper equation: x_{l+1} = H_post * (F(H_pre * x) + H_res * H_pre * x)

    TWO-STAGE API (Recommended - matches paper exactly):
        mhc = ManifoldHyperConnection(dims, expansion)
        x_pre = mhc.pre_scale(x)           # Apply H_pre
        layer_out = your_layer(x_pre)      # F(H_pre * x)
        output = mhc.post_combine(x, layer_out)

    LEGACY API (Backward compatible):
        output = mhc(x, layer_output)

    Args:
        dims: Hidden dimension of the input/output.
        expansion: Expansion factor for the hyper-connection width (default: 2).
        sinkhorn_iterations: Number of Sinkhorn-Knopp iterations (default: 10).
    """

    def __init__(
        self,
        dims: int,
        expansion: int = 2,
        sinkhorn_iterations: int = 20,
    ):
        super().__init__()

        self.dims = dims
        self.expansion = expansion
        self.sinkhorn_iterations = sinkhorn_iterations

        # Cache for eval mode (populated on first forward, cleared on train())
        self._cached_h_res = None
        self._cached_h_pre = None
        self._cached_h_post = None

        scale = 1.0 / math.sqrt(expansion)
        self.h_res_raw = mx.random.normal((expansion, expansion)) * scale
        self.h_pre_raw = mx.zeros((expansion,))
        self.h_pre_bias = mx.zeros((expansion,))
        self.h_post_raw = mx.zeros((expansion,))
        self.h_post_bias = mx.zeros((expansion,))

    def train(self, mode: bool = True):
        """Override to clear cache when switching to train mode."""
        if mode and not self.training:
            self._cached_h_res = None
            self._cached_h_pre = None
            self._cached_h_post = None
        return super().train(mode)

    def _project_h_res(self) -> mx.array:
        """Project H_res to doubly stochastic matrix using Sinkhorn-Knopp."""
        return sinkhorn_knopp(
            self.h_res_raw,
            max_iterations=self.sinkhorn_iterations,
            log_space=True,
        )

    def _project_h_pre(self) -> mx.array:
        """Project H_pre to non-negative via sigmoid."""
        return mx.sigmoid(self.h_pre_raw + self.h_pre_bias)

    def _project_h_post(self) -> mx.array:
        """Project H_post to non-negative [0, 2] via scaled sigmoid."""
        return 2.0 * mx.sigmoid(self.h_post_raw + self.h_post_bias)

    def pre_scale(self, x: mx.array) -> mx.array:
        """
        Apply H_pre scaling to input (first stage of two-stage API).

        Args:
            x: Input tensor of shape (batch, seq_len, dims)

        Returns:
            Scaled tensor of shape (batch, seq_len, dims): H_pre * x
        """
        batch_size, seq_len, dims = x.shape
        h_pre = self._project_h_pre()
        x_expanded = x.reshape(batch_size, seq_len, self.expansion, -1)
        x_pre = x_expanded * h_pre.reshape(1, 1, self.expansion, 1)
        return x_pre.reshape(batch_size, seq_len, dims)

    def _get_h_res(self) -> mx.array:
        """Get H_res, using cache in eval mode."""
        if not self.training and self._cached_h_res is not None:
            return self._cached_h_res
        h_res = self._project_h_res()
        if not self.training:
            self._cached_h_res = h_res
        return h_res

    def _get_h_pre(self) -> mx.array:
        """Get H_pre, using cache in eval mode."""
        if not self.training and self._cached_h_pre is not None:
            return self._cached_h_pre
        h_pre = self._project_h_pre()
        if not self.training:
            self._cached_h_pre = h_pre
        return h_pre

    def _get_h_post(self) -> mx.array:
        """Get H_post, using cache in eval mode."""
        if not self.training and self._cached_h_post is not None:
            return self._cached_h_post
        h_post = self._project_h_post()
        if not self.training:
            self._cached_h_post = h_post
        return h_post

    def post_combine(self, x: mx.array, layer_output: mx.array) -> mx.array:
        """
        Combine layer output with residual and apply H_post (second stage).

        Computes: H_post * (layer_output + H_res * H_pre * x)

        Args:
            x: Original input tensor of shape (batch, seq_len, dims)
            layer_output: Output from layer F applied to pre_scale(x)

        Returns:
            Output tensor of shape (batch, seq_len, dims)
        """
        batch_size, seq_len, dims = x.shape

        h_res = self._get_h_res()
        h_pre = self._get_h_pre()
        h_post = self._get_h_post()

        x_expanded = x.reshape(batch_size, seq_len, self.expansion, -1)
        x_pre = x_expanded * h_pre.reshape(1, 1, self.expansion, 1)
        x_res = mx.einsum('ij,...jd->...id', h_res, x_pre)

        layer_expanded = layer_output.reshape(batch_size, seq_len, self.expansion, -1)
        combined = layer_expanded + x_res
        output_expanded = combined * h_post.reshape(1, 1, self.expansion, 1)

        return output_expanded.reshape(batch_size, seq_len, dims)

    def __call__(self, x: mx.array, layer_output: mx.array) -> mx.array:
        """
        Apply manifold-constrained hyper-connection (legacy API).

        For correct paper behavior, use the two-stage API instead:
            x_pre = mhc.pre_scale(x)
            layer_out = your_layer(x_pre)
            output = mhc.post_combine(x, layer_out)

        Args:
            x: Input tensor of shape (batch, seq_len, dims)
            layer_output: Output from the layer of shape (batch, seq_len, dims)

        Returns:
            Output tensor of shape (batch, seq_len, dims)
        """
        return self.post_combine(x, layer_output)
