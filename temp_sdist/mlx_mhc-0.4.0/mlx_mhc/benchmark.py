"""Benchmark utilities for comparing mHC vs standard residuals."""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from typing import Dict, List, Any, Tuple

from .mhc import ManifoldHyperConnection


class GradientTracker:
    """Track gradient norms during training."""

    def __init__(self):
        self.history: List[float] = []

    def record(self, grads: Any) -> None:
        """Record the total gradient norm for this step."""
        norm = _compute_grad_norm(grads) ** 0.5
        self.history.append(norm)

    def stats(self) -> Dict[str, float]:
        """Compute mean and std of recorded gradient norms."""
        if not self.history:
            return {"mean": 0.0, "std": 0.0}
        mean = sum(self.history) / len(self.history)
        variance = sum((x - mean) ** 2 for x in self.history) / len(self.history)
        return {"mean": mean, "std": variance ** 0.5}


class BaselineBlock(nn.Module):
    """Transformer block with standard residual connections."""

    def __init__(self, dims: int, num_heads: int = 4):
        super().__init__()
        self.norm1 = nn.RMSNorm(dims)
        self.norm2 = nn.RMSNorm(dims)
        self.attn = nn.MultiHeadAttention(dims, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(dims, dims * 4),
            nn.GELU(),
            nn.Linear(dims * 4, dims),
        )

    def __call__(self, x: mx.array) -> mx.array:
        h = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        return h + self.mlp(self.norm2(h))


class MHCBlock(nn.Module):
    """Transformer block with mHC residual connections."""

    def __init__(self, dims: int, num_heads: int = 4, expansion: int = 2):
        super().__init__()
        self.norm1 = nn.RMSNorm(dims)
        self.norm2 = nn.RMSNorm(dims)
        self.attn = nn.MultiHeadAttention(dims, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(dims, dims * 4),
            nn.GELU(),
            nn.Linear(dims * 4, dims),
        )
        self.mhc_attn = ManifoldHyperConnection(dims, expansion)
        self.mhc_mlp = ManifoldHyperConnection(dims, expansion)

    def __call__(self, x: mx.array) -> mx.array:
        # Correct mHC: apply H_pre before layer F
        x_norm = self.norm1(x)
        x_pre = self.mhc_attn.pre_scale(x_norm)
        attn_out = self.attn(x_pre, x_pre, x_pre)
        h = self.mhc_attn.post_combine(x_norm, attn_out)

        h_norm = self.norm2(h)
        h_pre = self.mhc_mlp.pre_scale(h_norm)
        mlp_out = self.mlp(h_pre)
        return self.mhc_mlp.post_combine(h_norm, mlp_out)


class BaselineModel(nn.Module):
    """Multi-layer transformer with standard residuals."""

    def __init__(self, dims: int, num_layers: int, num_heads: int = 4):
        super().__init__()
        self.layers = [BaselineBlock(dims, num_heads) for _ in range(num_layers)]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


class MHCModel(nn.Module):
    """Multi-layer transformer with mHC residuals."""

    def __init__(self, dims: int, num_layers: int, num_heads: int = 4, expansion: int = 2):
        super().__init__()
        self.layers = [MHCBlock(dims, num_heads, expansion) for _ in range(num_layers)]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


def create_baseline_model(dims: int, num_layers: int, num_heads: int = 4) -> BaselineModel:
    """Create transformer with standard residual connections."""
    return BaselineModel(dims, num_layers, num_heads)


def create_mhc_model(dims: int, num_layers: int, num_heads: int = 4, expansion: int = 2) -> MHCModel:
    """Create transformer with mHC residual connections."""
    return MHCModel(dims, num_layers, num_heads, expansion)


def _count_recursive(obj: Any) -> int:
    """Recursively count parameters in nested structure."""
    if isinstance(obj, mx.array):
        return obj.size
    elif isinstance(obj, dict):
        return sum(_count_recursive(v) for v in obj.values())
    elif isinstance(obj, (list, tuple)):
        return sum(_count_recursive(v) for v in obj)
    return 0


def _compute_grad_norm(grads: Any) -> float:
    """Compute total squared gradient norm from nested structure."""
    if isinstance(grads, mx.array):
        return float(mx.sum(grads ** 2))
    elif isinstance(grads, dict):
        return sum(_compute_grad_norm(v) for v in grads.values())
    elif isinstance(grads, (list, tuple)):
        return sum(_compute_grad_norm(v) for v in grads)
    return 0.0


def count_params(model: nn.Module) -> int:
    """Count total parameters in model."""
    return _count_recursive(model.parameters())


def train_step(model: nn.Module, x: mx.array, y: mx.array) -> Tuple[mx.array, float]:
    """Perform a single training step, return loss and gradient norm."""
    def loss_fn(params):
        model.update(params)
        pred = model(x)
        return mx.mean((pred - y) ** 2)

    loss, grads = mx.value_and_grad(loss_fn)(model.trainable_parameters())
    mx.eval(loss, grads)
    grad_norm = _compute_grad_norm(grads) ** 0.5
    return loss, grad_norm


def train_model(
    model: nn.Module,
    num_steps: int,
    batch_size: int,
    seq_len: int,
    dims: int,
    lr: float = 1e-3,
) -> Tuple[List[float], GradientTracker]:
    """Train model for num_steps, tracking losses and gradients."""
    optimizer = optim.SGD(learning_rate=lr)
    losses = []
    tracker = GradientTracker()

    def step(model, x, y):
        def loss_fn(params):
            model.update(params)
            return mx.mean((model(x) - y) ** 2)
        return mx.value_and_grad(loss_fn)(model.trainable_parameters())

    for _ in range(num_steps):
        x = mx.random.normal((batch_size, seq_len, dims))
        y = mx.random.normal((batch_size, seq_len, dims))

        loss, grads = step(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), loss)

        losses.append(float(loss))
        tracker.record(grads)

    return losses, tracker


def compare_models(
    dims: int,
    num_layers: int,
    num_steps: int,
    batch_size: int,
    seq_len: int,
    num_heads: int = 4,
    expansion: int = 2,
    lr: float = 1e-3,
) -> Dict[str, Any]:
    """Compare training dynamics between baseline and mHC models."""
    mx.random.seed(42)

    # Create and train baseline
    baseline = create_baseline_model(dims, num_layers, num_heads)
    baseline_losses, baseline_tracker = train_model(
        baseline, num_steps, batch_size, seq_len, dims, lr
    )

    mx.random.seed(42)  # Reset for fair comparison

    # Create and train mHC
    mhc = create_mhc_model(dims, num_layers, num_heads, expansion)
    mhc_losses, mhc_tracker = train_model(
        mhc, num_steps, batch_size, seq_len, dims, lr
    )

    # Compute stats
    baseline_stats = baseline_tracker.stats()
    mhc_stats = mhc_tracker.stats()
    baseline_params = count_params(baseline)
    mhc_params = count_params(mhc)
    overhead_pct = ((mhc_params - baseline_params) / baseline_params) * 100

    return {
        "baseline": {
            "grad_mean": baseline_stats["mean"],
            "grad_std": baseline_stats["std"],
            "params": baseline_params,
            "final_loss": baseline_losses[-1] if baseline_losses else 0,
        },
        "mhc": {
            "grad_mean": mhc_stats["mean"],
            "grad_std": mhc_stats["std"],
            "params": mhc_params,
            "final_loss": mhc_losses[-1] if mhc_losses else 0,
        },
        "param_overhead_pct": overhead_pct,
    }
