"""
--------------------
Author: XYZ
Description: Evidence-Based (EB) Early Stopping Criterion.  Implementation of the early stopping criterion from:
Mahsereci et al., Early Stopping without a Validation Set, 2017: https://arxiv.org/pdf/1703.09580
The EB-criterion stops training when gradients become so small that they are likely just noise due to the finiteness of the dataset, rather than carrying
information about the true gradient. For more details, see equation (9) for SGD with mini batches.
Python version: 3.12.0
"""

import logging
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
from lightning.pytorch.callbacks import Callback

logger = logging.getLogger(__name__)


class EBCriterionCallback(Callback):
    """
    Evidence-Based Early Stopping Criterion.
    Args:
        batch_size: Mini-batch size m used in the criterion formula
        patience: Number of epochs to wait after criterion is met before stopping
        smoothing: Exponential moving average smoothing factor (0-1).
                   Higher = more smoothing. Default: 0.9
        min_epochs: Minimum number of epochs before allowing early stopping
        group_by_layer: If True, compute separate criterion for each layer/parameter group
        verbose: print criterion values during training
        stopping_threshold: Threshold for stopping. default: 0.
    """

    def __init__(
        self,
        batch_size: int,
        patience: int = 1,
        smoothing: float = 0.9,
        min_epochs: int = 10,
        group_by_layer: bool = False,
        verbose: bool = True,
        stopping_threshold: float = 0.0,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.patience = patience
        self.smoothing = smoothing
        self.min_epochs = min_epochs
        self.group_by_layer = group_by_layer
        self.verbose = verbose
        self.stopping_threshold = stopping_threshold

        # State
        self.wait_count = 0
        self.ema_criterion = None
        self.ema_criterions_by_layer: Dict[str, float] = {}

        # For tracking gradient statistics across batches within an epoch
        self.gradient_squared_sums: Dict[str, float] = {}
        self.variance_sums: Dict[str, float] = {}
        self.num_batches = 0

        # History for debugging
        self.history = []

    def on_train_epoch_start(self, trainer, pl_module):
        """Reset per-epoch accumulators."""
        self.gradient_squared_sums = {}
        self.variance_sums = {}
        self.num_batches = 0

    def on_train_start(self, trainer, pl_module):
        """Set up gradient hooks."""
        self.gradient_hooks = []

        def make_gradient_hook(name):
            def hook(module, grad_input, grad_output):
                # grad_input in a tuple of gradients wrt outputs and grad_output is a tuple of gradients wrt inputs
                pass
            return hook

        # Register hooks on all parameters
        for name, param in pl_module.named_parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(
                    self._make_grad_hook(name)
                )

    def _make_grad_hook(self, name: str):
        """Create a gradient hook for a specific parameter."""
        def hook(param: torch.Tensor):
            if param.grad is not None:
                grad = param.grad.data
                grad_flat = grad.flatten()

                # Compute gradient squared sum for this parameter
                grad_sq_sum = (grad_flat ** 2).sum().item()

                # Compute variance estimate: running estimate of the mean gradient
                if not hasattr(self, '_grad_means'):
                    self._grad_means = {}

                if name not in self._grad_means:
                    self._grad_means[name] = torch.zeros_like(grad_flat)

                # Update running mean (exponential moving average)
                grad_mean = self._grad_means[name]
                if grad_mean.device != grad_flat.device:
                    grad_mean = grad_mean.to(grad_flat.device)
                    self._grad_means[name] = grad_mean

                grad_mean.mul_(0.9).add_(grad_flat, alpha=0.1)

                # Variance
                variance = ((grad_flat - grad_mean) ** 2).sum().item()

                # Accumulate statistics
                if name not in self.gradient_squared_sums:
                    self.gradient_squared_sums[name] = 0.0
                    self.variance_sums[name] = 0.0

                self.gradient_squared_sums[name] += grad_sq_sum
                self.variance_sums[name] += variance

        return hook

    def on_train_epoch_end(self, trainer, pl_module):
        """
        Compute the EB-criterion at the end of each epoch and check stopping condition.
        This function implemetns accumulated gradient statistics from the epoch.
        """
        if self.num_batches == 0 and not self.gradient_squared_sums:
            # No gradients accumulated (first epoch or no training steps)
            return

        # Compute aggregate criterion across all parameters
        total_grad_sq = sum(self.gradient_squared_sums.values())
        total_variance = sum(self.variance_sums.values())

        if total_variance > 0:
            # EB-criterion formula (Equation 9 in paper)
            eb_criterion = (self.batch_size / total_grad_sq) * total_variance
        else:
            eb_criterion = float('inf')

        # Apply exponential moving average for stability
        if self.ema_criterion is None:
            self.ema_criterion = eb_criterion
        else:
            self.ema_criterion = (
                self.smoothing * self.ema_criterion +
                (1 - self.smoothing) * eb_criterion
            )

        current_epoch = trainer.current_epoch
        self.history.append({
            'epoch': current_epoch,
            'eb_criterion': eb_criterion,
            'ema_criterion': self.ema_criterion,
            'total_grad_sq': total_grad_sq,
            'total_variance': total_variance,
        })

        if self.verbose:
            logger.info(
                f"Epoch {current_epoch}: EB-criterion = {eb_criterion:.4f}, "
                f"EMA = {self.ema_criterion:.4f}"
            )

        # Check stopping condition
        if current_epoch >= self.min_epochs:
            if self.ema_criterion < self.stopping_threshold:
                self.wait_count += 1
                if self.verbose:
                    logger.info(
                        f"EB-criterion ({self.ema_criterion:.4f}) < threshold "
                        f"({self.stopping_threshold}). Wait count: {self.wait_count}/{self.patience}"
                    )

                if self.wait_count >= self.patience:
                    logger.info(
                        f"EB-criterion stopping condition met at epoch {current_epoch}. "
                        f"Stopping training..."
                    )
                    trainer.should_stop = True
            else:
                self.wait_count = 0


class EBCriterionCallbackV2(Callback):
    """
    Evidence-Based Early Stopping Criterion - V2 with per-layer computation.
    This version computes the EB-criterion separately for each parameter group – exactly as in the paper.
    The overall criterion is the mean of per-layer criteria.

    Args:
        batch_size: Mini-batch size m used in the criterion formula
        patience: Number of epochs to wait after criterion is met before stopping
        smoothing: Exponential moving average smoothing factor (0-1).
        min_epochs: Minimum number of epochs before allowing early stopping
        verbose: Whether to print criterion values during training
        stopping_threshold: Threshold for stopping. Default: 0 (as in paper)
    """

    def __init__(
        self,
        batch_size: int,
        patience: int = 1,
        smoothing: float = 0.9,
        min_epochs: int = 10,
        verbose: bool = True,
        stopping_threshold: float = 0.0,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.patience = patience
        self.smoothing = smoothing
        self.min_epochs = min_epochs
        self.verbose = verbose
        self.stopping_threshold = stopping_threshold

        # State
        self.wait_count = 0
        self.per_layer_ema: Dict[str, float] = {}
        self.overall_ema = None

        # Gradient statistics accumulators
        self.grad_stats: Dict[str, Dict[str, float]] = {}

        # History
        self.history = []

    def on_train_epoch_start(self, trainer, pl_module):
        """Reset per-epoch accumulators."""
        self.grad_stats = {}

    def on_train_start(self, trainer, pl_module):
        """Register gradient hooks on all parameters."""
        for name, param in pl_module.named_parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(
                    self._make_grad_hook(name)
                )

        if self.verbose:
            logger.info(f"[EB-criterion] Registered gradient hooks on {len(list(pl_module.named_parameters()))} parameters")

    def _make_grad_hook(self, name: str):
        # Create a gradient hook for a specific parameter.
        def hook(param: torch.Tensor):
            if param.grad is not None:
                grad = param.grad.data
                grad_flat = grad.flatten()

                # Gradient squared sum
                grad_sq_sum = (grad_flat ** 2).sum().item()

                # Compute running variance
                if not hasattr(self, '_running_means'):
                    self._running_means = {}

                if name not in self._running_means:
                    # Initialize with current gradient
                    self._running_means[name] = grad_flat.clone().detach()
                else:
                    # Update running mean with smaller learning rate for stability
                    mean = self._running_means[name]
                    if mean.device != grad_flat.device:
                        mean = grad_flat.new_zeros(grad_flat.shape)
                        self._running_means[name] = mean
                    mean.lerp_(grad_flat, 0.01)

                # Variance estimate
                variance = ((grad_flat - self._running_means[name]) ** 2).sum().item()

                # Store for this batch
                if name not in self.grad_stats:
                    self.grad_stats[name] = {'grad_sq': 0.0, 'variance': 0.0, 'count': 0}

                self.grad_stats[name]['grad_sq'] += grad_sq_sum
                self.grad_stats[name]['variance'] += variance
                self.grad_stats[name]['count'] += 1

        return hook

    def on_train_epoch_end(self, trainer, pl_module):
        """Compute EB-criterion and check stopping condition."""
        if not self.grad_stats:
            return

        # Compute per-layer EB-criterion
        per_layer_criteria = {}

        for name, stats in self.grad_stats.items():
            grad_sq = stats['grad_sq']
            variance = stats['variance']

            if variance > 0:
                # EB-criterion for this layer: 1/ (m * (∇^2 / σ^2)) (taken from the paper)
                # Inversion is needed
                layer_criterion = (self.batch_size * variance) / grad_sq
            else:
                layer_criterion = float('inf')

            per_layer_criteria[name] = layer_criterion

            # Update per-layer EMA
            if name not in self.per_layer_ema:
                self.per_layer_ema[name] = layer_criterion
            else:
                self.per_layer_ema[name] = (
                    self.smoothing * self.per_layer_ema[name] +
                    (1 - self.smoothing) * layer_criterion
                )

        # Overall criterion is the mean of per-layer criteria
        overall_criterion = sum(per_layer_criteria.values()) / len(per_layer_criteria)

        # Update overall EMA
        if self.overall_ema is None:
            self.overall_ema = overall_criterion
        else:
            self.overall_ema = (
                self.smoothing * self.overall_ema +
                (1 - self.smoothing) * overall_criterion
            )

        current_epoch = trainer.current_epoch
        self.history.append({
            'epoch': current_epoch,
            'overall_criterion': overall_criterion,
            'overall_ema': self.overall_ema,
            'per_layer_criteria': per_layer_criteria.copy(),
        })

        if self.verbose:
            logger.info(
                f"Epoch {current_epoch}: EB-criterion = {overall_criterion:.4f}, "
                f"EMA = {self.overall_ema:.4f}"
            )

        # Check stopping condition
        if current_epoch >= self.min_epochs:
            if self.overall_ema < self.stopping_threshold:
                self.wait_count += 1
                if self.verbose:
                    logger.info(
                        f"EB-criterion ({self.overall_ema:.4f}) < threshold. "
                        f"Wait: {self.wait_count}/{self.patience}"
                    )

                if self.wait_count >= self.patience:
                    logger.info(f"EB-criterion stopping at epoch {current_epoch}")
                    trainer.should_stop = True
            else:
                self.wait_count = 0

        # Log to logger if available
        if trainer.logger is not None:
            trainer.logger.log_metrics({
                'train/eb_criterion': overall_criterion,
                'train/eb_criterion_ema': self.overall_ema,
            }, step=trainer.current_epoch)


class EBSimpleStopping(Callback):
    """
    Simplified EB-inspired early stopping.

    Instead of implementing the full gradient variance computation,
    this is a simpler version: it stop when the training loss improvement
    falls below a threshold for multiple consecutive epochs.
    Args:
        patience: Number of epochs with no improvement after which to stop
        min_delta: Minimum change to qualify as improvement
        min_epochs: Minimum epochs before stopping can occur
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-4,
        min_epochs: int = 10,
        monitor: str = 'train_loss',
        verbose: bool = True,
    ):
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.monitor = monitor
        self.verbose = verbose

        self.wait_count = 0
        self.best_loss = None
        self.history = []

    def on_train_epoch_end(self, trainer, pl_module):
        """Check if training has plateaued."""
        # Get the monitored metric from logged metrics
        logs = trainer.callback_metrics or {}

        if self.monitor not in logs:
            return

        current_loss = logs[self.monitor].item()

        # Initialize best loss
        if self.best_loss is None:
            self.best_loss = current_loss
            return

        # Check for improvement
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait_count = 0
        else:
            self.wait_count += 1

        self.history.append({
            'epoch': trainer.current_epoch,
            'loss': current_loss,
            'best_loss': self.best_loss,
            'wait_count': self.wait_count,
        })

        if self.verbose and self.wait_count > 0:
            logger.info(
                f"Train loss hasn't improved for {self.wait_count} epochs. "
                f"Current: {current_loss:.6f}, Best: {self.best_loss:.6f}"
            )

        # Check stopping condition
        if trainer.current_epoch >= self.min_epochs and self.wait_count >= self.patience:
            if self.verbose:
                logger.info(
                    f"Early stopping at epoch {trainer.current_epoch}: "
                    f"{self.monitor} hasn't improved for {self.patience} epochs"
                )
            trainer.should_stop = True
