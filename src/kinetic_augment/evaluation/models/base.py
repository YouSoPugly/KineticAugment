"""
Base Model Classes for KineticAugment Evaluation.

Provides abstract base class for SLR classifiers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None
    torch = None


def check_torch_available() -> None:
    """Raise ImportError if PyTorch is not available."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for SLR models. "
            "Install with: pip install torch"
        )


class BaseSLRClassifier(nn.Module if TORCH_AVAILABLE else ABC):
    """
    Abstract base class for Sign Language Recognition classifiers.

    All SLR models should inherit from this class and implement
    the forward() method.
    """

    name: str = "BaseClassifier"

    def __init__(self, num_classes: int):
        """
        Initialize the classifier.

        Args:
            num_classes: Number of output classes
        """
        check_torch_available()
        super().__init__()
        self._num_classes = num_classes

    @property
    def num_classes(self) -> int:
        """Return number of classes."""
        return self._num_classes

    @abstractmethod
    def forward(
        self,
        x: 'torch.Tensor',
        lengths: Optional['torch.Tensor'] = None,
    ) -> 'torch.Tensor':
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, landmarks, 3) or
               (batch, seq_len, features)
            lengths: Optional sequence lengths for padding (batch,)

        Returns:
            Logits tensor of shape (batch, num_classes)
        """
        pass

    def predict(
        self,
        x: 'torch.Tensor',
        lengths: Optional['torch.Tensor'] = None,
    ) -> 'torch.Tensor':
        """
        Get class predictions.

        Args:
            x: Input tensor
            lengths: Optional sequence lengths

        Returns:
            Predicted class indices (batch,)
        """
        logits = self.forward(x, lengths)
        return torch.argmax(logits, dim=-1)

    def predict_proba(
        self,
        x: 'torch.Tensor',
        lengths: Optional['torch.Tensor'] = None,
    ) -> 'torch.Tensor':
        """
        Get class probabilities.

        Args:
            x: Input tensor
            lengths: Optional sequence lengths

        Returns:
            Class probabilities (batch, num_classes)
        """
        logits = self.forward(x, lengths)
        return torch.softmax(logits, dim=-1)

    def get_num_params(self, trainable_only: bool = True) -> int:
        """
        Count model parameters.

        Args:
            trainable_only: Only count trainable parameters

        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def freeze(self) -> None:
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
