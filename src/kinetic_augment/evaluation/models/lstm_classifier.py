"""
LSTM Classifier for KineticAugment Evaluation.

Simple bidirectional LSTM classifier for SLR evaluation.
Fast to train, suitable for ablation studies.
"""

from __future__ import annotations

from typing import Optional

from kinetic_augment.evaluation.models.base import (
    BaseSLRClassifier,
    check_torch_available,
    TORCH_AVAILABLE,
)

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn


class LSTMClassifier(BaseSLRClassifier):
    """
    Bidirectional LSTM classifier for sign language recognition.

    Architecture:
    1. Flatten landmarks to feature vector
    2. Bidirectional LSTM layers
    3. Take final hidden state
    4. Linear classifier head

    This is a simple baseline model for evaluation, not meant for SOTA.
    Training time: ~5 minutes on WLASL100 (CPU).

    Example:
        >>> model = LSTMClassifier(
        ...     num_classes=100,
        ...     input_dim=543 * 3,
        ...     hidden_dim=256,
        ...     num_layers=2,
        ... )
        >>> logits = model(x)  # x: (batch, seq_len, 543, 3)
    """

    name = "LSTM"

    def __init__(
        self,
        num_classes: int,
        input_dim: int = 1629,  # 543 * 3
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        """
        Initialize LSTM classifier.

        Args:
            num_classes: Number of output classes
            input_dim: Input feature dimension (landmarks * 3)
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Use bidirectional LSTM
        """
        check_torch_available()
        super().__init__(num_classes)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # Output dimension
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_output_dim),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, num_classes),
        )

    def forward(
        self,
        x: 'torch.Tensor',
        lengths: Optional['torch.Tensor'] = None,
    ) -> 'torch.Tensor':
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, landmarks, 3) or
               (batch, seq_len, features)
            lengths: Optional sequence lengths for packing

        Returns:
            Logits (batch, num_classes)
        """
        # Flatten landmarks if needed
        if x.dim() == 4:
            batch, seq_len, num_lm, coords = x.shape
            x = x.view(batch, seq_len, num_lm * coords)

        # Pack sequences if lengths provided
        if lengths is not None:
            # Sort by length for packing
            lengths_cpu = lengths.cpu()
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            lstm_out, (h_n, c_n) = self.lstm(x_packed)
        else:
            lstm_out, (h_n, c_n) = self.lstm(x)

        # Get final hidden state
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            # h_n shape: (num_layers * 2, batch, hidden_dim)
            forward_final = h_n[-2]  # Last layer, forward
            backward_final = h_n[-1]  # Last layer, backward
            final = torch.cat([forward_final, backward_final], dim=-1)
        else:
            final = h_n[-1]

        # Classify
        logits = self.classifier(final)

        return logits


class GRUClassifier(BaseSLRClassifier):
    """
    GRU-based classifier as an alternative to LSTM.

    Slightly faster than LSTM with similar performance.
    """

    name = "GRU"

    def __init__(
        self,
        num_classes: int,
        input_dim: int = 1629,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        """Initialize GRU classifier."""
        check_torch_available()
        super().__init__(num_classes)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        gru_output_dim = hidden_dim * (2 if bidirectional else 1)

        self.classifier = nn.Sequential(
            nn.LayerNorm(gru_output_dim),
            nn.Dropout(dropout),
            nn.Linear(gru_output_dim, num_classes),
        )

    def forward(
        self,
        x: 'torch.Tensor',
        lengths: Optional['torch.Tensor'] = None,
    ) -> 'torch.Tensor':
        """Forward pass."""
        if x.dim() == 4:
            batch, seq_len, num_lm, coords = x.shape
            x = x.view(batch, seq_len, num_lm * coords)

        if lengths is not None:
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, h_n = self.gru(x_packed)
        else:
            _, h_n = self.gru(x)

        if self.bidirectional:
            final = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            final = h_n[-1]

        return self.classifier(final)
