"""
PyTorch GRU (Gated Recurrent Unit) model for human activity detection.
"""

from __future__ import annotations
import torch
import torch.nn as nn


class GRUClassifier(nn.Module):
    """
    GRU Classifier for multi-channel time-series data.
    """
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        super().__init__()
        
        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        
        self.num_directions = 2 if bidirectional else 1
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * self.num_directions, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Expects x shape: (batch, seq_len, num_features)
        """
        # gru_out: (batch, seq_len, hidden_size * num_directions)
        gru_out, _ = self.gru(x)
        
        # Take the output of the last time step
        last_out = gru_out[:, -1, :]
        
        # Classification
        return self.classifier(last_out)
