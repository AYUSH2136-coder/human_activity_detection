"""
PyTorch Multi-Layer Perceptron (MLP) model for human activity detection.
"""

from __future__ import annotations
import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """
    Multi-Layer Perceptron Classifier.
    """
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_layers: list[int] = [256, 128, 64],
        dropout: float = 0.3,
        activation: str = "relu",
    ):
        super().__init__()
        
        # Determine activation function
        if activation.lower() == "relu":
            act_fn = nn.ReLU
        elif activation.lower() == "tanh":
            act_fn = nn.Tanh
        elif activation.lower() == "leaky_relu":
            act_fn = nn.LeakyReLU
        else:
            act_fn = nn.ReLU
            
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
            
        # Final classification layer
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        If input is 3D (batch, seq_len, features), flatten it first.
        """
        if x.dim() == 3:
            x = x.flatten(start_dim=1)
        return self.network(x)
