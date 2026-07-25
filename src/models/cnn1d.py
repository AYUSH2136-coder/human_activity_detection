"""
PyTorch 1D Convolutional Neural Network (CNN1D) model for human activity detection.
"""

from __future__ import annotations
import torch
import torch.nn as nn


class CNN1DClassifier(nn.Module):
    """
    1D CNN Classifier for multi-channel time-series data.
    """
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        num_filters: list[int] = [64, 128, 256],
        kernel_size: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        layers = []
        in_channels = num_features
        for out_channels in num_filters:
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(kernel_size=2))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_channels = out_channels
            
        self.features = nn.Sequential(*layers)
        
        # Adaptive pooling to handle variable sequence lengths gracefully
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(num_filters[-1], 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Expects x shape: (batch, seq_len, num_features)
        Transposes to: (batch, num_features, seq_len) for Conv1D
        """
        x = x.transpose(1, 2)
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.classifier(x)
