"""Temporal Convolutional Network backbone and task heads."""
from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.LayerNorm(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm2 = nn.LayerNorm(channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.dilation = dilation
        self.kernel_size = kernel_size

    def _trim(self, x: torch.Tensor) -> torch.Tensor:
        trim = (self.kernel_size - 1) * self.dilation
        if trim > 0:
            return x[:, :, :-trim]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        residual = x
        out = self.conv1(x)
        out = self._trim(out)
        out = out.transpose(1, 2)
        out = self.norm1(out)
        out = out.transpose(1, 2)
        out = self.act(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self._trim(out)
        out = out.transpose(1, 2)
        out = self.norm2(out)
        out = out.transpose(1, 2)
        out = self.act(out)
        out = self.dropout(out)

        return out + residual


class TCNBackbone(nn.Module):
    def __init__(
        self,
        n_features: int,
        channels: int = 64,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(n_features, channels, 1)
        self.blocks = nn.ModuleList(
            [ResidualBlock(channels, kernel_size, d, dropout) for d in dilations]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        # Global average pool over time -> (B, C)
        return x.mean(dim=-1)


class TCNModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        task: str = "classification",
        channels: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.task = task
        self.backbone = TCNBackbone(n_features, channels=channels, dropout=dropout)
        out_dim = 1
        self.head = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels // 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        out = self.head(h).squeeze(-1)
        return out

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.sigmoid(logits)


class SimpleLSTM(nn.Module):
    """Baseline LSTM for comparison."""

    def __init__(self, n_features: int, hidden: int = 64, task: str = "classification"):
        super().__init__()
        self.task = task
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True, num_layers=1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)
