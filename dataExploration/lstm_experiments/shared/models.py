"""LSTM model variants for SailGP experiments."""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMRegressor(nn.Module):
    """2-layer LSTM -> linear scalar output (last timestep)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class LSTMSeqClassifier(nn.Module):
    """2-layer LSTM -> per-timestep binary logits."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out).squeeze(-1)


class LSTMClassifier(nn.Module):
    """2-layer LSTM -> pooled multiclass logits."""

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        pooled = self.dropout(out.mean(dim=1))
        return self.head(pooled)


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM -> multiclass from last hidden state."""

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        pooled = self.dropout(out[:, -1, :])
        return self.head(pooled)


class LSTMFutureBinaryClassifier(nn.Module):
    """Encode input window; predict binary labels for next H timesteps."""

    def __init__(
        self,
        input_size: int,
        horizon: int = 10,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.horizon = horizon
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class AttentionLSTMRegressor(nn.Module):
    """LSTM with learned attention pooling -> scalar regression."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        weights = torch.softmax(self.attn(out).squeeze(-1), dim=1)
        context = torch.sum(out * weights.unsqueeze(-1), dim=1)
        context = self.dropout(context)
        return self.head(context).squeeze(-1)


class BubbleAttentionClassifier(nn.Module):
    """
    Ego BiLSTM + multi-head attention over neighbour tokens at the last timestep.

    Expects ego sequence (B, T, F_ego), neighbour tokens (B, K, F_nb), and mask (B, K).
    """

    def __init__(
        self,
        ego_input_size: int,
        neighbour_input_size: int,
        num_classes: int = 3,
        hidden_size: int = 128,
        num_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            ego_input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        d = hidden_size * 2
        self.neighbour_proj = nn.Linear(neighbour_input_size, d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d, num_classes),
        )
        self.last_attn_weights: torch.Tensor | None = None

    def forward(
        self,
        ego: torch.Tensor,
        neighbours: torch.Tensor,
        neighbour_mask: torch.Tensor,
    ) -> torch.Tensor:
        out, _ = self.lstm(ego)
        query = out[:, -1:, :]  # (B, 1, D)
        kv = self.neighbour_proj(neighbours)
        key_padding = ~neighbour_mask  # (B, K): True = ignore

        # When every neighbour slot is masked, PyTorch's softmax over all-(-inf)
        # returns NaN, corrupting gradients for ~16% of windows.
        # Fix: unmask slot 0 for those rows so attention has one valid position
        # (the zero-padded slot; its contribution is still learnable).
        all_masked = key_padding.all(dim=-1)  # (B,)
        if all_masked.any():
            key_padding = key_padding.clone()
            key_padding[all_masked, 0] = False

        attn_out, weights = self.attn(query, kv, kv, key_padding_mask=key_padding)
        self.last_attn_weights = weights.detach()
        combined = torch.cat([query.squeeze(1), attn_out.squeeze(1)], dim=-1)
        return self.head(self.dropout(combined))


class BubbleAttentionLSTMTokenClassifier(nn.Module):
    """
    Ego BiLSTM + per-neighbour 5s LSTM encoding + multi-head attention.

    Expects ego (B, T, F_ego), neighbour sequences (B, K, T_nb, F_nb), mask (B, K).
    """

    def __init__(
        self,
        ego_input_size: int,
        neighbour_seq_features: int,
        neighbour_seq_len: int = 5,
        neighbour_hidden: int = 32,
        num_classes: int = 3,
        hidden_size: int = 128,
        num_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.neighbour_seq_len = neighbour_seq_len
        self.lstm = nn.LSTM(
            ego_input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        d = hidden_size * 2
        self.neighbour_lstm = nn.LSTM(
            neighbour_seq_features,
            neighbour_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.neighbour_proj = nn.Linear(neighbour_hidden, d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d, num_classes),
        )
        self.last_attn_weights: torch.Tensor | None = None

    def forward(
        self,
        ego: torch.Tensor,
        neighbour_seq: torch.Tensor,
        neighbour_mask: torch.Tensor,
    ) -> torch.Tensor:
        out, _ = self.lstm(ego)
        query = out[:, -1:, :]

        b, k, t, f = neighbour_seq.shape
        nb_flat = neighbour_seq.reshape(b * k, t, f)
        _, (h_n, _) = self.neighbour_lstm(nb_flat)
        tokens = h_n[-1].reshape(b, k, -1)
        kv = self.neighbour_proj(tokens)

        key_padding = ~neighbour_mask
        all_masked = key_padding.all(dim=-1)
        if all_masked.any():
            key_padding = key_padding.clone()
            key_padding[all_masked, 0] = False

        attn_out, weights = self.attn(query, kv, kv, key_padding_mask=key_padding)
        self.last_attn_weights = weights.detach()
        combined = torch.cat([query.squeeze(1), attn_out.squeeze(1)], dim=-1)
        return self.head(self.dropout(combined))
