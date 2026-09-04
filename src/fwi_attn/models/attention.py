"""Additive (Bahdanau-style) attention pooling over a sequence of hidden states."""
from __future__ import annotations

import torch
from torch import nn


class BahdanauPoolingAttention(nn.Module):
    """Pools (B, T, H) hidden states to a single (B, H) context vector using a
    learned query: score_t = v^T tanh(W h_t), weights = softmax(score), context = sum_t weights_t * h_t.
    """

    def __init__(self, hidden_size: int, attn_size: int | None = None):
        super().__init__()
        attn_size = attn_size or hidden_size
        self.W = nn.Linear(hidden_size, attn_size, bias=True)
        self.v = nn.Linear(attn_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """hidden_states: (B, T, H) -> context: (B, H), weights: (B, T)"""
        scores = self.v(torch.tanh(self.W(hidden_states))).squeeze(-1)  # (B, T)
        weights = torch.softmax(scores, dim=1)
        context = torch.einsum("bt,bth->bh", weights, hidden_states)
        return context, weights
