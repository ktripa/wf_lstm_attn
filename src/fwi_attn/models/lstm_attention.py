"""LSTM + attention model for FWI attribution, with configurable active branches.

Branch A (concurrent weather) and Branch C (static) each go through a dense+ReLU
layer. Branch B (antecedent sequence) goes through a single-layer LSTM whose
hidden states are pooled either by additive attention (the main model) or by
taking the final hidden state (the "plain LSTM" baseline) -- same class, a
`pooling` flag, so the two share one implementation and can't drift apart.

`branches` controls which of {"A", "B", "C"} are active, so this one class
serves all 7 branch combinations needed for the ablation study: inactive
branches are simply omitted from the concatenation and the head's input size
is computed accordingly.
"""
from __future__ import annotations

import torch
from torch import nn

from fwi_attn.models.attention import BahdanauPoolingAttention


class DenseReLU(nn.Module):
    def __init__(self, in_size: int, out_size: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BranchBEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        attn_size: int | None = None,
        dropout: float = 0.0,
        pooling: str = "attention",
    ):
        super().__init__()
        if pooling not in ("attention", "final_state"):
            raise ValueError(f"Unknown pooling mode: {pooling!r}")
        self.pooling = pooling
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = BahdanauPoolingAttention(hidden_size, attn_size) if pooling == "attention" else None
        self.output_size = hidden_size

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """x: (B, T, input_size) -> context: (B, hidden_size), attn_weights: (B, T) or None"""
        hidden_states, (h_n, _) = self.lstm(x)
        if self.pooling == "attention":
            return self.attn(hidden_states)
        return h_n[-1], None  # final layer's final hidden state


class FWIAttnModel(nn.Module):
    def __init__(
        self,
        config,
        branch_a_in: int,
        branch_b_in: int,
        branch_c_in: int,
        branches: tuple[str, ...] = ("A", "B", "C"),
    ):
        super().__init__()
        self.branches = tuple(branches)
        if not self.branches:
            raise ValueError("At least one branch must be active")

        embed_size = 0
        self.branch_a = self.branch_b = self.branch_c = None

        if "A" in self.branches:
            self.branch_a = DenseReLU(branch_a_in, config.branch_a.hidden_size)
            embed_size += config.branch_a.hidden_size

        if "B" in self.branches:
            self.branch_b = BranchBEncoder(
                input_size=branch_b_in,
                hidden_size=config.branch_b.hidden_size,
                num_layers=config.branch_b.num_layers,
                attn_size=config.branch_b.get("attn_size", None),
                dropout=config.branch_b.dropout,
                pooling=config.branch_b.get("pooling", "attention"),
            )
            embed_size += config.branch_b.hidden_size

        if "C" in self.branches:
            self.branch_c = DenseReLU(branch_c_in, config.branch_c.hidden_size)
            embed_size += config.branch_c.hidden_size

        head_layers = []
        in_size = embed_size
        for h in config.head.hidden_sizes:
            head_layers += [nn.Linear(in_size, h), nn.ReLU(), nn.Dropout(config.head.dropout)]
            in_size = h
        head_layers.append(nn.Linear(in_size, 1))
        self.head = nn.Sequential(*head_layers)

    def forward(self, xa: torch.Tensor | None, xb: torch.Tensor | None, xc: torch.Tensor | None):
        embeds = []
        attn_weights = None
        if self.branch_a is not None:
            embeds.append(self.branch_a(xa))
        if self.branch_b is not None:
            ctx, attn_weights = self.branch_b(xb)
            embeds.append(ctx)
        if self.branch_c is not None:
            embeds.append(self.branch_c(xc))
        h = torch.cat(embeds, dim=-1)
        out = self.head(h).squeeze(-1)
        return out, attn_weights
