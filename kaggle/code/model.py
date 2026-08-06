"""DMRGT Phase 1 — Transformer encoder model (§7.1 of build spec).

The model is intentionally small. Phase 1 is about honest measurement,
not state-of-the-art accuracy; the graph is the expected value driver
in Phase 2.

Architecture (per spec §7.1):
    Input  X = [n_stocks, lookback, n_features]
      → Linear projection 8 → 64
      → + sinusoidal positional encoding
      → nn.TransformerEncoder: 2 layers, 4 heads, dropout 0.1,
        causal mask
      → Take final time step
      → MLP head 64 → 32 → 1
    Output [n_stocks] predicted excess returns

Key design choices the spec is explicit about:
  • One shared Transformer applied across all stocks. Same weights for
    every ticker. We do not build per-stock models.
  • Causal mask is mandatory, even though only the final time step is
    read out. The mask is the architectural leakage guard — without
    it, a position could attend to a future day.
  • Positional encoding is required. Attention is order-blind without
    it.
  • The forward() interface matches ``baselines.random_model`` /
    ``persistence_model``: ``forward(X) -> y_pred`` where X has shape
    ``[n_stocks, lookback, n_features]`` and y_pred has shape
    ``[n_stocks]``. This means the model drops into
    ``eval.evaluate_fold`` unchanged.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al. 2017).

    Adds a fixed (non-learned) sinusoid of varying frequency to the
    projected input. We register the pe buffer so it moves with the
    module on .to(device) but does not get updated by the optimiser.
    """

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # shape: [1, max_len, d_model] so it broadcasts against [B, T, D]
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        return x + self.pe[:, : x.size(1), :]


class DMRGTPhase1(nn.Module):
    """The Phase 1 Transformer. NO graph, NO R-GCN — temporal encoder only.

    Parameters
    ----------
    n_features : int
        Number of input features per (stock, time) cell. 8 for Phase 1.
    d_model : int
        Width of the Transformer. 64 per spec.
    n_heads : int
        Number of attention heads. 4 per spec.
    n_layers : int
        Number of TransformerEncoder layers. 2 per spec.
    dim_ff : int
        Feed-forward inner dimension. The spec doesn't fix this; we
        use the conventional 4× d_model.
    dropout : float
        Dropout in the Transformer. 0.1 per spec.
    """

    def __init__(
        self,
        n_features: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_ff: int = 256,
        dropout: float = 0.1,
        max_len: int = 64,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        # 8 → 64 projection
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model=d_model, max_len=max_len)
        self.input_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # MLP head: 64 → 32 → 1
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Predict excess returns for one or more trading days across all stocks.

        Parameters
        ----------
        X : torch.Tensor
            Either shape ``[n_stocks, lookback, n_features]`` (single day)
            or ``[batch, n_stocks, lookback, n_features]`` (batched days
            for training). The same tensor is fed for every stock —
            there is no per-stock embedding.

        Returns
        -------
        torch.Tensor
            Predicted excess returns, shape ``[n_stocks]`` (single day)
            or ``[batch, n_stocks]`` (batched).
        """
        # The single-day path is the eval-interface path (matches
        # ``baselines.random_model(X) -> y_pred``). The batched path
        # is for training efficiency.
        if X.dim() == 3:
            n_stocks, lookback, n_features = X.shape
            x = X.unsqueeze(0)  # [1, n_stocks, lookback, n_features]
            squeeze = True
        elif X.dim() == 4:
            n_stocks = X.size(1)
            lookback = X.size(2)
            n_features = X.size(3)
            x = X
            squeeze = False
        else:
            raise ValueError(
                f"Expected X of shape [n_stocks, lookback, n_features] "
                f"or [batch, n_stocks, lookback, n_features], got {tuple(X.shape)}"
            )
        assert n_features == self.n_features, (
            f"Model was built for n_features={self.n_features} but "
            f"got input with n_features={n_features}"
        )

        # Project + add positional encoding
        # x shape: [B, n_stocks, lookback, n_features]
        # Flatten the leading two so the Linear applies along the
        # feature axis: [B*n_stocks, lookback, n_features] -> d_model.
        B = x.size(0)
        h = x.reshape(B * n_stocks, lookback, n_features)
        h = self.input_proj(h)             # [B*n_stocks, lookback, d_model]
        h = self.pos_enc(h)
        h = self.input_dropout(h)

        # Causal mask: position t may only attend to positions <= t.
        # The mask is shape [lookback, lookback] and is True at
        # positions that should be MASKED (prevented from attending).
        causal_mask = torch.triu(
            torch.ones(lookback, lookback, dtype=torch.bool, device=x.device),
            diagonal=1,
        )

        # Shared Transformer across all stocks (and all batch elements)
        h = self.encoder(h, mask=causal_mask)   # [B*n_stocks, lookback, d_model]

        # Take the final time step. Even with a causal mask this is
        # the most informative single position — it has the most
        # context to draw on.
        h_last = h[:, -1, :]                    # [B*n_stocks, d_model]

        # MLP head
        out = self.head(h_last).squeeze(-1)     # [B*n_stocks]

        # Restore the [B, n_stocks] shape (or [n_stocks] if single day)
        out = out.reshape(B, n_stocks)
        if squeeze:
            out = out.squeeze(0)
        return out


if __name__ == "__main__":
    # Quick smoke test: random input, confirm output shape and that
    # the model is parameter-shared (no per-stock weights).
    import numpy as np
    torch.manual_seed(0)
    model = DMRGTPhase1()
    X = torch.randn(20, 30, 8)
    y = model(X)
    print(f"input  shape: {tuple(X.shape)}")
    print(f"output shape: {tuple(y.shape)}")
    assert y.shape == (20,), f"expected (20,), got {tuple(y.shape)}"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters:   {n_params:,}")
    print(f"output[:3]:   {y[:3].tolist()}")
