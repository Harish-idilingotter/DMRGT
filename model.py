"""DMRGT Phase 2 — Transformer encoder, LSTM baseline, R-GCN, and DMRGT.

TransformerEncoder: 2 layers, 4 heads, causal mask, 64-dim hidden.
  No graph, no R-GCN — temporal encoder only. Byte-identical to Phase 1.
LSTMBaseline: 2-layer LSTM, hidden 64, shared across all stocks.
  Strong sequence model with no graph — the fair-fight control.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al. 2017)."""

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


class TransformerEncoder(nn.Module):
    """Phase 1/2 Transformer: temporal encoder only, no graph, no R-GCN.
    Byte-identical to the Phase 1 DMRGTPhase1 class — renamed for clarity
    since Phase 2 adds the actual DMRGT model with graph components."""

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

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        """Run the Transformer encoder and return hidden states.
        X: [B, n_stocks, lookback, n_features] (always 4D from this method).
        Returns [B, n_stocks, d_model]."""
        if X.dim() != 4:
            raise ValueError(
                f"encode() expects 4D input [B, n_stocks, L, F], "
                f"got {tuple(X.shape)}"
            )
        B, n_stocks, lookback, n_features = X.shape
        assert n_features == self.n_features, (
            f"Model was built for n_features={self.n_features} but "
            f"got input with n_features={n_features}"
        )

        h = X.reshape(B * n_stocks, lookback, n_features)
        h = self.input_proj(h)
        h = self.pos_enc(h)
        h = self.input_dropout(h)

        causal_mask = torch.triu(
            torch.ones(lookback, lookback, dtype=torch.bool, device=X.device),
            diagonal=1,
        )
        h = self.encoder(h, mask=causal_mask)   # [B*n_stocks, lookback, d_model]
        h_last = h[:, -1, :]                    # [B*n_stocks, d_model]
        return h_last.reshape(B, n_stocks, self.d_model)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Predict excess returns. X: [n_stocks, L, F] or [B, n_stocks, L, F].
        Returns [n_stocks] or [B, n_stocks]."""
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

        B = x.size(0)
        h_last = self.encode(x)  # [B, n_stocks, d_model]

        # MLP head — applied per-stock
        out = self.head(h_last.reshape(B * n_stocks, self.d_model)).squeeze(-1)

        out = out.reshape(B, n_stocks)
        if squeeze:
            out = out.squeeze(0)
        return out


class LSTMBaseline(nn.Module):
    """2-layer LSTM baseline — strong sequence model with no graph.

    One shared LSTM across all stocks, same as the Transformer.
    Same interface: forward(X) -> [n_stocks] or [B, n_stocks].
    """

    def __init__(
        self,
        n_features: int = 8,
        hidden_size: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        # MLP head: hidden_size → 32 → 1
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Predict excess returns. X: [n_stocks, L, F] or [B, n_stocks, L, F].
        Returns [n_stocks] or [B, n_stocks]."""
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
            f"LSTM was built for n_features={self.n_features} but "
            f"got input with n_features={n_features}"
        )

        B = x.size(0)
        # Flatten batch and stocks: [B*n_stocks, lookback, n_features]
        h = x.reshape(B * n_stocks, lookback, n_features)
        # LSTM returns (output, (h_n, c_n)). output: [B*n_stocks, lookback, hidden]
        # h_n: [n_layers, B*n_stocks, hidden]
        _, (h_n, _) = self.lstm(h)
        # Take the final layer's hidden state
        h_last = h_n[-1]  # [B*n_stocks, hidden]

        # MLP head
        out = self.head(h_last).squeeze(-1)  # [B*n_stocks]

        # Restore the [B, n_stocks] shape (or [n_stocks] if single day)
        out = out.reshape(B, n_stocks)
        if squeeze:
            out = out.squeeze(0)
        return out


class RelationMessagePassing(nn.Module):
    """Hand-rolled R-GCN layer: 3 relation-specific weight matrices,
    message passing via index_add_. No PyG dependency.

    With 40 nodes and ~640 edges this is fast on CPU and GPU, and
    carries zero dependency risk when Kaggle bumps its torch version.
    """

    def __init__(
        self,
        in_dim: int = 64,
        out_dim: int = 64,
        num_relations: int = 3,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations

        # One weight matrix per relation type
        self.W = nn.ParameterList([
            nn.Parameter(torch.empty(in_dim, out_dim))
            for _ in range(num_relations)
        ])
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.reset_parameters()

        # Assert exactly 3 distinct weight matrices at construction
        assert len(self.W) == 3, (
            f"R-GCN must have exactly 3 relation weight matrices, got {len(self.W)}"
        )

    def reset_parameters(self):
        for w in self.W:
            nn.init.xavier_uniform_(w)

    def forward(
        self,
        x: torch.Tensor,           # [N, in_dim]
        edge_index: torch.Tensor,  # [2, E]
        edge_type: torch.Tensor,   # [E]
    ) -> torch.Tensor:
        """Compute Z = sum_r( aggregate_{j→i, type=r}( x_j @ W_r ) ) + bias.
        Returns [N, out_dim]."""
        N = x.size(0)
        device = x.device
        out = torch.zeros(N, self.out_dim, device=device)

        for r in range(self.num_relations):
            mask = (edge_type == r)
            if not mask.any():
                continue
            src = edge_index[0, mask]  # [E_r]
            dst = edge_index[1, mask]  # [E_r]
            # Messages: x[src] @ W_r  → [E_r, out_dim]
            msgs = x[src] @ self.W[r]
            # Aggregate into destination nodes
            out = out.index_add(0, dst, msgs)

        # Normalise by in-degree per relation? Not needed for top-k
        # graphs where every node has the same degree by construction.
        return out + self.bias


class RGCNDMRGT(nn.Module):
    """Full DMRGT model: TransformerEncoder + R-GCN + concat head.

    Architecture:
      X [n_stocks, 30, 8] → TransformerEncoder → H [n_stocks, 64]
      H, edge_index, edge_type → RelationMessagePassing → Z [n_stocks, 64]
      concat([H, Z]) = [n_stocks, 128] → MLP → [n_stocks]

    The TransformerEncoder is byte-identical to Phase 1's DMRGTPhase1.
    Ablations are implemented by filtering edge_index/edge_type BEFORE
    they reach this model — the model always sees exactly the relations
    it was configured for.
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
        num_relations: int = 3,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        # Byte-identical Transformer encoder from Phase 1
        self.transformer = TransformerEncoder(
            n_features=n_features,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_ff=dim_ff,
            dropout=dropout,
            max_len=max_len,
        )

        # Hand-rolled R-GCN layer
        self.rgcn = RelationMessagePassing(
            in_dim=d_model,
            out_dim=d_model,
            num_relations=num_relations,
        )

        # Fusion head: concat[H, Z] = 2*d_model → d_model → 32 → 1
        self.fusion_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """Predict excess returns.

        X: [n_stocks, L, F] or [B, n_stocks, L, F]
        edge_index: [2, E]
        edge_type: [E]
        Returns [n_stocks] or [B, n_stocks].
        """
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
            f"DMRGT was built for n_features={self.n_features} but "
            f"got input with n_features={n_features}"
        )

        B = x.size(0)

        # Transformer encoder → hidden states
        H = self.transformer.encode(x)  # [B, n_stocks, d_model]

        # R-GCN: apply per sample in the batch
        Z_list = []
        for b in range(B):
            H_b = H[b]  # [n_stocks, d_model]
            Z_b = self.rgcn(H_b, edge_index, edge_type)  # [n_stocks, d_model]
            Z_list.append(Z_b)
        Z = torch.stack(Z_list, dim=0)  # [B, n_stocks, d_model]

        # Concat and predict
        fused = torch.cat([H, Z], dim=-1)  # [B, n_stocks, 2*d_model]
        out = self.fusion_head(fused).squeeze(-1)  # [B, n_stocks]

        if squeeze:
            out = out.squeeze(0)
        return out


if __name__ == "__main__":
    # Quick smoke test: random input, confirm output shape and that
    # the model is parameter-shared (no per-stock weights).
    torch.manual_seed(0)

    # TransformerEncoder
    tx = TransformerEncoder()
    X = torch.randn(40, 30, 8)
    y = tx(X)
    print(f"TransformerEncoder:")
    print(f"  input  shape: {tuple(X.shape)}")
    print(f"  output shape: {tuple(y.shape)}")
    assert y.shape == (40,), f"expected (40,), got {tuple(y.shape)}"
    n_tx = sum(p.numel() for p in tx.parameters())
    print(f"  parameters:   {n_tx:,}")

    # LSTM
    lstm = LSTMBaseline()
    y_lstm = lstm(X)
    print(f"\nLSTMBaseline:")
    print(f"  input  shape: {tuple(X.shape)}")
    print(f"  output shape: {tuple(y_lstm.shape)}")
    assert y_lstm.shape == (40,), f"expected (40,), got {tuple(y_lstm.shape)}"
    n_lstm = sum(p.numel() for p in lstm.parameters())
    print(f"  parameters:   {n_lstm:,}")
    print(f"  ratio (LSTM/TX): {n_lstm/n_tx:.2f}")

    # RGCNDMRGT
    dmrgt = RGCNDMRGT()
    edge_index = torch.randint(0, 40, (2, 640))
    edge_type = torch.cat([
        torch.zeros(320, dtype=torch.long),
        torch.ones(160, dtype=torch.long),
        torch.full((160,), 2, dtype=torch.long),
    ])
    y_dmrgt = dmrgt(X, edge_index, edge_type)
    print(f"\nRGCNDMRGT:")
    print(f"  input  shape: {tuple(X.shape)}")
    print(f"  edges:        {edge_index.shape[1]}")
    print(f"  output shape: {tuple(y_dmrgt.shape)}")
    assert y_dmrgt.shape == (40,), f"expected (40,), got {tuple(y_dmrgt.shape)}"
    n_dmrgt = sum(p.numel() for p in dmrgt.parameters())
    print(f"  parameters:   {n_dmrgt:,}")
    print(f"  ratio (DMRGT/TX): {n_dmrgt/n_tx:.2f}")

    # Verify exactly 3 distinct weight matrices in the R-GCN
    w0 = dmrgt.rgcn.W[0].data_ptr()
    w1 = dmrgt.rgcn.W[1].data_ptr()
    w2 = dmrgt.rgcn.W[2].data_ptr()
    assert w0 != w1 and w1 != w2 and w0 != w2, \
        "R-GCN weight matrices are not distinct — they share memory"
    print(f"  R-GCN: 3 distinct weight matrices ✓")
