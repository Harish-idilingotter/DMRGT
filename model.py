"""DMRGT Phase 3 — the model (§7): TransformerEncoder + hand-rolled R-GCN
+ concat head. ONE architecture, no model_id branching (Rule 19).

The two Phase 3 arms — dmrgt_all3 (dynamic graph) and dmrgt_frozen
(frozen graph) — are THE SAME model class. They differ only in the edge
data train.py feeds them, which is exactly what makes the arm
comparison clean: any performance difference is attributable to the
graph, never to a code path selected by model id.

Architecture (X: [n_stocks, lookback=30, n_features=8]):
  TransformerEncoder (2 layers, 4 heads, causal mask, sinusoidal PE,
    8 → 64) → H [n_stocks, 64]
  RelationMessagePassing (3 relation-specific [64, 64] weight matrices,
    torch.index_add_ aggregation, no PyG) → Z [n_stocks, 64]
  concat([H, Z]) → [n_stocks, 128] → Linear(128, 32) → ReLU →
    Linear(32, 1) → prediction per stock

The temporal encoder is byte-identical to Phase 1/2's. The head is
deliberately SIMPLER than Phase 2's 3-block fusion head — Phase 3 tunes
trading rules and compares graph arms; a smaller head removes a source
of variance that contributed nothing in Phase 2.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        divterm = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * divterm)
        pe[:, 1::2] = torch.cos(position * divterm)
        # [1, max_len, d_model] so it broadcasts against [B, T, D]
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        return x + self.pe[:, : x.size(1), :]


class TransformerEncoder(nn.Module):
    """Temporal encoder: 2 layers, 4 heads, causal mask over the
    lookback axis, 8 → 64. Byte-identical to Phase 1/2 — every trained
    Phase 3 model reuses this exact backbone."""

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

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        """X: [B, n_stocks, lookback, n_features] → [B, n_stocks, d_model].

        Each stock's lookback window is encoded independently along the
        time axis with a causal mask (position t sees positions ≤ t);
        the last hidden state is the stock's embedding.
        """
        if X.dim() != 4:
            raise ValueError(
                f"encode() expects 4D input [B, n_stocks, L, F], "
                f"got {tuple(X.shape)}"
            )
        B, n_stocks, lookback, n_features = X.shape
        assert n_features == self.n_features, (
            f"Model built for n_features={self.n_features} but got "
            f"n_features={n_features}"
        )
        h = X.reshape(B * n_stocks, lookback, n_features)
        h = self.input_proj(h)
        h = self.pos_enc(h)
        h = self.input_dropout(h)

        causal_mask = torch.triu(
            torch.ones(lookback, lookback, dtype=torch.bool, device=X.device),
            diagonal=1,
        )
        h = self.encoder(h, mask=causal_mask)  # [B*n_stocks, lookback, d_model]
        h_last = h[:, -1, :]                   # [B*n_stocks, d_model]
        return h_last.reshape(B, n_stocks, self.d_model)


class RelationMessagePassing(nn.Module):
    """Hand-rolled R-GCN layer: 3 relation-specific [in_dim, out_dim]
    weight matrices, aggregation via index_add_. No PyG dependency —
    with 40 nodes and 640 edges this is fast on CPU and GPU and carries
    zero dependency risk when Kaggle bumps its torch version.
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

        # One weight matrix per relation type (asserted at construction)
        self.W = nn.ParameterList([
            nn.Parameter(torch.empty(in_dim, out_dim))
            for _ in range(num_relations)
        ])
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.reset_parameters()
        assert len(self.W) == 3, (
            f"R-GCN must have exactly 3 relation weight matrices, got "
            f"{len(self.W)}"
        )

    def reset_parameters(self) -> None:
        for w in self.W:
            nn.init.xavier_uniform_(w)

    def forward(
        self,
        x: torch.Tensor,            # [N, in_dim]
        edge_index: torch.Tensor,    # [2, E]
        edge_type: torch.Tensor,     # [E]
    ) -> torch.Tensor:
        """Z = Σ_r ( aggregate_{j→i, r} x_j @ W_r ) + bias → [N, out_dim].

        Every node has the same degree per relation by graph
        construction (k_pos, k_neg out-edges per stock; 4 sector peers),
        so mean-vs-sum aggregation only differs by a constant factor
        per relation — absorbed by the weight matrices.
        """
        N = x.size(0)
        device = x.device
        out = torch.zeros(N, self.out_dim, device=device)

        for r in range(self.num_relations):
            mask = edge_type == r
            if not mask.any():
                continue
            src = edge_index[0, mask]
            dst = edge_index[1, mask]
            msgs = x[src] @ self.W[r]     # [E_r, out_dim]
            out = out.index_add(0, dst, msgs)

        return out + self.bias


def stack_per_sample_edges(
    edge_indices: Sequence[torch.Tensor],
    edge_types: Sequence[torch.Tensor],
    n_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate per-sample edge lists, shifting sample b's node ids by
    b*n_nodes.

    Flattening a batch of B dates gives [B*n_nodes, d_model] with sample
    b's nodes occupying rows [b*n_nodes, (b+1)*n_nodes). Shifting its node
    ids by that offset lets ONE index_add cover the whole batch instead of
    one R-GCN call per sample.

    The graph and the arithmetic are identical either way — every
    destination node accumulates the same source messages in the same
    order. What changes is the number of kernel launches: with 40 nodes
    and ~640 edges the per-sample call is launch-bound, not compute-bound,
    so B sequential calls cost roughly B times one batched call.
    """
    if len(edge_indices) != len(edge_types):
        raise ValueError(
            f"{len(edge_indices)} edge_index tensors but "
            f"{len(edge_types)} edge_type tensors"
        )
    if not edge_indices:
        raise ValueError("no per-sample edge lists given")
    ei = torch.cat([e + b * n_nodes for b, e in enumerate(edge_indices)],
                   dim=1)
    et = torch.cat(list(edge_types), dim=0)
    return ei, et


class DMRGT(nn.Module):
    """The ONE Phase 3 model (Rule 19 — no model_id branching).

    X [n_stocks, 30, 8] → TransformerEncoder.encode → H [n_stocks, 64]
    H, edge_index, edge_type → RelationMessagePassing → Z [n_stocks, 64]
    concat([H, Z]) [n_stocks, 128] → Linear(128,32) → ReLU →
      Linear(32,1) → [n_stocks]
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

        self.transformer = TransformerEncoder(
            n_features=n_features,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_ff=dim_ff,
            dropout=dropout,
            max_len=max_len,
        )
        self.rgcn = RelationMessagePassing(
            in_dim=d_model,
            out_dim=d_model,
            num_relations=num_relations,
        )
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def _as_batched(self, X: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """Normalise X to [B, n_stocks, L, F], reporting whether the
        caller passed the single-sample [n_stocks, L, F] form."""
        if X.dim() == 3:
            x, squeeze = X.unsqueeze(0), True
        elif X.dim() == 4:
            x, squeeze = X, False
        else:
            raise ValueError(
                f"Expected X [n_stocks, L, F] or [B, n_stocks, L, F], "
                f"got {tuple(X.shape)}"
            )
        assert X.shape[-1] == self.n_features, (
            f"DMRGT built for n_features={self.n_features} but got "
            f"n_features={X.shape[-1]}"
        )
        return x, squeeze

    def _fuse(
        self,
        x: torch.Tensor,             # [B, n_stocks, L, F]
        edge_index: torch.Tensor,    # [2, E] over flattened batch nodes
        edge_type: torch.Tensor,     # [E]
    ) -> torch.Tensor:
        """Shared body of both entry points: encode, one batched
        message-passing pass, concat head. Returns [B, n_stocks]."""
        B, n_stocks = x.size(0), x.size(1)
        H = self.transformer.encode(x)          # [B, n_stocks, d_model]
        Z = self.rgcn(
            H.reshape(B * n_stocks, self.d_model), edge_index, edge_type
        ).reshape(B, n_stocks, self.d_model)    # [B, n_stocks, d_model]
        fused = torch.cat([H, Z], dim=-1)       # [B, n_stocks, 2*d_model]
        return self.head(fused).squeeze(-1)     # [B, n_stocks]

    def forward(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """X: [n_stocks, L, F] or [B, n_stocks, L, F]; edge_index [2, E];
        edge_type [E]. Returns [n_stocks] or [B, n_stocks].

        The ONE edge list is applied to every sample in the batch — the
        right semantics when all samples share a structure, which is how
        the tests and the smoke test call it. When samples have their own
        per-date edges (everything train.py does), use
        forward_with_edge_lists: same arithmetic, one R-GCN call instead
        of B.
        """
        x, squeeze = self._as_batched(X)
        B, n_stocks = x.size(0), x.size(1)
        if B == 1:
            ei, et = edge_index, edge_type   # nothing to offset
        else:
            ei, et = stack_per_sample_edges(
                [edge_index] * B, [edge_type] * B, n_stocks
            )
        out = self._fuse(x, ei, et)
        return out.squeeze(0) if squeeze else out

    def forward_with_edge_lists(
        self,
        X: torch.Tensor,                          # [B, n_stocks, L, F]
        edge_indices: Sequence[torch.Tensor],     # B × [2, E_b]
        edge_types: Sequence[torch.Tensor],       # B × [E_b]
    ) -> torch.Tensor:
        """One edge list PER sample — a batch of dates, each with its own
        graph. Returns [B, n_stocks].

        Mathematically identical to calling forward() once per sample and
        concatenating, but the R-GCN runs as a single batched index_add.
        That matters here: B=32 dates means 32 sequential forwards per
        optimizer step, each launching a handful of tiny kernels with only
        ~640 edges of work behind them, so the step is launch-bound. The
        encoder is unaffected — it was already called once on the whole
        batch, so dropout consumes the same RNG stream either way and a
        seeded run stays reproducible.
        """
        x, squeeze = self._as_batched(X)
        B, n_stocks = x.size(0), x.size(1)
        if len(edge_indices) != B:
            raise ValueError(
                f"{len(edge_indices)} edge lists for a batch of {B} — "
                f"exactly one per sample is required"
            )
        ei, et = stack_per_sample_edges(edge_indices, edge_types, n_stocks)
        out = self._fuse(x, ei, et)
        return out.squeeze(0) if squeeze else out


if __name__ == "__main__":
    # Smoke test: shapes, parameter counts, 3 distinct R-GCN matrices.
    torch.manual_seed(0)
    model = DMRGT()
    X = torch.randn(40, 30, 8)
    edge_index = torch.randint(0, 40, (2, 640))
    edge_type = torch.cat([
        torch.zeros(320, dtype=torch.long),
        torch.ones(160, dtype=torch.long),
        torch.full((160,), 2, dtype=torch.long),
    ])
    y = model(X, edge_index, edge_type)
    print(f"input  {tuple(X.shape)}  edges {tuple(edge_index.shape)}")
    print(f"output {tuple(y.shape)}")
    assert y.shape == (40,), f"expected (40,), got {tuple(y.shape)}"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,}")
    ptrs = [w.data_ptr() for w in model.rgcn.W]
    assert len(set(ptrs)) == 3, "R-GCN weight matrices share memory"
    print("R-GCN: 3 distinct weight matrices ✓")

    # Batched path
    Xb = torch.randn(4, 40, 30, 8)
    yb = model(Xb, edge_index, edge_type)
    assert yb.shape == (4, 40), f"expected (4, 40), got {tuple(yb.shape)}"
    print(f"batched output {tuple(yb.shape)} ✓")