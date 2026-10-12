"""DMRGT Phase 4 — models (spec §6.3).

Ported verbatim from Phase 3 model.py: PositionalEncoding, TransformerEncoder
(arm A), RelationMessagePassing (hand-rolled R-GCN), stack_per_sample_edges,
and the concat prediction head. New: MambaEncoder (pure PyTorch), and the
split forward (encode / propagate) on DMRGT so the Phase 4B graph decision
can happen between the two steps.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


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


# ─── Mamba encoder (new in Phase 4; pure PyTorch, no compiled kernels) ─────

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def selective_scan(u, delta, A, B, C, D):
    # u, delta: (Bt, L, Di); A: (Di, S); B, C: (Bt, L, S); D: (Di,)
    Bt, L, Di = u.shape
    h = torch.zeros(Bt, Di, A.shape[1], device=u.device, dtype=u.dtype)
    ys = []
    for t in range(L):
        dA = torch.exp(delta[:, t].unsqueeze(-1) * A)
        dBu = delta[:, t].unsqueeze(-1) * B[:, t].unsqueeze(1) * u[:, t].unsqueeze(-1)
        h = dA * h + dBu
        ys.append((h * C[:, t].unsqueeze(1)).sum(-1))
    return torch.stack(ys, dim=1) + u * D


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_inner, self.d_state = expand * d_model, d_state
        self.dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                groups=self.d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.last_delta = None                    # read-only, for Phase 4B signal + explainability

    def forward(self, x):
        L = x.shape[1]
        u, z = self.in_proj(x).chunk(2, dim=-1)
        u = F.silu(self.conv1d(u.transpose(1, 2))[..., :L].transpose(1, 2))   # causal conv
        dt, B, C = torch.split(self.x_proj(u), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt))
        self.last_delta = delta.detach()
        y = selective_scan(u, delta, -torch.exp(self.A_log), B, C, self.D)
        return self.out_proj(y * F.silu(z))


class MambaEncoder(nn.Module):
    def __init__(self, n_features=8, d_model=64, n_layers=2, d_state=16, d_conv=4,
                 expand=2, dropout=0.1, d_out=None):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.layers = nn.ModuleList([MambaBlock(d_model, d_state, d_conv, expand) for _ in range(n_layers)])
        self.norms = nn.ModuleList([RMSNorm(d_model) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.final_norm = RMSNorm(d_model)
        self.out = nn.Linear(d_model, d_out) if (d_out and d_out != d_model) else nn.Identity()
        self.return_sequence = False              # True only inside the causality test

    def forward(self, x):
        h = self.input_proj(x)
        for norm, block in zip(self.norms, self.layers):
            h = h + self.dropout(block(norm(h)))
        h = self.final_norm(h)
        return h if self.return_sequence else self.out(h[:, -1])

    def delta_signal(self):
        """Mean memory-reset strength at the final step (last layer), one number per call."""
        return float(self.layers[-1].last_delta[:, -1].mean())

    def delta_per_sample(self, n_groups: int):
        """Per-group version of delta_signal for a batch of n_groups days:
        last layer, final step, averaged over stocks and channels → [n_groups]."""
        d = self.layers[-1].last_delta[:, -1]                # [n_groups * n_stocks, d_inner]
        return d.reshape(n_groups, -1).mean(-1)


class MambaStockEncoder(nn.Module):
    """Applies MambaEncoder independently to every stock's lookback window.
    X: [B, n_stocks, L, F] → [B, n_stocks, d_model] (same contract as
    TransformerEncoder.encode)."""

    def __init__(self, n_features=8, d_model=64, n_layers=2, d_state=16, d_conv=4,
                 expand=2, dropout=0.1):
        super().__init__()
        self.n_features, self.d_model = n_features, d_model
        self.mamba = MambaEncoder(n_features, d_model, n_layers, d_state, d_conv,
                                  expand, dropout, d_out=d_model)

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        if X.dim() != 4:
            raise ValueError(f"encode() expects 4D input [B, n_stocks, L, F], got {tuple(X.shape)}")
        B, n_stocks, lookback, n_features = X.shape
        assert n_features == self.n_features
        h = self.mamba(X.reshape(B * n_stocks, lookback, n_features))   # [B*n_stocks, d_model]
        return h.reshape(B, n_stocks, self.d_model)

    def delta_signal(self):
        return self.mamba.delta_signal()

    def delta_per_day(self, n_days: int):
        return self.mamba.delta_per_sample(n_days)


# ─── DMRGT with a split forward ────────────────────────────────────────────

class DMRGT(nn.Module):
    """Same R-GCN + concat head as Phase 3; the temporal encoder is chosen
    by `encoder` ('transformer' → arm A, 'mamba' → arms B/C/D).

    Construction order (encoder → rgcn → head) is the Phase 3 order, so the
    transformer arm draws the same initial weights as Phase 3 for a seed.

      encode(x)               x [B, N, L, F]  → h [B, N, d]
      propagate(h, ei, et)    R-GCN + concat head → [B, N]
      forward                 propagate(encode(x), ...)
    """

    def __init__(self, encoder: str = "transformer", n_features: int = 8,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 dim_ff: int = 256, dropout: float = 0.1, max_len: int = 64,
                 num_relations: int = 3, mamba_cfg: dict | None = None) -> None:
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.encoder_type = encoder
        if encoder == "transformer":
            self.encoder = TransformerEncoder(
                n_features=n_features, d_model=d_model, n_heads=n_heads,
                n_layers=n_layers, dim_ff=dim_ff, dropout=dropout, max_len=max_len)
        elif encoder == "mamba":
            mc = dict(mamba_cfg or {})
            self.encoder = MambaStockEncoder(
                n_features=n_features, d_model=mc.get("d_model", d_model),
                n_layers=mc.get("n_layers", 2), d_state=mc.get("d_state", 16),
                d_conv=mc.get("d_conv", 4), expand=mc.get("expand", 2),
                dropout=mc.get("dropout", dropout))
            assert mc.get("d_model", d_model) == d_model
        else:
            raise ValueError(f"unknown encoder {encoder!r}")
        self.rgcn = RelationMessagePassing(in_dim=d_model, out_dim=d_model,
                                           num_relations=num_relations)
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def _as_batched(self, X: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if X.dim() == 3:
            x, squeeze = X.unsqueeze(0), True
        elif X.dim() == 4:
            x, squeeze = X, False
        else:
            raise ValueError(f"Expected X [n_stocks, L, F] or [B, n_stocks, L, F], got {tuple(X.shape)}")
        assert X.shape[-1] == self.n_features
        return x, squeeze

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        """X [N, L, F] or [B, N, L, F] → h [N, d] or [B, N, d]."""
        x, squeeze = self._as_batched(X)
        h = self.encoder.encode(x)
        return h.squeeze(0) if squeeze else h

    def propagate(self, h: torch.Tensor, edge_index: torch.Tensor,
                  edge_type: torch.Tensor) -> torch.Tensor:
        """h [N, d] or [B, N, d]; edges over the flattened batch nodes
        (stack_per_sample_edges). R-GCN + concat head → [N] or [B, N]."""
        squeeze = h.dim() == 2
        if squeeze:
            h = h.unsqueeze(0)
        B, n_stocks = h.size(0), h.size(1)
        Z = self.rgcn(h.reshape(B * n_stocks, self.d_model), edge_index,
                      edge_type).reshape(B, n_stocks, self.d_model)
        fused = torch.cat([h, Z], dim=-1)
        out = self.head(fused).squeeze(-1)
        return out.squeeze(0) if squeeze else out

    def propagate_with_edge_lists(self, h: torch.Tensor,
                                  edge_indices: Sequence[torch.Tensor],
                                  edge_types: Sequence[torch.Tensor]) -> torch.Tensor:
        B, n_stocks = h.size(0), h.size(1)
        if len(edge_indices) != B:
            raise ValueError(f"{len(edge_indices)} edge lists for a batch of {B}")
        ei, et = stack_per_sample_edges(edge_indices, edge_types, n_stocks)
        return self.propagate(h, ei, et)

    def forward(self, X: torch.Tensor, edge_index: torch.Tensor,
                edge_type: torch.Tensor) -> torch.Tensor:
        """One edge list applied to every sample in the batch."""
        x, squeeze = self._as_batched(X)
        B, n_stocks = x.size(0), x.size(1)
        if B == 1:
            ei, et = edge_index, edge_type
        else:
            ei, et = stack_per_sample_edges([edge_index] * B, [edge_type] * B, n_stocks)
        out = self.propagate(self.encoder.encode(x), ei, et)
        return out.squeeze(0) if squeeze else out

    def forward_with_edge_lists(self, X: torch.Tensor,
                                edge_indices: Sequence[torch.Tensor],
                                edge_types: Sequence[torch.Tensor]) -> torch.Tensor:
        """One edge list PER sample (a batch of dates). [B, n_stocks]."""
        x, squeeze = self._as_batched(X)
        out = self.propagate_with_edge_lists(self.encoder.encode(x), edge_indices, edge_types)
        return out.squeeze(0) if squeeze else out


def make_model(encoder: str, mamba_cfg: dict | None = None, n_features: int = 8) -> DMRGT:
    """The one constructor used by training, the probes and Phase 4B inference."""
    return DMRGT(encoder=encoder, n_features=n_features, mamba_cfg=mamba_cfg)
