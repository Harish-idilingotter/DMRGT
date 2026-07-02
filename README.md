DMRGT — Dynamic Multi-Relational Graph Transformer

Cross-asset market forecasting that models the market as a living graph, not a bag of isolated time series.

DMRGT represents the market as a dynamic, heterogeneous graph — nodes are securities, edges are typed relationships (price correlation, sector, and later news/supply-chain) that evolve over time. A Transformer encodes each asset's own history while a Graph Neural Network propagates information across these changing edges, so the model learns temporal patterns and cross-asset structure jointly.


Core claim: a time-varying, multi-relational market graph fused with a temporal encoder forecasts cross-asset returns better than (a) a temporal model alone and (b) the same model on a static graph.




Why this exists

Univariate models (LSTM, single-stock Transformers) ignore that securities influence each other — a supplier's earnings move its customers, a sector shock moves peers. Existing graph models usually fix the edges, so they degrade when market structure shifts. DMRGT lets the graph adapt: correlation edges slide over a trailing window, and additional relation types can be switched on as the evidence justifies them.

Key ideas


Heterogeneous relations — multiple edge types modelled simultaneously (start with correlation + sector).
Dynamic updates — edges and weights change per trading day from recent data.
Joint temporal + structural learning — Transformer encoder → heterogeneous GNN → fused prediction head.
Gated scope — the pipeline is built as a thin end-to-end slice first; complexity (news edges, regime module, auxiliary losses) is added only when ablations show it earns its place.


Architecture

Prices / market globals / sector labels
            │  (log-returns + ~6 indicators)
            ▼
   Dynamic graph  G_t   ── correlation (60d, top-k) + sector ──▶ HeteroData snapshot
            │
            ▼
   Temporal encoder (GRU → Transformer)  →  h_i
            │                                 │
            └──────────▶  Heterogeneous GNN (corr + sector)  →  z_i
                                              │
                          concat [h_i ‖ z_i] → MLP head → next-day return
            │
            ▼
   Walk-forward backtest · baselines · ablations

See Revised-Architecture.md / Revised-Architecture.pdf for the full design, rationale, and rendered diagrams.

Repository structure

dmrgt/
├── data/            # cached Parquet: prices, features, graph snapshots
├── src/
│   ├── data.py      # download + feature engineering (leakage-safe)
│   ├── graph.py     # correlation + sector edges → HeteroData snapshots
│   ├── model.py     # encoder (GRU/Transformer) + HeteroGNN + head
│   ├── train.py     # walk-forward training loop, single loss
│   └── eval.py      # metrics + vectorbt backtest + ablation runner
├── baselines.py     # persistence, no-graph, static-graph
├── configs/         # yaml per experiment (seed, universe, k, L, ...)
└── README.md

Quickstart

bash# 1. Environment
git clone https://github.com/<you>/dmrgt.git
cd dmrgt
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # torch, torch-geometric, pandas, numpy, ta, vectorbt

# 2. Build the dataset (downloads prices, engineers features, caches to Parquet)
python -m src.data --universe configs/sp100_subset.yaml

# 3. Construct dynamic graph snapshots
python -m src.graph --window 60 --topk 8

# 4. Train (walk-forward)
python -m src.train --config configs/mvp.yaml

# 5. Evaluate: metrics + backtest + ablations
python -m src.eval --config configs/mvp.yaml

Data sources (all free)

StreamSourceUseDaily OHLCVyfinancenode time-series featuresMarket globalsindex return, VIX (yfinance)global contextSector labelsGICS metadatastatic sector edges(later) news / earningspublic RSS / EDGARdynamic news edges (Phase 3)

All features and correlation windows use data strictly up to day t to predict day t+1; splits are chronological (walk-forward), never random — no look-ahead leakage.

Evaluation


Prediction: MSE/MAE, directional accuracy, rank IC (Spearman).
Backtest: daily rank, long top-decile / short bottom-decile, 0.1% cost + 0.05% slippage; annualized return, Sharpe, max drawdown vs buy-and-hold (vectorbt).
Baselines: last-return persistence, no-graph model, and the same model on a static graph (the headline comparison).
Ablations: no-GNN, static vs dynamic, drop each relation, GRU vs Transformer encoder.
Significance: paired Wilcoxon test on daily strategy returns.


Roadmap (accelerated — delivery by 30 Nov 2026)


 P0 · Vertical slice — end-to-end on 20 stocks (GRU, correlation-only, MSE)
 P1 · MVP — ~50 stocks (Transformer, corr + sector) → beat persistence & no-graph
 P2 · Validate novelty — static-vs-dynamic + per-relation ablations → dynamic beats static with significance
 P3 · Enhancements (optional) — news / regime / auxiliary losses, one at a time
 P4 · Explain & write — attention maps, edge-importance, case studies, paper draft


Tech stack

PyTorch · PyTorch Geometric · pandas / numpy · ta / pandas_ta · vectorbt · MLflow or Weights & Biases. Seeds fixed; versions pinned in requirements.txt.

Citing

If you build on this work:

bibtex@misc{dmrgt2026,
  title  = {DMRGT: A Dynamic Multi-Relational Graph Transformer for Cross-Asset Market Forecasting},
  author = {Sivakumar, Harish},
  year   = {2026},
  note   = {https://github.com/<you>/dmrgt}
}

Disclaimer

Research project for academic purposes only. Not investment advice. Uses publicly available data; no guarantees of predictive accuracy or trading profitability.