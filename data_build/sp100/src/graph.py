"""DMRGT Phase 3 — graph construction (§6): dynamic, frozen, and §6b variants.

Edge types (identical semantics to Phase 2):
  0 = +corr  — top k_pos correlated peers per stock, trailing corr_window
               Pearson return correlation ending at the edge date t
               (inclusive; only information from t and earlier — Rule 3)
  1 = -corr  — bottom k_neg correlated peers per stock, same window
  2 = sector — same GICS label, static, directed both ways (i→j and j→i)

Two products:

  build_daily_graph_all_dates  — the DYNAMIC graph (Phase 2's
      cache/graphs.parquet, rebuilt). The per-day edge computation is
      byte-identical to Phase 2's builder — the §6b identity test
      asserts the baseline-k rebuild equals the cached file exactly —
      because the frozen arm and every sensitivity variant must differ
      from it ONLY through k_pos / k_neg / corr_window, never through
      an accidental algorithm change.

  build_frozen_graph_for_fold  — the FROZEN arm (Rule 16): for each
      walk-forward fold, compute the graph ONCE from the training
      window's last day and stamp that edge list onto every test date.
      The frozen graph can only use information available at train end;
      the leakage guard asserts the correlation window ends at or
      before the last training date.

CLI: python graph.py            → rebuild graphs.parquet (cache/) +
                                    graphs_frozen.parquet (OUTPUT_DIR —
                                    artifacts/ locally, /kaggle/working/
                                    in-session) at the config's k, then
                                    update hashes.json
     python graph.py --frozen-only → rebuild only the frozen file

The frozen file is an OUTPUT, never part of the attached cache dataset
(§4 of the single-session rework): notebook Cell 4 builds it in-session
by calling build_all_frozen_graphs(prices, cfg_frozen, folds) on the
attached prices.parquet and writing straight to /kaggle/working/ — the
local CLI rebuild exists only for local runs and provenance pins.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

import numpy as np
import pandas as pd

import paths


# ─── universe / return preparation (shared by every builder) ─────────────

def _pct_change_no_fill(df: pd.DataFrame) -> pd.DataFrame:
    """pct_change with NO filling, across pandas 2.x and 3.x.

    pandas 2.x: pct_change(fill_method=None) — explicit no-fill.
    pandas 3.x: the fill_method keyword was removed BECAUSE no-fill
    became the only behaviour, so plain pct_change() is identical.
    The cache was built under pandas 2.x no-fill semantics; the §6b
    baseline-identity test proves the rebuild reproduces it byte-for-
    byte under either version.
    """
    try:
        return df.pct_change(fill_method=None)
    except TypeError:  # pandas ≥ 3: fill_method removed
        return df.pct_change()


def _universe_returns(prices: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Wide close prices for the 40 universe tickers → daily returns.

    rets.loc[d] is the return from d-1 to d; the first row (NaN) is
    dropped. Same op order as Phase 2 so the skip logic below lands on
    the same dates.
    """
    universe = list(cfg["universe"]["tickers"])
    px_wide = (prices.pivot(index="date", columns="ticker", values="close")
               .sort_index())
    missing = [t for t in universe if t not in px_wide.columns]
    if missing:
        raise ValueError(f"prices.parquet is missing universe tickers: {missing}")
    px_universe = px_wide[universe]
    rets = _pct_change_no_fill(px_universe)
    return rets.iloc[1:]


def _sector_edges(cfg: dict) -> tuple[list[int], list[int]]:
    """Static intra-sector edges (both directions), as index pairs.

    With 5 tickers per sector × 8 sectors: 8 × 20 = 160 directed edges.
    """
    universe = list(cfg["universe"]["tickers"])
    sectors_cfg = cfg.get("sectors", {})
    if not sectors_cfg:
        raise ValueError("configs.yaml missing 'sectors:' block — required for graph.py")
    include_sector = bool(cfg["graph"].get("include_sector", True))
    srcs: list[int] = []
    dsts: list[int] = []
    if not include_sector:
        return srcs, dsts
    n = len(universe)
    for i in range(n):
        for j in range(n):
            if i >= j:
                continue
            if sectors_cfg.get(universe[i]) == sectors_cfg.get(universe[j]):
                srcs.extend([i, j])  # directed both ways
                dsts.extend([j, i])
    sector_counts = Counter(sectors_cfg[t] for t in universe)
    expected = sum(c * (c - 1) for c in sector_counts.values())
    assert len(srcs) == expected, (
        f"Sector edge count mismatch: built {len(srcs)}, expected "
        f"{expected} from configs.yaml (5 per sector × 8 sectors → 160)"
    )
    return srcs, dsts


# ─── one day's edges — the single source of truth for both arms ──────────

def _edges_for_returns_window(
    window_rets: pd.DataFrame,
    as_of: pd.Timestamp,
    cfg: dict,
    sector_src: list[int],
    sector_dst: list[int],
) -> pd.DataFrame:
    """Edges for one date from its trailing corr_window return slice.

    Exact Phase 2 arithmetic: Pearson corr → nan_to_num → diagonal
    −inf → argsort(−corr) for type 0 → diagonal +inf → argsort(corr)
    for type 1. Deterministic: the (date, edge_type, src, dst) key is
    unique, so the later sort is order-independent.
    """
    universe = list(cfg["universe"]["tickers"])
    n_stocks = len(universe)
    corr_window = int(cfg["graph"]["corr_window"])
    k_pos = int(cfg["graph"]["k_pos"])
    k_neg = int(cfg["graph"]["k_neg"])

    assert len(window_rets) == corr_window, (
        f"Date {pd.Timestamp(as_of).date()}: window has {len(window_rets)} "
        f"returns, expected {corr_window}"
    )
    nan_count = window_rets.isna().sum().sum()
    assert nan_count == 0, (
        f"Date {pd.Timestamp(as_of).date()}: {nan_count} NaN returns in "
        f"correlation window — a NaN-corrupted variant must never reach "
        f"training (Rule 4)"
    )

    corr = window_rets.corr().values
    corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)

    rows: list[dict] = []
    # type 0: most positively correlated first; self excluded via −inf
    np.fill_diagonal(corr, -np.inf)
    pos_order = np.argsort(-corr, axis=1)
    for i in range(n_stocks):
        for rank in range(k_pos):
            j = int(pos_order[i, rank])
            rows.append({"date": pd.Timestamp(as_of), "src": i, "dst": j,
                         "edge_type": 0})
    # type 1: most negatively correlated first; self excluded via +inf
    np.fill_diagonal(corr, np.inf)
    neg_order = np.argsort(corr, axis=1)
    for i in range(n_stocks):
        for rank in range(k_neg):
            j = int(neg_order[i, rank])
            rows.append({"date": pd.Timestamp(as_of), "src": i, "dst": j,
                         "edge_type": 1})
    # type 2: static sector edges
    for s, d in zip(sector_src, sector_dst):
        rows.append({"date": pd.Timestamp(as_of), "src": int(s),
                     "dst": int(d), "edge_type": 2})
    return pd.DataFrame(rows)


def compute_daily_edges(
    prices: pd.DataFrame,
    as_of_date: pd.Timestamp,
    cfg: dict,
) -> pd.DataFrame:
    """Edge list for ONE date (spec §6 signature) — the frozen builder's
    unit. Uses only returns up to and including as_of_date; callers
    enforce the fold-level leakage guard.
    """
    rets = _universe_returns(prices, cfg)
    corr_window = int(cfg["graph"]["corr_window"])
    as_of = pd.Timestamp(as_of_date)
    if as_of not in rets.index:
        raise ValueError(
            f"{as_of.date()} is not a return date — the graph needs a "
            f"trading day with at least {corr_window} prior returns."
        )
    idx = rets.index.get_loc(as_of)
    if idx < corr_window - 1:
        raise ValueError(
            f"Only {idx + 1} returns available up to {as_of.date()}; the "
            f"correlation window needs {corr_window} (min_history_days "
            f"{cfg['graph'].get('min_history_days', corr_window + 1)} "
            f"prices)."
        )
    window_rets = rets.iloc[idx - corr_window + 1: idx + 1]
    sector_src, sector_dst = _sector_edges(cfg)
    df = _edges_for_returns_window(window_rets, as_of, cfg,
                                   sector_src, sector_dst)
    return df.sort_values(["edge_type", "src", "dst"]).reset_index(drop=True)


# ─── dynamic graph (all dates) — Phase 2 reproduction ─────────────────────

def _edge_persistence_diagnostic(df: pd.DataFrame, n_stocks: int) -> None:
    """Day-over-day stability of a stock's +corr partners. High
    persistence means the dynamic-vs-frozen comparison will find a
    small difference — worth knowing before interpreting a null result.
    """
    pos = df[df["edge_type"] == 0]
    dates_sorted = sorted(pos["date"].unique())
    vals = []
    for i in range(1, min(len(dates_sorted), 60)):
        prev = pos[pos["date"] == dates_sorted[i - 1]]
        curr = pos[pos["date"] == dates_sorted[i]]
        overlap = 0
        for stock in range(n_stocks):
            p = set(prev[prev["src"] == stock]["dst"])
            c = set(curr[curr["src"] == stock]["dst"])
            if p:
                overlap += len(p & c) / len(p)
        vals.append(overlap / n_stocks)
    mean_p = float(np.mean(vals)) if vals else float("nan")
    print(f"[graph] edge persistence (mean over {len(vals)} transitions): "
          f"{mean_p:.4f}")
    if mean_p > 0.95:
        print("[graph] WARNING: persistence > 95% — graph nearly static; "
              "dynamic-vs-frozen may show little difference.")


def _validate_edge_frame(df: pd.DataFrame, cfg: dict,
                         label: str = "graph") -> None:
    """Invariants every produced graph must satisfy (Rule 4 spirit):
    per-date/type counts, no self-loops, no isolated nodes, directed."""
    universe = list(cfg["universe"]["tickers"])
    n_stocks = len(universe)
    k_pos = int(cfg["graph"]["k_pos"])
    k_neg = int(cfg["graph"]["k_neg"])
    sector_src, _ = _sector_edges(cfg)
    include_sector = bool(cfg["graph"].get("include_sector", True))

    expected_per_date = n_stocks * k_pos + n_stocks * k_neg
    if include_sector:
        expected_per_date += len(sector_src)
    per_date = df.groupby("date").size()
    bad = per_date[per_date != expected_per_date]
    assert bad.empty, (
        f"[{label}] {len(bad)} dates have wrong edge count "
        f"(e.g. {bad.index[0].date()}: {bad.iloc[0]}, expected "
        f"{expected_per_date})"
    )
    type_counts = df.groupby("date")["edge_type"].value_counts().unstack(fill_value=0)
    for etype, expected in [(0, n_stocks * k_pos), (1, n_stocks * k_neg)]:
        assert (type_counts[etype] == expected).all(), (
            f"[{label}] type-{etype} edge count wrong on some dates "
            f"(expected {expected} per date)"
        )
    if include_sector:
        assert (type_counts[2] == len(sector_src)).all()
    assert (df["src"] != df["dst"]).all(), f"[{label}] self-loops found"
    assert int(df["src"].max()) < n_stocks and int(df["dst"].max()) < n_stocks, \
        f"[{label}] node index out of universe range"
    # No isolated nodes (spot-check a few dates)
    for d in list(df["date"].unique())[:5]:
        day = df[df["date"] == d]
        connected = set(day["src"].unique()) | set(day["dst"].unique())
        assert len(connected) == n_stocks, (
            f"[{label}] {pd.Timestamp(d).date()}: only {len(connected)}/"
            f"{n_stocks} nodes connected"
        )
    print(f"[{label}] per-date counts {expected_per_date} "
          f"({n_stocks * k_pos}+corr, {n_stocks * k_neg}-corr, "
          f"{len(sector_src) if include_sector else 0} sector), "
          f"no self-loops, no isolated nodes ✓")


def build_daily_graph_all_dates(prices: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """The DYNAMIC graph: one edge list per date with sufficient history.

    Identical algorithm and date coverage to Phase 2's builder — the
    §6b identity test asserts a baseline-k rebuild of this equals
    cache/graphs.parquet exactly, so any observed Phase 3 difference
    between arms is attributable to the graph arm, not a rebuild drift.
    """
    universe = list(cfg["universe"]["tickers"])
    n_stocks = len(universe)
    corr_window = int(cfg["graph"]["corr_window"])
    sector_src, sector_dst = _sector_edges(cfg)
    print(f"[graph] sector edges: {len(sector_src)} "
          f"(expected 160 from 8 sectors × 5 tickers)")

    rets = _universe_returns(prices, cfg)
    rows: list[dict] = []
    n_skipped = 0
    for idx, d in enumerate(rets.index.tolist()):
        if idx < corr_window - 1:
            n_skipped += 1
            continue
        window_rets = rets.iloc[idx - corr_window + 1: idx + 1]
        day_df = _edges_for_returns_window(window_rets, d, cfg,
                                           sector_src, sector_dst)
        rows.extend(day_df.to_dict("records"))

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "edge_type", "src", "dst"]).reset_index(drop=True)

    print(f"[graph] built {df['date'].nunique()} daily graphs "
          f"(skipped {n_skipped} dates with insufficient history)")
    _validate_edge_frame(df, cfg, label="dynamic")
    if not bool(cfg["graph"].get("symmetrise", False)):
        day0 = df["date"].iloc[0]
        day0_edges = df[df["date"] == day0]
        edge_set = set(zip(day0_edges["src"], day0_edges["dst"]))
        asymmetric = sum(1 for a, b in edge_set if (b, a) not in edge_set)
        assert asymmetric > 0, "Directed graph has no asymmetric edges"
        print(f"[graph] directed: {asymmetric} asymmetric edges on first date ✓")
    _edge_persistence_diagnostic(df, n_stocks)
    return df


# ─── frozen graph (Rule 16) — one snapshot per fold, stamped to test ──────

def build_frozen_graph_for_fold(
    prices: pd.DataFrame,
    fold,
    cfg: dict,
) -> pd.DataFrame:
    """Frozen arm for one fold: the graph as of the LAST TRAINING DAY,
    held constant across EVERY date the fold touches — train, val, and
    test alike. train.py trains dmrgt_frozen on this same edge list for
    all its samples, so the model is trained, early-stopped, and scored
    against one frozen structure (the arm's definition).

    Leakage guard (Rule 16): the snapshot's correlation window ends at
    train end — strictly before the val and test windows — so the frozen
    graph cannot encode val/test-period relationships. This is exactly
    what distinguishes it from the dynamic arm, whose edges recompute
    daily. (Early TRAIN dates reuse the train-end snapshot; that is
    look-ahead within the training window only — the same category as
    the train-fitted scaler — and never touches val/test data.)

    The frame carries a `fold` column: the folds' date ranges overlap by
    construction (train windows expand from the same start, so every
    fold's train period is inside the next fold's), so a date alone does
    not identify which snapshot applies — train.py resolves edges by
    (fold, date).
    """
    train_end = pd.Timestamp(fold.train_dates[-1])
    snapshot = compute_daily_edges(prices, train_end, cfg)
    snapshot_dates = pd.to_datetime(snapshot["date"])
    assert snapshot_dates.max() <= train_end, (
        f"Frozen snapshot for fold {fold.name!r} used data after train "
        f"end {train_end.date()} — leakage."
    )
    fold_dates = (pd.DatetimeIndex(fold.train_dates)
                  .append(pd.DatetimeIndex(fold.val_dates))
                  .append(pd.DatetimeIndex(fold.test_dates)))
    print(f"[graph:frozen] fold {fold.name}: snapshot at {train_end.date()} "
          f"({len(snapshot)} edges) stamped to {len(fold_dates)} fold dates "
          f"(train+val+test)")

    rows = []
    for d in fold_dates:
        day = snapshot.copy()
        day["date"] = pd.Timestamp(d)
        rows.append(day)
    df = pd.concat(rows, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["fold"] = fold.name
    return df.sort_values(["date", "edge_type", "src", "dst"]).reset_index(drop=True)


def build_all_frozen_graphs(
    prices: pd.DataFrame,
    cfg: dict,
    folds: list | None = None,
) -> pd.DataFrame:
    """Frozen arm over every fold → cache/graphs_frozen.parquet.

    One snapshot per fold (taken at that fold's train end), stamped to
    every date the fold touches — train.py's dmrgt_frozen run reads this
    file as its edge source for ALL samples (§8), so the training,
    early-stopping, and test samples of a fold all see one structure.
    """
    if folds is None:
        import splits
        feats_dates = pd.read_parquet(paths.FEATURES, columns=["date"])
        avail = pd.DatetimeIndex(feats_dates["date"].unique())
        folds = splits.make_folds(cfg, available_dates=avail)
    # Validate each fold's frame SEPARATELY: fold date ranges overlap
    # (expanding train windows), so the concatenated frame legitimately
    # carries the same date under several folds — per-fold counts must
    # still be exact.
    frames = []
    for f in folds:
        frame = build_frozen_graph_for_fold(prices, f, cfg)
        _validate_edge_frame(frame, cfg, label=f"frozen:{f.name}")
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["fold", "date", "edge_type", "src", "dst"]) \
        .reset_index(drop=True)
    # Frozen identity: within a fold, EVERY date has the same edges.
    for fold in folds:
        sub = df[df["fold"] == fold.name]
        first = sub[sub["date"] == sub["date"].min()].drop(columns="date") \
            .sort_values(["edge_type", "src", "dst"]).reset_index(drop=True)
        for d in sub["date"].unique()[1:5]:
            other = sub[sub["date"] == d].drop(columns="date") \
                .sort_values(["edge_type", "src", "dst"]).reset_index(drop=True)
            assert first.equals(other), (
                f"Frozen graph varies within fold {fold.name!r} at "
                f"{pd.Timestamp(d).date()} — a frozen graph must be "
                f"identical across the fold's dates."
            )
    n_pairs = df.groupby(["fold", "date"]).ngroups
    print(f"[graph:frozen] {n_pairs} (fold, date) pairs across "
          f"{len(folds)} folds, identical edges within each fold ✓")
    return df


# ─── CLI: rebuild both graph files at the config's k, re-hash ─────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="DMRGT Phase 3 graph builder")
    parser.add_argument("--frozen-only", action="store_true",
                        help="Rebuild only graphs_frozen.parquet")
    args = parser.parse_args()

    import data
    import splits

    cfg = data.load_config(str(paths.CONFIG))
    if not paths.PRICES.exists():
        print("[graph] prices.parquet missing — the cache is the only data "
              "source in Phase 3 (Rule 9); nothing to rebuild from.",
              file=sys.stderr)
        sys.exit(1)
    prices = pd.read_parquet(paths.PRICES)
    print(f"[graph] loaded prices: {len(prices):,} rows, "
          f"{prices['ticker'].nunique()} tickers")

    feats_dates = pd.read_parquet(paths.FEATURES, columns=["date"])
    folds = splits.make_folds(cfg, available_dates=pd.DatetimeIndex(
        feats_dates["date"].unique()))

    if not args.frozen_only:
        dyn = build_daily_graph_all_dates(prices, cfg)
        dyn.to_parquet(paths.GRAPHS, index=False)
        print(f"[graph] wrote {paths.GRAPHS}  rows={len(dyn):,}  "
              f"dates={dyn['date'].nunique()}")

    fro = build_all_frozen_graphs(prices, cfg, folds=folds)
    fro.to_parquet(paths.GRAPHS_FROZEN, index=False)
    print(f"[graph] wrote {paths.GRAPHS_FROZEN}  rows={len(fro):,}  "
          f"dates={fro['date'].nunique()}")

    hashes = paths.update_hashes()
    missing = [n for n in paths.CACHE_PARQUETS if n not in hashes]
    assert not missing, f"hashes.json incomplete after rebuild: missing {missing}"
    print(f"[graph] hashes.json updated ({len(paths.CACHE_PARQUETS)} cache files)")


if __name__ == "__main__":
    main()