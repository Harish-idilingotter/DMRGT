"""DMRGT Phase 2 — dynamic 3-relation graph construction (§5).

Builds daily edge lists from trailing 60-day return correlations.
Caches to cache/graphs.parquet in long format: date | src | dst | edge_type.

Edge types: 0 = +corr (top-8 per stock), 1 = -corr (bottom-4 per stock),
2 = sector (same GICS label, static across dates).

Usage: python graph.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import data as data_mod
import paths


def load_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path is not None else paths.CONFIG
    if not p.exists():
        raise FileNotFoundError(f"configs.yaml not found at {p}")
    return yaml.safe_load(p.read_text())


def build_daily_graphs(
    prices: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """Build one edge list per trading day. Returns long-format DataFrame
    with columns: date, src, dst, edge_type."""

    universe = list(cfg["universe"]["tickers"])
    n_stocks = len(universe)
    sectors_cfg = cfg.get("sectors", {})
    if not sectors_cfg:
        raise ValueError("configs.yaml missing 'sectors:' block — required for graph.py")

    corr_window = int(cfg["graph"]["corr_window"])
    k_pos = int(cfg["graph"]["k_pos"])
    k_neg = int(cfg["graph"]["k_neg"])
    symmetrise = bool(cfg["graph"]["symmetrise"])
    include_sector = bool(cfg["graph"]["include_sector"])
    min_history = int(cfg["graph"]["min_history_days"])

    # ── pivot prices to wide (date × ticker) ──────────────────────────
    px_wide = prices.pivot(index="date", columns="ticker",
                           values="close").sort_index()
    # Only the universe tickers (not ^GSPC, ^VIX)
    px_universe = px_wide[universe]

    # ── daily returns ──────────────────────────────────────────────────
    rets = px_universe.pct_change(fill_method=None)  # rets.loc[d] = return from d-1 to d
    # Drop the first row (NaN return)
    rets = rets.iloc[1:]

    # ── compute sector edges once (static across all dates) ────────────
    sector_edges: list[tuple[int, int]] = []
    if include_sector:
        for i, t_i in enumerate(universe):
            for j, t_j in enumerate(universe):
                if i >= j:
                    continue
                if sectors_cfg.get(t_i) == sectors_cfg.get(t_j):
                    sector_edges.append((i, j))
                    if not symmetrise:
                        # Directed: add both directions
                        sector_edges.append((j, i))
                    # If symmetrise, we add both below anyway
        if symmetrise:
            # Already added both directions above for directed case;
            # for symmetrise we also want both directions.
            pass  # sector_edges already has both (i,j) and (j,i)

    # Compute expected sector edge count from config
    from collections import Counter
    sector_counts = Counter(sectors_cfg[t] for t in universe)
    expected_sector_edges = 0
    for count in sector_counts.values():
        expected_sector_edges += count * (count - 1)  # n*(n-1) directed pairs
    # With 5 per sector × 8 sectors: 8 × 20 = 160
    assert len(sector_edges) == expected_sector_edges, (
        f"Sector edge count mismatch: built {len(sector_edges)}, "
        f"expected {expected_sector_edges} from config"
    )
    print(f"[graph] sector edges: {len(sector_edges)} "
          f"(expected {expected_sector_edges} from {len(sector_counts)} sectors)")

    # ── build per-date edge lists ──────────────────────────────────────
    rows: list[dict] = []
    dates = rets.index.tolist()
    n_dates = len(dates)
    n_skipped = 0

    for idx, d in enumerate(dates):
        # Need at least corr_window returns ending at d
        if idx < corr_window - 1:
            n_skipped += 1
            continue

        # Trailing corr_window returns: [idx - corr_window + 1 .. idx] inclusive
        start_idx = idx - corr_window + 1
        window_rets = rets.iloc[start_idx: idx + 1]

        # Assert exactly corr_window observations per stock
        assert len(window_rets) == corr_window, (
            f"Date {d.date()}: window has {len(window_rets)} returns, "
            f"expected {corr_window}"
        )
        # Assert no NaN returns in the window for any stock
        nan_count = window_rets.isna().sum().sum()
        assert nan_count == 0, (
            f"Date {d.date()}: {nan_count} NaN returns in correlation window"
        )

        # Pearson correlation matrix [n_stocks, n_stocks]
        corr = window_rets.corr().values
        # Zero-variance stocks produce NaN correlations; replace with 0
        # (no linear relationship) so argsort behaves deterministically.
        corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)
        # Diagonal = 1.0 (self-correlation); set to -inf so self is never selected
        np.fill_diagonal(corr, -np.inf)

        # ── positive edges (type 0): top-k_pos per stock ────────────────
        # argsort descending along each row
        pos_order = np.argsort(-corr, axis=1)  # most positive first
        for i in range(n_stocks):
            for rank in range(k_pos):
                j = pos_order[i, rank]
                rows.append({"date": d, "src": i, "dst": int(j), "edge_type": 0})

        # ── negative edges (type 1): bottom-k_neg per stock ────────────
        # Set diagonal to +inf so self is never selected as "most negative"
        np.fill_diagonal(corr, np.inf)
        neg_order = np.argsort(corr, axis=1)
        for i in range(n_stocks):
            for rank in range(k_neg):
                j = neg_order[i, rank]
                rows.append({"date": d, "src": i, "dst": int(j), "edge_type": 1})

        # ── sector edges (type 2) — same on every date ─────────────────
        if include_sector:
            for src_i, dst_i in sector_edges:
                rows.append({"date": d, "src": src_i, "dst": dst_i, "edge_type": 2})

    # ── build DataFrame ────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "edge_type", "src", "dst"]).reset_index(drop=True)

    # ── per-date edge count assertions ─────────────────────────────────
    n_dates_built = df["date"].nunique()
    print(f"[graph] built {n_dates_built} daily graphs "
          f"(skipped {n_skipped} dates with insufficient history)")

    # Check edge counts per date
    per_date = df.groupby("date").size()
    expected_per_date = n_stocks * k_pos + n_stocks * k_neg
    if include_sector:
        expected_per_date += len(sector_edges)
    bad_dates = per_date[per_date != expected_per_date].index.tolist()
    if bad_dates:
        for bd in bad_dates[:5]:
            print(f"  BAD DATE {bd.date()}: {per_date[bd]} edges, "
                  f"expected {expected_per_date}")
        raise AssertionError(
            f"{len(bad_dates)} dates have wrong edge count. "
            f"Expected {expected_per_date} per date."
        )
    print(f"[graph] edge count per date: {expected_per_date} "
          f"(verified on all {n_dates_built} dates)")

    # ── type-specific counts ───────────────────────────────────────────
    type_counts = df.groupby("date")["edge_type"].value_counts().unstack(fill_value=0)
    for etype, expected in [(0, n_stocks * k_pos), (1, n_stocks * k_neg)]:
        bad = type_counts[etype] != expected
        if bad.any():
            bad_dates_list = bad[bad].index.tolist()
            raise AssertionError(
                f"Type-{etype} edge count wrong on {len(bad_dates_list)} dates. "
                f"Expected {expected}."
            )
    print(f"[graph] type-0 (+corr): {n_stocks * k_pos} per date ✓")
    print(f"[graph] type-1 (-corr): {n_stocks * k_neg} per date ✓")
    if include_sector:
        print(f"[graph] type-2 (sector): {len(sector_edges)} per date ✓")

    # ── no self-loops ──────────────────────────────────────────────────
    n_self = (df["src"] == df["dst"]).sum()
    assert n_self == 0, f"Found {n_self} self-loops"
    print(f"[graph] self-loops: 0 ✓")

    # ── no isolated nodes ──────────────────────────────────────────────
    for d in df["date"].unique()[:5]:  # spot-check first 5 dates
        day_edges = df[df["date"] == d]
        connected = set(day_edges["src"].unique()) | set(day_edges["dst"].unique())
        assert len(connected) == n_stocks, (
            f"Date {pd.Timestamp(d).date()}: only {len(connected)}/{n_stocks} "
            f"nodes have edges"
        )
    print(f"[graph] no isolated nodes ✓ (spot-checked)")

    # ── directedness assertion ─────────────────────────────────────────
    if not symmetrise:
        # Verify asymmetry exists: at least one A→B edge where B→A is absent
        day0 = df["date"].iloc[0]
        day0_edges = df[df["date"] == day0]
        edge_set = set(zip(day0_edges["src"], day0_edges["dst"]))
        asymmetric = 0
        for a, b in edge_set:
            if (b, a) not in edge_set:
                asymmetric += 1
        assert asymmetric > 0, (
            "Directed graph has no asymmetric edges — graph may be "
            "accidentally symmetrised"
        )
        print(f"[graph] directed: {asymmetric} asymmetric edges on first date ✓")

    # ── edge persistence ──────────────────────────────────────────────
    # Fraction of a stock's top-8 partners that remain the same day-over-day
    pos_edges = df[df["edge_type"] == 0].copy()
    persistence_vals = []
    dates_sorted = sorted(pos_edges["date"].unique())
    for i in range(1, min(len(dates_sorted), 60)):  # first 60 transitions
        prev_d = dates_sorted[i - 1]
        curr_d = dates_sorted[i]
        prev = pos_edges[pos_edges["date"] == prev_d]
        curr = pos_edges[pos_edges["date"] == curr_d]
        overlap = 0
        for stock in range(n_stocks):
            prev_partners = set(prev[prev["src"] == stock]["dst"])
            curr_partners = set(curr[curr["src"] == stock]["dst"])
            if len(prev_partners) > 0:
                overlap += len(prev_partners & curr_partners) / len(prev_partners)
        persistence_vals.append(overlap / n_stocks)

    mean_persistence = np.mean(persistence_vals) if persistence_vals else float("nan")
    print(f"[graph] edge persistence (mean over {len(persistence_vals)} transitions): "
          f"{mean_persistence:.4f}")
    if mean_persistence > 0.95:
        print("[graph] WARNING: edge persistence > 95% — graph is nearly static. "
              "Phase 3's dynamic-vs-frozen comparison may find little difference.")
    elif mean_persistence < 0.3:
        print("[graph] WARNING: edge persistence < 30% — graph is highly unstable. "
              "Turnover may be elevated.")

    return df


def main():
    parser = argparse.ArgumentParser(description="DMRGT Phase 2 graph builder")
    parser.add_argument("--config", default=None,
                        help="Path to configs.yaml (default: paths.CONFIG)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if not paths.PRICES.exists():
        print("[graph] prices.parquet missing — run 'python data.py --download' first",
              file=sys.stderr)
        sys.exit(1)

    prices = pd.read_parquet(paths.PRICES)
    print(f"[graph] loaded prices: {len(prices):,} rows, "
          f"{prices['ticker'].nunique()} tickers")

    df = build_daily_graphs(prices, cfg)

    # ── cache ──────────────────────────────────────────────────────────
    df.to_parquet(paths.GRAPHS, index=False)
    print(f"[graph] wrote {paths.GRAPHS}  rows={len(df):,}  "
          f"dates={df['date'].nunique()}")

    # Update hashes.json
    data_mod.update_hashes(paths.HASHES_JSON)
    data_mod._verify_hashes_complete(paths.HASHES_JSON)
    print(f"[graph] hashes.json updated")


if __name__ == "__main__":
    main()
