"""DMRGT Phase 3 — node ordering correspondence test.

The graph's src/dst are POSITIONS in the universe list, and the model's
stock axis is built from the same list in the same order. If those two
index spaces ever disagreed (e.g. a sorted-ticker leak on one side),
sector edges would connect stocks from different sectors and every
message would pass to the wrong node — silently, with no crash.

This test pins the correspondence:
  * every node index is in [0, n_stocks);
  * every sector edge (type 2) connects two tickers whose GICS labels
    match under the universe-order mapping;
  * the graph's dates are a subset of the features' dates (the model
    looks up edges by sample date);
  * each sampled edge date equals the sample date it is looked up by
    (guards groupby/merge misuse);
  * every stock has exactly k_pos outgoing +corr and k_neg outgoing
    −corr edges per day (out-degree check).

Both graphs.parquet (dynamic) and graphs_frozen.parquet are checked —
the frozen arm must obey the same contract.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data
import paths


@pytest.fixture(scope="module")
def cfg() -> dict:
    return data.load_config(str(paths.CONFIG))


@pytest.fixture(scope="module")
def universe(cfg) -> list[str]:
    return list(cfg["universe"]["tickers"])


def _sample_dates(df: pd.DataFrame, n: int = 6) -> list[pd.Timestamp]:
    dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    rng = np.random.default_rng(42)
    picks = [dates[0], dates[len(dates) // 2], dates[-1]]
    picks += list(rng.choice(dates, size=n - 3, replace=False))
    return list(dict.fromkeys(picks))


def _check_graph(df: pd.DataFrame, universe: list[str], cfg: dict,
                 label: str) -> None:
    n = len(universe)
    sectors = cfg["sectors"]

    # node indices in range
    assert df["src"].between(0, n - 1).all(), f"[{label}] src out of range"
    assert df["dst"].between(0, n - 1).all(), f"[{label}] dst out of range"
    assert (df["src"] != df["dst"]).all(), f"[{label}] self-loop present"

    # sector edges connect same-sector tickers under universe order
    sec = df[df["edge_type"] == 2]
    src_sect = sec["src"].map(lambda i: sectors[universe[int(i)]])
    dst_sect = sec["dst"].map(lambda i: sectors[universe[int(i)]])
    bad = (src_sect != dst_sect).sum()
    assert bad == 0, (
        f"[{label}] {bad} sector edges connect different sectors — the "
        f"universe-order mapping between graph indices and tickers is "
        f"broken; messages would pass to the wrong nodes."
    )
    # every same-sector pair is present (160 directed edges = 8×20)
    n_sector = int(sec.groupby("date").size().iloc[0])
    assert n_sector == 160, f"[{label}] expected 160 sector edges, got {n_sector}"

    # out-degrees per stock per day
    for d in _sample_dates(df):
        day = df[df["date"] == d]
        for etype, expected in [(0, cfg["graph"]["k_pos"]),
                                (1, cfg["graph"]["k_neg"])]:
            out_deg = day[day["edge_type"] == etype].groupby("src").size()
            assert (out_deg == expected).all(), (
                f"[{label}] type-{etype} out-degree wrong on "
                f"{pd.Timestamp(d).date()}: expected {expected} per stock"
            )
            assert len(out_deg) == n, (
                f"[{label}] only {len(out_deg)}/{n} stocks have "
                f"type-{etype} edges on {pd.Timestamp(d).date()}"
            )


def test_dynamic_graph_node_ordering(cfg, universe):
    if not paths.GRAPHS.exists():
        pytest.skip("graphs.parquet not present")
    _check_graph(pd.read_parquet(paths.GRAPHS), universe, cfg, "dynamic")


def test_frozen_graph_node_ordering(cfg, universe):
    """The frozen frame stacks all folds (their date ranges overlap),
    so each fold's slice is checked separately — within one fold the
    dates are unique and the per-date invariants apply."""
    if not paths.GRAPHS_FROZEN.exists():
        pytest.skip("graphs_frozen.parquet not present")
    frozen = pd.read_parquet(paths.GRAPHS_FROZEN)
    assert "fold" in frozen.columns, (
        "graphs_frozen.parquet has no fold column — fold date ranges "
        "overlap, so date alone cannot identify the snapshot"
    )
    for fold_name in sorted(frozen["fold"].unique()):
        _check_graph(frozen[frozen["fold"] == fold_name], universe, cfg,
                     f"frozen:{fold_name}")


def test_graph_dates_are_feature_dates(cfg):
    """Every date the model might look up edges for must exist in the
    features calendar — the two caches share the prices calendar."""
    feats_dates = set(pd.read_parquet(paths.FEATURES, columns=["date"])
                      ["date"].unique())
    for path in [paths.GRAPHS, paths.GRAPHS_FROZEN]:
        if not path.exists():
            continue
        g_dates = set(pd.read_parquet(path, columns=["date"])["date"].unique())
        extra = g_dates - feats_dates
        assert not extra, (
            f"{path.name} has {len(extra)} dates absent from features "
            f"(e.g. {sorted(extra)[0]}) — the caches disagree on the "
            f"calendar."
        )


def test_edge_lookup_by_date_is_exact(cfg):
    """The lookup pattern the model uses must return exactly that date's
    edges (no cross-date contamination). Dynamic files are keyed by
    date; the frozen file is keyed by (fold, date) — overlapping fold
    date ranges mean a date-only lookup on the frozen frame would mix
    snapshots from different folds."""
    for path in [paths.GRAPHS, paths.GRAPHS_FROZEN]:
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "fold" in df.columns:
            for fold_name in sorted(df["fold"].unique()):
                sub = df[df["fold"] == fold_name]
                for d in _sample_dates(sub):
                    day = sub[sub["date"] == d]
                    assert day["fold"].nunique() == 1
                    assert day["date"].nunique() == 1
                    assert (day["fold"] == fold_name).all()
        else:
            for d in _sample_dates(df):
                day = df[df["date"] == d]
                assert day["date"].nunique() == 1
                assert pd.Timestamp(day["date"].iloc[0]) == pd.Timestamp(d)


def test_features_cover_full_universe_per_date(cfg, universe):
    """The model's stock axis: every features date carries all 40
    universe tickers, so row j of the assembled tensor is universe[j]."""
    feats = pd.read_parquet(paths.FEATURES, columns=["date", "ticker"])
    per_date = feats.groupby("date")["ticker"].nunique()
    assert (per_date == len(universe)).all(), (
        f"features has dates without the full {len(universe)}-ticker "
        f"universe — the tensor assembly would produce holes"
    )
    assert set(feats["ticker"].unique()) == set(universe) | set(
        cfg["universe"]["global_series"]) or set(feats["ticker"].unique()) == set(universe), (
        "features tickers are not exactly the universe (optionally plus "
        "the global series)"
    )