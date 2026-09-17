"""DMRGT Phase 3 — frozen graph tests (§6, Rule 16).

Four mandated checks on cache/graphs_frozen.parquet:
  1. within a fold, every test date has IDENTICAL edges (frozen = one
     snapshot stamped across the test window);
  2. different folds have DIFFERENT edges (each fold's snapshot is
     taken at its own train end, so folds must not share a graph);
  3. NaN-corruption — a frozen snapshot can never be built from a
     window containing NaN returns (the builder must raise);
  4. per-date counts: 320 +corr, 160 −corr, 160 sector, 640 total.

Plus the leakage guard: the snapshot for each fold uses only data up to
that fold's train end — verified structurally by checking that the
frozen fold's edge set differs from the DYNAMIC graph on that fold's
test dates (if the dynamic graph were also frozen-looking, the arm
comparison would be vacuous).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data
import graph
import paths
import splits


@pytest.fixture(scope="module")
def cfg() -> dict:
    return data.load_config(str(paths.CONFIG))


@pytest.fixture(scope="module")
def folds(cfg) -> list:
    feats_dates = pd.read_parquet(paths.FEATURES, columns=["date"])
    avail = pd.DatetimeIndex(feats_dates["date"].unique())
    return splits.make_folds(cfg, available_dates=avail)


@pytest.fixture(scope="module")
def frozen(cfg, folds) -> pd.DataFrame:
    if not paths.GRAPHS_FROZEN.exists():
        pytest.skip("graphs_frozen.parquet not built yet — run graph.py "
                    "after the frozen builder exists")
    return pd.read_parquet(paths.GRAPHS_FROZEN)


def test_frozen_graph_exists(frozen):
    assert len(frozen) > 0


# ─── 1. identical edges across test dates within each fold ───────────────

def test_identical_edges_within_fold(frozen, folds):
    """Fold date ranges overlap across folds (expanding trains), so every
    check slices by the fold column FIRST — date alone is ambiguous."""
    for fold in folds:
        test_dates = pd.DatetimeIndex(fold.test_dates)
        sub = frozen[(frozen["fold"] == fold.name)
                     & frozen["date"].isin(test_dates)]
        assert sub["date"].nunique() == len(test_dates), (
            f"Fold {fold.name!r}: frozen graph covers only "
            f"{sub['date'].nunique()}/{len(test_dates)} test dates"
        )
        # Compare every date's edge list to the fold's first date.
        first_day = sub[sub["date"] == test_dates[0]].drop(columns="date")
        first_sorted = first_day.sort_values(["edge_type", "src", "dst"]) \
                                 .reset_index(drop=True)
        # Sample dates across the window (all 250+ is slow to compare one
        # by one; sample first, middle, last plus randoms with fixed seed).
        rng = np.random.default_rng(0)
        check_dates = list(test_dates[[0, len(test_dates) // 2, -1]]) \
            + list(rng.choice(test_dates, size=5, replace=False))
        for d in check_dates:
            other = sub[sub["date"] == d].drop(columns="date")
            other_sorted = other.sort_values(["edge_type", "src", "dst"]) \
                                 .reset_index(drop=True)
            assert first_sorted.equals(other_sorted), (
                f"Frozen graph differs within fold {fold.name!r} between "
                f"{test_dates[0].date()} and {pd.Timestamp(d).date()} — "
                f"a frozen graph must be one snapshot stamped to all "
                f"test dates (Rule 16)."
            )


# ─── 2. different folds have different edges ─────────────────────────────

def test_differs_between_folds(frozen, folds):
    snapshots = {}
    for fold in folds:
        test_dates = pd.DatetimeIndex(fold.test_dates)
        sub = frozen[(frozen["fold"] == fold.name)
                     & frozen["date"].isin(test_dates)]
        day = sub[sub["date"] == test_dates[0]].drop(columns="date")
        snapshots[fold.name] = (frozenset(zip(day["src"], day["dst"],
                                              day["edge_type"])),
                                pd.Timestamp(fold.train_dates[-1]))
    names = [f.name for f in folds]
    train_ends = [snapshots[n][1] for n in names]
    # Sanity: the folds' train ends really are distinct dates.
    assert len(set(train_ends)) == len(train_ends), \
        "folds share a train end — splits are broken"
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = snapshots[names[i]][0]
            b = snapshots[names[j]][0]
            assert a != b, (
                f"Frozen graphs for folds {names[i]!r} and {names[j]!r} "
                f"are identical — each fold's snapshot is taken at its "
                f"own train end ({train_ends[i].date()} vs "
                f"{train_ends[j].date()}) so they must differ."
            )


# ─── 3. NaN corruption raises ─────────────────────────────────────────────

def test_nan_window_rejected(cfg):
    """A window with NaN returns must fail loudly, never silently emit
    a partial/garbage graph (Rule 4)."""
    universe = list(cfg["universe"]["tickers"])
    n = len(universe)
    corr_window = int(cfg["graph"]["corr_window"])
    # Synthetic prices with a NaN hole inside the correlation window.
    dates = pd.bdate_range("2020-01-01", periods=corr_window + 10)
    rng = np.random.default_rng(7)
    wide = pd.DataFrame(
        100 + np.cumsum(rng.standard_normal((len(dates), n)), axis=0),
        index=dates, columns=universe,
    )
    prices = (wide.stack().rename("close").reset_index())
    prices["ticker"] = prices["level_1"]
    prices["date"] = prices["level_0"]
    prices = prices[["date", "ticker", "close"]]
    # Corrupt one interior day entirely — the window ending at the last
    # date then contains NaNs.
    bad_day = dates[len(dates) // 2]
    mask = prices["date"] == bad_day
    prices.loc[mask, "close"] = np.nan

    with pytest.raises(AssertionError, match="NaN"):
        graph.compute_daily_edges(prices, dates[-1], cfg)


def test_nan_window_rejected_synthetic_graph_build(cfg):
    """Same guard through the all-dates builder: a single NaN day in the
    middle of the price history must abort the build."""
    universe = list(cfg["universe"]["tickers"])
    n = len(universe)
    corr_window = int(cfg["graph"]["corr_window"])
    dates = pd.bdate_range("2020-01-01", periods=corr_window + 10)
    rng = np.random.default_rng(11)
    arr = 100 + np.cumsum(rng.standard_normal((len(dates), n)), axis=0)
    arr[5, 0] = np.nan  # hole well inside the first window
    wide = pd.DataFrame(arr, index=dates, columns=universe)
    prices = (wide.stack().rename("close").reset_index())
    prices["ticker"] = prices["level_1"]
    prices["date"] = prices["level_0"]
    prices = prices[["date", "ticker", "close"]]
    with pytest.raises(AssertionError, match="NaN"):
        graph.build_daily_graph_all_dates(prices, cfg)


# ─── 4. per-date counts: 320 / 160 / 160 / 640 ────────────────────────────

def test_per_date_counts(cfg, frozen, folds):
    n_stocks = len(cfg["universe"]["tickers"])
    k_pos = int(cfg["graph"]["k_pos"])
    k_neg = int(cfg["graph"]["k_neg"])
    for fold in folds:
        test_dates = pd.DatetimeIndex(fold.test_dates)
        sub = frozen[(frozen["fold"] == fold.name)
                     & frozen["date"].isin(test_dates)]
        counts = sub.groupby(["date", "edge_type"]).size().unstack(fill_value=0)
        assert (counts[0] == n_stocks * k_pos).all(), \
            f"fold {fold.name}: type-0 count != {n_stocks * k_pos} on some date"
        assert (counts[1] == n_stocks * k_neg).all(), \
            f"fold {fold.name}: type-1 count != {n_stocks * k_neg} on some date"
        assert (counts[2] == 160).all(), \
            f"fold {fold.name}: sector count != 160 on some date"
        per_date = sub.groupby("date").size()
        assert (per_date == 640).all(), \
            f"fold {fold.name}: total edges != 640 on some date"


# ─── leakage guard: frozen snapshot predates the test window ──────────────

def test_frozen_snapshot_uses_only_train_data(cfg, frozen, folds):
    """Structural leakage check: the frozen fold graph must EQUAL the
    dynamic graph computed at that fold's train end (same builder, same
    as-of date), which by construction uses no test data."""
    prices = pd.read_parquet(paths.PRICES)
    for fold in folds:
        train_end = pd.Timestamp(fold.train_dates[-1])
        expected = graph.compute_daily_edges(prices, train_end, cfg) \
            .drop(columns="date").sort_values(["edge_type", "src", "dst"]) \
            .reset_index(drop=True)
        test_dates = pd.DatetimeIndex(fold.test_dates)
        sub = frozen[(frozen["fold"] == fold.name)
                     & frozen["date"].isin(test_dates)]
        frozen_day = sub[sub["date"] == test_dates[0]] \
            .drop(columns=["date", "fold"]) \
            .sort_values(["edge_type", "src", "dst"]) \
            .reset_index(drop=True)
        assert frozen_day.equals(expected), (
            f"Fold {fold.name!r}: frozen test-date edges are not the "
            f"snapshot computed at train end {train_end.date()} — the "
            f"frozen builder must use the train-end graph exactly."
        )


def test_frozen_differs_from_dynamic_on_test_dates(frozen, folds):
    """The frozen arm must actually be a DIFFERENT signal from the
    dynamic arm, otherwise the Phase 3 comparison is vacuous. The
    dynamic graph on a mid-test date (built from data including the test
    window so far) must differ from the fold's frozen snapshot for at
    least some fold."""
    prices = pd.read_parquet(paths.PRICES)
    dynamic = pd.read_parquet(paths.GRAPHS)
    for fold in folds:
        test_dates = pd.DatetimeIndex(fold.test_dates)
        # A late test date: the dynamic graph there has seen ~a year of
        # test-period data the frozen snapshot never saw.
        probe = test_dates[len(test_dates) // 2]
        dyn_day = dynamic[dynamic["date"] == probe] \
            .drop(columns="date").reset_index(drop=True)
        sub = frozen[(frozen["fold"] == fold.name)
                     & frozen["date"].isin(test_dates)]
        fro_day = sub[sub["date"] == test_dates[0]] \
            .drop(columns=["date", "fold"]).reset_index(drop=True)
        assert not dyn_day.equals(fro_day), (
            f"Fold {fold.name!r}: dynamic graph on {probe.date()} equals "
            f"the frozen snapshot — the two arms would be identical."
        )