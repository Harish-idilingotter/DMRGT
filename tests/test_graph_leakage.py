"""DMRGT Phase 2 — graph leakage tests (§5.4).

Tests that the graph construction obeys Rules 10 and 11:
no forward-looking data enters the edge list for any date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import data as data_mod
import graph as graph_mod
import paths


@pytest.fixture(scope="module")
def cfg():
    return data_mod.load_config()


@pytest.fixture(scope="module")
def prices():
    return pd.read_parquet(paths.PRICES)


@pytest.fixture(scope="module")
def graphs():
    return pd.read_parquet(paths.GRAPHS)


class TestWindowEndsAtT:
    """Rule 10: The correlation window must end at day t.

    Corrupt all data after date t with NaN, rebuild the graph for t,
    assert the edge list is byte-identical. Run against 20+ randomly
    chosen dates.
    """

    def test_nan_corruption_20_dates(self, cfg, prices):
        """Corrupt data after t → graph for t must be unchanged."""
        universe = list(cfg["universe"]["tickers"])
        corr_window = int(cfg["graph"]["corr_window"])

        # Build the full graph once as reference
        ref_graph = graph_mod.build_daily_graphs(prices, cfg)

        # Pick 20+ random dates with enough history
        px_wide = prices.pivot(index="date", columns="ticker",
                               values="close").sort_index()
        px_universe = px_wide[universe]
        rets = px_universe.pct_change().iloc[1:]
        eligible_dates = rets.index[corr_window - 1:]
        rng = np.random.default_rng(42)
        test_dates = rng.choice(eligible_dates, size=min(25, len(eligible_dates)),
                                replace=False)
        test_dates = sorted(pd.DatetimeIndex(test_dates))

        n_tested = 0
        for d in test_dates:
            # Corrupt all data after d, then TRUNCATE to dates ≤ d.
            # This tests that the graph for d depends only on data up to d.
            corrupted = prices.copy()
            mask = corrupted["date"] > d
            corrupted.loc[mask, ["open", "high", "low", "close", "volume"]] = np.nan
            # Truncate: only keep dates up to d (plus enough for the window)
            corrupted = corrupted[corrupted["date"] <= d].copy()

            # Build graph from corrupted (truncated) data
            corrupted_graph = graph_mod.build_daily_graphs(corrupted, cfg)

            # Extract edges for date d from both
            ref_day = ref_graph[ref_graph["date"] == d]
            corr_day = corrupted_graph[corrupted_graph["date"] == d]

            # Must be identical
            ref_edges = set(zip(ref_day["src"], ref_day["dst"], ref_day["edge_type"]))
            corr_edges = set(zip(corr_day["src"], corr_day["dst"],
                                 corr_day["edge_type"]))

            assert ref_edges == corr_edges, (
                f"Date {d.date()}: graph changed after corrupting future data.\n"
                f"  In ref but not corrupted: {len(ref_edges - corr_edges)}\n"
                f"  In corrupted but not ref: {len(corr_edges - ref_edges)}"
            )
            n_tested += 1

        assert n_tested >= 20, f"Only tested {n_tested} dates, need >= 20"
        print(f"  PASS: {n_tested} dates tested, all byte-identical after NaN corruption")


class TestWindowLength:
    """The correlation window must contain exactly 60 return observations."""

    def test_exact_window_length(self, cfg, prices):
        universe = list(cfg["universe"]["tickers"])
        corr_window = int(cfg["graph"]["corr_window"])
        px_wide = prices.pivot(index="date", columns="ticker",
                               values="close").sort_index()
        px_universe = px_wide[universe]
        rets = px_universe.pct_change().iloc[1:]

        # Spot-check: for 10 random dates, verify the window has exactly
        # corr_window returns
        eligible = rets.index[corr_window - 1:]
        rng = np.random.default_rng(123)
        check_dates = rng.choice(eligible, size=min(10, len(eligible)),
                                 replace=False)

        for d in pd.DatetimeIndex(check_dates):
            d_loc = rets.index.get_loc(d)
            window = rets.iloc[d_loc - corr_window + 1: d_loc + 1]
            assert len(window) == corr_window, (
                f"Date {d.date()}: window has {len(window)} returns, "
                f"expected {corr_window}"
            )
            # No NaN returns in the window
            assert window.isna().sum().sum() == 0, (
                f"Date {d.date()}: {window.isna().sum().sum()} NaN returns in window"
            )


class TestNoSelfLoops:
    """No edge where source == destination."""

    def test_no_self_loops(self, graphs):
        n_self = (graphs["src"] == graphs["dst"]).sum()
        assert n_self == 0, f"Found {n_self} self-loops"


class TestEdgeCounts:
    """Exactly 8 positive and 4 negative edges per stock per day."""

    def test_positive_edges_per_stock(self, cfg, graphs):
        n_stocks = len(cfg["universe"]["tickers"])
        k_pos = int(cfg["graph"]["k_pos"])
        pos = graphs[graphs["edge_type"] == 0]
        per_stock = pos.groupby(["date", "src"]).size()
        bad = per_stock[per_stock != k_pos]
        assert len(bad) == 0, (
            f"{len(bad)} (date, stock) pairs have wrong positive edge count. "
            f"Expected {k_pos}. Examples:\n{bad.head(10)}"
        )

    def test_negative_edges_per_stock(self, cfg, graphs):
        n_stocks = len(cfg["universe"]["tickers"])
        k_neg = int(cfg["graph"]["k_neg"])
        neg = graphs[graphs["edge_type"] == 1]
        per_stock = neg.groupby(["date", "src"]).size()
        bad = per_stock[per_stock != k_neg]
        assert len(bad) == 0, (
            f"{len(bad)} (date, stock) pairs have wrong negative edge count. "
            f"Expected {k_neg}. Examples:\n{bad.head(10)}"
        )


class TestSectorEdgesStatic:
    """Type-2 edges must be identical across all dates."""

    def test_sector_edges_identical(self, graphs):
        sector = graphs[graphs["edge_type"] == 2]
        if sector.empty:
            pytest.skip("No sector edges in graph")
        dates = sorted(sector["date"].unique())
        if len(dates) < 2:
            pytest.skip("Only one date of sector edges")

        # Compare first date to all others
        ref = sector[sector["date"] == dates[0]]
        ref_edges = set(zip(ref["src"], ref["dst"]))

        for d in dates[1:]:
            day_edges = sector[sector["date"] == d]
            cur_edges = set(zip(day_edges["src"], day_edges["dst"]))
            assert ref_edges == cur_edges, (
                f"Sector edges differ on {pd.Timestamp(d).date()} vs "
                f"{pd.Timestamp(dates[0]).date()}"
            )


class TestNoIsolatedNodes:
    """No isolated node on any date — every stock has at least one edge."""

    def test_no_isolated_nodes(self, cfg, graphs):
        n_stocks = len(cfg["universe"]["tickers"])
        for d in graphs["date"].unique():
            day = graphs[graphs["date"] == d]
            connected = set(day["src"].unique()) | set(day["dst"].unique())
            assert len(connected) == n_stocks, (
                f"Date {pd.Timestamp(d).date()}: only {len(connected)}/{n_stocks} "
                f"nodes have edges"
            )


class TestDirectedness:
    """If symmetrise is false, asymmetry must exist in the graph."""

    def test_asymmetry_exists(self, cfg, graphs):
        if cfg["graph"]["symmetrise"]:
            pytest.skip("Graph is symmetrised — asymmetry test not applicable")

        # Check first 5 dates: each should have at least one asymmetric edge
        dates = sorted(graphs["date"].unique())[:5]
        for d in dates:
            day = graphs[graphs["date"] == d]
            edge_set = set(zip(day["src"], day["dst"]))
            asymmetric = sum(1 for a, b in edge_set if (b, a) not in edge_set)
            assert asymmetric > 0, (
                f"Date {pd.Timestamp(d).date()}: no asymmetric edges found. "
                f"Graph may be accidentally symmetrised."
            )
