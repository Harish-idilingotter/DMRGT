"""Verify the target construction in data.build_targets.

Per §4 of the build spec, the following checks are MANDATORY before
proceeding to Step 3. Each is implemented as an individual test so a
failure pinpoints exactly which property is broken.

Tests are skipped if cache/targets.parquet does not exist; they assume
the cache has been built by `python data.py --targets`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the project root importable.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import data  # noqa: E402
import paths  # noqa: E402


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    if not paths.PRICES.exists():
        pytest.skip("cache/prices.parquet missing — run `python data.py --download`")
    return pd.read_parquet(paths.PRICES)


@pytest.fixture(scope="module")
def targets(prices: pd.DataFrame) -> pd.DataFrame:
    """Build targets fresh (not from disk) so a test author can iterate
    on build_targets without first re-running the CLI."""
    cfg = data.load_config()
    return data.build_targets(prices, cfg)


@pytest.fixture(scope="module")
def cfg() -> dict:
    return data.load_config()


# ── Test 1 — cross-sectional sum ≈ 0 on every date ─────────────────────

def test_cross_sectional_sum_is_zero(targets: pd.DataFrame) -> None:
    """For every date, the excess returns across all 20 stocks must sum
    to ≈ 0 (absolute value < 1e-10). This is the strongest arithmetic
    invariant of the target: it is what makes the labels 'excess'."""
    sums = targets.groupby("date")["excess_return"].sum().abs()
    assert (sums < 1e-10).all(), (
        f"Cross-sectional sum exceeds 1e-10 on at least one date.\n"
        f"  max sum: {sums.max():.3e}\n"
        f"  worst dates: {sums.nlargest(5).to_dict()}"
    )


# ── Test 2 — cross-sectional mean ≈ 0 on every date ────────────────────

def test_cross_sectional_mean_is_zero(targets: pd.DataFrame) -> None:
    """Equivalent to test 1 but stated on the mean. Both invariants are
    consequences of subtracting the equal-weight cross-sectional mean;
    we check both because they catch different arithmetic bugs (e.g.
    weighting that would zero the sum but not the mean)."""
    means = targets.groupby("date")["excess_return"].mean().abs()
    assert (means < 1e-6).mean() == 1.0, (  # extremely loose — mean of 20 values
        f"Cross-sectional mean is not ≈ 0 on every date.\n"
        f"  max |mean|: {means.max():.3e}"
    )


# ── Test 3 — manual hand-computation matches to 1e-9 ───────────────────

def test_manual_hand_computation_matches(targets: pd.DataFrame, prices: pd.DataFrame) -> None:
    """Pick 3 random (date, ticker) pairs, compute the excess return by
    hand from raw closes, and assert agreement to 1e-9. This is the
    'did we get the math right' check — it bypasses build_targets and
    re-derives the label from the underlying prices."""
    rng = np.random.default_rng(42)
    # Pick three dates that have enough history (skip first 60 days so
    # the underlying pct_change and shift arithmetic is well-defined).
    eligible_dates = sorted(targets["date"].unique())[60:200]
    chosen_dates = rng.choice(eligible_dates, size=3, replace=False)
    tickers = sorted(targets["ticker"].unique())
    chosen_tickers = rng.choice(tickers, size=3, replace=False)

    px_wide = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    # The label at date t is the excess return realised on t+1, so
    # for hand-computation we need the close at t, t+1, t+2.
    for d, t in zip(chosen_dates, chosen_tickers):
        d_ts = pd.Timestamp(d)
        d_next = px_wide.index[px_wide.index.get_loc(d_ts) + 1]
        d_next2 = px_wide.index[px_wide.index.get_loc(d_ts) + 2]
        # daily return at d_next = (close[d_next] - close[d_next-1]) / close[d_next-1]
        rets_at_next = (px_wide.loc[d_next] - px_wide.loc[d_next2].rename(
            {c: c for c in px_wide.columns}  # noop
        )) / px_wide.loc[d_next]
        # actually we need: return at d_next is from d_next-1 to d_next,
        # which is (close[d_next] / close[d_next-1]) - 1.
        rets_at_next = (px_wide.loc[d_next] / px_wide.shift(1).loc[d_next]) - 1
        # universe mean is over the 20 stock tickers only
        universe_tickers = [c for c in rets_at_next.index if not c.startswith("^")]
        mean_at_next = rets_at_next[universe_tickers].mean()
        hand = float(rets_at_next[t] - mean_at_next)
        # Compare against the target at date d
        actual = float(targets[(targets["date"] == d) & (targets["ticker"] == t)]["excess_return"].iloc[0])
        assert abs(hand - actual) < 1e-9, (
            f"Hand-computation mismatch for {t} on {d}:\n"
            f"  hand:  {hand:.12f}\n"
            f"  target: {actual:.12f}\n"
            f"  diff:   {abs(hand-actual):.3e}"
        )


# ── Test 4 — alignment: label at t = excess return realised on t+1 ─────

def test_alignment_label_describes_next_day(targets: pd.DataFrame, prices: pd.DataFrame) -> None:
    """The label at date t must equal the excess return realised on the
    *next* trading day. Verified by explicit index lookups, not by
    re-running the same shift logic (which would tautologically pass)."""
    cfg = data.load_config()
    universe_tickers = list(cfg["universe"]["tickers"])
    px_wide = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    rets = px_wide.pct_change()  # row t = return from t-1 to t
    excess = rets[universe_tickers].sub(rets[universe_tickers].mean(axis=1), axis=0)
    # excess.loc[date, ticker] = return[date] - mean[date]
    # The label at date t is excess.shift(-1) at date t = excess at t+1.

    # Check the first eligible date.
    t0 = sorted(targets["date"].unique())[60]  # skip warmup
    t1 = px_wide.index[px_wide.index.get_loc(pd.Timestamp(t0)) + 1]
    expected = float(excess.loc[t1, universe_tickers[0]])
    actual = float(targets[(targets["date"] == t0) & (targets["ticker"] == universe_tickers[0])]
                   ["excess_return"].iloc[0])
    assert abs(expected - actual) < 1e-12, (
        f"Alignment check failed for {universe_tickers[0]} on {t0}:\n"
        f"  expected (excess at t+1={t1.date()}): {expected:.12f}\n"
        f"  actual (label at t):                  {actual:.12f}"
    )


# ── Test 5 — no-lookahead: the last date's label is dropped ────────────

def test_no_lookahead_last_date_dropped(targets: pd.DataFrame, prices: pd.DataFrame) -> None:
    """The label at the final date cannot be computed (we don't know
    t+1's return yet). The targets table must therefore be missing the
    final date entirely. Also: there must be no labels at any date that
    would require data from beyond the last available date."""
    last_price_date = pd.Timestamp(prices["date"].max())
    last_target_date = pd.Timestamp(targets["date"].max())
    # The targets table's last date is one trading day before the
    # prices' last date (because the last prices date has no t+1).
    assert last_target_date < last_price_date, (
        f"Targets include the final price date ({last_target_date.date()}) — "
        f"this would be a lookahead. Last price date is {last_price_date.date()}."
    )
    # Also: the targets table must NOT contain a date later than the
    # last price date minus one business day.
    # Find the trading day before last_price_date:
    px_dates = sorted(prices["date"].unique())
    last_eligible_target_date = px_dates[-2]
    assert last_target_date == last_eligible_target_date, (
        f"Targets last date ({last_target_date.date()}) should be "
        f"{last_eligible_target_date} (one trading day before last price date)."
    )


# ── Test 6 — exactly 20 tickers per date (not 22, which would mean
#            ^GSPC and ^VIX leaked into the universe mean) ─────────────

def test_exactly_20_tickers_per_date(targets: pd.DataFrame) -> None:
    counts = targets.groupby("date").size()
    assert (counts == 20).all(), (
        f"Some dates don't have exactly 20 tickers. "
        f"min={counts.min()}, max={counts.max()}, "
        f"unique={counts.unique()}"
    )
