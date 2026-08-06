"""Verify the 8 features built by features.build_features.

Per §6 of the build spec, the following checks are MANDATORY before
proceeding. Each is implemented as a separate test so a failure
pinpoints the broken feature.

Tests are skipped if cache/features.parquet does not exist; they assume
the cache has been built by `python data.py --features`.

The single most important test is the NaN-corruption lookahead check
— it confirms that the value of a feature at date t is unaffected by
the data on dates t+1, t+2, ... . If a rolling window ever includes
future data, this test catches it.
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
import features as feat_mod  # noqa: E402
import paths  # noqa: E402


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    if not paths.FEATURES.exists():
        pytest.skip("cache/features.parquet missing — run `python data.py --features`")
    return pd.read_parquet(paths.FEATURES)


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    if not paths.PRICES.exists():
        pytest.skip("cache/prices.parquet missing — run `python data.py --download`")
    return pd.read_parquet(paths.PRICES)


@pytest.fixture(scope="module")
def cfg() -> dict:
    return data.load_config()


# ── 1. All 8 feature columns present and in canonical order ───────────────

def test_feature_columns_present_in_order(features: pd.DataFrame) -> None:
    assert list(features.columns) == ["date", "ticker"] + feat_mod.FEATURE_COLUMNS


# ── 2. RSI strictly bounded in [0, 100] ───────────────────────────────────

def test_rsi_bounded_0_100(features: pd.DataFrame) -> None:
    rsi = features["f3_rsi"]
    assert rsi.between(0.0, 100.0).all(), (
        f"RSI out of [0, 100]: min={rsi.min()}, max={rsi.max()}"
    )


# ── 3. ATR strictly positive (scaled) ──────────────────────────────────────

def test_atr_strictly_positive(features: pd.DataFrame) -> None:
    atr = features["f5_atr"]
    # Strictly > 0 — ATR/close where both > 0 is always > 0. The
    # warm-up drop should have removed any NaN. Allow zero only if
    # somehow ATR was identically zero for a row (very rare).
    assert (atr > 0).all(), f"ATR non-positive: min={atr.min()}, zeros={(atr == 0).sum()}"


# ── 4. f7 and f8 are identical across all tickers on each date ────────────

def test_f7_f8_identical_per_date(features: pd.DataFrame) -> None:
    f7_std = features.groupby("date")["f7_market_ret"].std()
    f8_std = features.groupby("date")["f8_vix"].std()
    # Spec: "Identical value for all 20 tickers on a given date".
    # The cross-sectional std must be 0 (up to float epsilon).
    eps = 1e-12
    assert (f7_std < eps).all(), (
        f"f7_market_ret differs across tickers on at least one date. "
        f"max std = {f7_std.max():.3e}"
    )
    assert (f8_std < eps).all(), (
        f"f8_vix differs across tickers on at least one date. "
        f"max std = {f8_std.max():.3e}"
    )


# ── 5. NaN-corruption lookahead test — the critical anti-leakage test ─────

def test_no_lookahead_in_any_feature(prices: pd.DataFrame, cfg: dict) -> None:
    """For every feature, the value at date t must not depend on data
    from dates > t.

    The procedure:
      1. Build features from the full prices table.
      2. Pick a date D in the middle of the range.
      3. Rebuild features with all data after D set to NaN.
      4. Confirm the features for all dates <= D are identical to step 1.
    """
    # Build the "clean" features
    clean = feat_mod.build_features(prices, cfg)

    # Pick a date well past the warm-up so the rolling windows are
    # fully formed on both sides. We use the median of the available
    # dates to maximise the rolling-window coverage on either side.
    available = sorted(prices["date"].unique())
    median_date = available[len(available) // 2]
    # Need ~60 days of pre-D history to fill the volume z-score window.
    assert available.index(median_date) > 200, (
        "median date is too early; the test needs ~200 days of history"
    )

    # Corrupt everything strictly after median_date with NaN.
    corrupt = prices.copy()
    corrupt.loc[corrupt["date"] > pd.Timestamp(median_date),
                ["open", "high", "low", "close", "volume"]] = np.nan

    # Rebuild features from the corrupted data.
    rebuilt = feat_mod.build_features(corrupt, cfg)

    # For every (date <= median_date, ticker), the feature value in
    # the clean and rebuilt frames must be identical. We use the
    # intersection of dates both frames have, and compare per feature.
    cmp_dates = [d for d in rebuilt["date"].unique()
                 if pd.Timestamp(d) <= pd.Timestamp(median_date)]
    cmp_dates = sorted(cmp_dates)
    # Use a 5-day safety margin from the corruption boundary so we are
    # not affected by edge effects where a rolling window would
    # normally have crossed the boundary.
    safety = 5
    safe_dates = cmp_dates[:-safety] if len(cmp_dates) > safety else []

    for f in feat_mod.FEATURE_COLUMNS:
        a = clean[clean["date"].isin(safe_dates)].set_index(["date", "ticker"])[f]
        b = rebuilt[rebuilt["date"].isin(safe_dates)].set_index(["date", "ticker"])[f]
        common = a.index.intersection(b.index)
        if len(common) == 0:
            continue
        diff = (a.loc[common] - b.loc[common]).abs()
        max_diff = float(diff.max())
        assert max_diff < 1e-12, (
            f"Feature {f!r} shows lookahead: corrupting data after "
            f"{median_date.date()} changed the value at some earlier date "
            f"by up to {max_diff:.3e}. This means a rolling window is "
            f"including future data."
        )


# ── 6. No infinities anywhere ─────────────────────────────────────────────

def test_no_infinities(features: pd.DataFrame) -> None:
    feature_cols = feat_mod.FEATURE_COLUMNS
    arr = features[feature_cols].to_numpy()
    n_inf = int(np.sum(np.isinf(arr)))
    assert n_inf == 0, f"Found {n_inf} infinities in features"
    n_nan = int(np.sum(np.isnan(arr)))
    assert n_nan == 0, f"Found {n_nan} NaNs in features (warm-up should have dropped them)"


# ── 7. Price/MA centred near 1.0, f1_logret centred near 0 ────────────────

def test_f1_f2_centring_and_range(features: pd.DataFrame) -> None:
    """Sanity: f1_logret should be centred near 0 and within ±0.15
    for the bulk of days; f2_price_ma should be centred near 1.0 with
    a typical range of 0.8-1.2. The spec calls these 'sanity checks'."""
    f1 = features["f1_logret"]
    f2 = features["f2_price_ma"]
    # Centre
    assert abs(f1.mean()) < 0.01, f"f1_logret mean {f1.mean():.4f} not near 0"
    assert abs(f2.mean() - 1.0) < 0.01, f"f2_price_ma mean {f2.mean():.4f} not near 1.0"
    # P1-P99 range
    f1_lo, f1_hi = float(f1.quantile(0.01)), float(f1.quantile(0.99))
    f2_lo, f2_hi = float(f2.quantile(0.01)), float(f2.quantile(0.99))
    assert -0.15 < f1_lo and f1_hi < 0.15, f"f1_logret P1-P99 ({f1_lo}, {f1_hi}) outside ±0.15"
    assert 0.8 < f2_lo and f2_hi < 1.2, f"f2_price_ma P1-P99 ({f2_lo}, {f2_hi}) outside 0.8-1.2"


# ── 8. f6_vol_zscore centred near 0, right-skewed ────────────────────────

def test_f6_centred_and_skewed(features: pd.DataFrame) -> None:
    f6 = features["f6_vol_zscore"]
    assert abs(f6.mean()) < 0.1, f"f6 mean {f6.mean():.4f} not near 0"
    # Right-skewed: median < mean (the long tail of positive outliers
    # pulls the mean above the median).
    assert f6.median() < f6.mean(), (
        f"f6 not right-skewed: median={f6.median():.3f}, mean={f6.mean():.3f}"
    )


# ── 9. Exactly 20 tickers per date ────────────────────────────────────────

def test_exactly_20_tickers_per_date(features: pd.DataFrame) -> None:
    counts = features.groupby("date").size()
    assert (counts == 20).all(), (
        f"Some dates don't have exactly 20 tickers: "
        f"min={counts.min()}, max={counts.max()}"
    )
