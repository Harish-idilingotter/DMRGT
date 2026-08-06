"""DMRGT Phase 1 — feature engineering (Step D / §6 of build spec).

Builds the 8 features listed in the spec, one at a time, and saves them
to cache/features.parquet UNSCALED. Scaling is fitted per fold inside
train.py (§7.2 of the spec) — never here, never in the cached Parquet.

Each feature is verified by hand against the spec's sanity checks
(distributions, ranges, cross-section equality for the global series).
The verification notebook (notebooks/verification.ipynb) plots every
feature; tests/test_features.py asserts the structural invariants.

Critical design constraints (all from §1 + §6 of the spec):
  • No future data may leak into a feature at date t. Every rolling
    window ends at day t.
  • Features are stored RAW. No scaler touches the cache.
  • The warm-up period (~60 days) is dropped and the count is logged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as ta


# Canonical column names. Order matters — this is the order the
# Transformer receives them in. Do not reorder without re-training.
FEATURE_COLUMNS: list[str] = [
    "f1_logret",       # 1. log-return
    "f2_price_ma",     # 2. price / 10-day MA
    "f3_rsi",          # 3. RSI(14)
    "f4_macd",         # 4. MACD
    "f5_atr",          # 5. ATR(14) scaled by close
    "f6_vol_zscore",   # 6. volume z-score (60-day)
    "f7_market_ret",   # 7. market return (^GSPC, identical per date)
    "f8_vix",          # 8. VIX level (identical per date)
]


def _apply_per_ticker(wide_close: pd.DataFrame, fn, _: str | None = None,
                      wide_high: pd.DataFrame | None = None,
                      wide_low: pd.DataFrame | None = None) -> pd.DataFrame:
    """Apply a single-ticker feature function to every column of a wide
    price DataFrame and return a wide DataFrame with the same columns.

    For features needing only close, ``wide_high``/``wide_low`` may be
    None. For features needing high/low/close, pass all three.
    """
    out = {}
    for col in wide_close.columns:
        s = wide_close[col]
        if wide_high is not None and wide_low is not None:
            res = fn(wide_high[col], wide_low[col], s)
        else:
            res = fn(s)
        out[col] = pd.Series(res, index=s.index)
    return pd.DataFrame(out, index=wide_close.index)


def _f1_logret(close: pd.Series) -> pd.Series:
    """Feature 1 — log-return at day t.
    log(close[t] / close[t-1]). NaN at t=0 by construction.
    """
    return np.log(close / close.shift(1))


def _f2_price_ma(close: pd.Series, window: int = 10) -> pd.Series:
    """Feature 2 — close[t] / rolling_mean(close, 10)[t].
    The rolling mean is computed with min_periods=window so a value at
    date t is only emitted when the window ending at t is complete — no
    partial windows, no future data.
    """
    ma = close.rolling(window=window, min_periods=window).mean()
    return close / ma


def _f3_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Feature 3 — RSI(14), strictly bounded in [0, 100]."""
    return ta.rsi(close, length=period)


def _f4_macd(close: pd.Series) -> pd.Series:
    """Feature 4 — MACD line (12, 26 default from pandas-ta).
    The MACD line, not the histogram — the histogram combines MACD and
    its signal, which double-counts information. Spec says 'MACD' with
    no further qualification, so we keep the line.
    """
    macd_df = ta.macd(close, fast=12, slow=26, signal=9)
    # macd_df columns: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
    return macd_df.iloc[:, 0]  # the MACD line


def _f5_atr(high: pd.Series, low: pd.Series, close: pd.Series,
            period: int = 14) -> pd.Series:
    """Feature 5 — ATR(14) scaled by close so it is comparable across
    tickers with very different price levels. Strictly > 0 (ATR itself
    is positive; the ratio to a positive close is positive too)."""
    raw_atr = ta.atr(high=high, low=low, close=close, length=period)
    return raw_atr / close


def _f6_vol_zscore(volume: pd.Series, window: int = 60) -> pd.Series:
    """Feature 6 — (vol - rolling_mean(vol, 60)) / rolling_std(vol, 60).
    Centred near 0, right-skewed (positive outliers on volume spikes).
    """
    mu = volume.rolling(window=window, min_periods=window).mean()
    sd = volume.rolling(window=window, min_periods=window).std(ddof=0)
    return (volume - mu) / sd


def _f7_market_ret(spx_close: pd.Series) -> pd.Series:
    """Feature 7 — log-return of ^GSPC. Identical for all 20 tickers
    on a given date because it is computed from a single global series
    and broadcast at the merge step.
    """
    return np.log(spx_close / spx_close.shift(1))


def _f8_vix(vix_close: pd.Series) -> pd.Series:
    """Feature 8 — VIX level, broadcast identically to all 20 tickers.
    NOT a return — the spec explicitly says 'VIX close level'."""
    return vix_close


def build_features(prices: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Build the 8 features per ticker for the whole date range.

    Parameters
    ----------
    prices : DataFrame
        Long-format prices with columns
        ``date | ticker | open | high | low | close | volume``.
        Must include ^GSPC and ^VIX as tickers — they are the source
        for features 7 and 8.
    cfg : dict
        The full configs.yaml dict. Reads:
          - universe.tickers (the 20 stocks)
          - universe.global_series (must contain ^GSPC and ^VIX)
          - features.ma_period
          - features.rsi_period
          - features.atr_period
          - features.volume_zscore_window

    Returns
    -------
    DataFrame
        Long format ``date | ticker | f1 ... f8``. First ~60 rows are
        dropped (warm-up). The cache stores this RAW — no scaling.
    """
    universe = list(cfg["universe"]["tickers"])
    globals_ = list(cfg["universe"]["global_series"])
    if "^GSPC" not in globals_ or "^VIX" not in globals_:
        raise ValueError(
            f"configs.yaml universe.global_series must contain both "
            f"^GSPC and ^VIX; got {globals_}"
        )
    ma_w = int(cfg["features"]["ma_period"])
    rsi_p = int(cfg["features"]["rsi_period"])
    atr_p = int(cfg["features"]["atr_period"])
    vol_w = int(cfg["features"]["volume_zscore_window"])

    # ── pivot each price column to wide (index=date, columns=ticker) ──
    needed = ["open", "high", "low", "close", "volume"]
    wide = {col: prices.pivot(index="date", columns="ticker",
                              values=col).sort_index()
            for col in needed}

    # Only compute per-stock features for the 20 tickers (skip ^GSPC,
    # ^VIX — they are handled as global features below).
    stock_close = wide["close"][universe]
    stock_high = wide["high"][universe]
    stock_low = wide["low"][universe]
    stock_volume = wide["volume"][universe]

    # ── features 1..6 — one per ticker, computed independently ──────
    f1 = _apply_per_ticker(stock_close,
                           lambda s: _f1_logret(s), "f1")
    f2 = _apply_per_ticker(stock_close,
                           lambda s: _f2_price_ma(s, window=ma_w), "f2")
    f3 = _apply_per_ticker(stock_close,
                           lambda s: _f3_rsi(s, period=rsi_p), "f3")
    f4 = _apply_per_ticker(stock_close, _f4_macd, "f4")
    f5 = _apply_per_ticker(stock_close, lambda h, l, c: _f5_atr(
        h, l, c, period=atr_p), "f5",
        wide_high=stock_high, wide_low=stock_low)
    f6 = _apply_per_ticker(stock_volume,
                           lambda s: _f6_vol_zscore(s, window=vol_w), "f6")

    # ── stack the per-ticker features into a single long DataFrame ───
    pieces = []
    for f_wide, fname in [(f1, "f1_logret"),
                          (f2, "f2_price_ma"),
                          (f3, "f3_rsi"),
                          (f4, "f4_macd"),
                          (f5, "f5_atr"),
                          (f6, "f6_vol_zscore")]:
        melted = f_wide.stack(future_stack=True).reset_index()
        melted.columns = ["date", "ticker", fname]
        pieces.append(melted)
    long = pieces[0]
    for p in pieces[1:]:
        long = long.merge(p, on=["date", "ticker"], how="outer")

    # ── features 7 & 8 — global series, broadcast to all tickers ─────
    spx = wide["close"]["^GSPC"]
    vix = wide["close"]["^VIX"]
    f7_series = _f7_market_ret(spx)
    f8_series = _f8_vix(vix)

    f7_df = pd.DataFrame({"date": f7_series.index,
                          "f7_market_ret": f7_series.values})
    f8_df = pd.DataFrame({"date": f8_series.index,
                          "f8_vix": f8_series.values})
    long = long.merge(f7_df, on="date", how="left")
    long = long.merge(f8_df, on="date", how="left")

    # ── drop warm-up rows where any feature is NaN ───────────────────
    feature_cols = FEATURE_COLUMNS
    n_before = len(long)
    long = long.dropna(subset=feature_cols).reset_index(drop=True)
    n_after = len(long)
    print(f"[features] dropped {n_before - n_after:,} warm-up rows with NaN "
          f"({n_before:,} → {n_after:,})")

    # ── ordering and final type hygiene ──────────────────────────────
    long = long[["date", "ticker"] + feature_cols]
    # Force float64 to ensure consistent parquet schema. pandas_ta can
    # return slightly different dtypes per platform.
    for c in feature_cols:
        long[c] = long[c].astype(np.float64)
    # Final sanity: no NaNs, no infinities anywhere in the features.
    assert long[feature_cols].isna().sum().sum() == 0, "NaNs after warm-up drop"
    assert np.isfinite(long[feature_cols].to_numpy()).all(), "infs in features"
    return long


if __name__ == "__main__":
    # Manual smoke run: `python features.py` rebuilds from the cache and
    # prints a one-line summary. Useful when iterating without going
    # through the data.py CLI.
    import data
    import paths
    if not paths.PRICES.exists():
        raise SystemExit("cache/prices.parquet missing — run "
                         "`python data.py --download` first")
    cfg = data.load_config()
    px = pd.read_parquet(paths.PRICES)
    out = build_features(px, cfg)
    print(out.head())
    print(f"rows={len(out):,}  tickers={out['ticker'].nunique()}  "
          f"date range={out['date'].min().date()} → {out['date'].max().date()}")
