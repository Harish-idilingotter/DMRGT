"""DMRGT Phase 5 — cache builder (Nasdaq-100) with a legacy-40 regression mode.

    # Regression mode (hard gate, run first)
    python build_cache.py --universe legacy40 --out /tmp/legacy40_rebuild --compare /tmp/dmrgt-phase-3-cache
    # Build the Nasdaq-100 cache
    python build_cache.py --universe ndx100 --out cache/ --compare /tmp/dmrgt-phase-3-cache \
                          --regression-json /tmp/legacy40_rebuild/regression_test_legacy40.json

Port provenance (old code read from tag `archive-before-ndx` / the commits below, pasted — not rewritten):

| function(s) here                                                      | old file     | old commit |
|-----------------------------------------------------------------------|--------------|------------|
| `_align_and_validate_prices` (calendar alignment + plausibility asserts) | data.py (`download_prices`) | fd8125d |
| `build_targets`                                                       | data.py      | fd8125d    |
| `_f1_logret` … `_f8_vix`, `_apply_per_ticker`, `build_features`, `FEATURE_COLUMNS` | features.py | fd8125d |
| `_pct_change_no_fill`, `_universe_returns`, `_sector_edges`, `_edges_for_returns_window`, `_validate_edge_frame`, `build_daily_graph_all_dates` | graph.py | 175768d (tag phase3-complete) |
| `_screen`, `_runs` (gap/listing screen, ffill ≤ 2 days)               | data_build/sp100/universe.py | a58dd21 (tag archive-before-ndx; S&P 100 WIP) |
| `compare_caches` (regression thresholds), `validate_cache`            | data_build/sp100/regression.py, validate.py | a58dd21 |
| `LEGACY40`                                                            | configs.yaml | fd8125d    |

Intentional changes vs the old code: configuration is an in-memory dict (no yaml/paths modules); the price download takes
pre-fetched frames so the same alignment code serves both universes; the 40-stock "no sector above 20%" design guard is
not applied (it cannot hold for the Nasdaq-100); the universe comes from rules, not a hand-typed list.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

START, END = "2015-01-01", "2024-12-31"
GLOBALS = ["^GSPC", "^VIX"]
MAX_GAP, MAX_MISSING_FRAC = 2, 0.01
GRAPH = {"corr_window": 60, "corr_method": "pearson", "k_pos": 8, "k_neg": 4, "symmetrise": False,
         "include_sector": True, "min_history_days": 61}
FEATS = {"rsi_period": 14, "atr_period": 14, "ma_period": 10, "volume_zscore_window": 60}
FEATURE_COLUMNS = ["f1_logret", "f2_price_ma", "f3_rsi", "f4_macd", "f5_atr", "f6_vol_zscore", "f7_market_ret", "f8_vix"]
CACHE_PARQUETS = ("prices.parquet", "features.parquet", "targets.parquet", "graphs_baseline.parquet")
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120 Safari/537.36",
      "Accept": "application/json"}

LEGACY40 = {  # ticker -> GICS sector, in the old config's order (ported from configs.yaml @ fd8125d)
    "AAPL": "Information Technology", "MSFT": "Information Technology", "NVDA": "Information Technology",
    "AVGO": "Information Technology", "CRM": "Information Technology",
    "JPM": "Financials", "BRK-B": "Financials", "GS": "Financials", "BAC": "Financials", "V": "Financials",
    "JNJ": "Health Care", "PFE": "Health Care", "UNH": "Health Care", "ABBV": "Health Care", "MRK": "Health Care",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy", "EOG": "Energy",
    "AMZN": "Consumer Discretionary", "HD": "Consumer Discretionary", "NKE": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary", "SBUX": "Consumer Discretionary",
    "PG": "Consumer Staples", "KO": "Consumer Staples", "WMT": "Consumer Staples", "COST": "Consumer Staples",
    "PEP": "Consumer Staples",
    "HON": "Industrials", "CAT": "Industrials", "BA": "Industrials", "UPS": "Industrials", "UNP": "Industrials",
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities", "AEP": "Utilities", "EXC": "Utilities",
}


def make_cfg(tickers: list[str], sectors: dict[str, str]) -> dict:
    return {"universe": {"tickers": list(tickers), "global_series": list(GLOBALS), "start_date": START, "end_date": END},
            "sectors": {**{t: sectors[t] for t in tickers}, **{g: "global" for g in GLOBALS}},
            "features": dict(FEATS), "graph": dict(GRAPH)}


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ───────────────────────── prices (data.py @ fd8125d) ─────────────────────────

def _align_and_validate_prices(raw: pd.DataFrame) -> pd.DataFrame:
    """Calendar alignment + validation, verbatim from download_prices."""
    raw = raw.sort_values(["ticker", "date"]).reset_index(drop=True)
    per_ticker_dates = {t: set(d["date"]) for t, d in raw.groupby("ticker")}
    common_dates = set.intersection(*per_ticker_dates.values()) if per_ticker_dates else set()
    if not common_dates:
        raise RuntimeError("No common trading dates across all tickers")
    print(f"[data] master calendar: {len(common_dates)} trading days")
    aligned = (raw[raw["date"].isin(common_dates)].sort_values(["date", "ticker"]).reset_index(drop=True))
    counts = aligned.groupby("ticker").size()
    assert counts.nunique() == 1, f"Calendar alignment failed — differing row counts: {counts.to_dict()}"
    n_dates = int(counts.iloc[0])
    assert aligned.duplicated(subset=["date", "ticker"]).sum() == 0, "duplicate (date, ticker) rows"
    assert aligned.isna().sum().sum() == 0, f"NaNs present after alignment:\n{aligned.isna().sum()}"
    assert 1500 <= n_dates <= 2600, f"Implausible row count per ticker: {n_dates}"
    return aligned


def _flatten_yf(df: pd.DataFrame) -> pd.DataFrame:
    """Column/index normalisation exactly as download_prices does it."""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={c: "close" for c in df.columns if c == "price"})
    df = df.reset_index()
    date_col = "Date" if "Date" in df.columns else ("Datetime" if "Datetime" in df.columns else df.columns[0])
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    keep = ["date", "open", "high", "low", "close", "volume"]
    df = df[[c for c in keep if c in df.columns]]
    if "volume" not in df.columns:
        df["volume"] = np.nan
    return df


def fetch_yf(t: str, work: Path) -> pd.DataFrame:
    """Same yf.download call as the old pipeline; raw frames cached in `work` so a rerun is resumable."""
    p = work / f"{t}.pkl"
    if p.exists():
        return pickle.load(open(p, "rb"))
    import yfinance as yf
    df = pd.DataFrame()
    for attempt in range(4):
        try:
            df = yf.download(t, start=START, end=END, auto_adjust=True, progress=False, actions=False)
            break
        except Exception as e:  # rate limit etc.
            print(f"  retry {t}: {e}")
            time.sleep(5 * (attempt + 1))
    pickle.dump(df, open(p, "wb"))
    return df


# ───────────────────────── targets (data.py @ fd8125d) ─────────────────────────

def build_targets(prices: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Next-day excess return: daily_return[t+1] - cross_sectional_mean[t+1]. Final row dropped."""
    universe_tickers = list(cfg["universe"]["tickers"])
    px = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    rets = px.pct_change()
    universe = rets[universe_tickers]
    mean_ret = universe.mean(axis=1)
    excess = universe.sub(mean_ret, axis=0)
    labels = excess.shift(-1)
    n_before = len(labels)
    labels = labels.dropna(how="all")
    print(f"[data] targets: dropped {n_before - len(labels)} row(s) with no t+1 label")
    long = (labels.stack(future_stack=True).rename("excess_return").reset_index().rename(columns={"level_1": "ticker"}))
    return long[["date", "ticker", "excess_return"]]


# ───────────────────────── features (features.py @ fd8125d) ─────────────────────────

def _apply_per_ticker(wide_close, fn, _=None, wide_high=None, wide_low=None):
    out = {}
    for col in wide_close.columns:
        s = wide_close[col]
        res = fn(wide_high[col], wide_low[col], s) if (wide_high is not None and wide_low is not None) else fn(s)
        out[col] = pd.Series(res, index=s.index)
    return pd.DataFrame(out, index=wide_close.index)


def _f1_logret(close): return np.log(close / close.shift(1))
def _f2_price_ma(close, window=10): return close / close.rolling(window=window, min_periods=window).mean()


def _f3_rsi(close, period=14):
    import pandas_ta as ta
    return ta.rsi(close, length=period)


def _f4_macd(close):
    import pandas_ta as ta
    return ta.macd(close, fast=12, slow=26, signal=9).iloc[:, 0]


def _f5_atr(high, low, close, period=14):
    import pandas_ta as ta
    return ta.atr(high=high, low=low, close=close, length=period) / close


def _f6_vol_zscore(volume, window=60):
    mu = volume.rolling(window=window, min_periods=window).mean()
    sd = volume.rolling(window=window, min_periods=window).std(ddof=0)
    return (volume - mu) / sd


def _f7_market_ret(spx_close): return np.log(spx_close / spx_close.shift(1))
def _f8_vix(vix_close): return vix_close


def build_features(prices: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    universe = list(cfg["universe"]["tickers"])
    ma_w, rsi_p = int(cfg["features"]["ma_period"]), int(cfg["features"]["rsi_period"])
    atr_p, vol_w = int(cfg["features"]["atr_period"]), int(cfg["features"]["volume_zscore_window"])
    wide = {col: prices.pivot(index="date", columns="ticker", values=col).sort_index()
            for col in ["open", "high", "low", "close", "volume"]}
    stock_close, stock_high = wide["close"][universe], wide["high"][universe]
    stock_low, stock_volume = wide["low"][universe], wide["volume"][universe]
    f1 = _apply_per_ticker(stock_close, lambda s: _f1_logret(s), "f1")
    f2 = _apply_per_ticker(stock_close, lambda s: _f2_price_ma(s, window=ma_w), "f2")
    f3 = _apply_per_ticker(stock_close, lambda s: _f3_rsi(s, period=rsi_p), "f3")
    f4 = _apply_per_ticker(stock_close, _f4_macd, "f4")
    f5 = _apply_per_ticker(stock_close, lambda h, l, c: _f5_atr(h, l, c, period=atr_p), "f5",
                           wide_high=stock_high, wide_low=stock_low)
    f6 = _apply_per_ticker(stock_volume, lambda s: _f6_vol_zscore(s, window=vol_w), "f6")
    pieces = []
    for f_wide, fname in [(f1, "f1_logret"), (f2, "f2_price_ma"), (f3, "f3_rsi"), (f4, "f4_macd"),
                          (f5, "f5_atr"), (f6, "f6_vol_zscore")]:
        melted = f_wide.stack(future_stack=True).reset_index()
        melted.columns = ["date", "ticker", fname]
        pieces.append(melted)
    long = pieces[0]
    for p in pieces[1:]:
        long = long.merge(p, on=["date", "ticker"], how="outer")
    f7_series, f8_series = _f7_market_ret(wide["close"]["^GSPC"]), _f8_vix(wide["close"]["^VIX"])
    long = long.merge(pd.DataFrame({"date": f7_series.index, "f7_market_ret": f7_series.values}), on="date", how="left")
    long = long.merge(pd.DataFrame({"date": f8_series.index, "f8_vix": f8_series.values}), on="date", how="left")
    n_before = len(long)
    long = long.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    print(f"[features] dropped {n_before - len(long):,} warm-up rows with NaN ({n_before:,} → {len(long):,})")
    long = long[["date", "ticker"] + FEATURE_COLUMNS]
    for c in FEATURE_COLUMNS:
        long[c] = long[c].astype(np.float64)
    assert long[FEATURE_COLUMNS].isna().sum().sum() == 0, "NaNs after warm-up drop"
    assert np.isfinite(long[FEATURE_COLUMNS].to_numpy()).all(), "infs in features"
    return long


# ───────────────────────── graph (graph.py @ 175768d) ─────────────────────────

def _pct_change_no_fill(df):
    try:
        return df.pct_change(fill_method=None)
    except TypeError:  # pandas >= 3
        return df.pct_change()


def _universe_returns(prices, cfg):
    universe = list(cfg["universe"]["tickers"])
    px_wide = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    missing = [t for t in universe if t not in px_wide.columns]
    if missing:
        raise ValueError(f"prices.parquet is missing universe tickers: {missing}")
    return _pct_change_no_fill(px_wide[universe]).iloc[1:]


def _sector_edges(cfg):
    universe = list(cfg["universe"]["tickers"])
    sectors_cfg = cfg.get("sectors", {})
    if not sectors_cfg:
        raise ValueError("cfg missing 'sectors'")
    srcs, dsts = [], []
    if not bool(cfg["graph"].get("include_sector", True)):
        return srcs, dsts
    n = len(universe)
    for i in range(n):
        for j in range(n):
            if i >= j:
                continue
            if sectors_cfg.get(universe[i]) == sectors_cfg.get(universe[j]):
                srcs.extend([i, j])
                dsts.extend([j, i])
    counts = Counter(sectors_cfg[t] for t in universe)
    expected = sum(c * (c - 1) for c in counts.values())
    assert len(srcs) == expected, f"Sector edge count mismatch: built {len(srcs)}, expected {expected}"
    return srcs, dsts


def _edges_for_returns_window(window_rets, as_of, cfg, sector_src, sector_dst):
    universe = list(cfg["universe"]["tickers"])
    n_stocks = len(universe)
    corr_window, k_pos, k_neg = int(cfg["graph"]["corr_window"]), int(cfg["graph"]["k_pos"]), int(cfg["graph"]["k_neg"])
    assert len(window_rets) == corr_window, f"{pd.Timestamp(as_of).date()}: window has {len(window_rets)} returns"
    nan_count = window_rets.isna().sum().sum()
    assert nan_count == 0, f"{pd.Timestamp(as_of).date()}: {nan_count} NaN returns in correlation window"
    corr = window_rets.corr().values
    corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)
    rows = []
    np.fill_diagonal(corr, -np.inf)
    pos_order = np.argsort(-corr, axis=1)
    for i in range(n_stocks):
        for rank in range(k_pos):
            rows.append({"date": pd.Timestamp(as_of), "src": i, "dst": int(pos_order[i, rank]), "edge_type": 0})
    np.fill_diagonal(corr, np.inf)
    neg_order = np.argsort(corr, axis=1)
    for i in range(n_stocks):
        for rank in range(k_neg):
            rows.append({"date": pd.Timestamp(as_of), "src": i, "dst": int(neg_order[i, rank]), "edge_type": 1})
    for s, d in zip(sector_src, sector_dst):
        rows.append({"date": pd.Timestamp(as_of), "src": int(s), "dst": int(d), "edge_type": 2})
    return pd.DataFrame(rows)


def _validate_edge_frame(df, cfg, label="graph"):
    n_stocks = len(cfg["universe"]["tickers"])
    k_pos, k_neg = int(cfg["graph"]["k_pos"]), int(cfg["graph"]["k_neg"])
    sector_src, _ = _sector_edges(cfg)
    expected_per_date = n_stocks * k_pos + n_stocks * k_neg + len(sector_src)
    per_date = df.groupby("date").size()
    bad = per_date[per_date != expected_per_date]
    assert bad.empty, f"[{label}] {len(bad)} dates have wrong edge count"
    tc = df.groupby("date")["edge_type"].value_counts().unstack(fill_value=0)
    assert (tc[0] == n_stocks * k_pos).all() and (tc[1] == n_stocks * k_neg).all() and (tc[2] == len(sector_src)).all()
    assert (df["src"] != df["dst"]).all(), f"[{label}] self-loops found"
    assert int(df["src"].max()) < n_stocks and int(df["dst"].max()) < n_stocks
    for d in list(df["date"].unique())[:5]:
        day = df[df["date"] == d]
        assert len(set(day["src"]) | set(day["dst"])) == n_stocks, f"[{label}] isolated nodes on {d}"
    print(f"[{label}] per-date counts {expected_per_date} ✓")


def build_daily_graph_all_dates(prices, cfg):
    corr_window = int(cfg["graph"]["corr_window"])
    sector_src, sector_dst = _sector_edges(cfg)
    rets = _universe_returns(prices, cfg)
    rows, n_skipped = [], 0
    for idx, d in enumerate(rets.index.tolist()):
        if idx < corr_window - 1:
            n_skipped += 1
            continue
        window_rets = rets.iloc[idx - corr_window + 1: idx + 1]
        rows.extend(_edges_for_returns_window(window_rets, d, cfg, sector_src, sector_dst).to_dict("records"))
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "edge_type", "src", "dst"]).reset_index(drop=True)
    print(f"[graph] built {df['date'].nunique()} daily graphs (skipped {n_skipped} dates)")
    _validate_edge_frame(df, cfg, label="dynamic")
    return df


# ───────────────────────── universe screening (sp100 universe.py @ a58dd21) ─────────────────────────

def _runs(missing_mask):
    out, n, start = [], len(missing_mask), None
    for i, m in enumerate(missing_mask):
        if m and start is None:
            start = i
        if (not m) and start is not None:
            out.append((start, i - start))
            start = None
    if start is not None:
        out.append((start, n - start))
    return out


def _ok_frame(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.dropna(subset=["open", "high", "low", "close"])


def _screen(raw, ref_cal):
    """-> (forward-filled frame or None, info, reason). Rules: data from the first calendar day, no gap longer than
    MAX_GAP, at most MAX_MISSING_FRAC missing days; gaps <= MAX_GAP are forward-filled."""
    if raw is None or len(raw) == 0:
        return None, {"first_date": "", "last_date": "", "missing_days": len(ref_cal)}, "no data returned by yfinance"
    ok = _ok_frame(raw)
    first, last = ok.index.min(), ok.index.max()
    aligned = ok.reindex(ref_cal)
    miss = aligned["close"].isna().to_numpy()
    info = {"first_date": str(first.date()), "last_date": str(last.date()), "missing_days": int(miss.sum())}
    rr = _runs(miss)
    if miss[0]:
        return None, info, f"no data at start of period (first data {first.date()}; listed after start)"
    if any(l > MAX_GAP for _, l in rr):
        return None, info, f"gap longer than {MAX_GAP} trading days ({max(l for _, l in rr)})"
    if miss.mean() > MAX_MISSING_FRAC:
        return None, info, f"{miss.mean():.2%} missing trading days (> {MAX_MISSING_FRAC:.0%})"
    filled = aligned.ffill()
    assert filled[["open", "high", "low", "close"]].notna().all().all()
    filled["volume"] = filled["volume"].fillna(0)
    return filled, info, ""


def _to_prices_long(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for t, df in frames.items():
        d = _flatten_yf(df)
        d["ticker"] = t
        parts.append(d[["date", "ticker", "open", "high", "low", "close", "volume"]])
    return pd.concat(parts, ignore_index=True)


# ───────────────────────── Nasdaq-100 universe (rules §3.2) ─────────────────────────

def fetch_ndx_snapshot() -> pd.DataFrame:
    """Current Nasdaq-100 components. Wikipedia's Nasdaq-100 page no longer carries a components table, so the
    list comes from Nasdaq's own index endpoint and the sector / industry from Nasdaq's stock screener (recorded in
    the manifest as a source deviation). Never typed from memory."""
    import requests
    r = requests.get("https://api.nasdaq.com/api/quote/list-type/nasdaq100", headers=UA, timeout=30)
    r.raise_for_status()
    comp = pd.DataFrame(r.json()["data"]["data"]["rows"])[["symbol", "companyName", "marketCap"]]
    s = requests.get("https://api.nasdaq.com/api/screener/stocks",
                     params={"tableonly": "true", "limit": "10000", "download": "true"}, headers=UA, timeout=90)
    s.raise_for_status()
    scr = pd.DataFrame(s.json()["data"]["rows"])[["symbol", "sector", "industry"]]
    snap = comp.merge(scr, on="symbol", how="left")
    snap["fetched_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    snap["source_url"] = "api.nasdaq.com/api/quote/list-type/nasdaq100 + api.nasdaq.com/api/screener/stocks"
    return snap


def _company_key(name: str) -> str:
    n = re.sub(r"\s+(Class|Series)\s+[A-Z]\b", "", name)
    n = re.sub(r"\s+(Common Stock|Capital Stock|Ordinary Shares?|American Depositary (Shares|Receipts)|"
               r"Common Shares)\b.*$", "", n, flags=re.I)
    return re.sub(r"[^a-z0-9]", "", n.lower())


def build_ndx_universe(work: Path, out: Path):
    snap = fetch_ndx_snapshot()
    snap.to_csv(out / "universe_raw_snapshot.csv", index=False)
    snap["ticker"] = snap["symbol"].str.strip().str.replace(".", "-", regex=False)
    snap["company"] = snap["companyName"]
    snap["sector"] = snap["sector"].fillna("").replace("", "Unclassified")
    snap["sub_sector"] = snap["industry"].fillna("").replace("", "Unclassified")
    snap["key"] = snap["company"].map(_company_key)

    raw = {t: fetch_yf(t, work) for t in GLOBALS + snap["ticker"].tolist()}
    for g in GLOBALS:
        assert raw[g] is not None and len(raw[g]) > 0, f"global series {g} missing"
    ref_cal = _ok_frame(raw["^GSPC"]).index

    def median_dv(t):
        d = _ok_frame(raw[t]) if raw[t] is not None and len(raw[t]) else pd.DataFrame()
        return float((d["close"] * d["volume"]).median()) if len(d) else float("nan")
    snap["median_dollar_volume"] = snap["ticker"].map(median_dv)

    snap["included"], snap["exclusion_reason"] = True, ""
    for key, g in snap.groupby("key"):
        if len(g) > 1:
            keep = g["median_dollar_volume"].fillna(-1).idxmax()
            for i in g.index.drop(keep):
                snap.loc[i, ["included", "exclusion_reason"]] = [False, f"second share class of {snap.loc[keep, 'ticker']} "
                                                                  f"(lower median dollar volume)"]
    screened, infos = {}, {}
    for g in GLOBALS:
        f, _, why = _screen(raw[g], ref_cal)
        assert f is not None, f"global series {g} fails the rules: {why}"
        screened[g] = f
    for i, rw in snap[snap["included"]].iterrows():
        t = rw["ticker"]
        f, info, why = _screen(raw[t], ref_cal)
        infos[t] = info
        if f is None:
            snap.loc[i, ["included", "exclusion_reason"]] = [False, why]
        else:
            screened[t] = f
        print(f"{t:6s} {'ok  ' if f is not None else 'EXCL'} {why}", flush=True)
    for c in ("first_date", "last_date", "missing_days"):
        snap[c] = [infos.get(t, {}).get(c, "") for t in snap["ticker"]]
    snap = snap.sort_values("ticker").reset_index(drop=True)
    U = snap[["ticker", "company", "sector", "sub_sector", "included", "exclusion_reason", "first_date", "last_date",
              "missing_days", "median_dollar_volume"]]
    U.to_csv(out / "universe.csv", index=False)
    inc = U[U["included"]]
    print(f"\nincluded {len(inc)} / {len(U)}; sectors: {inc['sector'].value_counts().to_dict()}")
    frames = {t: screened[t][["open", "high", "low", "close", "volume"]].rename(columns=str.capitalize).rename_axis("Date")
              for t in GLOBALS + inc["ticker"].tolist()}
    return U, frames


# ───────────────────────── regression + validation ─────────────────────────

def compare_caches(new_dir: Path, old_dir: Path) -> dict:
    """Spec §3.4 thresholds."""
    R = {"thresholds": {"feature_pearson_min": 0.999, "feature_median_abs_diff_max_in_col_std": 1e-6,
                        "target_pearson_min": 0.9999, "target_frac_within_1e-6_min": 0.999,
                        "graph_identical_dates_min": 0.99}}
    ok = True
    load = lambda d, n: pd.read_parquet(d / f"{n}.parquet")
    dt = {}
    for n in ("prices", "features", "targets"):
        a, b = load(new_dir, n), load(old_dir, n)
        same = (set(a["date"]) == set(b["date"])) and (set(a["ticker"]) == set(b["ticker"]))
        dt[n] = {"dates_identical": set(a["date"]) == set(b["date"]), "tickers_identical": set(a["ticker"]) == set(b["ticker"]),
                 "rows_new": len(a), "rows_old": len(b)}
        ok &= same
    R["dates_and_tickers"] = dt
    fa, fb = load(new_dir, "features"), load(old_dir, "features")
    m = fa.merge(fb, on=["date", "ticker"], suffixes=("_n", "_o"))
    assert len(m) == len(fb) == len(fa), "feature rows do not line up"
    feat = {}
    for c in FEATURE_COLUMNS:
        x, y = m[c + "_n"].to_numpy(), m[c + "_o"].to_numpy()
        corr = float(np.corrcoef(x, y)[0, 1])
        mad, sd = float(np.median(np.abs(x - y))), float(np.std(y))
        p = corr >= 0.999 and mad <= 1e-6 * sd
        feat[c] = {"pearson": corr, "median_abs_diff": mad, "col_std": sd, "max_abs_diff": float(np.abs(x - y).max()), "pass": bool(p)}
        ok &= p
    R["features"] = feat
    ta_, tb = load(new_dir, "targets"), load(old_dir, "targets")
    m = ta_.merge(tb, on=["date", "ticker"], suffixes=("_n", "_o"))
    assert len(m) == len(tb) == len(ta_), "target rows do not line up"
    x, y = m["excess_return_n"].to_numpy(), m["excess_return_o"].to_numpy()
    corr, frac = float(np.corrcoef(x, y)[0, 1]), float(np.mean(np.abs(x - y) <= 1e-6))
    R["targets"] = {"pearson": corr, "frac_within_1e-6": frac, "max_abs_diff": float(np.abs(x - y).max()),
                    "pass": bool(corr >= 0.9999 and frac >= 0.999)}
    ok &= R["targets"]["pass"]
    ga, gb = load(new_dir, "graphs_baseline"), load(old_dir, "graphs_baseline")
    sets = lambda g: {d: frozenset(zip(x.edge_type, x.src, x.dst)) for d, x in g.groupby("date")}
    sa, sb = sets(ga), sets(gb)
    common = sorted(set(sa) & set(sb))
    same = sum(sa[d] == sb[d] for d in common)
    R["graphs"] = {"dates_new": len(sa), "dates_old": len(sb), "dates_common": len(common), "identical_edge_set_dates": same,
                   "fraction": same / max(1, len(common)),
                   "pass": bool(len(sa) == len(sb) and same / max(1, len(common)) >= 0.99)}
    ok &= R["graphs"]["pass"]
    R["passed"] = bool(ok)
    return R


def price_revision_stats(new_dir: Path, old_dir: Path) -> dict:
    a, b = pd.read_parquet(new_dir / "prices.parquet"), pd.read_parquet(old_dir / "prices.parquet")
    m = a.merge(b, on=["date", "ticker"], suffixes=("_n", "_o"))
    rel = ((m["close_n"] - m["close_o"]).abs() / m["close_o"]).to_numpy()
    return {"close_rel_diff_max": float(rel.max()), "close_rel_diff_median": float(np.median(rel)),
            "frac_rows_with_rel_diff_gt_1e-9": float((rel > 1e-9).mean())}


def validate_cache(out: Path, cfg: dict, U: pd.DataFrame, legacy_dir: Path | None) -> dict:
    tick = cfg["universe"]["tickers"]
    N = len(tick)
    k_pos, k_neg = cfg["graph"]["k_pos"], cfg["graph"]["k_neg"]
    P, F, T, G = (pd.read_parquet(out / f"{n}.parquet") for n in ("prices", "features", "targets", "graphs_baseline"))
    R, checks = {"n_stocks": N}, {}

    def chk(name, passed, **detail):
        checks[name] = {"pass": bool(passed), **detail}

    inc = U[U["included"]]
    chk("universe_matches_included", set(tick) == set(inc["ticker"]) and set(F["ticker"]) == set(tick)
        and set(T["ticker"]) == set(tick), n_included=len(inc))
    chk("no_nan_inf", F[FEATURE_COLUMNS].notna().all().all() and np.isfinite(F[FEATURE_COLUMNS].to_numpy()).all()
        and T["excess_return"].notna().all() and np.isfinite(T["excess_return"].to_numpy()).all())
    rsi = F["f3_rsi"]
    per_t_vz = F.groupby("ticker")["f6_vol_zscore"].apply(lambda s: float(((s > -5) & (s < 5)).mean()))
    const_cols = [c for c in FEATURE_COLUMNS if F[c].std() == 0]
    const_tk = sorted({t for t, g in F.groupby("ticker") if any(g[c].std() == 0 for c in FEATURE_COLUMNS[:6])})
    chk("feature_ranges", rsi.between(0, 100).all() and per_t_vz.min() >= 0.97 and not const_cols and not const_tk,
        rsi_min=float(rsi.min()), rsi_max=float(rsi.max()), vol_z_frac_in_pm5_worst_ticker=float(per_t_vz.min()),
        worst_ticker=str(per_t_vz.idxmin()), constant_columns=const_cols, constant_tickers=const_tk)
    close = P.pivot(index="date", columns="ticker", values="close").sort_index()
    nxt = close[tick].shift(-1) / close[tick] - 1
    ex = nxt.sub(nxt.mean(axis=1), axis=0)
    rng = np.random.default_rng(0)
    sample = T.iloc[rng.choice(len(T), 20, replace=False)]
    cal, bad = list(close.index), []
    for _, r in sample.iterrows():
        d, t = r["date"], r["ticker"]
        tdate = cal[cal.index(d) + 1]
        ok = (d < tdate) and np.isclose(r["excess_return"], ex.loc[d, t], atol=1e-12)
        fr = F[(F["date"] == d) & (F["ticker"] == t)]
        if len(fr):
            i = cal.index(d)
            ok = ok and np.isclose(fr.iloc[0]["f1_logret"], np.log(close.loc[d, t] / close.loc[cal[i - 1], t]), atol=1e-12)
        if not ok:
            bad.append([str(d.date()), t])
    chk("alignment_20_rows", not bad, failures=bad)
    dm = T.groupby("date")["excess_return"].mean().abs()
    chk("targets_demeaned", dm.max() < 1e-9, max_abs_mean=float(dm.max()))
    cnt = G.groupby(["date", "edge_type"]).size().unstack(fill_value=0)
    sec = dict(zip(inc["ticker"], inc["sector"]))
    exp_sec = sum(c * (c - 1) for c in pd.Series(list(sec.values())).value_counts())
    exp_pairs = {(tick.index(a), tick.index(b)) for a in tick for b in tick if a != b and sec[a] == sec[b]}
    g0 = G[G["date"] == G["date"].iloc[0]]
    got_pairs = set(zip(g0[g0.edge_type == 2]["src"], g0[g0.edge_type == 2]["dst"]))
    sector_const = G[G.edge_type == 2].groupby("date").apply(
        lambda x: frozenset(zip(x["src"], x["dst"])), include_groups=False).nunique() == 1
    chk("graph_structure", (cnt[0] == N * k_pos).all() and (cnt[1] == N * k_neg).all() and (cnt[2] == exp_sec).all()
        and got_pairs == exp_pairs and sector_const and (G["src"] != G["dst"]).all(),
        pos_per_date=int(N * k_pos), neg_per_date=int(N * k_neg), sector_edges=int(exp_sec), dates=int(G["date"].nunique()))
    per_t_f, per_t_t = F.groupby("ticker").size(), T.groupby("ticker").size()
    chk("row_counts", len(F) == int(per_t_f.sum()) and len(T) == int(per_t_t.sum()) and per_t_f.nunique() == 1
        and per_t_t.nunique() == 1 and len(P) == int(P.groupby("ticker").size().sum()),
        features_rows=len(F), targets_rows=len(T), prices_rows=len(P))
    if legacy_dir is not None:
        cal_ok = {n: bool(set(df["date"]) == set(pd.read_parquet(legacy_dir / f"{n}.parquet", columns=["date"])["date"]))
                  for n, df in (("prices", P), ("features", F), ("targets", T), ("graphs_baseline", G))}
        chk("calendar_identical_to_legacy_cache", all(cal_ok.values()), per_file=cal_ok)
    R["checks"], R["passed"] = checks, all(c["pass"] for c in checks.values())
    (out / "validation_report.json").write_text(json.dumps(R, indent=2))
    return R


# ───────────────────────── driver ─────────────────────────

def write_core(out: Path, cfg: dict, prices: pd.DataFrame):
    prices.to_parquet(out / "prices.parquet", index=False)
    targets = build_targets(prices, cfg)
    targets.to_parquet(out / "targets.parquet", index=False)
    feats = build_features(prices, cfg)
    feats.to_parquet(out / "features.parquet", index=False)
    g = build_daily_graph_all_dates(prices, cfg)
    g.to_parquet(out / "graphs_baseline.parquet", index=False)
    print(f"[build] prices={len(prices):,} targets={len(targets):,} features={len(feats):,} graph_rows={len(g):,}")


def write_hashes(out: Path) -> dict:
    h = {n: sha256_of_file(out / n) for n in CACHE_PARQUETS}
    h["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    (out / "hashes.json").write_text(json.dumps(h, indent=2, sort_keys=True))
    return h


def lib_versions() -> dict:
    import importlib.metadata as md
    v = {"python": sys.version.split()[0]}
    for p in ("pandas", "numpy", "pyarrow", "yfinance", "requests", "lxml", "pandas_ta", "PyYAML"):
        try:
            v[p] = md.version(p)
        except md.PackageNotFoundError:
            pass
    return v


def run_legacy40(out: Path, work: Path, compare: Path, prices_from: Path | None):
    cfg = make_cfg(list(LEGACY40), LEGACY40)
    out.mkdir(parents=True, exist_ok=True)
    if prices_from is not None:
        prices = pd.read_parquet(prices_from / "prices.parquet")
        src = f"legacy prices from {prices_from}"
    else:
        frames = {t: fetch_yf(t, work) for t in cfg["universe"]["tickers"] + GLOBALS}
        for t, f in frames.items():
            if f is None or f.empty:
                raise RuntimeError(f"yfinance returned no data for {t!r}")
        prices = _align_and_validate_prices(_to_prices_long(frames))
        src = "fresh yfinance download"
    write_core(out, cfg, prices)
    R = compare_caches(out, compare)
    R["source"] = src
    R["price_revision_vs_legacy"] = price_revision_stats(out, compare)
    U = pd.DataFrame({"ticker": list(LEGACY40), "sector": list(LEGACY40.values()), "included": True})
    validate_cache(out, cfg, U, compare)
    (out / "regression_test_legacy40.json").write_text(json.dumps(R, indent=2))
    print("regression passed:", R["passed"], "| targets:", R["targets"], "| graphs:", R["graphs"])
    return R


def run_ndx(out: Path, work: Path, compare: Path | None, regression_json: Path | None):
    out.mkdir(parents=True, exist_ok=True)
    U, frames = build_ndx_universe(work, out)
    inc = U[U["included"]]
    cfg = make_cfg(inc["ticker"].tolist(), dict(zip(inc["ticker"], inc["sector"])))
    prices = _align_and_validate_prices(_to_prices_long(frames))
    write_core(out, cfg, prices)
    V = validate_cache(out, cfg, U, compare)
    print({k: v["pass"] for k, v in V["checks"].items()})
    if regression_json is not None and regression_json.exists():
        (out / "regression_test_legacy40.json").write_text(regression_json.read_text())
    h = write_hashes(out)
    manifest = {
        "universe": "ndx100", "n_candidates": int(len(U)), "n_included": int(len(inc)),
        "n_excluded": int((~U["included"]).sum()),
        "exclusions": U[~U["included"]][["ticker", "exclusion_reason"]].to_dict("records"),
        "snapshot_fetched_utc": str(pd.read_csv(out / "universe_raw_snapshot.csv")["fetched_utc"].iloc[0]),
        "sector_classification": "Nasdaq stock screener `sector` (sector) and `industry` (sub_sector); the spec's GICS/ICB "
                                 "snapshot-table columns are unavailable because Wikipedia's Nasdaq-100 page no longer "
                                 "has a components table. Constituent list: Nasdaq index endpoint.",
        "sector_counts": inc["sector"].value_counts().to_dict(),
        "sub_sector_count": int(inc["sub_sector"].nunique()),
        "date_range": [START, END], "global_series": GLOBALS,
        "graph_baseline": GRAPH, "library_versions": lib_versions(),
        "ported_from": {"data.py/features.py": "fd8125d", "graph.py": "175768d (tag phase3-complete)",
                        "universe screening, regression, validation": "a58dd21 (tag archive-before-ndx)"},
        "limitations": ["Survivorship bias: today's Nasdaq-100 members are used back to 2015; the index adds winners and "
                        "removes losers, so this favours survivors more strongly than the S&P 100 universe did.",
                        "Full-history rule drops recent listings, so N is well below 100.",
                        "Prices are Yahoo auto-adjusted closes; Yahoo revises history, so rebuilds can differ slightly.",
                        "Dollar volume (share-class rule) is computed from adjusted prices."],
        "hashes_generated_at": h["generated_at"],
    }
    (out / "build_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (out / "dataset-metadata.json").write_text(json.dumps({
        "title": "dmrgt-ndx-cache", "id": "harishsivakumar1912/dmrgt-ndx-cache",
        "licenses": [{"name": "other"}], "isPrivate": True}, indent=2))
    if not V["passed"]:
        print("VALIDATION FAILED — cache must not be published", file=sys.stderr)
        sys.exit(1)
    allowed = set(CACHE_PARQUETS) | {"universe.csv", "universe_raw_snapshot.csv", "regression_test_legacy40.json",
                                     "validation_report.json", "build_manifest.json", "hashes.json", "dataset-metadata.json"}
    extra = sorted(set(p.name for p in out.iterdir()) - allowed)
    assert not extra, f"unexpected files in cache folder: {extra}"
    print(f"[build] cache ready in {out} ({len(inc)} stocks)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", choices=["legacy40", "ndx100"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", default=None, help="legacy cache dir (regression reference / calendar check)")
    ap.add_argument("--prices-from", default=None, help="legacy40 only: build from this cache's prices.parquet instead of downloading")
    ap.add_argument("--regression-json", default=None, help="ndx100: copy this regression result into the cache folder")
    ap.add_argument("--work", default=None, help="scratch dir for raw downloads (default /tmp/dmrgt_build_work/<universe>)")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    work = Path(a.work or f"/tmp/dmrgt_build_work/{a.universe}")
    work.mkdir(parents=True, exist_ok=True)
    cmp_ = Path(a.compare).resolve() if a.compare else None
    if a.universe == "legacy40":
        assert cmp_ is not None, "--compare is required in legacy40 mode"
        R = run_legacy40(out, work, cmp_, Path(a.prices_from).resolve() if a.prices_from else None)
        sys.exit(0 if R["passed"] else 2)
    run_ndx(out, work, cmp_, Path(a.regression_json) if a.regression_json else None)


if __name__ == "__main__":
    main()
