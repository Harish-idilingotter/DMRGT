"""DMRGT Phase 1 — data pipeline.

python data.py --download   # Step 1: raw prices (idempotent)
python data.py --targets    # Step 2: next-day excess returns
python data.py --features   # Step 4: 8 raw, unscaled features
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import paths

# features.py is introduced in Step E. The import lives inside the
# --features branch (where it is actually used) so that Pylance and
# other static analysers don't flag an unresolved import before the
# module exists, and so that --download and --targets have zero
# dependency on features.py.



def load_config(path: str = "configs.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def all_tickers(cfg: dict) -> list[str]:
    """Universe tickers plus the global series (^GSPC, ^VIX)."""
    return list(cfg["universe"]["tickers"]) + list(cfg["universe"]["global_series"])



def sha256_of_file(path: Path) -> str:
    """SHA-256 hex digest of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_hashes(hashes_path: Path) -> dict:
    """Read the existing hashes.json (returns {} if absent or unreadable).
    Each stage UPDATES this dict rather than overwriting it — running
    `data.py --features` alone must not drop the prices.parquet entry
    written by an earlier `--download`."""
    import json
    if not hashes_path.exists():
        return {}
    try:
        return json.loads(hashes_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def update_hashes(hashes_path: Path,
                  parquet_paths: list[Path] | None = None) -> dict:
    """Compute SHA-256 of each existing parquet and merge into
    `hashes.json`, preserving entries for files this stage did not
    touch. Adds a `generated_at` timestamp.

    Every parquet that exists in CACHE_DIR at the end of the stage
    must have a hash entry — see `_verify_hashes_complete`.
    """
    import json
    from datetime import datetime, timezone

    if parquet_paths is None:
        parquet_paths = [paths.PRICES, paths.FEATURES, paths.TARGETS, paths.GRAPHS]

    hashes = _read_hashes(hashes_path)
    for p in parquet_paths:
        if p.exists():
            hashes[p.name] = sha256_of_file(p)
    hashes["generated_at"] = (
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    hashes_path.write_text(json.dumps(hashes, indent=2, sort_keys=True))
    return hashes


def _verify_hashes_complete(hashes_path: Path) -> None:
    """Hard assertion: every parquet currently in CACHE_DIR has a
    matching entry in hashes.json. A failure here means a stage wrote
    a partial file and dropped an entry for a parquet that still
    exists on disk — that would be a divergence bug."""
    import json
    hashes = json.loads(hashes_path.read_text())
    missing = []
    for p in [paths.PRICES, paths.FEATURES, paths.TARGETS, paths.GRAPHS]:
        if p.exists() and p.name not in hashes:
            missing.append(p.name)
    assert not missing, (
        f"hashes.json is missing entries for: {missing}. "
        f"This means a stage of data.py wrote a partial hashes.json "
        f"without preserving existing entries. Re-run the stages in "
        f"order: --download, --targets, --features."
    )



def download_prices(cfg: dict) -> pd.DataFrame:
    """Download OHLCV, align to common calendar, cache as long-format Parquet.
    Idempotent: returns cached file if date range and ticker set match."""
    tickers = all_tickers(cfg)
    start = cfg["universe"]["start_date"]
    end = cfg["universe"]["end_date"]

    # ── idempotency check ────────────────────────────────────────────
    if paths.PRICES.exists():
        cached = pd.read_parquet(paths.PRICES)
        cached_hash = sha256_of_file(paths.PRICES)
        cached_min = cached["date"].min()
        cached_max = cached["date"].max()
        # Allow up to 5 trading days of slack at each end so market
        # holidays (e.g. 2015-01-01, 2024-12-31) don't trigger re-downloads.
        slack = pd.tseries.offsets.BDay(5)
        start_deadline = pd.Timestamp(start) + slack
        end_deadline = pd.Timestamp(end) - slack
        covers_window = (cached_min <= start_deadline) and (cached_max >= end_deadline)
        tickers_match = set(cached["ticker"].unique()) == set(tickers)
        if covers_window and tickers_match:
            print(f"[data] cache hit: {paths.PRICES}")
            print(f"[data] sha256: {cached_hash}")
            print(f"[data] rows: {len(cached):,}  tickers: {cached['ticker'].nunique()}")
            print(f"[data] date range: {cached_min.date()} → {cached_max.date()}")
            return cached
        print(f"[data] cache exists but stale — re-downloading "
              f"(covers_window={covers_window}, tickers_match={tickers_match})")

    # ── download ─────────────────────────────────────────────────────
    import yfinance as yf
    print(f"[data] downloading {len(tickers)} series from {start} → {end}")

    frames = []
    for t in tickers:
        # auto_adjust=True folds splits/dividends into the close so
        # the price series is what we actually want for returns.
        df = yf.download(
            t,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            actions=False,
        )
        if df.empty:
            raise RuntimeError(f"yfinance returned no data for {t!r}")
        # yfinance sometimes returns a MultiIndex columns frame for a
        # single ticker; flatten to plain columns.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        # yfinance 0.2.x returns "price" instead of "close" sometimes;
        # guard against that.
        rename = {}
        for c in df.columns:
            if c == "price":
                rename[c] = "close"
        if rename:
            df = df.rename(columns=rename)
        df = df.reset_index()
        # yfinance uses "Date" or "Datetime" depending on version.
        date_col = "Date" if "Date" in df.columns else (
            "Datetime" if "Datetime" in df.columns else df.columns[0]
        )
        df = df.rename(columns={date_col: "date"})
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df["ticker"] = t
        # Keep only the columns we need.
        keep = ["date", "ticker", "open", "high", "low", "close", "volume"]
        df = df[[c for c in keep if c in df.columns]]
        if "volume" not in df.columns:
            df["volume"] = np.nan
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)
    raw = raw.sort_values(["ticker", "date"]).reset_index(drop=True)

    # ── calendar alignment ──────────────────────────────────────────
    # Master calendar = intersection of all tickers' date indices.
    # For each ticker we drop dates where ANY other ticker is missing.
    # This guarantees every ticker has the same number of rows and
    # identical date indices.
    per_ticker_dates = {t: set(d["date"]) for t, d in raw.groupby("ticker")}
    common_dates = set.intersection(*per_ticker_dates.values()) \
        if per_ticker_dates else set()
    if not common_dates:
        raise RuntimeError("No common trading dates across all tickers")
    print(f"[data] master calendar: {len(common_dates)} trading days")

    aligned = (raw[raw["date"].isin(common_dates)]
               .sort_values(["date", "ticker"])
               .reset_index(drop=True))

    # ── validation ──────────────────────────────────────────────────
    counts = aligned.groupby("ticker").size()
    assert counts.nunique() == 1, (
        f"Calendar alignment failed — tickers have differing row counts: "
        f"{counts.to_dict()}"
    )
    n_dates = int(counts.iloc[0])
    assert aligned["date"].is_unique == False or aligned.duplicated(
        subset=["date", "ticker"]).sum() == 0, "duplicate (date, ticker) rows"
    assert aligned.isna().sum().sum() == 0, (
        f"NaNs present after alignment:\n{aligned.isna().sum()}"
    )
    # Plausibility: ~252 trading days per calendar year over 2015-2024
    # is ~2510 days, but the intersection of all 22 series is a bit
    # smaller. Anything in [1500, 2600] is plausible.
    assert 1500 <= n_dates <= 2600, f"Implausible row count per ticker: {n_dates}"

    # ── save ────────────────────────────────────────────────────────
    aligned.to_parquet(paths.PRICES, index=False)

    # ── summary table ───────────────────────────────────────────────
    print("\n[data] validation summary:")
    summary_rows = []
    sectors = _sector_lookup(cfg)
    for t in tickers:
        sub = aligned[aligned["ticker"] == t]
        summary_rows.append({
            "ticker": t,
            "sector": sectors.get(t, "global"),
            "first": str(sub["date"].min().date()),
            "last": str(sub["date"].max().date()),
            "rows": len(sub),
            "NaNs": int(sub.isna().sum().sum()),
        })
    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False))

    h = sha256_of_file(paths.PRICES)
    print(f"\n[data] wrote {paths.PRICES}  sha256={h}")
    # Hashes for the cache are centralised in hashes.json (updated by
    # update_hashes() in main() after this returns); we don't write a
    # separate prices.sha256 sidecar — two sources of truth will
    # eventually disagree (see KAGGLE.md §7).

    return aligned


def _sector_lookup(cfg: dict) -> dict[str, str]:
    """Map ticker → GICS sector. Reads from configs.yaml sectors: block
    so there is one source of truth shared with graph.py."""
    sectors_cfg = cfg.get("sectors", {})
    if sectors_cfg:
        # Validate: every universe ticker must have a sector entry
        universe = list(cfg["universe"]["tickers"])
        missing = [t for t in universe if t not in sectors_cfg]
        if missing:
            raise ValueError(
                f"configs.yaml sectors: block missing entries for: {missing}"
            )
        # Validate: no sector holds more than 20% of the universe (§3.1)
        from collections import Counter
        counts = Counter(sectors_cfg[t] for t in universe)
        n = len(universe)
        for sector, count in counts.items():
            pct = count / n
            if pct > 0.20:
                raise ValueError(
                    f"Sector {sector!r} holds {pct:.1%} of the universe "
                    f"({count}/{n}), exceeding the 20% limit (§3.1)."
                )
        return dict(sectors_cfg)
    # Fallback for Phase 1 compatibility (should not be hit in Phase 2)
    return {
        "AAPL": "Information Technology", "MSFT": "Information Technology",
        "JPM":  "Financials",             "BRK-B": "Financials", "GS": "Financials",
        "JNJ":  "Health Care",            "PFE":   "Health Care", "UNH": "Health Care",
        "XOM":  "Energy",                 "CVX":   "Energy",
        "AMZN": "Consumer Discretionary", "HD":    "Consumer Discretionary",
        "NKE":  "Consumer Discretionary",
        "PG":   "Consumer Staples",       "KO":    "Consumer Staples",
        "HON":  "Industrials",            "CAT":   "Industrials", "BA": "Industrials",
        "NEE":  "Utilities",              "DUK":   "Utilities",
        "^GSPC": "global", "^VIX": "global",
    }



def build_targets(prices: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Next-day excess return: daily_return[t+1] - cross_sectional_mean[t+1].
    Label at date t describes what happens on t+1. Final row dropped."""
    # Resolve which tickers belong to the universe vs the global
    # series (^GSPC, ^VIX). When cfg is provided we use the configured
    # ticker list; otherwise we infer "global" by the leading caret.
    if cfg is not None:
        universe_tickers = list(cfg["universe"]["tickers"])
    else:
        universe_tickers = [t for t in prices["ticker"].unique() if not t.startswith("^")]

    # 1) daily returns per ticker
    px = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    rets = px.pct_change()  # row t = return from t-1 to t

    # 2) cross-sectional mean return per date — universe only, NOT globals
    universe = rets[universe_tickers]
    mean_ret = universe.mean(axis=1)

    # 3) excess return, then SHIFT so label at date t = excess at t+1
    excess = universe.sub(mean_ret, axis=0)
    # After this, excess.loc[date, ticker] = return[t] - mean[t].
    # We want the label at date t to describe what happens on t+1:
    # label[t] = excess[t+1]. Use shift(-1).
    labels = excess.shift(-1)

    # 4) last row has no label; drop it explicitly.
    n_before = len(labels)
    labels = labels.dropna(how="all")
    n_after = len(labels)
    print(f"[data] targets: dropped {n_before - n_after} row(s) with no t+1 label")

    # 5) reshape to long
    long = (labels.stack(future_stack=True)
                  .rename("excess_return")
                  .reset_index()
                  .rename(columns={"level_1": "ticker"}))
    long = long[["date", "ticker", "excess_return"]]
    return long



def main():
    parser = argparse.ArgumentParser(description="DMRGT Phase 1 data pipeline")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--download", action="store_true", help="Step 1: download + cache prices")
    g.add_argument("--targets",  action="store_true", help="Step 2: build targets.parquet")
    g.add_argument("--features", action="store_true", help="Step 4: build features.parquet (Step E)")
    args = parser.parse_args()

    cfg = load_config()

    if args.download:
        download_prices(cfg)
        # Update hashes.json so it reflects the new prices file even if
        # targets/features don't exist yet (entries for those are
        # preserved from any previous run).
        update_hashes(paths.HASHES_JSON)
        _verify_hashes_complete(paths.HASHES_JSON)
        return

    if args.targets:
        if not paths.PRICES.exists():
            print("[data] prices.parquet missing — run --download first", file=sys.stderr)
            sys.exit(1)
        prices = pd.read_parquet(paths.PRICES)
        targets = build_targets(prices, cfg)
        targets.to_parquet(paths.TARGETS, index=False)
        print(f"[data] wrote {paths.TARGETS}  rows={len(targets):,}")
        update_hashes(paths.HASHES_JSON)
        _verify_hashes_complete(paths.HASHES_JSON)
        return

    if args.features:
        if not paths.PRICES.exists():
            print("[data] prices.parquet missing — run --download first", file=sys.stderr)
            sys.exit(1)
        # Lazy import — features.py is introduced in Step E. Importing
        # here keeps --download / --targets independent of features.py
        # and silences the static-analyser warning about an unresolved
        # top-level import.
        try:
            from features import build_features, FEATURE_COLUMNS
        except ImportError:
            print("[data] features.py not yet implemented (Step E)", file=sys.stderr)
            sys.exit(1)
        prices = pd.read_parquet(paths.PRICES)
        features = build_features(prices, cfg)
        features.to_parquet(paths.FEATURES, index=False)
        print(f"[data] wrote {paths.FEATURES}  rows={len(features):,}")
        print(f"[data] features: {FEATURE_COLUMNS}")
        update_hashes(paths.HASHES_JSON)
        _verify_hashes_complete(paths.HASHES_JSON)
        return


if __name__ == "__main__":
    main()
