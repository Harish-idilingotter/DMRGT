"""Driver for the ported cache pipeline. Calls the VERBATIM functions in src/:

    data.download_prices  → prices.parquet
    data.build_targets    → targets.parquet
    features.build_features → features.parquet
    graph.build_daily_graph_all_dates → graphs_baseline.parquet

and writes hashes.json. Nothing here re-implements a pipeline step; the only
adaptations are in the data source (an optional pre-fetched yfinance shim for
the S&P 100 build, so that ≤2-day gaps can be forward-filled per spec §2.3)
and the printed-summary sector lookup (see PORTING_LOG.md).

  DMRGT_BUILD_CACHE=<out dir> DMRGT_BUILD_CONFIG=<yaml> python build.py [--yf-cache DIR] [--relax-sector-cap]
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd

import paths      # noqa: E402  (needs the env vars)
import data       # noqa: E402
import features   # noqa: E402
import graph      # noqa: E402


def install_yf_shim(yf_cache: Path):
    """Make `yf.download(ticker, ...)` return the pre-fetched, screened frames.
    Same return shape as yfinance (Date index, Price/Ticker MultiIndex columns)."""
    import yfinance as yf

    def cached_download(t, start=None, end=None, auto_adjust=True, progress=False, actions=False, **kw):
        with open(yf_cache / f"{t}.pkl", "rb") as f:
            return pickle.load(f)

    yf.download = cached_download


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yf-cache", default=None)
    ap.add_argument("--relax-sector-cap", action="store_true",
                    help="S&P 100 only: skip the 40-stock design guard that no sector exceeds 20%% of the universe")
    a = ap.parse_args()

    cfg = data.load_config(str(paths.CONFIG))
    if a.yf_cache:
        install_yf_shim(Path(a.yf_cache))
    if a.relax_sector_cap:
        data._sector_lookup = lambda c: dict(c["sectors"])    # printed summary only

    prices = data.download_prices(cfg)
    data.update_hashes(paths.HASHES_JSON)

    targets = data.build_targets(prices, cfg)
    targets.to_parquet(paths.TARGETS, index=False)
    print(f"[build] wrote {paths.TARGETS}  rows={len(targets):,}")

    feats = features.build_features(prices, cfg)
    feats.to_parquet(paths.FEATURES, index=False)
    print(f"[build] wrote {paths.FEATURES}  rows={len(feats):,}")

    g = graph.build_daily_graph_all_dates(prices, cfg)
    g.to_parquet(paths.GRAPHS, index=False)
    print(f"[build] wrote {paths.GRAPHS}  rows={len(g):,}  dates={g['date'].nunique()}")

    h = paths.update_hashes()
    print("[build] hashes:", {k: (v[:12] if k != 'generated_at' else v) for k, v in h.items()})


if __name__ == "__main__":
    main()
