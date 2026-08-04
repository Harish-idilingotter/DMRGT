"""DMRGT Phase 1 — feature engineering (Step E stub).

The real implementation lands in Step E (8 features, one at a time,
plotted and inspected). This stub exists so that Pylance and other
static analysers can resolve the import in data.py without warnings,
and so that `python data.py --features` fails with a clear message
until the real implementation arrives.

When Step E is implemented, this stub is replaced with the real
build_features() function and the FEATURE_COLUMNS list (8 names:
f1_logret, f2_price_ma, f3_rsi, f4_macd, f5_atr, f6_vol_zscore,
f7_market_ret, f8_vix).
"""

from __future__ import annotations

import pandas as pd


FEATURE_COLUMNS: list[str] = [
    "f1_logret",
    "f2_price_ma",
    "f3_rsi",
    "f4_macd",
    "f5_atr",
    "f6_vol_zscore",
    "f7_market_ret",
    "f8_vix",
]


def build_features(prices: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Stub — implemented in Step E."""
    raise NotImplementedError(
        "build_features is implemented in Step E of the DMRGT Phase 1 "
        "build. See DMRGT-Phase1-Build-Spec.md §6."
    )
