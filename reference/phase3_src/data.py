"""DMRGT Phase 3 — deliberately minimal data module (§4.2).

The cache already contains prices, features, targets, and the dynamic
graph; Phase 3 never regenerates any of them. This file exists only so
train.py and graph.py have a config loader on both local and Kaggle
sides.

By design it contains NO download logic and NO regeneration logic, so it
can never fetch fresh data from inside a Kaggle notebook (Rule 9).
"""

from __future__ import annotations

import yaml


def load_config(path: str = "configs.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def all_tickers(cfg: dict) -> list[str]:
    """Universe tickers plus the global series (^GSPC, ^VIX)."""
    return list(cfg["universe"]["tickers"]) + list(cfg["universe"]["global_series"])