"""Slimmed-down data module for Kaggle. Only load_config + all_tickers."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str = "configs.yaml") -> dict:
    """Read the YAML config from disk and return it as a Python dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def all_tickers(cfg: dict) -> list[str]:
    """The 20 universe tickers plus the 2 global series."""
    return list(cfg["universe"]["tickers"]) + list(cfg["universe"]["global_series"])


if __name__ == "__main__":
    print(load_config())
