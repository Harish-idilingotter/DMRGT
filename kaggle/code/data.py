"""Slimmed-down data module for the Kaggle training notebook.

On Kaggle, the only thing we need from ``data.py`` is the
``load_config`` function (reads ``configs.yaml``). The price/target/
features pipelines all run LOCALLY before the cache is uploaded; the
Kaggle notebook only loads the pre-built Parquet files.

This file is a copy of the relevant part of the main-directory
``data.py`` so the notebook can import it without needing the rest of
the data pipeline.
"""

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
