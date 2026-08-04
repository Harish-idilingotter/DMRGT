"""Environment-aware path resolution.

Every module in this project imports paths from here. No module may build a
path by hand. This is the single source of truth for where data lives in
local development and on Kaggle.

The Kaggle branch is detected by the presence of /kaggle/input — that path
is mounted by Kaggle whenever a notebook has a dataset attached. On Kaggle,
/kaggle/input is read-only; everything we generate goes to /kaggle/working.
"""

from __future__ import annotations

import os
from pathlib import Path


def is_kaggle() -> bool:
    """True when running inside a Kaggle notebook environment."""
    return os.path.exists("/kaggle/input")


if is_kaggle():
    # Kaggle: input dataset is read-only, working dir is writable.
    CACHE_DIR: Path = Path("/kaggle/input/dmrgt-cache")
    OUTPUT_DIR: Path = Path("/kaggle/working")
    # On Kaggle we never create the read-only input dir.
else:
    ROOT: Path = Path(__file__).resolve().parent
    CACHE_DIR = ROOT / "cache"
    OUTPUT_DIR = ROOT / "artifacts"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Canonical file locations. Add new ones here — never inline a path elsewhere.
PRICES: Path = CACHE_DIR / "prices.parquet"
FEATURES: Path = CACHE_DIR / "features.parquet"
TARGETS: Path = CACHE_DIR / "targets.parquet"
HASHES_JSON: Path = CACHE_DIR / "hashes.json"

# Local-only artifacts (downloaded from Kaggle runs).
PREDICTIONS: Path = OUTPUT_DIR / "predictions.parquet"
TRAIN_METRICS: Path = OUTPUT_DIR / "train_metrics.json"
RUN_MANIFEST: Path = OUTPUT_DIR / "run_manifest.json"
CHECKPOINTS_DIR: Path = OUTPUT_DIR / "checkpoints"


def describe() -> dict[str, str]:
    """Return a dict of the resolved paths, for logging."""
    return {
        "env": "kaggle" if is_kaggle() else "local",
        "CACHE_DIR": str(CACHE_DIR),
        "OUTPUT_DIR": str(OUTPUT_DIR),
        "PRICES": str(PRICES),
        "FEATURES": str(FEATURES),
        "TARGETS": str(TARGETS),
        "HASHES_JSON": str(HASHES_JSON),
        "PREDICTIONS": str(PREDICTIONS),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(describe(), indent=2))
