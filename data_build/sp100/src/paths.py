"""Build-time paths for the data_build pipeline (the ONLY file in src/ that is
not a verbatim copy; it replaces Phase 2/3 paths.py, whose Kaggle/local
resolution is irrelevant here).

Everything the ported modules read from `paths` is defined: PRICES, FEATURES,
TARGETS, GRAPHS, HASHES_JSON, CONFIG, GRAPHS_FROZEN, CACHE_PARQUETS,
update_hashes. Output location comes from the environment so the same code
builds the 40-stock regression cache and the S&P 100 cache.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

CACHE_DIR: Path = Path(os.environ["DMRGT_BUILD_CACHE"]).resolve()
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR: Path = CACHE_DIR
CONFIG: Path = Path(os.environ["DMRGT_BUILD_CONFIG"]).resolve()

PRICES: Path = CACHE_DIR / "prices.parquet"
FEATURES: Path = CACHE_DIR / "features.parquet"
TARGETS: Path = CACHE_DIR / "targets.parquet"
GRAPHS: Path = CACHE_DIR / "graphs_baseline.parquet"   # Phase 3: graphs.parquet == graphs_baseline.parquet
GRAPHS_FROZEN: Path = CACHE_DIR / "graphs_frozen.parquet"   # never built here
HASHES_JSON: Path = CACHE_DIR / "hashes.json"
CACHE_PARQUETS: tuple[str, ...] = ("prices.parquet", "features.parquet", "targets.parquet",
                                   "graphs_baseline.parquet")


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def update_hashes(hashes_path: Path = HASHES_JSON) -> dict:
    from datetime import datetime, timezone
    hashes = {}
    for n in CACHE_PARQUETS:
        if (CACHE_DIR / n).exists():
            hashes[n] = sha256_of_file(CACHE_DIR / n)
    hashes["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    hashes_path.write_text(json.dumps(hashes, indent=2, sort_keys=True))
    return hashes
