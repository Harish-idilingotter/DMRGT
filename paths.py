"""Environment-aware path resolution.

Detects local vs Kaggle via /kaggle/input. Every module imports paths from
here — no hardcoded paths anywhere.
"""

from __future__ import annotations

import os
from pathlib import Path


KAGGLE_INPUT = Path("/kaggle/input")


def is_kaggle() -> bool:
    """True when running inside a Kaggle notebook environment."""
    return KAGGLE_INPUT.exists()


def _resolve_kaggle_dataset(slug: str) -> Path:
    """Locate an attached Kaggle dataset. Tries known mount layouts, then
    falls back to rglob. Raises with directory listing on failure."""
    for c in (KAGGLE_INPUT / slug,
              KAGGLE_INPUT / "datasets" / slug):
        if c.is_dir():
            return c

    matches = [p for p in KAGGLE_INPUT.rglob(slug) if p.is_dir()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous: multiple directories named '{slug}':\n  "
            + "\n  ".join(str(m) for m in matches)
        )

    listing = sorted(str(p) for p in KAGGLE_INPUT.rglob("*"))[:40]
    raise FileNotFoundError(
        f"Kaggle dataset '{slug}' not found under {KAGGLE_INPUT}.\n"
        f"Attach it via Add Input, then RESTART THE SESSION — attaching "
        f"a dataset does not mount it into an already-running kernel.\n"
        f"Actual contents (first 40 entries):\n  " + "\n  ".join(listing)
    )


if is_kaggle():
    CACHE_DIR: Path = _resolve_kaggle_dataset("dmrgt-cache")
    CODE_DIR: Path = _resolve_kaggle_dataset("dmrgt-code")
    OUTPUT_DIR: Path = Path("/kaggle/working")
else:
    ROOT: Path = Path(__file__).resolve().parent
    CACHE_DIR = ROOT / "cache"
    CODE_DIR = ROOT
    OUTPUT_DIR = ROOT / "artifacts"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Inputs
PRICES: Path = CACHE_DIR / "prices.parquet"
FEATURES: Path = CACHE_DIR / "features.parquet"
TARGETS: Path = CACHE_DIR / "targets.parquet"
HASHES_JSON: Path = CACHE_DIR / "hashes.json"
CONFIG: Path = CODE_DIR / "configs.yaml"

# Outputs
PREDICTIONS: Path = OUTPUT_DIR / "predictions.parquet"
TRAIN_METRICS: Path = OUTPUT_DIR / "train_metrics.json"
RUN_MANIFEST: Path = OUTPUT_DIR / "run_manifest.json"
CHECKPOINTS_DIR: Path = OUTPUT_DIR / "checkpoints"

# The three data files whose hashes are verified. Single source of
# truth — used by train.py and the notebook so they cannot disagree.
CACHE_PARQUETS: tuple[str, ...] = (
    "prices.parquet",
    "features.parquet",
    "targets.parquet",
)


def describe() -> dict:
    """Diagnostic snapshot printed by notebook Cell 1."""
    return {
        "environment": "kaggle" if is_kaggle() else "local",
        "CACHE_DIR":   str(CACHE_DIR),
        "CODE_DIR":    str(CODE_DIR),
        "OUTPUT_DIR":  str(OUTPUT_DIR),
        "CONFIG":      str(CONFIG),
        "config_exists":  CONFIG.exists(),
        "cache_exists":   CACHE_DIR.exists(),
        "cache_contents": sorted(p.name for p in CACHE_DIR.iterdir())
                          if CACHE_DIR.exists() else [],
        "prices_exists":   PRICES.exists(),
        "features_exists": FEATURES.exists(),
        "targets_exists":  TARGETS.exists(),
        "hashes_exists":   HASHES_JSON.exists(),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(describe(), indent=2))
