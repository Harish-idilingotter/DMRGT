"""Environment-aware path resolution + cache hash management.

Detects local vs Kaggle via /kaggle/input. Every module imports paths from
here — no hardcoded paths anywhere (Rule 8).

Hash utilities (sha256_of_file / read_hashes / update_hashes) live here
rather than data.py because Phase 3's data.py is deliberately minimal
(§4.2): it holds only the config loader, so it can never fetch or
regenerate data from inside a Kaggle notebook (Rule 9).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


KAGGLE_INPUT = Path("/kaggle/input")


def is_kaggle() -> bool:
    """True when running inside a Kaggle notebook environment."""
    return KAGGLE_INPUT.exists()


def _resolve_kaggle_dataset(slug: str) -> Path:
    """Locate an attached Kaggle dataset. Tries the known mount layouts,
    then falls back to rglob. Raises with a directory listing on failure.

    Observed layouts:
      /kaggle/input/<slug>
      /kaggle/input/datasets/<slug>
      /kaggle/input/datasets/<username>/<slug>   <- current default
    """
    for c in (KAGGLE_INPUT / slug,
              KAGGLE_INPUT / "datasets" / slug):
        if c.is_dir():
            return c
    # /kaggle/input/datasets/<username>/<slug> — username is not known
    # ahead of time, so glob for the slug one level down.
    user_scoped = [c for c in KAGGLE_INPUT.glob(f"datasets/*/{slug}") if c.is_dir()]
    if len(user_scoped) == 1:
        return user_scoped[0]
    if len(user_scoped) > 1:
        raise RuntimeError(
            f"Ambiguous: multiple datasets named '{slug}':\n  "
            + "\n  ".join(str(m) for m in user_scoped)
        )

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
    CACHE_DIR: Path = _resolve_kaggle_dataset("dmrgt-phase-3-cache")
    CODE_DIR: Path = _resolve_kaggle_dataset("dmrgt-phase-3-code")
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
GRAPHS: Path = CACHE_DIR / "graphs.parquet"
# The frozen edge file is an OUTPUT, not cache (§4 of the single-session
# rework): it is built in-session by the notebook's frozen-graph cell from
# the attached prices.parquet and written to the WRITABLE working directory
# — never part of the read-only attached cache dataset. This is what removes
# the mid-run dataset-version-and-reattach cycle: dmrgt_frozen reads its
# edges from here, so no stage of the run needs a re-upload.
GRAPHS_FROZEN: Path = OUTPUT_DIR / "graphs_frozen.parquet"
HASHES_JSON: Path = CACHE_DIR / "hashes.json"
CONFIG: Path = CODE_DIR / "configs.yaml"

# Sensitivity-grid variants (§6b) — built locally by sensitivity.py,
# uploaded inside the dmrgt-phase-3-cache dataset, read by the Kaggle grid session.
#
# They sit in the cache ROOT, not in a `graph_variants/` subdirectory.
# `kaggle datasets version -p cache` defaults to `--dir-mode skip`, which
# SILENTLY omits subdirectories: the upload reports success, the files are
# simply absent, and the grid session then dies on a missing variant with
# nothing pointing back at the upload as the cause. Flattened, the plain
# command is the correct one — no --dir-mode flag, no unzip step on
# Kaggle — and the local and Kaggle layouts are identical, so there is a
# single path expression to keep right.
#
# Names are graphs_<config>.parquet, which cannot collide with the five
# canonical cache files above.
VARIANT_PREFIX: str = "graphs_"
# Matches the prefix but is not a grid cell. graphs_frozen.parquet is an
# output (built in-session, GRAPHS_FROZEN above) — this guard keeps a
# stale copy sitting in CACHE_DIR from ever being mistaken for a variant.
_NON_VARIANT_GRAPHS: frozenset[str] = frozenset({"graphs_frozen.parquet"})


def graph_variant_name(path: Path) -> str:
    """`.../graphs_kpos6.parquet` → `"kpos6"` (the §6b variant name)."""
    return Path(path).stem[len(VARIANT_PREFIX):]


def graph_variant_path(name: str) -> Path:
    """Path of the §6b variant `name` — e.g. "baseline", "kpos6", "win45"."""
    return CACHE_DIR / f"{VARIANT_PREFIX}{name}.parquet"


def graph_variant_files() -> list[Path]:
    """Every §6b variant present in the cache root, sorted by filename.

    Discovered rather than hardcoded: `configs.yaml [graph_grid]` is the
    source of truth for which sweep cells exist, and sensitivity.py writes
    exactly those. An empty list means none have been built yet.
    """
    return [p for p in sorted(CACHE_DIR.glob(f"{VARIANT_PREFIX}*.parquet"))
            if p.name not in _NON_VARIANT_GRAPHS]

# Outputs.
#
# Deliberately NO bare `predictions.parquet` / `train_metrics.json`
# constants (§6.1 of the known-issues doc). A Phase 1 file with exactly
# that name once sat in artifacts/ and got read by name in preference to
# the current phase's per-model files, producing a schema error far from
# the cause. Every Phase 3 output name is therefore qualified — by model
# and, for grid runs, by variant:
#   predictions_{model_name}.parquet          train.py
#   train_metrics_{model_name}.json           train.py
#   predictions_dmrgt_all3_grid_{variant}.…   §6b grid runs
# and every locally-written artifact carries a `phase3_` prefix
# (compare.py / sensitivity.py). Do not reintroduce a bare-name
# constant here.
RUN_MANIFEST: Path = OUTPUT_DIR / "run_manifest.json"
CHECKPOINTS_DIR: Path = OUTPUT_DIR / "checkpoints"

# The cache data files whose hashes are verified before every training
# run (§3.2 of the single-session rework). Single source of truth — used
# by train.py, graph.py, verify_env.py and the notebook so they cannot
# disagree. The counting guard in verify_cache_hashes iterates this
# tuple, so adding a file here automatically extends the hash check.
#
# The 7 grid variants are named EXPLICITLY here, not globbed: the hash
# gate must demand exactly the sweep cells configs.yaml declares, so a
# cache upload missing one variant halts in Cell 1 instead of dying two
# hours into the grid. The layout is FLAT — variants live directly in
# CACHE_DIR, never a subfolder (kaggle datasets version's default
# --dir-mode skip silently drops subdirectories; see the §6b note above).
#
# graphs_frozen.parquet is deliberately NOT in this list — it is built
# in-session to /kaggle/working/ (GRAPHS_FROZEN above), never part of the
# read-only attached cache dataset.
GRID_VARIANT_NAMES: tuple[str, ...] = (
    "baseline", "kpos6", "kpos10", "kneg3", "kneg5", "win45", "win90",
)
CACHE_PARQUETS: tuple[str, ...] = (
    "prices.parquet",
    "features.parquet",
    "targets.parquet",
    "graphs.parquet",
    *(f"graphs_{name}.parquet" for name in GRID_VARIANT_NAMES),
)
assert len(CACHE_PARQUETS) == 11, (
    f"CACHE_PARQUETS must be 11 (4 base + 7 flat grid variants), "
    f"got {len(CACHE_PARQUETS)}"
)


# ─── cache hash management ──────────────────────────────────────────────

def sha256_of_file(path: Path) -> str:
    """SHA-256 hex digest of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_hashes(hashes_path: Path = HASHES_JSON) -> dict:
    """Read the existing hashes.json. Returns {} if absent or unreadable."""
    if not hashes_path.exists():
        return {}
    try:
        return json.loads(hashes_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def update_hashes(hashes_path: Path = HASHES_JSON) -> dict:
    """Compute SHA-256 of every cached parquet and merge into hashes.json,
    preserving entries this call did not touch (e.g. rebuilding only the
    graphs must not drop the prices.parquet entry).

    CACHE_PARQUETS already names the 7 grid variants (flat layout), so
    one call describes the entire cache — there is no second location to
    remember to pass in, and no way for a caller to update hashes.json
    while leaving the variants unhashed. graphs_frozen.parquet is NOT
    hashed here: it is an output built in-session (GRAPHS_FROZEN), never
    part of the attached cache. Adds a `generated_at` timestamp.
    """
    from datetime import datetime, timezone

    hashes = read_hashes(hashes_path)
    for p in [CACHE_DIR / n for n in CACHE_PARQUETS]:
        if p.exists():
            hashes[p.name] = sha256_of_file(p)
    hashes["generated_at"] = (
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    hashes_path.write_text(json.dumps(hashes, indent=2, sort_keys=True))
    return hashes


def describe() -> dict:
    """Diagnostic snapshot printed by notebook Cell 1."""
    return {
        "environment": "kaggle" if is_kaggle() else "local",
        "CACHE_DIR":   str(CACHE_DIR),
        "CODE_DIR":    str(CODE_DIR),
        "OUTPUT_DIR":  str(OUTPUT_DIR),
        "CONFIG":      str(CONFIG),
        "config_exists":    CONFIG.exists(),
        "cache_exists":     CACHE_DIR.exists(),
        "cache_contents":   sorted(p.name for p in CACHE_DIR.iterdir())
                            if CACHE_DIR.exists() else [],
        "prices_exists":    PRICES.exists(),
        "features_exists":  FEATURES.exists(),
        "targets_exists":   TARGETS.exists(),
        "graphs_exists":    GRAPHS.exists(),
        "graphs_frozen_exists": GRAPHS_FROZEN.exists(),
        "graph_variants":   [p.name for p in graph_variant_files()],
        "hashes_exists":    HASHES_JSON.exists(),
    }


if __name__ == "__main__":
    print(json.dumps(describe(), indent=2))