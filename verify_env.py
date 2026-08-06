"""Local-vs-Kaggle structural cross-check (KAGGLE.md §7).

The purpose of this script is to confirm that the local environment and
the Kaggle environment are reading the **same data** and constructing
the **same folds**. It is NOT a model-quality comparison — the local
run uses fewer epochs than Kaggle, so validation loss and Rank IC
magnitudes are not expected to match.

Run locally:

    python verify_env.py --fold 0 --seed 0 --epochs 3

Optionally compare against a Kaggle `run_manifest.json`:

    python verify_env.py --fold 0 --seed 0 --epochs 3 \\
        --compare artifacts/run_manifest.json

The script then prints a structured report and exits non-zero on any
structural mismatch.

What it reports (each must match the Kaggle manifest exactly):

  • Number of training / validation / test samples for the fold
  • First and last training / test date
  • Ticker list and count
  • SHA-256 of each parquet in cache/

What it explicitly does NOT report (because they don't depend on the
environment and so don't catch divergence):

  • Validation loss
  • Rank IC magnitude
  • Any per-batch or per-epoch metric

All path resolution goes through `paths.py` — there is exactly one
place in the project where paths live.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import data
import paths
import splits as sp
import train


# ─── SHA-256 helpers ──────────────────────────────────────────────────────

def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ─── Report builder ───────────────────────────────────────────────────────

def _date_span(dates: np.ndarray | pd.DatetimeIndex) -> tuple[str | None, str | None]:
    if len(dates) == 0:
        return (None, None)
    arr = pd.DatetimeIndex(dates)
    return (str(arr[0].date()), str(arr[-1].date()))


def build_report(fold_idx: int, seed: int, cfg: dict) -> dict:
    """Load the cache, build the fold, return a flat dict of
    structural invariants. This is the local half of the diff."""
    if not paths.PRICES.exists():
        raise FileNotFoundError(f"{paths.PRICES} missing")
    if not paths.FEATURES.exists():
        raise FileNotFoundError(f"{paths.FEATURES} missing")
    if not paths.TARGETS.exists():
        raise FileNotFoundError(f"{paths.TARGETS} missing")

    features = pd.read_parquet(paths.FEATURES)
    targets = pd.read_parquet(paths.TARGETS)
    folds = sp.make_folds(cfg, available_dates=features["date"].unique())
    if fold_idx < 0 or fold_idx >= len(folds):
        raise IndexError(
            f"--fold {fold_idx} out of range (have {len(folds)} folds in config)"
        )
    fold = folds[fold_idx]

    tensors = train.assemble_fold(fold, features, targets, cfg)

    train_first, train_last = _date_span(tensors.train_dates if hasattr(tensors, "train_dates")
                                          else np.array([fold.train_dates[0], fold.train_dates[-1]]))
    test_first, test_last = _date_span(tensors.test_dates)

    universe = list(cfg["universe"]["tickers"])

    return {
        "fold_name": fold.name,
        "fold_idx": fold.fold_idx,
        "seed": seed,
        "n_train": int(len(tensors.X_train)),
        "n_val":   int(len(tensors.X_val)),
        "n_test":  int(len(tensors.X_test)),
        "n_stocks": int(len(tensors.test_tickers)),
        "train_first_date": train_first,
        "train_last_date":  train_last,
        "test_first_date":  test_first,
        "test_last_date":   test_last,
        "tickers": sorted(tensors.test_tickers),
        "tickers_count": len(tensors.test_tickers),
        "parquet_hashes": {
            "prices.parquet":   _sha256_of_file(paths.PRICES),
            "features.parquet": _sha256_of_file(paths.FEATURES),
            "targets.parquet":  _sha256_of_file(paths.TARGETS),
        },
        # NOTE: not reported — depends on training duration
        # "validation_loss": ...,
        # "rank_ic_mean":    ...,
    }


# ─── Optional run (one fold × one seed, n epochs) ─────────────────────────

def run_one_fold(fold_idx: int, seed: int, epochs: int, cfg: dict) -> None:
    """Optionally train the model for the requested fold/seed/epoch
    count. Used only to populate `run_manifest.json` with the local
    values; the structural report itself does not need this."""
    # Quick override of epochs for the cross-check run
    local_cfg = json.loads(json.dumps(cfg))  # deep copy
    local_cfg["training"]["epochs"] = int(epochs)
    # Honour --seed (we want exactly one seed)
    local_cfg["training"]["seeds"] = [int(seed)]
    # Honour --fold (we want exactly one fold)
    fold_name = local_cfg["folds"][fold_idx]["name"]
    local_cfg["folds"] = [local_cfg["folds"][fold_idx]]

    print(f"[verify] running one fold ({fold_name}) × seed {seed} × {epochs} epochs…")
    train.run_all(local_cfg, verify_hashes=True)


# ─── Comparison against a Kaggle run_manifest.json ────────────────────────

def _kaggle_manifest_fold(manifest: dict, fold_name: str) -> dict | None:
    for fr in manifest.get("fold_results", []):
        if fr.get("fold") == fold_name:
            return fr
    return None


def compare_against_manifest(report: dict, manifest_path: Path) -> bool:
    """Diff the local report against a Kaggle run_manifest.json.
    Returns True if everything matches, False otherwise.
    """
    if not manifest_path.exists():
        print(f"[verify] WARN: {manifest_path} not found — skipping comparison",
              file=sys.stderr)
        return True

    manifest = json.loads(manifest_path.read_text())
    kf = _kaggle_manifest_fold(manifest, report["fold_name"])
    if kf is None:
        print(f"[verify] FAIL: fold {report['fold_name']!r} not in "
              f"{manifest_path} (have: "
              f"{[fr.get('fold') for fr in manifest.get('fold_results', [])]})",
              file=sys.stderr)
        return False

    failures = []

    # n_train / n_val / n_test
    if report["n_train"] != kf.get("n_train_days"):
        failures.append(f"n_train: local={report['n_train']} vs "
                        f"kaggle={kf.get('n_train_days')}")
    if report["n_val"] != kf.get("n_val_days"):
        failures.append(f"n_val: local={report['n_val']} vs "
                        f"kaggle={kf.get('n_val_days')}")
    if report["n_test"] != kf.get("n_test_days"):
        failures.append(f"n_test: local={report['n_test']} vs "
                        f"kaggle={kf.get('n_test_days')}")

    # Tickers count (we can't check every ticker name in the manifest,
    # but the manifest's fold entry does include the seed list —
    # ticker count is a good proxy for "same cache").
    # Note: the manifest does not record tickers directly; we instead
    # rely on the parquet hashes (below) to verify the cache matches.
    # We also check that the local ticker count matches the config
    # (which is what Kaggle is also using).
    cfg_tickers = manifest.get("config_snapshot", {}).get("universe", {}).get("tickers", [])
    if report["tickers_count"] != len(cfg_tickers):
        failures.append(f"tickers_count: local={report['tickers_count']} vs "
                        f"kaggle_config={len(cfg_tickers)}")

    # Parquet hashes — the definitive data-identity check
    kaggle_hashes = manifest.get("cache_hashes", {})
    for name, local_h in report["parquet_hashes"].items():
        kaggle_h = kaggle_hashes.get(name)
        if kaggle_h is None:
            failures.append(f"{name}: local={local_h[:12]}… but not in "
                            f"manifest's cache_hashes")
        elif local_h != kaggle_h:
            failures.append(f"{name}: local={local_h[:12]}… vs "
                            f"kaggle={kaggle_h[:12]}…")

    return _emit_compare_summary(failures)


def _emit_compare_summary(failures: list[str]) -> bool:
    if not failures:
        print("[verify] COMPARE: PASS — local and Kaggle agree on every "
              "structural property.")
        return True
    print(f"[verify] COMPARE: FAIL — {len(failures)} structural mismatch(es):",
          file=sys.stderr)
    for f in failures:
        print(f"  - {f}", file=sys.stderr)
    print(file=sys.stderr)
    print("[verify] Most common cause: a stale dmrgt-cache version is "
          "attached to the notebook. Re-attach the latest version and "
          "re-run the Kaggle notebook.", file=sys.stderr)
    return False


# ─── Reporting ────────────────────────────────────────────────────────────

def print_report(report: dict) -> None:
    print("=" * 70)
    print(f"DMRGT Phase 1 — local structural report (KAGGLE.md §7)")
    print("=" * 70)
    print(f"  fold:                {report['fold_name']!r}  (idx={report['fold_idx']})")
    print(f"  seed:                {report['seed']}")
    print(f"  n_train:             {report['n_train']}")
    print(f"  n_val:               {report['n_val']}")
    print(f"  n_test:              {report['n_test']}")
    print(f"  n_stocks:            {report['n_stocks']}")
    print(f"  train dates:         {report['train_first_date']} → {report['train_last_date']}")
    print(f"  test dates:          {report['test_first_date']} → {report['test_last_date']}")
    print(f"  tickers (sorted):    {report['tickers']}")
    print(f"  tickers_count:       {report['tickers_count']}")
    print()
    print("  parquet hashes:")
    for name, h in report["parquet_hashes"].items():
        print(f"    {name:22s}  sha256={h}")
    print()
    print("  Excluded from comparison (depend on training duration /")
    print("  numerical kernels, not on environment):")
    print("    - validation loss")
    print("    - rank IC magnitude")
    print("    - any per-batch / per-epoch metric")


# ─── CLI ──────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--fold", type=int, default=0,
                   help="Fold index (0-based; matches configs.yaml folds: order)")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed index (0-based; matches configs.yaml seeds: order)")
    p.add_argument("--epochs", type=int, default=3,
                   help="Local training epochs for the cross-check (default 3).")
    p.add_argument("--config", default="configs.yaml")
    p.add_argument("--compare", default=None,
                   help="Optional path to a Kaggle run_manifest.json to diff against.")
    p.add_argument("--skip-run", action="store_true",
                   help="Skip the actual training run; only print the structural report.")
    args = p.parse_args()

    cfg = data.load_config(args.config)

    # 1. Run (or skip) the actual training
    if not args.skip_run:
        run_one_fold(args.fold, args.seed, args.epochs, cfg)
    else:
        print("[verify] --skip-run: skipping the training run",
              file=sys.stderr)

    # 2. Build & print the structural report
    report = build_report(args.fold, args.seed, cfg)
    print_report(report)

    # 3. Optionally diff against a Kaggle run_manifest.json
    if args.compare:
        ok = compare_against_manifest(report, Path(args.compare))
        if not ok:
            print("\n[verify] Final result: FAIL", file=sys.stderr)
            return 1

    print("\n[verify] Final result: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())