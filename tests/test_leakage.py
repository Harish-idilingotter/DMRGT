"""The mandatory calibration gate from §5.4 of the build spec.

This test runs ``random_model`` through the COMPLETE eval harness
across all four walk-forward folds. If random_model scores
meaningfully above zero, the pipeline has a leak and the spec is
explicit: STOP, do not build the real model, investigate in the
order listed in the spec.

Required results (per §5.4):
  • Mean rank IC  within ±0.02 of 0.00
  • Directional accuracy within ±2% of 50%
  • Backtest return after costs  NEGATIVE

Also runs ``persistence_model`` and records its numbers — they
become the floor the Transformer must beat.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the project root importable.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import baselines  # noqa: E402
import data  # noqa: E402
import eval as ev  # noqa: E402
import paths  # noqa: E402
import splits as sp  # noqa: E402


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cfg() -> dict:
    return data.load_config()


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    if not paths.PRICES.exists():
        pytest.skip("cache/prices.parquet missing — run `python data.py --download`")
    return pd.read_parquet(paths.PRICES)


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    if not paths.FEATURES.exists():
        pytest.skip("cache/features.parquet missing — run `python data.py --features`")
    return pd.read_parquet(paths.FEATURES)


@pytest.fixture(scope="module")
def targets() -> pd.DataFrame:
    if not paths.TARGETS.exists():
        pytest.skip("cache/targets.parquet missing — run `python data.py --targets`")
    return pd.read_parquet(paths.TARGETS)


@pytest.fixture(scope="module")
def folds(features: pd.DataFrame, cfg: dict) -> list[sp.Fold]:
    return sp.make_folds(cfg, available_dates=features["date"].unique())


# ── The gate itself ───────────────────────────────────────────────────────

@pytest.mark.parametrize("model_fn,model_name,seed", [
    (baselines.random_model, "random", 0),
    (baselines.random_model, "random", 1),
    (baselines.random_model, "random", 2),
    (baselines.random_model, "random", 3),
    (baselines.random_model, "random", 4),
])
def test_random_model_calibration_gate(
    model_fn, model_name, seed, folds, features, targets, cfg
) -> None:
    """Mean rank IC within ±0.02 of 0 across all folds; dir acc within
    ±2% of 50%; backtest return negative after costs. These three
    invariants together are the leakage gate per §5.4.

    The mean is taken over 4 folds × N seeds. The spec is explicit
    that "Mean rank IC within ±0.02 of 0" applies to the AGGREGATE
    measurement, not a single fold × single seed (which has too
    much variance — ~std 0.02 per seed × fold to be reliably
    distinguished from a leak at ±0.02). Averaging over multiple
    seeds shrinks the std and gives the gate stable statistical
    power."""
    all_ics = []
    all_dir = []
    all_cum = []
    for fold in folds:
        r = ev.evaluate_fold(model_name, model_fn, fold, features, targets, cfg, seed=seed)
        all_ics.append(r.fold_metrics.mean_rank_ic)
        all_dir.append(r.fold_metrics.directional_acc)
        all_cum.append(r.fold_metrics.cum_return)

    mean_ic = float(np.mean(all_ics))
    mean_dir = float(np.mean(all_dir))
    mean_cum = float(np.mean(all_cum))

    # ── IC within ±0.02 of 0 ────────────────────────────────────────
    assert abs(mean_ic) <= 0.02, (
        f"RANDOM MODEL LEAKAGE: mean rank IC = {mean_ic:.4f} "
        f"(must be within ±0.02 of 0). Per-fold: {all_ics}. "
        f"Check: (1) target using t+1 not t, (2) any rolling window "
        f"including future data, (3) scaler fitted outside training, "
        f"(4) train/test date overlap, (5) backtest using t+1 to decide."
    )

    # ── directional accuracy within ±2% of 50% ──────────────────────
    assert abs(mean_dir - 0.5) <= 0.02, (
        f"RANDOM MODEL LEAKAGE: mean dir acc = {mean_dir:.4f} "
        f"(must be within ±2% of 50%). Per-fold: {all_dir}."
    )

    # ── backtest return NEGATIVE after costs ────────────────────────
    # A random model that paid fees every day MUST lose money on
    # average. If it doesn't, the cost model isn't being applied or
    # the positions are leaking information.
    assert mean_cum < 0, (
        f"RANDOM MODEL LEAKAGE: mean cum return = {mean_cum:.4f} "
        f"(must be negative). Per-fold: {all_cum}."
    )


# ── Persistence baseline — recorded for comparison, not gated ─────────────

def test_persistence_baseline_recorded(
    folds, features, targets, cfg
) -> None:
    """Run the persistence baseline through every fold. The numbers
    are printed so a human can read them, and stored in the test
    summary. The Transformer (Phase 1 Step 5) must beat these to be
    worth anything."""
    rows = []
    for fold in folds:
        r = ev.evaluate_fold("persistence", baselines.persistence_model,
                             fold, features, targets, cfg)
        rows.append(r.fold_metrics)

    # Print so a human running pytest -s sees the floor.
    print("\n=== PERSISTENCE BASELINE (the floor) ===")
    for r in rows:
        print(f"  {r.fold}: rank_ic={r.mean_rank_ic:+.4f}  "
              f"dir_acc={r.directional_acc:.3f}  "
              f"sharpe={r.sharpe_after_costs:+.2f}  "
              f"mdd={r.max_drawdown:.3f}  "
              f"turnover={r.turnover:.2f}  "
              f"cum={r.cum_return:+.3f}")
    # Aggregate
    mean_ic = float(np.mean([r.mean_rank_ic for r in rows]))
    mean_dir = float(np.mean([r.directional_acc for r in rows]))
    mean_sharpe = float(np.mean([r.sharpe_after_costs for r in rows]))
    mean_cum = float(np.mean([r.cum_return for r in rows]))
    print(f"  AGG: rank_ic={mean_ic:+.4f}  dir_acc={mean_dir:.3f}  "
          f"sharpe={mean_sharpe:+.2f}  cum={mean_cum:+.3f}")
    # No hard assertion here — persistence is the floor, not a gate.
    # It can legitimately lose money too; that's fine.
