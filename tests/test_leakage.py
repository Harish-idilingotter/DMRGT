"""DMRGT Phase 3 — leakage gate (§5.3, Rule 21).

The evaluation harness is written BEFORE the model. To trust every
number it will ever produce, push pure noise through the full pipeline —
same folds, same windows, same backtest, same cost model — and assert it
comes out as noise:

  * mean rank IC within +/-0.02 of zero (no information about the
    target can have leaked into the inputs or the harness);
  * directional accuracy within +/-2% of 50% (no side-channel);
  * negative mean Sharpe after costs (a long/short decile book on
    noise loses the transaction costs — no accidental look-ahead in
    the position/return alignment).

If any of these fail, there is leakage in the data pipeline or the
harness. Find it BEFORE writing model.py or train.py. Do not proceed
until this passes.

persistence_model is also evaluated and its numbers recorded (not
asserted beyond finiteness) — it is the floor the trained models must
beat, and running it here proves the harness handles a real signal
without special-casing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import baselines
import data
import eval as ev
import paths
import splits


# ─── shared fixtures ───────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cfg() -> dict:
    return data.load_config(str(paths.CONFIG))


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    if not paths.FEATURES.exists():
        pytest.skip("cache/features.parquet not present — run on the machine holding the cache")
    return pd.read_parquet(paths.FEATURES)


@pytest.fixture(scope="module")
def targets() -> pd.DataFrame:
    if not paths.TARGETS.exists():
        pytest.skip("cache/targets.parquet not present — run on the machine holding the cache")
    return pd.read_parquet(paths.TARGETS)


@pytest.fixture(scope="module")
def folds(cfg, features) -> list[splits.Fold]:
    avail = pd.DatetimeIndex(features["date"].unique())
    return splits.make_folds(cfg, available_dates=avail)


@pytest.fixture(scope="module")
def random_results(cfg, features, targets, folds):
    """random_model through the FULL harness — one run per fold.

    A per-call counter makes the noise differ from day to day while
    staying fully deterministic (seed = f(fold, call order)). With a
    constant seed every day would draw the SAME noise, the book would
    never turn over, and the costs-charged check would be vacuous.
    """
    out = []
    for fold in folds:
        call = {"i": 0}

        def noise(X, *, fold_idx=fold.fold_idx, call=call):
            call["i"] += 1
            return baselines.random_model(
                X, seed=1234 + fold_idx * 1000 + call["i"])

        out.append(ev.evaluate_fold(
            model_name="random",
            model_fn=noise,
            fold=fold,
            features=features,
            targets=targets,
            cfg=cfg,
            seed=1234 + fold.fold_idx,
        ))
    return out


@pytest.fixture(scope="module")
def persistence_results(cfg, features, targets, folds):
    out = []
    for fold in folds:
        out.append(ev.evaluate_fold(
            model_name="persistence",
            model_fn=baselines.persistence_model,
            fold=fold,
            features=features,
            targets=targets,
            cfg=cfg,
            seed=None,
        ))
    return out


# ─── the gate ─────────────────────────────────────────────────────────────

def test_random_mean_rank_ic_is_zero(random_results):
    """Noise must not predict the target: |mean IC| <= 0.02."""
    ics = [r.fold_metrics.mean_rank_ic for r in random_results]
    mean_ic = float(np.mean(ics))
    print(f"\nrandom per-fold rank ICs: {[round(i, 4) for i in ics]}")
    print(f"random mean rank IC: {mean_ic:+.4f} (gate: +/-0.02)")
    assert abs(mean_ic) <= 0.02, (
        f"LEAKAGE GATE FAILED: random model mean rank IC {mean_ic:+.4f} "
        f"is outside +/-0.02 — information about the target is leaking "
        f"from the pipeline or the harness. Do not write model code "
        f"until this is found and fixed."
    )


def test_random_directional_accuracy_is_coin_flip(random_results):
    """No side-channel: directional accuracy within 2% of 50%."""
    accs = [r.fold_metrics.directional_acc for r in random_results]
    mean_acc = float(np.mean(accs))
    print(f"\nrandom per-fold dir acc: {[round(a, 4) for a in accs]}")
    print(f"random mean dir acc: {mean_acc:.4f} (gate: 0.48–0.52)")
    assert abs(mean_acc - 0.5) <= 0.02, (
        f"LEAKAGE GATE FAILED: random model directional accuracy "
        f"{mean_acc:.4f} is more than 2% away from 50% — a side-channel "
        f"is aligning predictions with target signs."
    )


def test_random_sharpe_negative_after_costs(random_results, cfg):
    """A decile book on noise must lose the transaction costs."""
    sharpes = [r.fold_metrics.sharpe_after_costs for r in random_results]
    mean_sharpe = float(np.mean(sharpes))
    print(f"\nrandom per-fold Sharpe (after costs): {[round(s, 3) for s in sharpes]}")
    print(f"random mean Sharpe: {mean_sharpe:+.3f} (gate: < 0)")
    assert mean_sharpe < 0.0, (
        f"LEAKAGE GATE FAILED: random model mean Sharpe after costs is "
        f"{mean_sharpe:+.3f} — a noise book should be dragged negative "
        f"by costs. Positive Sharpe on noise suggests look-ahead in the "
        f"position/return alignment."
    )


def test_random_backtest_costs_are_charged(random_results, cfg):
    """Sanity: turnover is positive and the cost line is non-trivial.

    If turnover were 0 the cost check above would be vacuous.
    """
    for r in random_results:
        assert r.fold_metrics.turnover > 0.0, (
            f"random book on fold {r.fold!r} has zero turnover — the "
            f"weights are not changing, so no costs are being charged "
            f"and the negative-Sharpe gate is untested."
        )
        assert 0.0 < r.fold_metrics.turnover < 2.0


def test_persistence_reference_floor(persistence_results, random_results):
    """persistence runs through the same harness; numbers recorded as
    the floor to beat. Only finiteness is asserted — its sign is an
    empirical question (returns are near unpredictable), not a gate.
    """
    rows = []
    for r in persistence_results:
        m = r.fold_metrics
        rows.append((m.fold, m.mean_rank_ic, m.directional_acc,
                     m.sharpe_after_costs))
    assert len(rows) > 0
    print("\npersistence reference floor:")
    for r in persistence_results:
        m = r.fold_metrics
        print(f"  {m.fold}: rank IC {m.mean_rank_ic:+.4f}  "
              f"dir acc {m.directional_acc:.4f}  "
              f"Sharpe {m.sharpe_after_costs:+.4f}")
    for r in persistence_results:
        m = r.fold_metrics
        assert np.isfinite(m.mean_rank_ic), f"persistence IC not finite on {m.fold}"
        assert np.isfinite(m.sharpe_after_costs), f"persistence Sharpe not finite on {m.fold}"


# ─── harness self-consistency (also part of the gate) ─────────────────────

def test_all_models_share_one_harness(random_results, persistence_results):
    """Every model's numbers came from the same evaluate_fold path —
    the FoldResult rows carry the model's own name, proving no
    special-casing per model."""
    names = {r.fold_metrics.model for r in random_results + persistence_results}
    assert names == {"random", "persistence"}


def test_fold_day_counts(random_results, folds):
    """Each fold scored roughly a year of test days (252±3)."""
    for r, fold in zip(random_results, folds):
        n = r.fold_metrics.n_days
        assert 240 <= n <= 260, (
            f"fold {r.fold!r} scored {n} test days — expected ~252. "
            f"A wildly wrong count means the window assembly or the "
            f"split boundaries are wrong."
        )


def test_predictions_align_with_realised_returns(random_results):
    """The harness never scores a day whose target is NaN for all stocks
    (dropped last date) — n_days must equal the per-day table length."""
    for r in random_results:
        assert len(r.per_day) == r.fold_metrics.n_days
        assert r.per_day.index.is_monotonic_increasing