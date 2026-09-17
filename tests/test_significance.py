"""§6 of the single-session rework — the in-session significance API.

stats.run_tests_from_predictions() is what notebook Cell 6 calls (and
compare.py re-runs locally as a cross-check), so its shape is pinned
here: the JSON keys the spec names (wilcoxon / bootstrap_sharpe /
winning_graph_config / conclusion), the fields the conclusion string
interpolates (wins, n_pairs, p_value, bootstrap_p_value), and the
(fold, seed) pairing that makes the Wilcoxon test honest.

The frames are synthetic but scored through the REAL evaluation
pipeline (eval.evaluate_predictions on the real targets file), because
the helpers must consume exactly what the training stage produces.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data
import paths
import stats


@pytest.fixture(scope="module")
def cfg() -> dict:
    return data.load_config(str(paths.CONFIG))


FOLDS = ["fold_1", "fold_2", "fold_3", "fold_4"]
FOLD_YEARS = [2019, 2020, 2021, 2022]  # disjoint, like the real folds
SEEDS = [0, 1]
N_DATES = 3


def _arm_frame(cfg: dict, model: str, skill: float, seed_shift: int = 0) -> pd.DataFrame:
    """One arm's TEST-split predictions with a given directional skill.

    ``skill`` scales the signal in y_pred: 1.0 tracks the target, 0.0
    is noise. Built on real dates/tickers from the cached targets so the
    backtest's raw-return recovery works; one fold per calendar year so
    the folds' date ranges are disjoint, like the real ones.
    """
    targets = pd.read_parquet(paths.TARGETS)
    tickers = list(cfg["universe"]["tickers"])
    all_dates = sorted(targets["date"].unique())
    rng = np.random.default_rng(0 + seed_shift)
    rows = []
    for fold, year in zip(FOLDS, FOLD_YEARS):
        dates = [d for d in all_dates if pd.Timestamp(d).year == year][:N_DATES]
        assert len(dates) == N_DATES, f"{year}: not enough dates"
        wide = (targets.pivot(index="date", columns="ticker",
                              values="excess_return")
                .reindex(index=dates, columns=tickers))
        for seed in SEEDS:
            noise = rng.standard_normal(wide.shape) * (1.0 - skill)
            pred = skill * wide.to_numpy() + noise
            for i, d in enumerate(dates):
                for j, t in enumerate(tickers):
                    rows.append({
                        "model": model, "fold": fold, "seed": seed,
                        "split": "test", "date": pd.Timestamp(d),
                        "ticker": t,
                        "y_pred": float(pred[i, j]),
                        "y_true": float(wide.iloc[i, j]),
                    })
    return pd.DataFrame(rows)


def test_run_tests_from_predictions_shape_and_conclusion(cfg):
    """The spec's exact JSON shape, and the conclusion interpolating
    wins / n_pairs / p_value / bootstrap_p_value."""
    dyn = _arm_frame(cfg, "dmrgt_all3_grid_baseline", skill=1.0)
    fro = _arm_frame(cfg, "dmrgt_frozen", skill=0.0)

    out = stats.run_tests_from_predictions(
        dyn, fro, cfg,
        winning_graph_config={"k_pos": 8, "k_neg": 4, "corr_window": 60})

    assert set(out) == {"wilcoxon", "bootstrap_sharpe",
                        "winning_graph_config", "conclusion"}
    wil, boot = out["wilcoxon"], out["bootstrap_sharpe"]
    for key in ("wins", "n_pairs", "p_value"):
        assert key in wil
    for key in ("bootstrap_p_value", "p_one_sided_dynamic_greater",
                "observed_sharpe_difference", "ci_95"):
        assert key in boot
    assert out["winning_graph_config"] == {
        "k_pos": 8, "k_neg": 4, "corr_window": 60}

    # The skilled arm wins every pair; conclusion reflects the numbers.
    assert wil["n_pairs"] == len(FOLDS) * len(SEEDS)
    assert wil["wins"] == wil["n_pairs"]
    assert wil["p_value"] <= wil["alpha"]
    assert out["conclusion"] == (
        f"Dynamic graph rank IC exceeds frozen in {wil['wins']}/"
        f"{wil['n_pairs']} folds (p={wil['p_value']:.3f}); "
        f"Sharpe difference p={boot['bootstrap_p_value']:.3f}"
    )
    # JSON round-trips (the notebook writes exactly this dict).
    json.loads(json.dumps(out))


def test_per_fold_seed_ic_pairs_on_fold_seed_keys(cfg):
    """Both arms must produce the same (fold, seed) key set — that is
    what makes the pairing honest."""
    dyn = stats.per_fold_seed_ic(_arm_frame(cfg, "dyn", skill=1.0), cfg)
    fro = stats.per_fold_seed_ic(_arm_frame(cfg, "fro", skill=0.0), cfg)
    assert set(dyn) == {(f, s) for f in FOLDS for s in SEEDS}
    assert set(dyn) == set(fro)
    # The skilled arm's ICs dominate pairwise.
    assert all(dyn[k] > fro[k] for k in dyn)


def test_wilcoxon_test_aligns_keyed_inputs(cfg):
    """Keyed dicts are aligned by (fold, seed) — a missing key in one
    arm must be refused, not silently dropped."""
    dyn = stats.per_fold_seed_ic(_arm_frame(cfg, "dyn", skill=1.0), cfg)
    fro = stats.per_fold_seed_ic(_arm_frame(cfg, "fro", skill=0.0), cfg)
    del fro[next(iter(fro))]  # drop one pair
    with pytest.raises(AssertionError, match="unpaired"):
        stats.wilcoxon_test(dynamic_ic=dyn, frozen_ic=fro, cfg=cfg)


def test_test_rows_refuses_multi_arm_frames(cfg):
    """A frame holding both arms' rows must be refused before any test
    statistic is computed — arms are evaluated one at a time."""
    mixed = pd.concat([_arm_frame(cfg, "dyn", skill=1.0),
                       _arm_frame(cfg, "fro", skill=0.0)])
    with pytest.raises(AssertionError, match="one arm"):
        stats._test_rows(mixed)


def test_daily_returns_pools_chronologically(cfg):
    """One series per arm, seed-averaged per fold, folds in date order."""
    dyn = _arm_frame(cfg, "dyn", skill=1.0)
    ret = stats.daily_returns(dyn, cfg)
    assert ret.index.is_monotonic_increasing
    assert len(ret) == N_DATES * len(FOLDS)  # seed-averaging collapses seeds


def test_stationary_bootstrap_deterministic_and_paired():
    """Same seed → identical replicate diffs; the two arms see the SAME
    index draws (paired), which is what makes the difference isolable."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal(120) * 0.01
    y = rng.standard_normal(120) * 0.01
    d1 = stats._stationary_bootstrap_diffs(x, y, 10, 50, seed=7)
    d2 = stats._stationary_bootstrap_diffs(x, y, 10, 50, seed=7)
    assert d1.shape == (50,) and np.array_equal(d1, d2)
    idx = stats._stationary_bootstrap_indices(120, 10, 200, seed=7)
    # Every replicate used the same indices for both arms — build diffs
    # by hand from the drawn indices and compare with the function's.
    with np.errstate(divide="ignore", invalid="ignore"):
        hand = (stats._sharpe_rows(x[idx]) - stats._sharpe_rows(y[idx]))
    assert np.allclose(stats._stationary_bootstrap_diffs(x, y, 10, 200, seed=7),
                       hand)


def test_hand_rolled_bootstrap_agrees_with_arch_on_decisive_case():
    """Local insurance: the numpy implementation of the Politis–Romano
    scheme must reach the same verdict as arch's StationaryBootstrap on
    a case with a decisive difference (dyn = fro + drift). Skipped where
    arch isn't installed — the in-session path never needs it."""
    arch_bs = pytest.importorskip("arch.bootstrap")
    rng = np.random.default_rng(3)
    n = 251
    fro = rng.standard_normal(n) * 0.01
    dyn = fro + 0.002
    mine = stats.bootstrap_sharpe_test(
        pd.Series(dyn), pd.Series(fro),
        {"significance": {"bootstrap_iterations": 2000,
                          "bootstrap_block_length": 10,
                          "wilcoxon_alpha": 0.05}}, seed=7)
    from arch.bootstrap import StationaryBootstrap

    def _sharpe(v):
        return float(np.mean(v) / np.std(v, ddof=1) * np.sqrt(252))

    bs = arch_bs.StationaryBootstrap(10, dyn, fro, seed=7)
    diffs = bs.apply(lambda a, b: _sharpe(a) - _sharpe(b), reps=2000).ravel()
    p_arch = float(np.mean(diffs <= 0))
    assert mine["bootstrap_p_value"] < 0.05
    assert p_arch < 0.05