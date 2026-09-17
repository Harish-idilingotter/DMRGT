"""DMRGT Phase 3 — significance testing (§6 of the single-session rework).

Two tests, both on the FINAL strategy (identity post-processing applied
identically to both arms — Rule 19), run IN-SESSION by notebook Cell 6
against the prediction files already sitting in /kaggle/working/, and
re-run locally by compare.py as a cross-check:

  1. Wilcoxon signed-rank on the PAIRED (fold, seed) mean rank ICs —
     4 folds × 5 seeds = 20 paired observations per arm. This is the
     primary significance test on predictive power (alpha from
     configs.yaml [significance] wilcoxon_alpha).

  2. Stationary bootstrap (Politis–Romano) on the Sharpe DIFFERENCE
     between arms — a Sharpe difference could be driven by a handful of
     days, so the test statistic is resampled at the DAILY-return level
     with blocks, not at the run level.

     Implemented directly in numpy (see _stationary_bootstrap_diffs),
     not via arch: arch is not in Kaggle's base image and the session
     runs with Internet OFF, and Cell 6 runs IN-SESSION — an arch
     import there would kill a ~2.5-hour run at the very last step.
     Same scheme: expected block length 10, circular indexing, one
     index draw per replicate applied to BOTH arms (paired).

     Documented construction choice: for each (model, fold) the 5
     seeds' daily after-cost return series are AVERAGED (one series per
     fold), then the 4 folds are concatenated chronologically (the
     folds cover disjoint calendar years, so pooling preserves time
     order; block length 10 is far shorter than any fold, so the 3
     stitch points add negligible distortion). Both arms are resampled
     with the SAME index draws (paired bootstrap) so the difference
     isolates the arms, not the resample.

Reading test rows is legitimate ONLY here — every caller of this module
runs after sensitivity.select_winner has returned and the winner is
locked (§1: the winner's already-existing grid test rows become the
final dynamic-arm result, read for the first time only then).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon as scipy_wilcoxon

import data
import eval as ev
import paths


# ─── inputs assembled FROM prediction frames ───────────────────────────────

def _as_frame(pred) -> pd.DataFrame:
    """Accept a predictions DataFrame or a path to its parquet."""
    if isinstance(pred, pd.DataFrame):
        df = pred
    else:
        df = pd.read_parquet(pred)
    return df


def _test_rows(pred) -> pd.DataFrame:
    """The test-split rows of a predictions file/frame.

    Callers run strictly after winner selection is locked (§1), so this
    is the first and only place test rows are read.
    """
    df = _as_frame(pred)
    assert df["model"].nunique() == 1, (
        "a significance-test input must hold exactly one arm's rows"
    )
    out = df[df["split"] == "test"]
    assert len(out) > 0, "no test rows in the predictions input"
    return out


def per_fold_seed_ic(pred_df: pd.DataFrame, cfg: dict) -> dict:
    """{(fold, seed): mean rank IC} for one arm's TEST rows — the 20
    paired observations the Wilcoxon test consumes.

    Scores through the same evaluation pipeline as everything else
    (identity post-processing, the declared final configuration), so the
    ICs here are exactly the per-run mean_rank_ic values compare.py
    reports.
    """
    targets = pd.read_parquet(paths.TARGETS)
    runs = ev.evaluate_predictions(_test_rows(pred_df), targets, cfg,
                                   split=None)
    keyed = {(r.fold, r.seed): float(r.fold_metrics.mean_rank_ic)
             for r in runs}
    assert len(keyed) == len(runs), "duplicate (fold, seed) keys"
    return keyed


def daily_returns(pred_df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Pooled daily after-cost return series for one arm's TEST rows.

    Seed-averaged per (model, fold), then folds concatenated
    chronologically (by each series' first date — disjoint calendar
    years make the stitch points unambiguous).
    """
    targets = pd.read_parquet(paths.TARGETS)
    runs = ev.evaluate_predictions(_test_rows(pred_df), targets, cfg,
                                   split=None)
    by_fold: dict[str, list[pd.Series]] = {}
    for r in runs:
        by_fold.setdefault(r.fold, []).append(r.backtest_returns)
    per_fold = []
    for fold, series in by_fold.items():
        df = pd.concat(series, axis=1)
        assert df.index.is_monotonic_increasing, f"{fold}: unordered index"
        per_fold.append(df.mean(axis=1))
    per_fold.sort(key=lambda s: s.index[0])
    return pd.concat(per_fold)


# ─── the two tests (§6) ────────────────────────────────────────────────────

def _sharpe(daily_ret: np.ndarray) -> float:
    """Annualised Sharpe of a daily return series (risk-free 0)."""
    r = np.asarray(daily_ret, dtype=float)
    if r.size < 2:
        return float("nan")
    sd = float(np.std(r, ddof=1))
    return float(np.mean(r) / sd * np.sqrt(252)) if sd > 0 else 0.0


def _sharpe_rows(returns: np.ndarray) -> np.ndarray:
    """Annualised Sharpe of each row of a [reps, n] return matrix."""
    sd = returns.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = returns.mean(axis=1) / sd * np.sqrt(252)
    return np.where(sd > 0, s, 0.0)


def _stationary_bootstrap_indices(n: int, block: int, reps: int,
                                  seed: int) -> np.ndarray:
    """Politis–Romano stationary bootstrap index draws, shape [reps, n].

    Each replicate starts at a uniform index; every subsequent index
    follows the previous one (t+1 mod n — circular, as in the original
    scheme) with probability 1 − 1/block, else jumps to a fresh uniform
    index. Expected block length is therefore ``block``.
    """
    if not (1 <= block <= n):
        raise ValueError(f"block length {block} out of range for n={n}")
    rng = np.random.default_rng(seed)
    out = np.empty((reps, n), dtype=np.int64)
    cur = rng.integers(0, n, size=reps)
    out[:, 0] = cur
    p_follow = 1.0 - 1.0 / float(block)
    for t in range(1, n):
        follow = rng.random(reps) < p_follow
        cur = np.where(follow, (cur + 1) % n, rng.integers(0, n, size=reps))
        out[:, t] = cur
    return out


def _stationary_bootstrap_diffs(dyn: np.ndarray, fro: np.ndarray,
                                block: int, reps: int, seed: int) -> np.ndarray:
    """Paired stationary bootstrap on the Sharpe difference, [reps].

    One index draw per replicate, applied to BOTH arms — the replicate
    isolates the arms, not the resample. numpy instead of arch's
    StationaryBootstrap because Cell 6 runs in-session with Internet
    OFF and arch is not in Kaggle's base image; same Politis–Romano
    scheme (expected block length ``block``, circular indexing).
    """
    dyn = np.asarray(dyn, dtype=float)
    fro = np.asarray(fro, dtype=float)
    idx = _stationary_bootstrap_indices(len(dyn), block, reps, seed)
    return _sharpe_rows(dyn[idx]) - _sharpe_rows(fro[idx])


def wilcoxon_test(dynamic_ic, frozen_ic, cfg: dict) -> dict:
    """Wilcoxon signed-rank on paired (fold, seed) mean rank ICs.

    ``dynamic_ic`` / ``frozen_ic`` are the per_fold_seed_ic() dicts (or
    equal-length sequences, paired by sort order). The result carries
    ``wins`` — how many paired observations the dynamic arm won — which
    the §6 conclusion string reports.
    """
    if isinstance(dynamic_ic, dict):
        assert isinstance(frozen_ic, dict)
        keys = sorted(set(dynamic_ic) & set(frozen_ic))
        missing = set(dynamic_ic) ^ set(frozen_ic)
        assert not missing, (
            f"unpaired (fold, seed) runs between arms: {sorted(missing)} — "
            f"the Wilcoxon test needs every run in BOTH arms"
        )
        dyn = [dynamic_ic[k] for k in keys]
        fro = [frozen_ic[k] for k in keys]
    else:
        dyn, fro = list(dynamic_ic), list(frozen_ic)
    assert len(dyn) == len(fro), (
        f"unpaired samples: {len(dyn)} dynamic vs {len(fro)} frozen"
    )

    alpha = float(cfg["significance"]["wilcoxon_alpha"])
    stat, p = scipy_wilcoxon(dyn, fro)
    return {
        "test": "Wilcoxon signed-rank, paired on (fold, seed)",
        "n_pairs": len(dyn),
        "wins": int(sum(1 for a, b in zip(dyn, fro) if a > b)),
        "statistic": float(stat),
        "p_value": float(p),
        "alpha": alpha,
        "significant": bool(p < alpha),
        "dynamic_mean_rank_ic": float(np.mean(dyn)),
        "frozen_mean_rank_ic": float(np.mean(fro)),
        "mean_paired_difference": float(np.mean(
            [a - b for a, b in zip(dyn, fro)])),
    }


def bootstrap_sharpe_test(dynamic_returns: pd.Series,
                          frozen_returns: pd.Series,
                          cfg: dict, seed: int = 7) -> dict:
    """Stationary bootstrap on the Sharpe difference between arms.

    Paired resampling: each replicate draws one index sequence and
    applies it to BOTH arms; the replicate statistic is
    Sharpe(dyn_resample) − Sharpe(fro_resample).

    ``bootstrap_p_value`` is the ONE-SIDED p under the pre-declared H1
    (dynamic > frozen) — the number the §6 conclusion string reports.
    """
    sig = cfg["significance"]
    n_iterations = int(sig["bootstrap_iterations"])
    block = int(sig["bootstrap_block_length"])
    alpha = float(sig["wilcoxon_alpha"])

    dyn = np.asarray(dynamic_returns, dtype=float)
    fro = np.asarray(frozen_returns, dtype=float)
    assert len(dyn) == len(fro) > 0, "pooled return series misaligned"

    observed = _sharpe(dyn) - _sharpe(fro)

    # Paired resampling: each replicate draws ONE index sequence and
    # applies it to BOTH arms, so the replicate statistic isolates the
    # arms, not the resample.
    diffs = _stationary_bootstrap_diffs(dyn, fro, block, n_iterations, seed)
    assert diffs.shape == (n_iterations,)

    p_one = float(np.mean(diffs <= 0))          # H1: dynamic > frozen
    p_two = float(min(1.0, 2 * min(p_one, 1 - p_one)))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "test": "Stationary bootstrap (Politis–Romano) on Sharpe difference",
        "n_iterations": n_iterations,
        "block_length": block,
        "construction": (
            "daily after-cost returns, seed-averaged per (model, fold), "
            "pooled across folds chronologically, paired resampling"
        ),
        "observed_sharpe_dynamic": _sharpe(dyn),
        "observed_sharpe_frozen": _sharpe(fro),
        "observed_sharpe_difference": float(observed),
        "ci_95": [float(lo), float(hi)],
        "p_one_sided_dynamic_greater": p_one,
        "p_two_sided": p_two,
        "bootstrap_p_value": p_one,
        "alpha": alpha,
        "significant_one_sided": bool(p_one < alpha),
        "bootstrap_mean_difference": float(np.mean(diffs)),
    }


# ─── the composite the notebook's Cell 6 and compare.py both call ──────────

def run_tests_from_predictions(dynamic_pred, frozen_pred, cfg: dict,
                               winning_graph_config: dict | None = None,
                               ) -> dict:
    """Both tests from the two arms' prediction files/frames, returning
    the dict phase3_significance_results.json holds (§6's exact shape:
    wilcoxon, bootstrap_sharpe, winning_graph_config, conclusion).

    The dynamic input is the WINNING grid variant's prediction file —
    its test rows are the final dmrgt_all3 result (§1: never retrained).
    """
    dyn = _test_rows(dynamic_pred)
    fro = _test_rows(frozen_pred)

    wilcoxon_result = wilcoxon_test(
        dynamic_ic=per_fold_seed_ic(dyn, cfg),
        frozen_ic=per_fold_seed_ic(fro, cfg),
        cfg=cfg,
    )
    bootstrap_result = bootstrap_sharpe_test(
        dynamic_returns=daily_returns(dyn, cfg),
        frozen_returns=daily_returns(fro, cfg),
        cfg=cfg,
    )

    conclusion = (
        f"Dynamic graph rank IC exceeds frozen in {wilcoxon_result['wins']}/"
        f"{wilcoxon_result['n_pairs']} folds (p={wilcoxon_result['p_value']:.3f}); "
        f"Sharpe difference p={bootstrap_result['bootstrap_p_value']:.3f}"
    )
    return {
        "wilcoxon": wilcoxon_result,
        "bootstrap_sharpe": bootstrap_result,
        "winning_graph_config": winning_graph_config,
        "conclusion": conclusion,
    }


# ─── CLI (local cross-check of the in-session result) ─────────────────────

def main() -> None:
    """Re-run the significance tests locally on the downloaded outputs.

    This is the same computation notebook Cell 6 ran in-session; its
    numbers must agree (compare.py halts on any mismatch). The dynamic
    arm is the winner's grid prediction file, read from
    phase3_frozen_graph_config.json.
    """
    cfg = data.load_config(str(paths.CONFIG))
    winner_record = json.loads(
        (paths.OUTPUT_DIR / "phase3_frozen_graph_config.json").read_text())
    dynamic_path = paths.OUTPUT_DIR / (
        f"predictions_dmrgt_all3_grid_{winner_record['winner']}.parquet")
    frozen_path = paths.OUTPUT_DIR / "predictions_dmrgt_frozen.parquet"

    print(f"[stats] dynamic arm: {dynamic_path.name} (test rows, winner "
          f"{winner_record['winner']!r} — selection already locked)")
    print(f"[stats] frozen arm:  {frozen_path.name} (test rows)")
    out = run_tests_from_predictions(
        dynamic_path, frozen_path, cfg,
        winning_graph_config=winner_record["winning_graph_params"])

    wil, boot = out["wilcoxon"], out["bootstrap_sharpe"]
    print("\n[stats] Wilcoxon (paired rank IC):")
    print(f"  n={wil['n_pairs']}  wins={wil['wins']}  "
          f"statistic={wil['statistic']:.4f}  p={wil['p_value']:.4f}  "
          f"(alpha={wil['alpha']})  significant={wil['significant']}")
    print(f"  dynamic mean IC {wil['dynamic_mean_rank_ic']:+.5f}  vs "
          f"frozen {wil['frozen_mean_rank_ic']:+.5f}  "
          f"(paired diff {wil['mean_paired_difference']:+.5f})")
    print("\n[stats] Stationary bootstrap (Sharpe difference):")
    print(f"  observed Sharpe dyn {boot['observed_sharpe_dynamic']:+.3f} vs "
          f"fro {boot['observed_sharpe_frozen']:+.3f} "
          f"(diff {boot['observed_sharpe_difference']:+.3f})")
    print(f"  {boot['n_iterations']} iterations, block {boot['block_length']}, "
          f"95% CI [{boot['ci_95'][0]:+.3f}, {boot['ci_95'][1]:+.3f}]")
    print(f"  one-sided p (dyn>fro)={boot['bootstrap_p_value']:.4f}  "
          f"two-sided p={boot['p_two_sided']:.4f}")
    print(f"\n[stats] CONCLUSION: {out['conclusion']}")

    out_path = paths.OUTPUT_DIR / "phase3_significance_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[stats] wrote {out_path}")


if __name__ == "__main__":
    main()