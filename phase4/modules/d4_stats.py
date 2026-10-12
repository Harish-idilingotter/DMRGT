"""DMRGT Phase 4 — significance testing (spec §6.4).

Ported verbatim from Phase 3 stats.py: Wilcoxon signed-rank on paired
(fold, seed) rank IC, and the NumPy Politis–Romano stationary bootstrap on
the Sharpe difference (seed-averaged per fold, folds pooled, paired
resampling). Phase 4 edits: imports (paths/data → d4_data, eval → d4_eval),
the local CLI was dropped, and the Holm helper + generic wrappers at the
bottom were added. The "dynamic"/"frozen" key names inside the ported
functions are kept as-is: in Phase 4, "dynamic" = arm/policy A under test
and "frozen" = the reference it is compared to.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon as scipy_wilcoxon

import d4_data
import d4_eval as ev


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
    targets = pd.read_parquet(d4_data.resolve_cache() / "targets.parquet")
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
    targets = pd.read_parquet(d4_data.resolve_cache() / "targets.parquet")
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


# ─── Phase 4 additions ─────────────────────────────────────────────────────

def holm_correction(p_values: dict, alpha: float = 0.05) -> dict:
    """Holm–Bonferroni step-down. Returns {name: {p_raw, p_adj, reject}}."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for rank, (name, p) in enumerate(items):
        adj = min(1.0, (m - rank) * p)
        running = max(running, adj)          # enforce monotonicity
        out[name] = {"p_raw": float(p), "p_adj": float(running), "reject": bool(running < alpha)}
    return out


def compare_arms(test_pred, ref_pred, cfg: dict) -> dict:
    """Wilcoxon + bootstrap for one comparison of two prediction frames
    (test rows). `test_pred` plays the 'dynamic' role, `ref_pred` the
    'frozen' role of the ported functions."""
    a, b = _test_rows(test_pred), _test_rows(ref_pred)
    wil = wilcoxon_test(per_fold_seed_ic(a, cfg), per_fold_seed_ic(b, cfg), cfg)
    boot = bootstrap_sharpe_test(daily_returns(a, cfg), daily_returns(b, cfg), cfg)
    return {"wilcoxon": wil, "bootstrap_sharpe": boot}
