"""DMRGT Phase 3 — evaluation harness (§5.2).

Rank IC (Spearman, per day, averaged), directional accuracy, Huber/MSE,
and the cost-aware long/short decile backtest. Every model — random,
persistence, dmrgt_all3, dmrgt_frozen — flows through the exact same
functions here; no model is special-cased.

Metric formulas are identical to Phase 1/2 with ONE deliberate, decided
correction, recorded here so nobody re-litigates it silently:

  LEG DIRECTION. The Phase 1/2 specs define the strategy as "long the
  top decile, short the bottom decile" — the top-predicted stocks are
  bought. Phase 2's implementation accidentally inverted this
  (`ranks <= k` on an ASCENDING rank selects the LOWEST predictions),
  which is visible in the Phase 2 results as a Sharpe ordering exactly
  inverted relative to the rank IC ordering (LSTM: IC −0.0016, best
  Sharpe; DMRGT all3: best IC, worst Sharpe). Phase 3 fixes the legs to
  match the written definition. Consequences: rank IC, directional
  accuracy, turnover and the cost model are UNCHANGED and remain
  comparable across phases; Sharpe and cumulative return are NOT
  numerically comparable to the Phase 2 table (they were leg-inverted
  there). Phase 3's numbers are internally consistent and economically
  meaningful, which matters because the §6b graph grid selects on Sharpe.

Two Phase 3 additions (per §5.2):
  * evaluate_predictions(..., split=...) — filters to "val" or "test"
    rows before scoring, so selection can be locked to validation.
  * backtest(..., postprocessing=...) — applies an optional
    post-processing dict (no_trade_band / decile_fraction /
    smoothing_days) to predictions before ranking; None is a no-op.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import paths


# ─── per-day metrics (formulas unchanged from Phase 1/2) ─────────────────

def _huber_loss(residuals: np.ndarray, delta: float) -> np.ndarray:
    """Per-element Huber loss: quadratic for |r|<=delta, linear otherwise."""
    r = np.asarray(residuals, dtype=np.float64)
    abs_r = np.abs(r)
    quadratic = 0.5 * r * r
    linear = delta * (abs_r - 0.5 * delta)
    return np.where(abs_r <= delta, quadratic, linear)


def per_day_rank_ic(pred: np.ndarray, true: np.ndarray) -> float:
    """Spearman rank IC for a single cross-section. NaN if either side
    has zero variance; the aggregator drops NaN days."""
    if pred.size < 2:
        return float("nan")
    if np.std(pred) == 0 or np.std(true) == 0:
        return float("nan")
    rho, _ = spearmanr(pred, true)
    return float(rho) if np.isfinite(rho) else float("nan")


def per_day_directional_acc(pred: np.ndarray, true: np.ndarray) -> float:
    """Fraction of stocks with sign(pred) == sign(true), over valid pairs."""
    mask = np.isfinite(pred) & np.isfinite(true)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.sign(pred[mask]) == np.sign(true[mask])))


def per_day_huber(pred: np.ndarray, true: np.ndarray, delta: float) -> float:
    mask = np.isfinite(pred) & np.isfinite(true)
    if not mask.any():
        return float("nan")
    return float(np.mean(_huber_loss(pred[mask] - true[mask], delta)))


def per_day_mse(pred: np.ndarray, true: np.ndarray) -> float:
    mask = np.isfinite(pred) & np.isfinite(true)
    if not mask.any():
        return float("nan")
    return float(np.mean((pred[mask] - true[mask]) ** 2))


# ─── dataclasses ──────────────────────────────────────────────────────────

@dataclass
class FoldMetrics:
    """All metrics for one model × fold × seed × split."""
    model: str
    fold: str
    seed: int | None
    split: str
    n_days: int
    # Prediction metrics — aggregated across the split's days
    mean_rank_ic: float
    std_rank_ic: float
    directional_acc: float
    huber_loss: float
    mse: float
    # Backtest metrics
    sharpe_after_costs: float
    max_drawdown: float
    turnover: float
    cum_return: float


@dataclass
class FoldResult:
    """Full output of evaluating one model on one fold.

    ``per_day`` is the un-aggregated per-day metric table;
    ``backtest_returns`` is the daily portfolio return series after
    costs (used by stats.py's bootstrap).
    """
    model: str
    fold: str
    seed: int | None
    split: str
    per_day: pd.DataFrame
    fold_metrics: FoldMetrics
    backtest_returns: pd.Series


def summarise(results: Iterable[FoldResult]) -> pd.DataFrame:
    """Flatten FoldResults into a tidy DataFrame."""
    return pd.DataFrame([asdict(r.fold_metrics) for r in results])


# ─── realised raw returns ─────────────────────────────────────────────────

def raw_returns_from_targets(targets: pd.DataFrame,
                             universe: list[str]) -> pd.DataFrame:
    """Recover realised RAW daily returns from the excess-return target.

    The backtest earns the raw return, not the demeaned target; raw[t] =
    excess[t] + cross-sectional mean[t]. Wide frame, date × universe.
    """
    tgt_wide = (targets.pivot(index="date", columns="ticker",
                              values="excess_return")
                .sort_index()
                .reindex(columns=universe))
    return tgt_wide.add(tgt_wide.mean(axis=1), axis=0)


# ─── post-processing (§9, applied identically to both arms — Rule 19) ─────

DEFAULT_POSTPROCESSING: dict = {
    "no_trade_band": 0.0,
    "decile_fraction": None,   # None → fall back to cfg backtest.decile_fraction
    "smoothing_days": 1,
}


def apply_postprocessing(preds: pd.DataFrame,
                         postprocessing: dict | None) -> pd.DataFrame:
    """Apply the Tier-1 post-processing rules to a long predictions frame.

    Rules, in order:
      1. smoothing_days — causal (trailing) rolling mean of y_pred over
         `smoothing_days` dates, per ticker, within each (model, fold,
         seed) group. Never looks ahead. 1 → no-op.
      2. no_trade_band — stocks whose |smoothed prediction| < band are
         flagged untradable on that date (``tradable`` column, False).
         The backtest excludes them from leg selection; rank IC keeps
         scoring them (the band is a trading rule, not a model change).

    decile_fraction travels in the same dict but is a backtest parameter
    (leg size), not a per-row transformation — backtest() reads it.

    Returns a new DataFrame; the input is not mutated.
    """
    if postprocessing is None:
        pp = dict(DEFAULT_POSTPROCESSING)
    else:
        pp = {**DEFAULT_POSTPROCESSING, **{
            k: v for k, v in postprocessing.items()
            if k in DEFAULT_POSTPROCESSING and v is not None
        }}

    df = preds.copy()
    if "tradable" not in df.columns:
        df["tradable"] = True

    # ── 1. causal smoothing ──────────────────────────────────────────
    smooth = int(pp.get("smoothing_days", 1) or 1)
    if smooth > 1:
        # Rolling per ticker within each (model, fold, seed) run — rows
        # of different tickers must never be mixed into one window.
        keys = [c for c in ("model", "fold", "seed", "ticker")
                if c in df.columns]
        df = df.sort_values(keys + ["date"])
        df["y_pred"] = (df.groupby(keys, sort=False)["y_pred"]
                        .transform(lambda s: s.rolling(smooth,
                                                       min_periods=1).mean()))
        df = df.reset_index(drop=True)

    # ── 2. no-trade band ─────────────────────────────────────────────
    band = float(pp.get("no_trade_band", 0.0) or 0.0)
    df["tradable"] = df["y_pred"].abs() >= band

    return df


# ─── the cost-aware backtest (§5.2) ───────────────────────────────────────

def backtest_daily_deciles(
    predictions_by_day: pd.DataFrame,   # date × ticker, values = y_pred
    true_returns_by_day: pd.DataFrame,  # date × ticker, values = realised raw return
    tickers: list[str],
    commission_bps: float,
    slippage_bps: float,
    decile_fraction: float,
    tradable_by_day: pd.DataFrame | None = None,  # date × ticker bool; None = all tradable
) -> tuple[pd.Series, dict]:
    """Long the top decile, short the bottom decile, equal-weighted within
    each leg. Legs are selected among the day's tradable stocks only.

    Positions decided from day-t predictions earn day-t+1 returns
    (weights are shifted one day before multiplying realised returns —
    no same-day lookahead). Commission + slippage are charged on
    weight changes: fee = (commission_bps + slippage_bps) / 1e4, applied
    to half the absolute daily weight change (one-way turnover).

    Returns (daily portfolio return series after costs, stats dict).
    """
    pred = predictions_by_day[tickers]
    rets = true_returns_by_day[tickers]
    trad = tradable_by_day[tickers] if tradable_by_day is not None else None

    n = len(tickers)
    fee = (commission_bps + slippage_bps) / 10_000.0

    weights = pd.DataFrame(0.0, index=pred.index, columns=tickers)
    for i, d in enumerate(pred.index):
        row = pred.loc[d].to_numpy(dtype=float)
        ok = (trad.loc[d].to_numpy(dtype=bool)
              if trad is not None else np.ones(n, dtype=bool))
        idx = np.flatnonzero(ok & np.isfinite(row))
        if idx.size == 0:
            continue  # nothing tradable → flat that day
        scores = row[idx]
        k = max(1, int(round(idx.size * decile_fraction)))
        k = min(k, idx.size)  # degenerate wide legs cancel to ~0 net
        order = np.argsort(-scores, kind="stable")  # descending: best first
        longs = idx[order[:k]]
        shorts = idx[order[-k:]]
        w = np.zeros(n)
        w[longs] += 1.0 / len(longs)
        w[shorts] -= 1.0 / len(shorts)
        weights.iloc[i] = w

    # Positions taken at close t earn returns realised at t+1.
    raw_port = (weights.shift(1).fillna(0) * rets).sum(axis=1)
    # Cost: one-way fee on each unit of weight change.
    w_change = weights.diff().abs().sum(axis=1).fillna(0)
    cost = fee * w_change / 2
    daily_ret = (raw_port - cost).astype(float)
    daily_ret.index = pd.DatetimeIndex(daily_ret.index)

    # Sharpe — annualised, risk-free 0, sample std.
    sd = float(daily_ret.std(ddof=1))
    sharpe = float(daily_ret.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0
    # Max drawdown from the equity curve.
    eq = (1 + daily_ret.fillna(0)).cumprod()
    peak = eq.cummax()
    dd = (eq - peak) / peak
    max_dd = float(-dd.min()) if dd.notna().any() else float("nan")
    # Turnover — one-way, fraction of NAV changing per day.
    turnover = float(w_change.mean() / 2)
    cum = float(eq.iloc[-1] - 1) if len(eq) > 0 else float("nan")

    return daily_ret, {
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "turnover": turnover,
        "cum_return": cum,
    }


def _effective_decile_fraction(cfg: dict, postprocessing: dict | None) -> float:
    """Post-processing decile_fraction overrides the config default."""
    if postprocessing and postprocessing.get("decile_fraction") is not None:
        return float(postprocessing["decile_fraction"])
    return float(cfg["backtest"]["decile_fraction"])


def _score_wide_group(
    wide_pred: pd.DataFrame,
    wide_true: pd.DataFrame,
    raw_wide: pd.DataFrame,
    tradable_wide: pd.DataFrame | None,
    cfg: dict,
    decile_fraction: float,
    universe: list[str],
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Per-day metrics + backtest for one (model, fold, seed) wide group."""
    dates = wide_pred.index
    per_day_rows = []
    for d in dates:
        y_pred = wide_pred.loc[d, universe].to_numpy()
        y_true = wide_true.loc[d, universe].to_numpy()
        per_day_rows.append({
            "date": d,
            "rank_ic": per_day_rank_ic(y_pred, y_true),
            "dir_acc": per_day_directional_acc(y_pred, y_true),
            "huber": per_day_huber(y_pred, y_true,
                                   delta=cfg["training"]["huber_delta"]),
            "mse": per_day_mse(y_pred, y_true),
        })
    per_day = pd.DataFrame(per_day_rows).set_index("date")

    bt_raw = raw_wide.reindex(dates).reindex(columns=universe)
    daily_ret, bt_stats = backtest_daily_deciles(
        predictions_by_day=wide_pred,
        true_returns_by_day=bt_raw,
        tickers=universe,
        commission_bps=cfg["backtest"]["commission_bps"],
        slippage_bps=cfg["backtest"]["slippage_bps"],
        decile_fraction=decile_fraction,
        tradable_by_day=tradable_wide,
    )
    return per_day, daily_ret, bt_stats


# ─── baseline path: model_fn over a fold's split (§5.3 gate) ──────────────

def evaluate_fold(
    model_name: str,
    model_fn: Callable[[np.ndarray], np.ndarray],
    fold,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    cfg: dict,
    seed: int | None = None,
    lookback: int | None = None,
) -> FoldResult:
    """Run one closed-form model through the per-day harness on one fold.

    Assembles the same windows the trained model sees: for prediction
    date d, features from [d-lookback+1 .. d] INCLUSIVE of d (Rule 3 —
    features come from day t and earlier). Note: Phase 2's baseline
    harness used a window ending at d-1; the trained models' assembly
    always ended at d. Phase 3 unifies on the spec-correct inclusive
    window, so persistence numbers shift slightly vs Phase 2's recorded
    floor; the random gate is unaffected (noise ignores input).
    """
    universe = list(cfg["universe"]["tickers"])
    n_stocks = len(universe)
    if lookback is None:
        lookback = int(cfg["features"]["lookback_days"])
    feature_cols = list(cfg["features"]["columns"])

    # Reindex to guarantee the feature level follows configs.yaml order —
    # pivot's MultiIndex columns are sorted lexicographically, which would
    # silently reorder X's feature axis if configs ever listed features
    # out of alphabetical order.
    feat_wide = (features.pivot(index="date", columns="ticker",
                                values=feature_cols)
                 .sort_index())
    feat_wide = feat_wide.reindex(columns=pd.MultiIndex.from_product(
        [feature_cols, list(feat_wide.columns.get_level_values("ticker")
                            .unique())],
        names=["feature", "ticker"]))
    tgt_wide = (targets.pivot(index="date", columns="ticker",
                              values="excess_return")
                .sort_index())
    raw_wide = raw_returns_from_targets(targets, universe)

    test_dates = list(fold.test_dates)
    pred_rows, true_rows, daily_metrics = [], [], []
    for d in test_dates:
        if d not in feat_wide.index or d not in tgt_wide.index:
            continue  # e.g. last date dropped from targets
        d_loc = feat_wide.index.get_loc(d)
        if d_loc < lookback - 1:
            continue  # not enough history for an inclusive window
        window_dates = feat_wide.index[d_loc - lookback + 1: d_loc + 1]
        assert len(window_dates) == lookback
        assert window_dates[-1] == d
        X = np.empty((n_stocks, lookback, len(feature_cols)), dtype=np.float64)
        for j, t in enumerate(universe):
            X[j] = feat_wide.loc[window_dates, (slice(None), t)].values
        if not np.isfinite(X).all():
            continue
        y_pred = model_fn(X)
        y_true = tgt_wide.loc[d, universe].values
        pred_rows.append(pd.Series(y_pred, index=universe, name=d))
        true_rows.append(pd.Series(y_true, index=universe, name=d))
        daily_metrics.append({
            "date": d,
            "rank_ic": per_day_rank_ic(y_pred, y_true),
            "dir_acc": per_day_directional_acc(y_pred, y_true),
            "huber": per_day_huber(y_pred, y_true,
                                   delta=cfg["training"]["huber_delta"]),
            "mse": per_day_mse(y_pred, y_true),
        })

    if not daily_metrics:
        raise RuntimeError(
            f"No valid test days for fold {fold.name!r} — check that "
            f"features/targets cover the test window."
        )
    daily_df = pd.DataFrame(daily_metrics).set_index("date")
    pred_df = pd.DataFrame(pred_rows)
    true_df = pd.DataFrame(true_rows)

    _, daily_ret, bt_stats = _score_wide_group(
        pred_df, true_df, raw_wide, None, cfg,
        decile_fraction=float(cfg["backtest"]["decile_fraction"]),
        universe=universe,
    )

    metrics = FoldMetrics(
        model=model_name,
        fold=fold.name,
        seed=seed,
        split="test",
        n_days=len(daily_df),
        mean_rank_ic=float(np.nanmean(daily_df["rank_ic"])),
        std_rank_ic=float(np.nanstd(daily_df["rank_ic"], ddof=1)),
        directional_acc=float(np.nanmean(daily_df["dir_acc"])),
        huber_loss=float(np.nanmean(daily_df["huber"])),
        mse=float(np.nanmean(daily_df["mse"])),
        sharpe_after_costs=bt_stats["sharpe"],
        max_drawdown=bt_stats["max_drawdown"],
        turnover=bt_stats["turnover"],
        cum_return=bt_stats["cum_return"],
    )
    return FoldResult(
        model=model_name,
        fold=fold.name,
        seed=seed,
        split="test",
        per_day=daily_df,
        fold_metrics=metrics,
        backtest_returns=daily_ret,
    )


# ─── predictions-parquet path (trained models) ────────────────────────────

def load_predictions(model_id: str | None = None,
                     split: str | None = None,
                     source: str | Path | None = None) -> pd.DataFrame:
    """Load a Phase 3 predictions parquet.

    Resolution order: explicit `source` (file or directory) →
    artifacts/predictions_{model_id}.parquet. Optionally filters to one
    `split` ("val" / "test").
    """
    src = Path(source) if source is not None else paths.OUTPUT_DIR
    if src.is_dir():
        if model_id is None:
            matches = sorted(src.glob("predictions_*.parquet"))
            if not matches:
                raise FileNotFoundError(f"No predictions_*.parquet in {src}")
            if len(matches) > 1 and model_id is None:
                raise ValueError(
                    f"Multiple prediction files in {src}; pass model_id "
                    f"to disambiguate: {[m.name for m in matches]}"
                )
            src = matches[0]
        else:
            src = src / f"predictions_{model_id}.parquet"
    if not src.exists():
        raise FileNotFoundError(
            f"Predictions not found: {src}. Download the Kaggle output "
            f"into artifacts/ first."
        )
    df = pd.read_parquet(src)
    required = {"model", "fold", "seed", "split", "date", "ticker",
                "y_pred", "y_true"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"predictions parquet {src.name} missing columns: {missing}. "
            f"Have: {list(df.columns)}"
        )
    if split is not None:
        df = df[df["split"] == split]
        if df.empty:
            raise ValueError(
                f"{src.name} has no rows with split == {split!r}. "
                f"Splits present: {sorted(pd.read_parquet(src)['split'].unique())}"
            )
    return df


def evaluate_predictions(
    source: str | Path | pd.DataFrame,
    targets: pd.DataFrame,
    cfg: dict,
    split: str | None = None,
    postprocessing: dict | None = None,
    universe: list[str] | None = None,
) -> list[FoldResult]:
    """Re-evaluate a predictions parquet (Phase 3 schema with a `split`
    column) through the same code path as evaluate_fold. One FoldResult
    per (model, fold, seed) — the pairing stats.py's Wilcoxon test uses.
    """
    if universe is None:
        universe = list(cfg["universe"]["tickers"])
    pred_df = (source if isinstance(source, pd.DataFrame)
               else load_predictions(source=source, split=split))
    pred_df = apply_postprocessing(pred_df, postprocessing)

    if split is not None:
        pred_df = pred_df[pred_df["split"] == split]
    if pred_df.empty:
        raise ValueError(f"No prediction rows to evaluate (split={split!r})")
    if pred_df["model"].nunique() > 1:
        raise ValueError(
            "evaluate_predictions refuses to pool multiple models into one "
            "result set — evaluate one arm at a time so the two arms of "
            "the comparison can never be silently mixed."
        )

    raw_wide = raw_returns_from_targets(targets, universe)
    frac = _effective_decile_fraction(cfg, postprocessing)

    results: list[FoldResult] = []
    for (model_name, fold_name, seed), grp in pred_df.groupby(
            ["model", "fold", "seed"]):
        # A group that mixed splits would silently blend the val and test
        # windows into one return series — refuse loudly instead.
        assert grp["split"].nunique() == 1, (
            f"group model={model_name} fold={fold_name} seed={seed} spans "
            f"splits {sorted(grp['split'].unique())} — filter with "
            f"split= before evaluating; never pool the tuning and scoring "
            f"windows."
        )
        wide_pred = (grp.pivot(index="date", columns="ticker",
                               values="y_pred").sort_index()
                     .reindex(columns=universe))
        wide_true = (grp.pivot(index="date", columns="ticker",
                               values="y_true").sort_index()
                     .reindex(columns=universe))
        tradable_wide = None
        if "tradable" in grp.columns:
            tradable_wide = (grp.pivot(index="date", columns="ticker",
                                       values="tradable").sort_index()
                             .reindex(columns=universe))

        per_day, daily_ret, bt_stats = _score_wide_group(
            wide_pred, wide_true, raw_wide, tradable_wide, cfg,
            decile_fraction=frac, universe=universe,
        )
        split_val = str(grp["split"].iloc[0])
        metrics = FoldMetrics(
            model=str(model_name),
            fold=str(fold_name),
            seed=int(seed),
            split=split_val,
            n_days=len(per_day),
            mean_rank_ic=float(np.nanmean(per_day["rank_ic"])),
            std_rank_ic=float(np.nanstd(per_day["rank_ic"], ddof=1)),
            directional_acc=float(np.nanmean(per_day["dir_acc"])),
            huber_loss=float(np.nanmean(per_day["huber"])),
            mse=float(np.nanmean(per_day["mse"])),
            sharpe_after_costs=bt_stats["sharpe"],
            max_drawdown=bt_stats["max_drawdown"],
            turnover=bt_stats["turnover"],
            cum_return=bt_stats["cum_return"],
        )
        results.append(FoldResult(
            model=str(model_name),
            fold=str(fold_name),
            seed=int(seed),
            split=split_val,
            per_day=per_day,
            fold_metrics=metrics,
            backtest_returns=daily_ret,
        ))
    return results


# ─── aggregate backtest — the grid's / compare.py interface ───────────────

def backtest(
    predictions: pd.DataFrame | str | Path,
    cfg: dict,
    split: str | None = None,
    postprocessing: dict | None = None,
    targets: pd.DataFrame | None = None,
) -> dict:
    """Score a predictions frame through the cost-aware backtest and
    aggregate across every (model, fold, seed) group it contains.

    Aggregation is the MEAN across groups (4 folds × 5 seeds = 20 runs
    per arm): mean of per-run mean rank ICs, mean of per-run Sharpes,
    mean of per-run max drawdowns and turnovers. Returns one flat dict;
    `sharpe` is the key the grid's selection_metric reads.

    Refuses to pool more than one model — a pooled aggregate across the
    two arms would be meaningless.
    """
    df = (predictions if isinstance(predictions, pd.DataFrame)
          else load_predictions(source=predictions, split=split))
    if targets is None:
        targets = pd.read_parquet(paths.TARGETS)
    results = evaluate_predictions(df, targets, cfg, split=split,
                                   postprocessing=postprocessing)
    if not results:
        raise ValueError("No (model, fold, seed) groups to backtest")

    per_run = []
    for r in results:
        m = asdict(r.fold_metrics)
        m["sharpe"] = m["sharpe_after_costs"]
        per_run.append(m)
    keys = ["mean_rank_ic", "std_rank_ic", "directional_acc", "huber_loss",
            "mse", "sharpe_after_costs", "max_drawdown", "turnover",
            "cum_return"]
    agg = {k: float(np.mean([run[k] for run in per_run])) for k in keys}
    agg["sharpe"] = agg["sharpe_after_costs"]
    agg["n_runs"] = len(per_run)
    agg["n_days"] = int(sum(run["n_days"] for run in per_run))
    agg["per_run"] = per_run
    agg["split"] = split
    return agg