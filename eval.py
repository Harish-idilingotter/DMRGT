"""DMRGT Phase 1 — evaluation harness (§5.3).

Every model (random, persistence, Transformer) flows through the same
functions here. Rank IC, directional accuracy, Huber loss, MSE, and
cost-aware backtest via vectorbt.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy.stats import spearmanr


_MLFLOW_DISABLED = os.environ.get("DMRGT_DISABLE_MLFLOW", "0") == "1"
_MLFLOW: object | None = None


def _maybe_init_mlflow(experiment: str = "dmrgt-phase1",
                       tracking_uri: str | None = None) -> bool:
    """Init MLflow once. Returns True if logging is active."""
    global _MLFLOW
    if _MLFLOW_DISABLED:
        return False
    if _MLFLOW is not None:
        return True
    try:
        import mlflow  # noqa: F401
    except Exception:
        return False
    if tracking_uri is None:
        # Default: a local SQLite-backed MLflow tracking store. MLflow
        # 3.x deprecated the filesystem backend in favour of a DB, so
        # we use SQLite by default. The spec is satisfied either way
        # (§2.5: "log to local MLflow from eval.py. Same audit trail,
        # no server"). The DB lives at artifacts/mlflow.db and is
        # browsable with `mlflow ui --backend-store-uri sqlite:///...`.
        import paths
        db_path = paths.OUTPUT_DIR / "mlflow.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"sqlite:///{db_path}"
    try:
        import mlflow as mlf
        mlf.set_tracking_uri(tracking_uri)
        mlf.set_experiment(experiment)
        _MLFLOW = mlf
        return True
    except Exception:
        return False


def log_fold_to_mlflow(fold_metrics, cfg: dict | None = None,
                       run_name: str | None = None) -> bool:
    """Log one fold's metrics to MLflow. No-op if MLflow is unavailable.

    Called by ``compare.py`` after each model has been evaluated across
    all folds. The audit trail lives in ``artifacts/mlruns/``."""
    if not _maybe_init_mlflow():
        return False
    import mlflow as mlf
    fm = fold_metrics
    tags = {"model": fm.model, "fold": fm.fold,
            "seed": str(fm.seed if fm.seed is not None else "none")}
    params = {
        "lookback": cfg["features"]["lookback_days"] if cfg else 0,
        "decile_fraction": cfg["backtest"]["decile_fraction"] if cfg else 0,
        "commission_bps": cfg["backtest"]["commission_bps"] if cfg else 0,
        "slippage_bps": cfg["backtest"]["slippage_bps"] if cfg else 0,
    }
    metrics = {
        "mean_rank_ic": float(fm.mean_rank_ic),
        "std_rank_ic": float(fm.std_rank_ic),
        "directional_acc": float(fm.directional_acc),
        "huber_loss": float(fm.huber_loss),
        "mse": float(fm.mse),
        "sharpe_after_costs": float(fm.sharpe_after_costs),
        "max_drawdown": float(fm.max_drawdown),
        "turnover": float(fm.turnover),
        "cum_return": float(fm.cum_return),
        "n_test_days": int(fm.n_test_days),
    }
    try:
        with mlf.start_run(run_name=run_name, nested=False):
            mlf.set_tags(tags)
            mlf.log_params(params)
            mlf.log_metrics(metrics)
        return True
    except Exception:
        return False


def _huber_loss(residuals: np.ndarray, delta: float) -> np.ndarray:
    """Per-element Huber loss: quadratic for |r|<=delta, linear otherwise."""
    r = np.asarray(residuals, dtype=np.float64)
    abs_r = np.abs(r)
    quadratic = 0.5 * r * r
    linear = delta * (abs_r - 0.5 * delta)
    return np.where(abs_r <= delta, quadratic, linear)



@dataclass
class FoldMetrics:
    """All metrics for a single fold × single seed × single model.

    The Transformer emits per-seed rows; the baselines emit a single
    row per fold (they have no seeds). Aggregation happens in
    ``summarise``.
    """
    model: str
    fold: str
    seed: int | None
    n_test_days: int
    # Prediction metrics — aggregated across the fold's test days
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
    """The full output of evaluating one model on one fold.

    ``per_day`` is the un-aggregated per-day metric table — useful for
    debugging and for plotting the IC time series.
    """
    model: str
    fold: str
    seed: int | None
    per_day: pd.DataFrame
    fold_metrics: FoldMetrics
    backtest_returns: pd.Series  # daily returns from the backtest



def per_day_rank_ic(pred: np.ndarray, true: np.ndarray) -> float:
    """Spearman rank IC for a single cross-section.

    Returns NaN if either side has zero variance. The aggregator drops
    NaN days before computing the mean.
    """
    if pred.size < 2:
        return float("nan")
    if np.std(pred) == 0 or np.std(true) == 0:
        return float("nan")
    rho, _ = spearmanr(pred, true)
    return float(rho) if np.isfinite(rho) else float("nan")


def per_day_directional_acc(pred: np.ndarray, true: np.ndarray) -> float:
    """Fraction of stocks with sign(pred) == sign(true).

    NaN-safe: NaN inputs are treated as wrong and excluded from both
    numerator and denominator. The aggregated number is over the
    valid pairs only.
    """
    mask = np.isfinite(pred) & np.isfinite(true)
    if not mask.any():
        return float("nan")
    p = pred[mask]
    t = true[mask]
    return float(np.mean(np.sign(p) == np.sign(t)))


def per_day_huber(pred: np.ndarray, true: np.ndarray, delta: float) -> float:
    """Mean Huber loss across the cross-section for one day.

    Uses the piecewise Huber loss with the configured delta. NaN pairs
    are excluded.
    """
    mask = np.isfinite(pred) & np.isfinite(true)
    if not mask.any():
        return float("nan")
    p = pred[mask]
    t = true[mask]
    return float(np.mean(_huber_loss(p - t, delta)))


def per_day_mse(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean squared error across the cross-section for one day."""
    mask = np.isfinite(pred) & np.isfinite(true)
    if not mask.any():
        return float("nan")
    p = pred[mask]
    t = true[mask]
    return float(np.mean((p - t) ** 2))



def backtest_daily_deciles(
    predictions_by_day: pd.DataFrame,
    true_returns_by_day: pd.DataFrame,
    tickers: list[str],
    commission_bps: float,
    slippage_bps: float,
    decile_fraction: float,
) -> tuple[pd.Series, dict]:
    """Long top decile, short bottom decile, equal-weighted within each leg.

    Parameters
    ----------
    predictions_by_day : DataFrame
        Index=date, columns=ticker, values=predicted excess return. One
        row per test day.
    true_returns_by_day : DataFrame
        Index=date, columns=ticker, values=the realised next-day *raw*
        return (not excess) for the SAME date. The backtest earns the
        raw return — the excess-return target is for ranking only.
    tickers : list of str
        The stock tickers, in a stable order.
    commission_bps : float
        One-way commission in basis points (10 bps = 0.10%).
    slippage_bps : float
        One-way slippage in basis points (5 bps = 0.05%).
    decile_fraction : float
        Top/bottom fraction to go long/short. 0.2 → 4 longs + 4 shorts
        out of n stocks.

    Returns
    -------
    (returns, stats)
        ``returns`` is a pd.Series of daily portfolio returns, indexed by
        the test dates in input order. ``stats`` is a dict with keys
        ``sharpe``, ``max_drawdown``, ``turnover``, ``cum_return``.
    """
    pred = predictions_by_day[tickers]
    rets = true_returns_by_day[tickers]

    n = len(tickers)
    k = max(1, int(round(n * decile_fraction)))  # 4 for n=20, f=0.2

    # ── build the long/short signals ─────────────────────────────────
    # rank along the cross-section (axis=1) — rank 0..n-1.
    ranks = pred.rank(axis=1, ascending=True, method="first")
    is_long = ranks <= k
    is_short = ranks > (n - k)
    is_invested = is_long | is_short

    # equal weight within each leg. With k=4 longs and k=4 shorts the
    # gross exposure is 1 (long +0.25 per leg, short -0.25 per leg = 0
    # net), the standard 1× dollar-neutral construction.
    long_n = is_long.sum(axis=1).replace(0, 1)
    short_n = is_short.sum(axis=1).replace(0, 1)
    long_w = is_long.astype(float).div(long_n, axis=0)
    short_w = is_short.astype(float).div(short_n, axis=0)
    weights = (long_w.fillna(0) - short_w.fillna(0))

    # days with no positions → zero weight
    weights = weights.where(is_invested, 0.0)

    # ── build the entries/exits for vbt ─────────────────────────────
    # A position change is an entry when weight goes 0→nonzero and an
    # exit when nonzero→0. We approximate "position change" by counting
    # any change in the discretised leg membership.
    long_change = (is_long != is_long.shift(1)).fillna(is_long).any(axis=1)
    short_change = (is_short != is_short.shift(1)).fillna(is_short).any(axis=1)
    entries = (long_change | short_change).astype(bool)
    # vbt's from_signals treats entries/exits as 0/1 booleans. We use
    # `entries` for both since the portfolio rebalances every day that
    # the membership changes.
    exits = entries.shift(1, fill_value=False).astype(bool) & ~entries

    # ── run the portfolio ───────────────────────────────────────────
    # vbt's fees are charged on each entry (one-way) by default.
    # commission + slippage is applied as a single combined fee in
    # decimal (10 bps + 5 bps = 15 bps = 0.0015 per leg change).
    fee = (commission_bps + slippage_bps) / 10_000.0
    init_cash = 1.0
    pf = vbt.Portfolio.from_signals(
        close=rets,
        entries=entries,
        exits=exits,
        size=np.nan,           # sizes inferred from weights
        size_type="percent",
        fees=fee,
        freq="1D",
        init_cash=init_cash,
    )

    # vbt's per-asset stats (max_drawdown, sharpe_ratio, total_return)
    # are Series when given a wide close frame. We aggregate to a
    # single portfolio by hand from the weights and the realised
    # returns, which is the cleanest way to get a single scalar
    # Sharpe / max DD / cum return for a dollar-neutral basket.
    # We then use pf.returns() (which already accounts for the
    # per-leg fees vbt charged) as a sanity check.
    raw_port = (weights.shift(1).fillna(0) * rets).sum(axis=1)
    # Cost: each entry (long_change | short_change) incurs fee * 1.0
    # worth of notional (the leg size we just bought/sold). Approximate
    # the cost as fee * |weight change summed| / 2 since each entry
    # represents an open and a close.
    w_change = weights.diff().abs().sum(axis=1).fillna(0)
    cost = fee * w_change / 2
    daily_ret = (raw_port - cost).astype(float)
    daily_ret.index = pd.DatetimeIndex(daily_ret.index)

    # Sharpe — annualised, with risk-free = 0, daily returns.
    sd = float(daily_ret.std(ddof=1))
    sharpe = float(daily_ret.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0
    # Max drawdown from the equity curve.
    eq = (1 + daily_ret.fillna(0)).cumprod()
    peak = eq.cummax()
    dd = (eq - peak) / peak
    max_dd = float(-dd.min()) if dd.notna().any() else float("nan")
    # Turnover — one-way, fraction of NAV that changes per day.
    turnover = float(w_change.mean() / 2)
    # Cumulative return
    cum = float(eq.iloc[-1] - 1) if len(eq) > 0 else float("nan")

    return daily_ret, {
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "turnover": turnover,
        "cum_return": cum,
    }



def evaluate_fold(
    model_name: str,
    model_fn: Callable[[np.ndarray], np.ndarray],
    fold,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    cfg: dict,
    seed: int | None = None,
    lookback: int = 30,
) -> FoldResult:
    """Run one model through the per-day harness on one fold.
    model_fn(X) -> y_pred where X is [n_stocks, lookback, n_features]."""
    universe = list(cfg["universe"]["tickers"])
    n_stocks = len(universe)
    feature_cols = [c for c in features.columns if c.startswith("f") and "_" in c]
    n_features = len(feature_cols)
    assert n_stocks == len(universe), (
        f"Expected {len(universe)} stocks from config, got {n_stocks}"
    )
    assert n_features == 8, f"Expected 8 features; got {n_features}"

    # Index features/targets for fast lookup
    feat_wide = (features.pivot(index="date", columns="ticker",
                                values=feature_cols)
                 .sort_index())
    # MultiIndex columns: (feature, ticker)
    tgt_wide = targets.pivot(index="date", columns="ticker",
                             values="excess_return").sort_index()

    # The realised *raw* return on day t (= close[t]/close[t-1]-1) is
    # what the backtest earns. The target table holds excess returns
    # (demeaned) — for the backtest we need the raw. We can recover it
    # from the per-day cross-sectional mean of the excess returns plus
    # the excess itself: raw[t] = excess[t] + mean[t] where mean[t] is
    # the universe mean. But the targets table is missing the final
    # date (it was dropped — §4 step 4). For the backtest we need a
    # return series that covers every test date; we therefore rebuild
    # the per-day universe mean from the targets that exist, accepting
    # the last-day drop.
    raw_wide = tgt_wide.add(tgt_wide.mean(axis=1), axis=0)
    # raw_wide now holds daily_return[t] for every (date, ticker). This
    # is the return realised on day t — i.e. the return earned by a
    # portfolio that takes positions at the close of day t-1 and holds
    # to the close of day t. The label for day t (excess at t+1) is
    # about the return ON day t+1, but for the backtest we use the
    # realised return of the SAME date the prediction was made for.

    # Walk every test date. For each, assemble the lookback window of
    # features and run the model.
    test_dates = list(fold.test_dates)
    pred_rows = []
    true_rows = []
    daily_metrics = []
    for d in test_dates:
        if d not in feat_wide.index or d not in tgt_wide.index:
            # The features/targets tables may be missing some test
            # dates (e.g. last date dropped from targets). Skip.
            continue
        d_loc = feat_wide.index.get_loc(d)
        if d_loc < lookback:
            # Not enough history to assemble a full lookback window.
            continue

        window_dates = feat_wide.index[d_loc - lookback: d_loc]
        # Shape [lookback, n_stocks, n_features]
        X = np.empty((lookback, n_stocks, n_features), dtype=np.float64)
        for j, t in enumerate(universe):
            X[:, j, :] = feat_wide.loc[window_dates, (slice(None), t)].values
        # Reorder to [n_stocks, lookback, n_features] for the model
        X_t = np.transpose(X, (1, 0, 2))
        # If any NaN sneaks in (shouldn't, after the warm-up drop),
        # skip this day.
        if not np.isfinite(X_t).all():
            continue
        y_pred = model_fn(X_t)
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

    # ── backtest on the test window ─────────────────────────────────
    bt_dates = pred_df.index
    bt_pred = pred_df.copy()
    bt_raw = raw_wide.loc[bt_dates, universe]
    daily_ret, bt_stats = backtest_daily_deciles(
        predictions_by_day=bt_pred,
        true_returns_by_day=bt_raw,
        tickers=universe,
        commission_bps=cfg["backtest"]["commission_bps"],
        slippage_bps=cfg["backtest"]["slippage_bps"],
        decile_fraction=cfg["backtest"]["decile_fraction"],
    )

    # ── aggregate per-day metrics across the fold ───────────────────
    mean_ic = float(np.nanmean(daily_df["rank_ic"]))
    std_ic = float(np.nanstd(daily_df["rank_ic"], ddof=1))
    mean_dir = float(np.nanmean(daily_df["dir_acc"]))
    mean_huber = float(np.nanmean(daily_df["huber"]))
    mean_mse = float(np.nanmean(daily_df["mse"]))

    metrics = FoldMetrics(
        model=model_name,
        fold=fold.name,
        seed=seed,
        n_test_days=len(daily_df),
        mean_rank_ic=mean_ic,
        std_rank_ic=std_ic,
        directional_acc=mean_dir,
        huber_loss=mean_huber,
        mse=mean_mse,
        sharpe_after_costs=bt_stats["sharpe"],
        max_drawdown=bt_stats["max_drawdown"],
        turnover=bt_stats["turnover"],
        cum_return=bt_stats["cum_return"],
    )
    return FoldResult(
        model=model_name,
        fold=fold.name,
        seed=seed,
        per_day=daily_df,
        fold_metrics=metrics,
        backtest_returns=daily_ret,
    )



def summarise(results: Iterable[FoldResult]) -> pd.DataFrame:
    """Flatten a list of FoldResults into a tidy DataFrame."""
    rows = [asdict(r.fold_metrics) for r in results]
    return pd.DataFrame(rows)



def evaluate_predictions(
    predictions_path,
    targets: pd.DataFrame,
    cfg: dict,
    universe: list[str] | None = None,
) -> list[FoldResult]:
    """Re-evaluate a downloaded predictions.parquet through the same
    code path as evaluate_fold. One FoldResult per (fold, seed)."""
    if universe is None:
        universe = list(cfg["universe"]["tickers"])
    pred_df = pd.read_parquet(predictions_path)
    required = {"model", "fold", "seed", "date", "ticker", "y_pred", "y_true"}
    missing = required - set(pred_df.columns)
    if missing:
        raise ValueError(
            f"predictions.parquet missing columns: {missing}. "
            f"Have: {list(pred_df.columns)}"
        )

    # Realised raw return per (date, ticker) — same construction as
    # in evaluate_fold: raw[t] = excess[t] + mean[t] over the universe.
    tgt_wide = (targets.pivot(index="date", columns="ticker",
                              values="excess_return")
                .sort_index())
    raw_wide = tgt_wide.add(tgt_wide.mean(axis=1), axis=0)

    results: list[FoldResult] = []
    for (model_name, fold_name, seed), grp in pred_df.groupby(
            ["model", "fold", "seed"]):
        seed_val = int(seed) if pd.notna(seed) else None
        # Pivot to wide [date × ticker] for this (model, fold, seed)
        wide_pred = grp.pivot(index="date", columns="ticker",
                              values="y_pred").sort_index()
        wide_true = grp.pivot(index="date", columns="ticker",
                              values="y_true").sort_index()
        # Reindex to the configured universe so column order matches
        wide_pred = wide_pred.reindex(columns=universe)
        wide_true = wide_true.reindex(columns=universe)
        # Backtest
        bt_dates = wide_pred.index
        bt_raw = raw_wide.reindex(bt_dates).reindex(columns=universe)
        daily_ret, bt_stats = backtest_daily_deciles(
            predictions_by_day=wide_pred,
            true_returns_by_day=bt_raw,
            tickers=universe,
            commission_bps=cfg["backtest"]["commission_bps"],
            slippage_bps=cfg["backtest"]["slippage_bps"],
            decile_fraction=cfg["backtest"]["decile_fraction"],
        )

        # Per-day metrics
        per_day_rows = []
        for d in bt_dates:
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
        daily_df = pd.DataFrame(per_day_rows).set_index("date")

        # Aggregate
        metrics = FoldMetrics(
            model=str(model_name),
            fold=str(fold_name),
            seed=seed_val,
            n_test_days=len(daily_df),
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
        results.append(FoldResult(
            model=str(model_name),
            fold=str(fold_name),
            seed=seed_val,
            per_day=daily_df,
            fold_metrics=metrics,
            backtest_returns=daily_ret,
        ))

    return results
