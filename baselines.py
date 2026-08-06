"""DMRGT Phase 1 — baselines (Step F / §5.2).

The two baseline models the spec requires us to run before any real
training:

  • random_model       — ignores input, returns pure noise. The
    calibration check (§5.4) demands that this scores rank IC ≈ 0 and
    loses money after costs. If it doesn't, the pipeline has a leak.
  • persistence_model  — predicts that tomorrow's excess return equals
    today's. Uses only the most recent day's log-return feature. This
    is the floor the Transformer must beat to be worth anything.

Both models accept and return the same shapes the real model will
(``[n_stocks, lookback, n_features]`` → ``[n_stocks]``), so they drop
into eval.py unchanged.
"""

from __future__ import annotations

import numpy as np


def random_model(X: np.ndarray, seed: int | None = None) -> np.ndarray:
    """Ignore all input. Return pure noise.

    Parameters
    ----------
    X : ndarray, shape [n_stocks, lookback, n_features]
        The feature tensor for a single trading day. Ignored.
    seed : int, optional
        Seed for the RNG. None means a fresh, un-seeded draw.

    Returns
    -------
    ndarray, shape [n_stocks]
        A standard-normal sample. The mean is NOT subtracted — that
        would make the predictions correlated with the cross-section
        in a way that the cross-sectional subtraction in
        build_targets does not (the real excess return can have a
        non-zero mean over a small sample). The harness subtracts
        the cross-sectional mean itself when computing rank IC.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal(X.shape[0])


def persistence_model(X: np.ndarray, feature_idx_logret: int = 0) -> np.ndarray:
    """Predict that tomorrow's excess return equals today's.

    Uses only the most recent day's log-return feature (the spec's
    default ``feature_idx_logret=0`` lines up with the first column
    of the feature tensor — f1_logret).

    The persistence model returns a *demeaned* prediction
    (today's return minus the cross-sectional mean of today's returns)
    so that its output is directly comparable to the excess-return
    target. This is the natural unit for ranking.

    Parameters
    ----------
    X : ndarray, shape [n_stocks, lookback, n_features]
        The feature tensor. Only ``X[:, -1, feature_idx_logret]`` is used.
    feature_idx_logret : int
        Index along the feature axis that holds the log-return
        feature. Default 0 (the first feature column, f1_logret).

    Returns
    -------
    ndarray, shape [n_stocks]
        Today's log-return for each stock, minus the cross-sectional
        mean. NaNs in the input propagate as NaNs in the output —
        eval.py drops NaN days before computing rank IC.
    """
    today_return = X[:, -1, feature_idx_logret]
    # The mean subtraction is critical: the target is *excess* return
    # (cross-sectionally demeaned). To be comparable, the prediction
    # must be demeaned in the same way.
    return today_return - np.nanmean(today_return)
