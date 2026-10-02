"""DMRGT Phase 3 — closed-form baselines (§5.1).

random_model: ignores input, returns noise (the leakage gate, Rule 21).
persistence_model: predicts tomorrow = today (the floor to beat).
Both accept [n_stocks, lookback, n_features] → [n_stocks], the same
interface the trained model exposes, so they drop into the harness
unchanged.
"""

from __future__ import annotations

import numpy as np


def random_model(X: np.ndarray, seed: int | None = None) -> np.ndarray:
    """Ignore input, return standard-normal noise of shape [n_stocks]."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(X.shape[0])


def persistence_model(X: np.ndarray, feature_idx_logret: int = 0) -> np.ndarray:
    """Predict tomorrow's excess return = today's log-return, demeaned.

    Uses X[:, -1, feature_idx_logret] only — the most recent day's
    log-return feature (the last row of the lookback window, i.e. the
    prediction date itself). The mean subtraction matches the target's
    cross-sectional demeaning.
    """
    today_return = X[:, -1, feature_idx_logret]
    return today_return - np.nanmean(today_return)