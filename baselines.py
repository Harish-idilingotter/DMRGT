"""DMRGT Phase 1 — baselines (§5.2).

random_model: ignores input, returns noise (the leakage gate).
persistence_model: predicts tomorrow = today (the floor to beat).
Both accept [n_stocks, lookback, n_features] → [n_stocks].
"""

from __future__ import annotations

import numpy as np


def random_model(X: np.ndarray, seed: int | None = None) -> np.ndarray:
    """Ignore input, return standard-normal noise of shape [n_stocks]."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(X.shape[0])


def persistence_model(X: np.ndarray, feature_idx_logret: int = 0) -> np.ndarray:
    """Predict tomorrow's excess return = today's log-return, demeaned.
    Uses X[:, -1, feature_idx_logret] only."""
    today_return = X[:, -1, feature_idx_logret]
    # The mean subtraction is critical: the target is *excess* return
    # (cross-sectionally demeaned). To be comparable, the prediction
    # must be demeaned in the same way.
    return today_return - np.nanmean(today_return)
