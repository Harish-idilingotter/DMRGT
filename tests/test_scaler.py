"""Regression tests for the per-fold StandardScaler in
train.assemble_fold() (defect 6).

The spec (§7.2) requires "StandardScaler per feature column" — i.e.
ONE mean and ONE std per feature (8 fitted parameters total for
Phase 1's 8 features). The earlier code reshaped the input to
``[N, lookback * n_features]`` before fitting, which gave the scaler
``lookback * n_features = 240`` parameters — each (lag, feature) cell
treated as a distinct column.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import data  # noqa: E402
import paths  # noqa: E402
import splits as sp  # noqa: E402
import train  # noqa: E402


# Reload paths so any Kaggle-branch reload done by test_paths.py
# earlier in this run can't leak into these tests — we always want
# the local cache here.
import importlib  # noqa: E402
paths = importlib.reload(paths)
paths.__dict__["CACHE_DIR"] = ROOT / "cache"
paths.__dict__["OUTPUT_DIR"] = ROOT / "artifacts"
paths.__dict__["PRICES"] = ROOT / "cache" / "prices.parquet"
paths.__dict__["FEATURES"] = ROOT / "cache" / "features.parquet"
paths.__dict__["TARGETS"] = ROOT / "cache" / "targets.parquet"
paths.__dict__["HASHES_JSON"] = ROOT / "cache" / "hashes.json"


FOLD_IDX = 0


def _build_fold_tensors():
    # Force local paths (defensive override — see comment at module top).
    paths.__dict__["CACHE_DIR"] = ROOT / "cache"
    paths.__dict__["OUTPUT_DIR"] = ROOT / "artifacts"
    paths.__dict__["PRICES"] = ROOT / "cache" / "prices.parquet"
    paths.__dict__["FEATURES"] = ROOT / "cache" / "features.parquet"
    paths.__dict__["TARGETS"] = ROOT / "cache" / "targets.parquet"
    paths.__dict__["HASHES_JSON"] = ROOT / "cache" / "hashes.json"
    cfg = data.load_config()
    features = pd.read_parquet(paths.FEATURES)
    targets = pd.read_parquet(paths.TARGETS)
    folds = sp.make_folds(cfg, available_dates=features["date"].unique())
    return train.assemble_fold(folds[FOLD_IDX], features, targets, cfg)


def test_scaler_has_exactly_n_features_parameters():
    """The fitted scaler must have exactly n_features (= 8) parameters,
    not lookback*n_features (= 240)."""
    cfg = data.load_config()
    n_feat = len(cfg["features"]["columns"])
    t = _build_fold_tensors()
    assert len(t.scaler_mean) == n_feat, (
        f"Scaler has {len(t.scaler_mean)} params, expected {n_feat}. "
        f"defect 6 (per-cell scaling) has regressed — the reshape is "
        f"wrong and gives one mean per (lag, feature) cell."
    )
    assert len(t.scaler_scale) == n_feat


def test_scaler_fitted_only_on_training_data():
    """The fitted scaler stats must be derived from training data
    only — not val, not test. We verify by recomputing the mean from
    the raw X_train tensor (before transform) and comparing to
    scaler_mean within numerical tolerance."""
    import pandas as pd
    cfg = data.load_config()
    n_feat = len(cfg["features"]["columns"])

    # Reload the raw X_train (before scaling) — re-assemble without
    # using assemble_fold's scaler.
    features = pd.read_parquet(paths.FEATURES)
    targets = pd.read_parquet(paths.TARGETS)
    folds = sp.make_folds(cfg, available_dates=features["date"].unique())

    universe = list(cfg["universe"]["tickers"])
    feature_cols = list(cfg["features"]["columns"])
    lookback = int(cfg["features"]["lookback_days"])
    feat_wide = features.pivot(index="date", columns="ticker",
                               values=feature_cols).sort_index()

    fold = folds[FOLD_IDX]
    train_Xs = []
    for d in fold.train_dates:
        if d not in feat_wide.index:
            continue
        loc = feat_wide.index.get_loc(d)
        start = loc - lookback + 1
        if start < 0:
            continue
        window = feat_wide.index[start: loc + 1]
        X = np.empty((len(universe), lookback, n_feat), dtype=np.float64)
        for j, t in enumerate(universe):
            X[j] = feat_wide.loc[window, (slice(None), t)].values
        train_Xs.append(X)
    X_train_raw = np.stack(train_Xs, axis=0)

    # Mean across days, stocks, and lags — per feature
    expected_mean = X_train_raw.reshape(-1, n_feat).mean(axis=0)
    expected_std = X_train_raw.reshape(-1, n_feat).std(axis=0)

    t = _build_fold_tensors()
    np.testing.assert_allclose(t.scaler_mean, expected_mean, rtol=1e-4,
                               err_msg="scaler_mean doesn't match raw X_train mean")
    np.testing.assert_allclose(t.scaler_scale, expected_std, rtol=1e-4,
                               err_msg="scaler_scale doesn't match raw X_train std")


def test_transformed_training_data_is_centred_and_unit_std():
    """After scaling, training data per feature must have ~zero mean
    and ~unit std across days, stocks, and lags (because the scaler
    pools all of them)."""
    t = _build_fold_tensors()
    cfg = data.load_config()
    n_feat = len(cfg["features"]["columns"])

    if len(t.X_train) == 0:
        pytest.skip("no training samples")

    flat = t.X_train.reshape(-1, n_feat)
    means = flat.mean(axis=0)
    stds = flat.std(axis=0)
    np.testing.assert_allclose(means, np.zeros(n_feat), atol=1e-6)
    np.testing.assert_allclose(stds,  np.ones(n_feat),  atol=1e-3)


def test_val_and_test_use_training_scaler_only():
    """The same scaler stats must be applied to val and test. We
    verify by checking that the mean/std of scaled val/test data is
    NOT zero/one (because they weren't used to fit the scaler)."""
    t = _build_fold_tensors()
    cfg = data.load_config()
    n_feat = len(cfg["features"]["columns"])

    if len(t.X_val) == 0:
        pytest.skip("no val samples")
    if len(t.X_test) == 0:
        pytest.skip("no test samples")

    val_flat = t.X_val.reshape(-1, n_feat)
    test_flat = t.X_test.reshape(-1, n_feat)
    # Means should NOT all be exactly zero (they would be if val/test
    # had been used to fit the scaler).
    assert not np.allclose(val_flat.mean(axis=0),  np.zeros(n_feat), atol=1e-6)
    assert not np.allclose(test_flat.mean(axis=0), np.zeros(n_feat), atol=1e-6)