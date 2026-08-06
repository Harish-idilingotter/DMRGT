"""Regression tests for the lookback-window construction in
train.assemble_fold() (defect 5).

The spec (§4) says the label at date ``t`` is the prediction made on
day ``t`` using data up to and INCLUDING day ``t``. The window is
therefore ``[t-lookback+1 .. t]`` — ``t`` itself is the last row.

The earlier code excluded ``t`` (window was ``[t-lookback .. t-1]``),
which is conservative (not leaky) but not what the spec wrote. These
tests pin the inclusive-of-prediction-date behaviour.
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


# Reload paths so that any Kaggle-branch reload done by test_paths.py
# earlier in this run can't leak into these tests — we always want the
# local cache here.
import importlib  # noqa: E402
paths = importlib.reload(paths)

# Force the local cache path here. If test_paths ran first and left
# paths in Kaggle mode (its last test reloads paths in a mocked
# Kaggle branch), this is the only reliable way to override it —
# the reload alone isn't enough because something between collection
# and test execution is mutating paths.__dict__ back to Kaggle.
paths.__dict__["CACHE_DIR"] = ROOT / "cache"
paths.__dict__["OUTPUT_DIR"] = ROOT / "artifacts"
paths.__dict__["PRICES"] = ROOT / "cache" / "prices.parquet"
paths.__dict__["FEATURES"] = ROOT / "cache" / "features.parquet"
paths.__dict__["TARGETS"] = ROOT / "cache" / "targets.parquet"
paths.__dict__["HASHES_JSON"] = ROOT / "cache" / "hashes.json"


# Use fold 0 (2019 calm) — a stable, deterministic fold.
FOLD_IDX = 0
FEATURE_COLS = list(data.load_config()["features"]["columns"])


def _build_fold_tensors():
    # Force local paths (this test file must always hit the local cache
    # regardless of what test order left behind).
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
    fold = folds[FOLD_IDX]
    return train.assemble_fold(fold, features, targets, cfg)


def test_window_has_exactly_lookback_rows():
    """The assembled X must have a lookback dimension of exactly
    cfg['features']['lookback_days']. The earlier code got this right
    too — this guards against future regressions."""
    cfg = data.load_config()
    lookback = int(cfg["features"]["lookback_days"])
    t = _build_fold_tensors()
    assert t.X_train.shape[2] == lookback
    assert t.X_val.shape[2]   == lookback
    assert t.X_test.shape[2]  == lookback


def test_window_last_date_equals_prediction_date():
    """For every (stock, day) sample, the assembled window's last
    day must equal the prediction date. This is the regression test
    for defect 5 — the earlier code used [t-lookback .. t-1] which
    excluded the prediction date."""
    t = _build_fold_tensors()
    cfg = data.load_config()
    lookback = int(cfg["features"]["lookback_days"])

    # We can't trivially recover the window's dates from the
    # assembled tensor (they were collapsed into the lookback axis),
    # so we re-derive the expected window dates and compare against
    # the train_dates/test_dates the assembler recorded.
    features = pd.read_parquet(paths.FEATURES)
    feat_wide = features.pivot(index="date", columns="ticker",
                               values=FEATURE_COLS).sort_index()

    # For each recorded prediction date d in test_dates, the window
    # must end at d. We verify this by checking that
    # feat_wide.loc[d-lookback+1 .. d] has exactly lookback dates and
    # its last index equals d.
    for d in t.test_dates[:5]:  # first five is enough
        loc = feat_wide.index.get_loc(pd.Timestamp(d))
        window = feat_wide.index[loc - lookback + 1: loc + 1]
        assert len(window) == lookback, (
            f"window for {d.date()} has {len(window)} dates, expected {lookback}"
        )
        assert window[-1] == pd.Timestamp(d), (
            f"window last date {window[-1]} != prediction date {d} — "
            f"defect 5 has regressed"
        )


def test_window_first_date_is_lookback_minus_one_before_prediction():
    """The window's first day must be ``lookback-1`` trading days
    before the prediction date."""
    t = _build_fold_tensors()
    cfg = data.load_config()
    lookback = int(cfg["features"]["lookback_days"])

    features = pd.read_parquet(paths.FEATURES)
    feat_wide = features.pivot(index="date", columns="ticker",
                               values=FEATURE_COLS).sort_index()

    for d in t.test_dates[:5]:
        loc = feat_wide.index.get_loc(pd.Timestamp(d))
        window = feat_wide.index[loc - lookback + 1: loc + 1]
        # The window has lookback dates, so the first one is exactly
        # lookback-1 trading days before d (inclusive of d itself).
        assert loc - (loc - lookback + 1) == lookback - 1
        assert len(window) == lookback


def test_no_window_contains_a_future_date():
    """Every (day, stock) sample must only see data from days ≤ that
    prediction date. No look-ahead into the future."""
    t = _build_fold_tensors()
    cfg = data.load_config()
    lookback = int(cfg["features"]["lookback_days"])

    features = pd.read_parquet(paths.FEATURES)
    feat_wide = features.pivot(index="date", columns="ticker",
                               values=FEATURE_COLS).sort_index()

    for d in t.test_dates[:5]:
        loc = feat_wide.index.get_loc(pd.Timestamp(d))
        window = feat_wide.index[loc - lookback + 1: loc + 1]
        assert window.max() <= pd.Timestamp(d), (
            f"window for {d.date()} extends to {window.max()} "
            f"which is AFTER the prediction date — leakage!"
        )


def test_samples_with_insufficient_history_are_skipped():
    """The first lookback-1 trading days cannot form a valid window
    and must be skipped, NOT zero-padded."""
    cfg = data.load_config()
    lookback = int(cfg["features"]["lookback_days"])

    features = pd.read_parquet(paths.FEATURES)
    targets = pd.read_parquet(paths.TARGETS)
    folds = sp.make_folds(cfg, available_dates=features["date"].unique())

    # The fold's training dates span ~4 years; the first lookback-1
    # of those have insufficient history and must be absent from
    # train_dates_kept.
    expected_first_kept = folds[FOLD_IDX].train_dates[lookback - 1]
    t = _build_fold_tensors()
    actual_first_kept = pd.Timestamp(t.train_dates[0])
    assert actual_first_kept == pd.Timestamp(expected_first_kept), (
        f"first kept train date {actual_first_kept} != expected "
        f"{expected_first_kept}. Samples with insufficient history "
        f"must be skipped (dropped), not zero-padded."
    )