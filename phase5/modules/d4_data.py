"""DMRGT Phase 5 — data module (Phase 4 spec §6.1 + Phase 5 spec §3/§6).

Ported from Phase 3 (reference/phase3_src/): splits.py (make_folds, Fold),
train.assemble_fold (windows, label alignment, train-only scaler, graph-
coverage drop). Phase 4 changes kept: lookback is a parameter; the test split
is gated. Phase 5 changes: the gate needs FOUR lock files; the universe and
sector map come from the cache's universe.csv (via the config snapshot); six
folds (1-4 selection, 5-6 holdout); each fold's validation is split into
val_es / val_sel halves (Phase 4.1 F2); graph variants are keyed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

WORK = Path(os.environ.get("DMRGT4_WORK", "/kaggle/working"))
LOCKS = [WORK / "phase5_graph_winner.json", WORK / "phase5_postproc.json",
         WORK / "phase5_winner.json", WORK / "phase5b_policy.json"]

CANDIDATES = [
    Path("/kaggle/input/dmrgt-sp100-cache"),
    Path("/kaggle/input/datasets/harishsivakumar1912/dmrgt-sp100-cache"),
]


def resolve_cache() -> Path:
    env = os.environ.get("DMRGT4_CACHE")
    if env:
        return Path(env)
    for c in CANDIDATES:
        if (c / "features.parquet").exists():
            return c
    hits = list(Path("/kaggle/input").rglob("features.parquet"))
    if len(hits) == 1:
        return hits[0].parent
    raise FileNotFoundError(f"Cache not found or ambiguous: {hits}")


# ─── config (from the Cell 0 snapshot) ─────────────────────────────────────

def load_snapshot() -> dict:
    return json.loads((WORK / "config_snapshot.json").read_text())


def _graph_params(snap: dict, variant: str | None) -> dict:
    C = snap["CONFIG"]
    if variant is None:
        variant = snap.get("GRAPH_WINNER", {}).get("name", "baseline")
    return dict(C["graph_grid"]["variants"][variant])


def load_cfg(variant: str | None = None) -> dict:
    """Phase-3-shaped cfg dict assembled from the snapshot, so the ported
    eval / stats / fold code runs unchanged. `variant` selects the graph
    parameters (default: the S1 winner once locked, else baseline)."""
    snap = load_snapshot()
    C, S, Uv = snap["CONFIG"], snap["STATIC"], snap["UNIVERSE"]
    sig = C["significance"]
    gp = _graph_params(snap, variant)
    return {
        "universe": {"tickers": list(Uv["tickers"])},
        "sectors": dict(Uv["sectors"]),
        "features": {"lookback_days": 30, "columns": list(S["feature_columns"])},
        "graph": {"corr_window": gp["corr_window"], "k_pos": gp["k_pos"], "k_neg": gp["k_neg"],
                  "symmetrise": False, "include_sector": True, "min_history_days": gp["corr_window"] + 1},
        "training": {"huber_delta": C["training"]["huber_delta_fixed"]},
        "backtest": dict(C["backtest"]),
        "significance": {"wilcoxon_alpha": sig["alpha"],
                         "bootstrap_iterations": sig["bootstrap_iterations"],
                         "bootstrap_block_length": sig["bootstrap_block_length"]},
        "folds": list(S["folds"]),
    }


def split_val(val_dates):
    """Phase 4.1 F2.1: chronological halves of a fold's validation period.
    val_es → early stopping / LR scheduler only; val_sel → selection only."""
    k = len(val_dates) // 2
    return val_dates[:k], val_dates[k:]


# ─── folds (ported verbatim from splits.py) ────────────────────────────────

@dataclass
class Fold:
    """One walk-forward fold with train_dates, val_dates, test_dates."""
    name: str
    fold_idx: int
    train_dates: pd.DatetimeIndex
    val_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex

    def __repr__(self) -> str:
        def span(idx: pd.DatetimeIndex) -> str:
            if len(idx) == 0:
                return "empty"
            return f"{idx[0].date()}→{idx[-1].date()} (n={len(idx)})"
        return (f"Fold({self.name!r}, train={span(self.train_dates)}, "
                f"val={span(self.val_dates)}, test={span(self.test_dates)})")


def make_folds(
    cfg: dict,
    available_dates: Iterable[pd.Timestamp] | None = None,
    val_fraction: float = 0.15,
) -> list[Fold]:
    """Build walk-forward folds from configs.yaml.

    `available_dates` restricts each configured window to dates that
    actually exist in the cache (the features table's calendar). The
    latest val_fraction of each training window becomes validation.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    avail = (pd.DatetimeIndex(sorted(set(pd.Timestamp(d) for d in available_dates)))
             if available_dates is not None else None)
    folds_cfg = cfg.get("folds", [])
    if not folds_cfg:
        raise ValueError("configs.yaml has no 'folds' section")

    out: list[Fold] = []
    for i, fc in enumerate(folds_cfg):
        name = str(fc.get("name", f"fold_{i}"))
        train_start = pd.Timestamp(fc["train_start"])
        train_end = pd.Timestamp(fc["train_end"])
        test_start = pd.Timestamp(fc["test_start"])
        test_end = pd.Timestamp(fc["test_end"])

        # ── hard chronological invariant (Rule 2) ─────────────────────
        if train_end >= test_start:
            raise AssertionError(
                f"Fold {name!r}: train_end ({train_end.date()}) must be "
                f"strictly before test_start ({test_start.date()}) — "
                f"otherwise the model sees the test window during training."
            )

        # ── build the full training and test index ─────────────────────
        if avail is not None:
            # Configured window, intersected with dates the cache has.
            # The boundary itself is owned by the config; the data
            # restricts what we can actually use.
            train_mask = (avail >= train_start) & (avail <= train_end)
            test_mask = (avail >= test_start) & (avail <= test_end)
            train_full = avail[train_mask]
            test_idx = avail[test_mask]
        else:
            # No availability constraint (tests / running splits without
            # the cache): synthesise business-day dates. The next consumer
            # that loads the features table intersects again.
            train_full = pd.bdate_range(train_start, train_end)
            test_idx = pd.bdate_range(test_start, test_end)

        if len(train_full) == 0:
            raise ValueError(f"Fold {name!r}: no train dates after intersecting with available dates")
        if len(test_idx) == 0:
            raise ValueError(f"Fold {name!r}: no test dates after intersecting with available dates")

        # ── split off the validation portion ───────────────────────────
        # The latest val_fraction of the training window becomes the
        # validation set. Chronological, not random.
        n_train_full = len(train_full)
        n_val = int(round(n_train_full * val_fraction))
        if n_val <= 0 or n_val >= n_train_full:
            raise ValueError(
                f"Fold {name!r}: val_fraction={val_fraction} produced "
                f"n_val={n_val} from n_train_full={n_train_full}; "
                f"need at least 1 train and 1 val date"
            )
        train_idx = train_full[:-n_val]
        val_idx = train_full[-n_val:]

        # ── zero-overlap asserts — fail loudly, never silently ─────────
        assert len(set(train_idx).intersection(val_idx)) == 0
        assert len(set(train_idx).intersection(test_idx)) == 0
        assert len(set(val_idx).intersection(test_idx)) == 0
        # Chronological: last train date precedes first val date, which
        # precedes the first test date.
        assert train_idx.max() < val_idx.min()
        assert val_idx.max() < test_idx.min()
        assert train_idx.max() < test_idx.min()

        out.append(Fold(
            name=name,
            fold_idx=i,
            train_dates=train_idx,
            val_dates=val_idx,
            test_dates=test_idx,
        ))

    return out


# ─── test access gate (Change 2) ───────────────────────────────────────────

def test_unlocked() -> bool:
    return all(p.exists() for p in LOCKS)


def assert_test_unlocked() -> None:
    if not test_unlocked():
        raise PermissionError(
            "Test split is locked until phase5_graph_winner.json, phase5_postproc.json, "
            "phase5_winner.json and phase5b_policy.json all exist.")


# ─── tables, loaded once per process ───────────────────────────────────────

_TABLES: dict = {}


def tables() -> dict:
    """features / targets pivots + folds for this process (cached)."""
    if _TABLES:
        return _TABLES
    cache = resolve_cache()
    cfg = load_cfg()
    features = pd.read_parquet(cache / "features.parquet")
    targets = pd.read_parquet(cache / "targets.parquet")
    feature_cols = list(cfg["features"]["columns"])
    missing = [c for c in feature_cols if c not in features.columns]
    assert not missing, f"config lists feature columns absent from features.parquet: {missing}"
    assert len(feature_cols) == 8, f"Expected 8 features, got {len(feature_cols)}"
    feat_wide = (features.pivot(index="date", columns="ticker", values=feature_cols).sort_index())
    # Fix the feature level to config order (pivot sorts lexicographically).
    feat_wide = feat_wide.reindex(columns=pd.MultiIndex.from_product(
        [feature_cols, list(feat_wide.columns.get_level_values("ticker").unique())],
        names=["feature", "ticker"]))
    tgt_wide = (targets.pivot(index="date", columns="ticker", values="excess_return").sort_index())
    folds = make_folds(cfg, available_dates=features["date"].unique())
    _TABLES.update(cfg=cfg, cache=cache, features=features, targets=targets,
                   feat_wide=feat_wide, tgt_wide=tgt_wide, folds=folds,
                   feature_cols=feature_cols)
    return _TABLES


def fold_by_name(name: str) -> Fold:
    for f in tables()["folds"]:
        if f.name == name:
            return f
    raise KeyError(name)


@dataclass
class SplitData:
    name: str
    split: str
    X: np.ndarray            # [n_days, n_stocks, lookback, n_features], scaled
    y: np.ndarray            # [n_days, n_stocks]
    dates: np.ndarray
    tickers: list


class FoldBuilder:
    """Per-fold, per-lookback tensors. Train and val are built eagerly;
    test is built lazily and only after both lock files exist."""

    def __init__(self, fold: Fold, lookback: int, graph_dates=None):
        T = tables()
        self.fold, self.lookback, self.graph_dates = fold, lookback, graph_dates
        self.cfg = T["cfg"]
        self.universe = list(self.cfg["universe"]["tickers"])
        self.feature_cols = T["feature_cols"]
        self.feat_wide, self.tgt_wide = T["feat_wide"], T["tgt_wide"]
        self.dropped_no_graph = {"train": [], "val": [], "test": []}

        X_tr, y_tr, d_tr = self._build(fold.train_dates, "train")
        X_va, y_va, d_va = self._build(fold.val_dates, "val")
        assert len(X_tr) > 0, f"{fold.name}: no training windows at lookback {lookback}"

        # ── bounded lookback drop (Change 1) ──────────────────────────
        # Dates Phase 3's 30-day windows kept that this lookback loses.
        locs = [self.feat_wide.index.get_loc(d) for d in fold.train_dates
                if d in self.feat_wide.index]
        base = sum(1 for l in locs if l >= 30 - 1)
        now = sum(1 for l in locs if l >= lookback - 1)
        self.lookback_dropped = base - now
        self.lookback_drop_frac = self.lookback_dropped / max(1, base)
        assert self.lookback_drop_frac < 0.05, (
            f"{fold.name}: lookback {lookback} drops {self.lookback_dropped}/{base} "
            f"train days ({self.lookback_drop_frac:.2%}) — over the 5% bound")
        print(f"[d4_data] {fold.name} lookback={lookback}: dropped {self.lookback_dropped}/{base} "
              f"train days vs the 30-day window ({self.lookback_drop_frac:.2%})", flush=True)

        # ── bounded graph-coverage check (ported from Phase 3 §3.5) ───
        total_dropped = sum(len(v) for v in self.dropped_no_graph.values())
        total_dates = len(fold.train_dates) + len(fold.val_dates)
        if graph_dates is not None and total_dropped > 0:
            drop_frac = total_dropped / total_dates
            assert drop_frac < 0.05, (
                f"{total_dropped} dates ({drop_frac:.2%}) missing graph coverage — "
                f"check which graph file this run was pointed at.")

        # ── scaler fitted on training dates ONLY ──────────────────────
        n_feat = X_tr.shape[3]
        scaler = StandardScaler().fit(X_tr.reshape(-1, n_feat))
        assert scaler.mean_.shape == (n_feat,)
        self._scaler = scaler
        self.scaler_mean = scaler.mean_.astype(float)
        self.scaler_scale = scaler.scale_.astype(float)
        self.scaler_fit_dates = (pd.Timestamp(d_tr[0]), pd.Timestamp(d_tr[-1]))
        self._splits = {
            "train": SplitData(fold.name, "train", self._transform(X_tr), y_tr, d_tr, self.universe),
            "val": SplitData(fold.name, "val", self._transform(X_va), y_va, d_va, self.universe),
        }
        assert pd.Timestamp(d_tr.max()) < pd.Timestamp(d_va.min()), "train/val chronology violated"
        # Phase 4.1 F2.1: val_es (first half of the fold's validation dates) / val_sel (second half)
        es_dates, sel_dates = split_val(fold.val_dates)
        d_idx = pd.DatetimeIndex(d_va)
        in_es = np.asarray(d_idx.isin(es_dates))
        in_sel = np.asarray(d_idx.isin(sel_dates))
        assert in_es.sum() > 0 and in_sel.sum() > 0 and not (in_es & in_sel).any()
        assert pd.Timestamp(d_va[in_es].max()) < pd.Timestamp(d_va[in_sel].min())
        v = self._splits["val"]
        self._splits["val_es"] = SplitData(fold.name, "val_es", v.X[in_es], v.y[in_es], v.dates[in_es], self.universe)
        self._splits["val_sel"] = SplitData(fold.name, "val_sel", v.X[in_sel], v.y[in_sel], v.dates[in_sel], self.universe)

    def _transform(self, X):
        if len(X) == 0:
            return X
        shape = X.shape
        return self._scaler.transform(X.reshape(-1, shape[3])).reshape(shape).astype(np.float32)

    def _build(self, date_set, which: str):
        lookback, feat_wide, tgt_wide = self.lookback, self.feat_wide, self.tgt_wide
        universe = self.universe
        Xs, ys, kept = [], [], []
        for d in date_set:
            if d not in feat_wide.index:
                continue
            if d not in tgt_wide.index:
                continue
            loc = feat_wide.index.get_loc(d)
            start = loc - lookback + 1
            if start < 0:
                continue  # not enough history — drop, never zero-pad
            if self.graph_dates is not None and d not in self.graph_dates:
                self.dropped_no_graph[which].append(d)
                continue
            window_dates = feat_wide.index[start: loc + 1]
            assert len(window_dates) == lookback
            assert window_dates[-1] == d, "window last date != prediction date"
            X = np.empty((len(universe), lookback, len(self.feature_cols)), dtype=np.float64)
            for j, t in enumerate(universe):
                X[j] = feat_wide.loc[window_dates, (slice(None), t)].values
            Xs.append(X)
            kept.append(d)
            ys.append(tgt_wide.loc[d, universe].values)
        X = (np.stack(Xs, axis=0) if Xs
             else np.empty((0, len(universe), lookback, len(self.feature_cols))))
        y = np.stack(ys, axis=0) if ys else np.empty((0, len(universe)))
        return X, y, np.array(kept)

    def get(self, split: str) -> SplitData:
        if split == "test":
            assert_test_unlocked()
            if "test" not in self._splits:
                X, y, d = self._build(self.fold.test_dates, "test")
                assert len(X) > 0
                assert pd.Timestamp(self._splits["val"].dates.max()) < pd.Timestamp(d.min()), \
                    "val/test chronology violated"
                self._splits["test"] = SplitData(self.fold.name, "test", self._transform(X), y, d,
                                                 self.universe)
        return self._splits[split]


from collections import OrderedDict

_BUILDERS: "OrderedDict" = OrderedDict()
_BUILDER_CAP = 2     # a fold at lookback 60 is ~0.5-0.7 GB; keep the working set small


def get_builder(fold_name: str, lookback: int = 30, graph_dates=None, graph_key: str = "baseline") -> FoldBuilder:
    key = (fold_name, lookback, graph_key)
    if key in _BUILDERS:
        _BUILDERS.move_to_end(key)
    else:
        _BUILDERS[key] = FoldBuilder(fold_by_name(fold_name), lookback, graph_dates)
        while len(_BUILDERS) > _BUILDER_CAP:
            _BUILDERS.popitem(last=False)
    return _BUILDERS[key]


def get_split(fold, split, lookback: int = 30, graph_dates=None, graph_key: str = "baseline") -> SplitData:
    """The only way tensors leave this module. split='test' raises
    PermissionError until all four phase-5 lock files exist."""
    if split == "test":
        assert_test_unlocked()
    name = fold if isinstance(fold, str) else fold.name
    return get_builder(name, lookback, graph_dates, graph_key).get(split)
