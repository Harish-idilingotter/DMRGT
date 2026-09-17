"""DMRGT Phase 3 — trainer (§8).

Kaggle-side training for the two Phase 3 arms (Rule 18 — exactly two):

  dmrgt_all3   dynamic daily graph  → cache/graphs.parquet
  dmrgt_frozen  per-fold frozen graph → cache/graphs_frozen.parquet

The model class is ONE DMRGT (no model_id branching in model.py — Rule 19);
the arms differ ONLY in the edge file train.py feeds them. The §6b grid
session reuses this entry point with `--edge-source` pointing at a
variant parquet and `--output-suffix` marking the outputs.

Before any training: verify SHA-256 of all eleven cache parquets (4
canonical + 7 flat grid variants) against hashes.json and HALT on any
mismatch. After training: predict on BOTH the val and test splits (best
weights restored), emitting a predictions parquet with a `split` column
— winner selection and tuning read val rows only (Rule 17), the one-shot
final scoring reads test rows exactly once (Rule 18).

run_manifest.json ACCUMULATES: every run appends its record to the
existing list instead of overwriting the file (the Phase 2 defect).

Usage (Kaggle / local with the cache present):
  python train.py --model dmrgt_all3
  python train.py --model dmrgt_frozen
  python train.py --model dmrgt_all3 \\
      --edge-source cache/graphs_kpos6.parquet \\
      --output-suffix _grid_kpos6
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

import data
import model as model_mod
import paths
import splits as sp


# ─── cache integrity gate (§8 step 1) ─────────────────────────────────────

def verify_cache_hashes(stop_on_mismatch: bool = True) -> dict[str, str]:
    """Recompute SHA-256 of each cached parquet, compare against
    hashes.json. Halts on ANY mismatch, missing file, or missing entry.

    The counting guard: a truncated hashes.json must not pass by
    verifying nothing — all len(CACHE_PARQUETS) files must be verified.
    """
    if not paths.HASHES_JSON.exists():
        raise RuntimeError(
            f"HALT: {paths.HASHES_JSON} not found. Cannot verify data "
            f"integrity. Regenerate the cache locally and re-upload."
        )
    expected = paths.read_hashes()
    print(f"[train] cache generated at: "
          f"{expected.get('generated_at', 'unknown')}")

    found: dict[str, str] = {}
    problems: list[str] = []
    for name in paths.CACHE_PARQUETS:
        if name not in expected:
            problems.append(f"{name} has no entry in hashes.json")
            continue
        p = paths.CACHE_DIR / name
        if not p.exists():
            problems.append(f"{name} listed in hashes.json but missing from cache")
            continue
        got = paths.sha256_of_file(p)
        found[name] = got
        if got != expected[name]:
            problems.append(
                f"{name} hash mismatch\n"
                f"      expected: {expected[name]}\n"
                f"      got:      {got}"
            )
        else:
            print(f"[train] hash OK: {name}  {got[:12]}…")

    if len(found) != len(paths.CACHE_PARQUETS) and not problems:
        problems.append(
            f"verified {len(found)} files, expected {len(paths.CACHE_PARQUETS)}"
        )

    if problems and stop_on_mismatch:
        raise RuntimeError(
            "HALT: cache verification failed. Refusing to train.\n  "
            + "\n  ".join(problems)
            + "\nMost common cause: an older dmrgt-phase-3-cache version is "
              "attached. Re-attach the latest version and RESTART the "
              "Kaggle session."
        )
    for p_ in problems:
        print(f"[train] WARNING: {p_}")

    print(f"[train] all {len(found)} parquet hashes verified")
    return found


def verify_aux_hash(parquet_path: Path) -> str | None:
    """Hash-verify a NON-canonical edge file (a §6b grid variant).
    Halts if the file's hash disagrees with hashes.json; warns if the
    file has no entry (unhashed upload) — canonical files halt instead."""
    hashes = paths.read_hashes()
    got = paths.sha256_of_file(parquet_path)
    key = parquet_path.name
    if key not in hashes:
        print(f"[train] WARNING: {key} has no entry in hashes.json — "
              f"cannot verify integrity (got {got[:12]}…)")
        return None
    if got != hashes[key]:
        raise RuntimeError(
            f"HALT: {key} hash mismatch.\n      expected: {hashes[key]}\n"
            f"      got:      {got}"
        )
    print(f"[train] hash OK: {key}  {got[:12]}…")
    return got


# ─── fold tensor assembly (reference semantics) ───────────────────────────

@dataclass
class FoldTensors:
    name: str
    fold_idx: int
    X_train: np.ndarray   # [n_train_days, n_stocks, lookback, n_features]
    y_train: np.ndarray   # [n_train_days, n_stocks]
    X_val:   np.ndarray
    y_val:   np.ndarray
    X_test:  np.ndarray
    y_test:  np.ndarray
    test_dates: np.ndarray
    train_dates: np.ndarray
    val_dates: np.ndarray
    test_tickers: list[str]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    fitted_scaler_before_test: bool
    feature_cols: list[str]
    edge_index_by_date: dict | None
    edge_type_by_date: dict | None


def assemble_fold(
    fold: sp.Fold,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    cfg: dict,
    graphs: pd.DataFrame | None = None,
) -> FoldTensors:
    """Per-fold train/val/test tensors. Scaler fitted on train only.

    The window for prediction date t is [t-lookback+1 .. t] INCLUSIVE of
    t (Rule 3 — features from day t and earlier); the label is t+1's
    excess return. Edge lookup is by sample date — the caller's graph
    file (dynamic or frozen) defines what each date sees.

    Dates the edge file does not cover are DROPPED, never crash the run
    (§3.5): a wider corr_window (win90) starts a handful of dates later
    than features.parquet's warm-up, so the earliest train dates of a
    fold legitimately lack a graph entry. The drop is logged and bounded
    (must stay under 5% of the fold's dates) — a larger gap means the
    wrong edge file reached this run.
    """
    universe = list(cfg["universe"]["tickers"])

    # Feature columns from config only — NEVER inferred from parquet
    # columns. A schema change must fail loudly.
    feature_cols = list(cfg["features"]["columns"])
    missing = [c for c in feature_cols if c not in features.columns]
    assert not missing, (
        f"configs.yaml lists feature columns absent from features.parquet: "
        f"{missing}. Available: {sorted(features.columns)}"
    )
    assert len(feature_cols) == 8, (
        f"Expected 8 features, got {len(feature_cols)}"
    )

    lookback = int(cfg["features"]["lookback_days"])

    feat_wide = (features.pivot(index="date", columns="ticker",
                                values=feature_cols).sort_index())
    # Fix the feature level to configs order (pivot sorts columns
    # lexicographically — see eval.py for the same reindex).
    feat_wide = feat_wide.reindex(columns=pd.MultiIndex.from_product(
        [feature_cols, list(feat_wide.columns.get_level_values("ticker").unique())],
        names=["feature", "ticker"]))
    tgt_wide = (targets.pivot(index="date", columns="ticker",
                              values="excess_return").sort_index())

    def build(date_set, with_labels: bool):
        Xs, ys, kept_dates, dropped_no_graph = [], [], [], []
        for d in date_set:
            if d not in feat_wide.index:
                continue
            if with_labels and d not in tgt_wide.index:
                continue
            loc = feat_wide.index.get_loc(d)
            start = loc - lookback + 1
            if start < 0:
                continue  # not enough history — drop, never zero-pad
            if graph_dates is not None and d not in graph_dates:
                # Expected window-boundary effect, not a bug or a leak
                # (§3.5): a wider corr_window — win90 in the grid — needs
                # more trailing history before its first valid graph than
                # features.parquet's own warm-up covers, so a handful of
                # dates at the very start of a fold's training window has
                # no edge entry. Checked AFTER the lookback check so the
                # count below reflects only dates that would otherwise
                # have been used — the true boundary gap, not warm-up
                # noise. Measured on the real cache: win90 drops 2 kept
                # train dates per fold, every other variant drops 0.
                dropped_no_graph.append(d)
                continue
            window_dates = feat_wide.index[start: loc + 1]
            assert len(window_dates) == lookback, (
                f"window has {len(window_dates)} dates, expected {lookback}"
            )
            assert window_dates[-1] == d, (
                f"window last date {window_dates[-1]} != prediction date {d}"
            )
            X = np.empty((len(universe), lookback, len(feature_cols)),
                         dtype=np.float64)
            for j, t in enumerate(universe):
                X[j] = feat_wide.loc[window_dates, (slice(None), t)].values
            Xs.append(X)
            kept_dates.append(d)
            if with_labels:
                ys.append(tgt_wide.loc[d, universe].values)
        X = (np.stack(Xs, axis=0) if Xs
             else np.empty((0, len(universe), lookback, len(feature_cols))))
        y = np.stack(ys, axis=0) if ys else np.empty((0, len(universe)))
        return X, y, np.array(kept_dates), dropped_no_graph

    # ── graph lookup by date ─────────────────────────────────────────
    # Resolved BEFORE tensor assembly so build() can drop uncovered
    # dates. The frozen edge file carries a `fold` column (its date
    # ranges overlap across folds — each fold stamps its own snapshot
    # onto shared train dates); resolve THIS fold's edges first. The
    # dynamic file has no fold column and passes through unchanged.
    edge_index_by_date = None
    edge_type_by_date = None
    graph_dates = None
    if graphs is not None:
        if "fold" in graphs.columns:
            graphs = graphs[graphs["fold"] == fold.name]
            assert len(graphs) > 0, (
                f"edge file has no rows for fold {fold.name!r} — wrong "
                f"edge source or a stale frozen graph build"
            )
        edge_index_by_date = {}
        edge_type_by_date = {}
        for d in graphs["date"].unique():
            day = graphs[graphs["date"] == d]
            ei = torch.tensor(day[["src", "dst"]].values.T, dtype=torch.long)
            et = torch.tensor(day["edge_type"].values, dtype=torch.long)
            edge_index_by_date[pd.Timestamp(d)] = ei
            edge_type_by_date[pd.Timestamp(d)] = et
        graph_dates = set(edge_index_by_date.keys())

    X_train, y_train, train_dates_kept, dropped_train = \
        build(fold.train_dates, True)
    X_val,   y_val,   val_dates_kept,   dropped_val   = \
        build(fold.val_dates, True)
    X_test,  y_test,  test_dates_kept,  dropped_test  = \
        build(fold.test_dates, True)

    # ── bounded graph-coverage sanity check (§3.5) ───────────────────
    # win90 measured 3 dropped train dates per fold against the real
    # cache (single digits) and every other variant drops zero. Anything
    # approaching 5% means the wrong edge file reached this run.
    total_dropped = len(dropped_train) + len(dropped_val) + len(dropped_test)
    total_dates = (len(fold.train_dates) + len(fold.val_dates)
                   + len(fold.test_dates))
    if graphs is not None and total_dropped > 0:
        drop_frac = total_dropped / total_dates
        print(f"[assemble_fold] {fold.name}: dropped {total_dropped} dates "
              f"({drop_frac:.2%}) with no graph entry — first: "
              f"{[str(pd.Timestamp(d).date()) for d in sorted(dropped_train + dropped_val + dropped_test)[:3]]}",
              flush=True)
        assert drop_frac < 0.05, (
            f"{total_dropped} dates ({drop_frac:.2%}) missing graph coverage "
            f"— far more than a window-boundary effect. Check which graph "
            f"file this run was pointed at."
        )

    # ── hard asserts ────────────────────────────────────────────────
    train_idx = pd.DatetimeIndex(train_dates_kept)
    val_idx = pd.DatetimeIndex(val_dates_kept)
    test_idx = pd.DatetimeIndex(test_dates_kept)
    if len(train_idx) > 0 and len(test_idx) > 0:
        assert len(train_idx.intersection(test_idx)) == 0, (
            "test dates appear in the training set"
        )
        assert train_idx.max() < test_idx.min(), "chronology violated"
    if len(val_idx) > 0 and len(test_idx) > 0:
        assert val_idx.max() < test_idx.min(), "val/test chronology violated"

    # ── scaler fitted on training dates ONLY ─────────────────────────
    if len(X_train) > 0:
        n_feat = X_train.shape[3]
        flat = X_train.reshape(-1, n_feat)
        scaler = StandardScaler().fit(flat)
        assert scaler.mean_.shape == (n_feat,), (
            f"Scaler fitted {scaler.mean_.shape[0]} params, expected "
            f"{n_feat} — pooling should give one mean per feature"
        )

        def transform(X):
            if len(X) == 0:
                return X
            shape = X.shape
            return scaler.transform(X.reshape(-1, n_feat)).reshape(shape)

        X_train = transform(X_train)
        X_val = transform(X_val)
        X_test = transform(X_test)
        scaler_fitted_before_test = True
        scaler_mean = scaler.mean_.astype(float)
        scaler_scale = scaler.scale_.astype(float)
    else:
        scaler_fitted_before_test = False
        scaler_mean = np.array([])
        scaler_scale = np.array([])

    for arr_name, arr in [("X_train", X_train), ("X_val", X_val),
                          ("X_test", X_test)]:
        if len(arr) > 0:
            assert arr.shape[2] == lookback, f"{arr_name} lookback dim wrong"
            assert arr.shape[3] == len(feature_cols), f"{arr_name} feature dim wrong"
            assert arr.shape[1] == len(universe), f"{arr_name} n_stocks dim wrong"

    return FoldTensors(
        name=fold.name,
        fold_idx=fold.fold_idx,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        test_dates=test_dates_kept,
        train_dates=train_dates_kept,
        val_dates=val_dates_kept,
        test_tickers=universe,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        fitted_scaler_before_test=scaler_fitted_before_test,
        feature_cols=feature_cols,
        edge_index_by_date=edge_index_by_date,
        edge_type_by_date=edge_type_by_date,
    )


# ─── seeding + init assertion (Rule 15) ───────────────────────────────────

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _param_fingerprint(model: torch.nn.Module) -> str:
    """Byte-level fingerprint of a model's trainable parameters.

    Why bytes and not sums: float addition is not associative, so CUDA's
    parallel reduction and the CPU's sequential sum give different last
    bits for identical weights. ``.cpu()`` preserves the bit pattern
    exactly, making the hash stable on any device.
    """
    import hashlib
    h = hashlib.sha256()
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def make_model(n_features: int = 8) -> torch.nn.Module:
    """The ONE Phase 3 model (Rule 19). No model_id branching — the arms
    differ only in the edge data they are fed."""
    return model_mod.DMRGT(n_features=n_features)


# ─── training one (fold, seed) ─────────────────────────────────────────────

def train_one_seed(
    tensors: FoldTensors,
    cfg: dict,
    seed: int,
    device: torch.device,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    """Train one seed on one fold. Returns (epoch metrics, val_pred,
    test_pred, best_state) — predictions are made with the BEST weights
    (lowest val loss), for BOTH splits."""
    epochs = int(cfg["training"]["epochs"])
    batch_size = int(cfg["training"]["batch_size"])
    lr = float(cfg["training"]["learning_rate"])
    huber_delta = float(cfg["training"]["huber_delta"])
    patience = int(cfg["training"]["early_stopping_patience"])

    n_stocks = (tensors.X_train.shape[1] if len(tensors.X_train) > 0
                else len(cfg["universe"]["tickers"]))
    n_features = tensors.X_train.shape[3] if len(tensors.X_train) > 0 else 8

    # ── Rule 15: random-init assertion, on CPU ────────────────────────
    # Same seed twice → identical byte fingerprints (determinism);
    # different seed → different fingerprint (weights really are drawn
    # from the seed, not loaded from anywhere).
    set_seed(seed)
    net = make_model(n_features=n_features)          # CPU
    fp1 = _param_fingerprint(net)

    set_seed(seed)
    net2 = make_model(n_features=n_features)         # CPU
    fp2 = _param_fingerprint(net2)
    assert fp1 == fp2, (
        f"Random-init assertion failed: same seed produced different "
        f"params (seed={seed}). Seeding is not deterministic."
    )

    set_seed(seed + 9999)
    net3 = make_model(n_features=n_features)         # CPU
    fp3 = _param_fingerprint(net3)
    assert fp1 != fp3, (
        f"Random-init assertion failed: different seeds produced "
        f"identical params (seed={seed} vs {seed + 9999}). The model may "
        f"be ignoring the seed, or weights were loaded."
    )
    del net2, net3, fp2, fp3

    # Checks passed — move the training model to the device, then
    # restore the RNG state consumed by the two extra constructions so
    # dropout and shuffling start from the seeded point.
    net = net.to(device)
    set_seed(seed)

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.HuberLoss(delta=huber_delta)

    X_train_t = torch.from_numpy(tensors.X_train).float().to(device)
    y_train_t = torch.from_numpy(tensors.y_train).float().to(device)
    X_val_t = torch.from_numpy(tensors.X_val).float().to(device)
    y_val_t = torch.from_numpy(tensors.y_val).float().to(device)

    def edge_for(dates, edge_by_date, edge_type_by_date):
        ei_list = [edge_by_date[pd.Timestamp(d)].to(device) for d in dates]
        et_list = [edge_type_by_date[pd.Timestamp(d)].to(device) for d in dates]
        return ei_list, et_list

    train_ei, train_et = edge_for(tensors.train_dates,
                                  tensors.edge_index_by_date,
                                  tensors.edge_type_by_date)
    val_ei, val_et = edge_for(tensors.val_dates,
                              tensors.edge_index_by_date,
                              tensors.edge_type_by_date)

    metrics = []
    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    n_train = len(X_train_t)

    for epoch in range(epochs):
        net.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n_train, batch_size):
            idx = perm[i: i + batch_size]
            Xb = X_train_t[idx]
            yb = y_train_t[idx]
            opt.zero_grad()
            # Each day has its own edge list. Handing the whole batch to
            # forward_with_edge_lists keeps one R-GCN call per step instead
            # of one per day — the per-day loop was launch-bound and made
            # the grid ~20x slower than the time budget assumed.
            rows = idx.tolist()
            yp = net.forward_with_edge_lists(
                Xb, [train_ei[j] for j in rows], [train_et[j] for j in rows]
            )                                           # [B, n_stocks]
            loss = loss_fn(yp, yb)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        train_loss = epoch_loss / max(1, n_batches)

        net.eval()
        with torch.no_grad():
            val_loss = 0.0
            n_vb = 0
            for i in range(0, len(X_val_t), batch_size):
                Xb = X_val_t[i: i + batch_size]
                yb = y_val_t[i: i + batch_size]
                n_b = len(Xb)
                yp = net.forward_with_edge_lists(
                    Xb, val_ei[i: i + n_b], val_et[i: i + n_b]
                )
                val_loss += float(loss_fn(yp, yb).item())
                n_vb += 1
            val_loss /= max(1, n_vb)

        metrics.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone()
                          for k, v in net.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    # Restore best weights (early stopping checkpoint)
    if best_state is not None:
        net.load_state_dict(best_state)

    # ── predict BOTH splits with the best weights ─────────────────────
    net.eval()
    def predict(X: np.ndarray, dates: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.empty((0, n_stocks))
        X_t = torch.from_numpy(X).float().to(device)
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_t), batch_size):
                Xb = X_t[i: i + batch_size]
                ei_list = [tensors.edge_index_by_date[pd.Timestamp(d)]
                           .to(device)
                           for d in dates[i: i + len(Xb)]]
                et_list = [tensors.edge_type_by_date[pd.Timestamp(d)]
                           .to(device)
                           for d in dates[i: i + len(Xb)]]
                yp = net.forward_with_edge_lists(Xb, ei_list, et_list)
                preds.append(yp.cpu().numpy())
        return np.concatenate(preds, axis=0)

    val_pred = predict(tensors.X_val, tensors.val_dates)
    test_pred = predict(tensors.X_test, tensors.test_dates)
    return (pd.DataFrame(metrics), val_pred, test_pred, best_state)


# ─── run orchestration ─────────────────────────────────────────────────────

def resolve_edge_source(model_id: str, cfg: dict,
                        edge_source: str | Path | None = None) -> Path:
    """§8: the arm's edge file. dmrgt_all3 → dynamic graphs.parquet;
    dmrgt_frozen → per-fold frozen graphs_frozen.parquet. An explicit
    edge_source override (the §6b grid variants) wins over the default.
    """
    if edge_source is not None:
        p = Path(edge_source)
        if not p.is_absolute():
            p = (paths.CACHE_DIR / edge_source
                 if not Path(edge_source).exists()
                 else p.resolve())
        if not p.exists():
            raise FileNotFoundError(f"edge source not found: {p}")
        return p
    if model_id == "dmrgt_all3":
        return paths.GRAPHS
    if model_id == "dmrgt_frozen":
        return paths.GRAPHS_FROZEN
    raise ValueError(
        f"Unknown model_id {model_id!r} — must be one of "
        f"{cfg.get('models', {}).get('ids', [])}"
    )


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _load_manifest() -> list:
    """run_manifest.json ACCUMULATES (Rule: one file, all runs). Load
    the existing list if present; start a new list otherwise."""
    if paths.RUN_MANIFEST.exists():
        try:
            existing = json.loads(paths.RUN_MANIFEST.read_text())
            if isinstance(existing, list):
                return existing
            return [existing]  # a Phase 2-style single dict — wrap it
        except (json.JSONDecodeError, OSError):
            print("[train] WARNING: run_manifest.json unreadable — "
                  "starting a fresh manifest")
    return []


def _assign_folds(folds: list, n_devices: int) -> list[list]:
    """Split folds across devices, balanced by train-window length.

    A fold's wall cost grows with its train window (more dates per
    epoch at the same epoch cap), so window length is a static,
    deterministic cost proxy — no timing information needed. Two
    devices get the longest+shortest pairing (for the 4 walk-forward
    folds that is the perfectly balanced split: {f1, f4} vs {f2, f3});
    any other device count gets round-robin over the descending order.
    """
    if n_devices <= 1:
        return [list(folds)]
    order = sorted(folds, key=lambda f: len(f.train_dates), reverse=True)
    buckets: list[list] = [[] for _ in range(n_devices)]
    rest = order
    if n_devices == 2 and len(order) >= 4:
        buckets[0].append(order[0])
        buckets[1].append(order[1])
        buckets[0].append(order[-1])
        buckets[1].append(order[-2])
        rest = order[2:-2]
    for i, f in enumerate(rest):
        buckets[i % n_devices].append(f)
    return buckets


def _train_fold(
    fold,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    cfg: dict,
    graphs: pd.DataFrame,
    model_id: str,
    model_name: str,
    seeds: list[int],
    ckpt_dir: Path,
    device: torch.device,
) -> tuple[list[dict], dict[str, list], dict]:
    """Train every seed of ONE fold on ONE device (see run_all).

    Thread-safe by construction: touches only its own fold's tensors,
    its own checkpoint files, and its own local containers — the caller
    merges the outputs after all folds join. Every run reseeds itself
    (`train_one_seed` → `set_seed(seed)`), so the schedule cannot
    change any number.
    """
    print(f"\n[train] === fold {fold.fold_idx} {fold.name!r} on {device} ===")
    print(f"  train: {fold.train_dates[0].date()} → "
          f"{fold.train_dates[-1].date()}  (n={len(fold.train_dates)})")
    print(f"  val:   {fold.val_dates[0].date()} → "
          f"{fold.val_dates[-1].date()}  (n={len(fold.val_dates)})")
    print(f"  test:  {fold.test_dates[0].date()} → "
          f"{fold.test_dates[-1].date()}  (n={len(fold.test_dates)})")

    tensors = assemble_fold(fold, features, targets, cfg, graphs)
    print(f"  X_train {tensors.X_train.shape}  "
          f"X_val {tensors.X_val.shape}  "
          f"X_test {tensors.X_test.shape}")
    assert tensors.fitted_scaler_before_test, (
        f"fold {fold.name}: scaler was not fitted before test data "
        f"was touched"
    )

    n_val = len(tensors.val_dates)
    n_test = len(tensors.test_dates)
    n_stocks = len(tensors.test_tickers)

    pred_rows: list[dict] = []
    metrics_entries: dict[str, list] = {}

    for seed in seeds:
        print(f"  --- seed {seed} ---")
        t0 = time.time()
        metrics_df, val_pred, test_pred, best_state = train_one_seed(
            tensors, cfg, seed, device)
        dt = time.time() - t0
        print(f"      epochs {len(metrics_df)}  "
              f"best/final val_loss "
              f"{metrics_df['val_loss'].min():.6f} — "
              f"elapsed {dt:.1f}s")
        metrics_entries[f"{fold.name}/seed_{seed}"] = \
            metrics_df.to_dict(orient="records")

        assert val_pred.shape == (n_val, n_stocks), (
            f"val prediction shape {val_pred.shape} != "
            f"({n_val}, {n_stocks})"
        )
        assert test_pred.shape == (n_test, n_stocks), (
            f"test prediction shape {test_pred.shape} != "
            f"({n_test}, {n_stocks})"
        )

        for split_name, preds, y, dates in [
            ("val", val_pred, tensors.y_val, tensors.val_dates),
            ("test", test_pred, tensors.y_test, tensors.test_dates),
        ]:
            for i, d in enumerate(dates):
                for j, t in enumerate(tensors.test_tickers):
                    pred_rows.append({
                        "model": model_name,
                        "fold": fold.name,
                        "seed": int(seed),
                        "split": split_name,
                        "date": pd.Timestamp(d),
                        "ticker": t,
                        "y_pred": float(preds[i, j]),
                        "y_true": float(y[i, j]),
                    })

        ckpt_path = ckpt_dir / f"{model_name}__{fold.name}__seed{seed}.pt"
        torch.save({
            "model_id": model_id,
            "model_name": model_name,
            "fold": fold.name,
            "seed": seed,
            "model_state_dict": best_state,
            "fold_meta": {
                "n_train_days": int(len(tensors.X_train)),
                "n_val_days": int(n_val),
                "n_test_days": int(n_test),
                "test_tickers": tensors.test_tickers,
            },
            "scaler_mean": tensors.scaler_mean,
            "scaler_scale": tensors.scaler_scale,
            "feature_cols": tensors.feature_cols,
        }, ckpt_path)

    fold_result = {
        "fold_idx": int(fold.fold_idx),
        "fold": fold.name,
        "n_train_days": int(len(tensors.X_train)),
        "n_val_days": int(n_val),
        "n_test_days": int(n_test),
        "n_stocks": int(n_stocks),
        "train_start": str(tensors.train_dates[0].date()),
        "train_end": str(tensors.train_dates[-1].date()),
        "val_start": str(tensors.val_dates[0].date()),
        "val_end": str(tensors.val_dates[-1].date()),
        "test_start": str(tensors.test_dates[0].date()),
        "test_end": str(tensors.test_dates[-1].date()),
        "feature_cols": tensors.feature_cols,
        "scaler_mean": tensors.scaler_mean.tolist(),
        "scaler_scale": tensors.scaler_scale.tolist(),
        "seeds": seeds,
    }
    return pred_rows, metrics_entries, fold_result


def run_all(
    cfg: dict,
    model_id: str,
    edge_source: str | Path | None = None,
    output_suffix: str = "",
    verify_hashes: bool = True,
    hashes: dict[str, str] | None = None,
    devices: list[str] | None = None,
) -> pd.DataFrame:
    """Train model_id over all folds × seeds; write
    predictions_{model_id}{suffix}.parquet (val AND test rows),
    train_metrics_{model_id}{suffix}.json, checkpoints, and append to the
    accumulating run_manifest.json. Returns the predictions frame.

    `devices` schedules FOLDS across devices (e.g. ["cuda:0", "cuda:1"]
    on the T4 x2 accelerator — the grid's ~11 h single-GPU projection
    only cleared the session cap once the second T4 carried half the
    folds). Pure scheduling: every run still reseeds itself
    (`train_one_seed` → `set_seed(seed)`), so results are bitwise
    identical to the serial schedule; folds are merged in their original
    chronological order. None → the one default device, exactly the old
    behaviour."""
    model_name = f"{model_id}{output_suffix}"
    paths.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_dir = paths.CHECKPOINTS_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── model id gate (Rule 18 — exactly two models) ─────────────────
    allowed = list(cfg.get("models", {}).get("ids", []))
    if allowed and model_id not in allowed:
        raise ValueError(
            f"model_id {model_id!r} is not in configs.yaml models.ids "
            f"({allowed}). Phase 3 trains exactly the two arms — nothing "
            f"else."
        )

    # ── 5-hash verification before anything else (§8 step 1) ─────────
    if verify_hashes:
        print("[train] verifying cache hashes…")
        hash_check = verify_cache_hashes(stop_on_mismatch=True)
    elif hashes is not None:
        hash_check = dict(hashes)
        print(f"[train] using pre-verified hashes ({len(hash_check)} files)")
    else:
        raise ValueError(
            "verify_hashes=False requires hashes= from a prior "
            "verify_cache_hashes() call."
        )

    if devices:
        device_list = [torch.device(d) for d in devices]
    else:
        device_list = [torch.device("cuda" if torch.cuda.is_available()
                                    else "cpu")]
    print(f"[train] device(s): "
          + ", ".join(
              f"{d} ({torch.cuda.get_device_name(d)})"
              if d.type == "cuda" else str(d)
              for d in device_list))

    # ── edge source by arm (§8) ───────────────────────────────────────
    edge_path = resolve_edge_source(model_id, cfg, edge_source)
    print(f"[train] model {model_name!r} edge source: {edge_path}")
    if edge_path != paths.GRAPHS and edge_path != paths.GRAPHS_FROZEN:
        verify_aux_hash(edge_path)  # a grid variant — verify its hash too
    graphs = pd.read_parquet(edge_path)
    print(f"[train] loaded edges: {len(graphs):,} rows, "
          f"{graphs['date'].nunique()} dates")

    print("[train] loading features & targets…")
    features = pd.read_parquet(paths.FEATURES)
    targets = pd.read_parquet(paths.TARGETS)

    folds = sp.make_folds(cfg, available_dates=features["date"].unique())
    seeds = list(cfg["training"]["seeds"])

    buckets = _assign_folds(folds, len(device_list))

    def _run_bucket(bucket_folds: list, device: torch.device) -> tuple:
        rows: list[dict] = []
        entries: dict[str, list] = {}
        results: list[dict] = []
        for fold in bucket_folds:
            r, m, fr = _train_fold(fold, features, targets, cfg, graphs,
                                   model_id, model_name, seeds, ckpt_dir,
                                   device)
            rows.extend(r)
            entries.update(m)
            results.append(fr)
        return rows, entries, results

    if len(device_list) == 1:
        all_predictions, all_metrics, fold_results = _run_bucket(
            buckets[0], device_list[0])
    else:
        # One thread per device; each thread trains its folds' seeds
        # sequentially. Every run reseeds itself (`train_one_seed` →
        # `set_seed(seed)`), so scheduling cannot change any number —
        # this is wall-clock engineering only.
        errors: list[tuple[str, BaseException]] = []
        per_device: dict[int, tuple] = {}

        def _worker(idx: int) -> None:
            try:
                per_device[idx] = _run_bucket(buckets[idx],
                                              device_list[idx])
            except BaseException as e:  # noqa: BLE001 — re-raised below
                errors.append((f"device {device_list[idx]}", e))

        threads = [threading.Thread(target=_worker, args=(i,),
                                    name=f"train-device-{i}")
                   for i in range(len(device_list))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            where, first = errors[0]
            raise RuntimeError(
                f"fold training failed on {where}") from first
        all_predictions, all_metrics, fold_results = [], {}, []
        for i in range(len(device_list)):
            rows, entries, results = per_device[i]
            all_predictions.extend(rows)
            all_metrics.update(entries)
            fold_results.extend(results)
        # Restore the chronological order the serial schedule produces —
        # devices finish in whatever order their folds' costs dictate.
        fold_order = {f.name: i for i, f in enumerate(folds)}
        fold_results.sort(key=lambda fr: fold_order[fr["fold"]])
        all_predictions.sort(key=lambda r: fold_order[r["fold"]])

    # ── write predictions ────────────────────────────────────────────
    pred_df = pd.DataFrame(all_predictions)
    pred_path = paths.OUTPUT_DIR / f"predictions_{model_name}.parquet"
    pred_df.to_parquet(pred_path, index=False)

    # ── row-count assertion (§8): ACTUAL per-fold day counts, val+test ─
    n_seeds = len(seeds)
    n_stocks_counts = {int(fr["n_stocks"]) for fr in fold_results}
    assert len(n_stocks_counts) == 1, (
        f"inconsistent stock counts across folds: {n_stocks_counts}"
    )
    n_stocks = n_stocks_counts.pop()
    total_val_days = sum(int(fr["n_val_days"]) for fr in fold_results)
    total_test_days = sum(int(fr["n_test_days"]) for fr in fold_results)
    expected_rows = (total_val_days + total_test_days) * n_seeds * n_stocks
    actual_rows = len(pred_df)
    per_fold = " + ".join(str(int(fr["n_test_days"])) for fr in fold_results)
    assert actual_rows == expected_rows, (
        f"Row count mismatch for {model_name}: expected {expected_rows:,} "
        f"(test days {per_fold} = {total_test_days}, plus "
        f"{total_val_days} val days, across {len(folds)} folds × "
        f"{n_seeds} seeds × {n_stocks} stocks), got {actual_rows:,}. "
        f"A fold, seed, or split may have silently failed."
    )
    # Both splits present per (fold, seed) — Rule 17/18 depend on it.
    split_counts = pred_df.groupby(["fold", "seed", "split"]).size()
    for fold in folds:
        for seed in seeds:
            for split_name, want_days in [("val", None), ("test", None)]:
                key = (fold.name, seed, split_name)
                assert key in split_counts.index, (
                    f"missing predictions for fold={fold.name} seed={seed} "
                    f"split={split_name}"
                )
    print(f"\n[train] wrote {pred_path}  rows={actual_rows:,}  "
          f"(test [{per_fold}]={total_test_days} days + "
          f"{total_val_days} val days) × {n_seeds} seeds × {n_stocks} stocks")

    # ── write per-model train metrics (never clobber another model's) ─
    metrics_path = (paths.OUTPUT_DIR
                    / f"train_metrics_{model_name}.json")
    metrics_path.write_text(json.dumps(all_metrics, indent=2))
    print(f"[train] wrote {metrics_path}")

    # ── accumulate the run manifest ──────────────────────────────────
    record = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": (torch.cuda.get_device_name(0)
                      if torch.cuda.is_available() else "none"),
        "devices": [str(d) for d in device_list],
        "model_id": model_id,
        "model_name": model_name,
        "output_suffix": output_suffix,
        "edge_source": str(edge_path),
        "init": "random",
        "config_snapshot": cfg,
        "cache_hashes": hash_check,
        "n_prediction_rows": int(actual_rows),
        "fold_results": fold_results,
    }
    manifest = _load_manifest()
    manifest.append(record)
    paths.RUN_MANIFEST.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"[train] appended run to {paths.RUN_MANIFEST} "
          f"(now {len(manifest)} run records)")

    # ── summary table ─────────────────────────────────────────────────
    print(f"\n[train] === {model_name} SUMMARY ===")
    rows = []
    for fr in fold_results:
        for seed in fr["seeds"]:
            m = all_metrics[f"{fr['fold']}/seed_{seed}"]
            rows.append({
                "fold": fr["fold"],
                "seed": seed,
                "n_epochs": len(m),
                "best_val_loss": min(x["val_loss"] for x in m),
                "final_train_loss": m[-1]["train_loss"],
            })
    with pd.option_context("display.width", 120):
        print(pd.DataFrame(rows).to_string(index=False))
    return pred_df


def main() -> None:
    parser = argparse.ArgumentParser(description="DMRGT Phase 3 trainer")
    parser.add_argument("--config", default=None,
                        help="Path to configs.yaml (default: paths.CONFIG)")
    parser.add_argument("--model", default=None,
                        help="model_id from configs.yaml models.ids "
                             "(dmrgt_all3 or dmrgt_frozen)")
    parser.add_argument("--edge-source", default=None,
                        help="Override the arm's edge file (§6b grid "
                             "variants: cache/graphs_<name>.parquet)")
    parser.add_argument("--output-suffix", default="",
                        help="Suffix for output filenames and the model "
                             "column (grid runs: _grid_<variant>)")
    parser.add_argument("--folds", type=int, nargs="*", default=None,
                        help="Restrict to fold indices (0-based)")
    parser.add_argument("--no-verify-hashes", action="store_true",
                        help="Skip the 5-file hash verification "
                             "(NOT recommended)")
    args = parser.parse_args()

    cfg = data.load_config(args.config or str(paths.CONFIG))
    allowed = list(cfg.get("models", {}).get("ids", []))
    if args.model is None:
        parser.error(
            f"--model is required. Phase 3 arms: {allowed}"
        )
    if args.model not in allowed:
        parser.error(
            f"--model must be one of configs.yaml models.ids: {allowed} "
            f"(got {args.model!r}) — Rule 18: exactly two models."
        )
    if args.folds is not None:
        cfg = dict(cfg)
        cfg["folds"] = [cfg["folds"][i] for i in args.folds
                        if 0 <= i < len(cfg["folds"])]

    if args.no_verify_hashes:
        hashes = verify_cache_hashes(stop_on_mismatch=False)
        run_all(cfg, model_id=args.model, edge_source=args.edge_source,
                output_suffix=args.output_suffix,
                verify_hashes=False, hashes=hashes)
    else:
        run_all(cfg, model_id=args.model, edge_source=args.edge_source,
                output_suffix=args.output_suffix)


if __name__ == "__main__":
    main()