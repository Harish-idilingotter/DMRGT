"""DMRGT Phase 2 — Multi-model trainer (§7).

Kaggle-side training: load cached features/targets/graphs, train the
selected model, emit predictions.parquet with a `model` column.
Does NOT run backtests or compute rank IC — those belong to eval.py
running locally.

Usage:
  python train.py --model transformer_nograph
  python train.py --model lstm
  python train.py --model dmrgt_all3
  python train.py --model dmrgt_pos_only
  python train.py --model dmrgt_no_sector
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.preprocessing import StandardScaler

import data
import model as model_mod
import paths
import splits as sp


def load_config(path: str | Path | None = None) -> dict:
    """Load configs.yaml. Defaults to paths.CONFIG."""
    p = Path(path) if path is not None else paths.CONFIG
    if not p.exists():
        raise FileNotFoundError(
            f"configs.yaml not found at {p}. On Kaggle this should "
            f"resolve to the dmrgt-code dataset via paths.CONFIG."
        )
    return yaml.safe_load(p.read_text())


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_cache_hashes(stop_on_mismatch: bool = True) -> dict[str, str]:
    """Recompute SHA-256 of each cached parquet, compare against hashes.json.
    Returns {filename: hash} for the files on disk. Halts on mismatch
    unless stop_on_mismatch=False."""
    if not paths.HASHES_JSON.exists():
        raise RuntimeError(
            f"HALT: {paths.HASHES_JSON} not found. Cannot verify data "
            f"integrity. Regenerate the cache locally and re-upload."
        )

    expected = json.loads(paths.HASHES_JSON.read_text())
    print(f"[train] cache generated at: {expected.get('generated_at', 'unknown')}")

    found: dict[str, str] = {}
    problems: list[str] = []

    # Iterate only the canonical parquet names, NOT every key in
    # hashes.json — metadata keys (e.g. generated_at) would otherwise
    # be treated as filenames.
    for name in paths.CACHE_PARQUETS:
        if name not in expected:
            problems.append(f"{name} has no entry in hashes.json")
            continue
        p = paths.CACHE_DIR / name
        if not p.exists():
            problems.append(f"{name} listed in hashes.json but missing from cache")
            continue
        got = sha256_of_file(p)
        found[name] = got
        if got != expected[name]:
            problems.append(
                f"{name} hash mismatch\n"
                f"      expected: {expected[name]}\n"
                f"      got:      {got}"
            )
        else:
            print(f"[train] hash OK: {name}  {got[:12]}…")

    # Counting guard: a truncated hashes.json (missing entries for
    # parquets that DO exist) must not pass by verifying nothing.
    if len(found) != len(paths.CACHE_PARQUETS) and not problems:
        problems.append(
            f"verified {len(found)} files, expected {len(paths.CACHE_PARQUETS)}"
        )

    if problems and stop_on_mismatch:
        raise RuntimeError(
            "HALT: cache verification failed. Refusing to train.\n  "
            + "\n  ".join(problems)
            + "\nMost common cause: an older dmrgt-cache version is attached. "
              "Re-attach the latest version and restart the session."
        )
    if problems:
        for p_ in problems:
            print(f"[train] WARNING: {p_}")

    print(f"[train] all {len(found)} parquet hashes verified")
    return found



@dataclass
class FoldTensors:
    """The per-fold tensors we feed into the training loop."""
    name: str
    fold_idx: int
    X_train: np.ndarray   # [n_train_days, n_stocks, lookback, n_features]
    y_train: np.ndarray   # [n_train_days, n_stocks]
    X_val:   np.ndarray
    y_val:   np.ndarray
    X_test:  np.ndarray
    y_test:  np.ndarray
    test_dates: np.ndarray  # [n_test_days] timestamps
    train_dates: np.ndarray  # [n_train_days] timestamps
    val_dates:   np.ndarray  # [n_val_days] timestamps
    test_tickers: list[str]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    fitted_scaler_before_test: bool  # assertion flag
    feature_cols: list[str]  # columns actually used (from cfg)
    # Graph data — None for non-graph models
    edge_index_by_date: dict | None = None  # pd.Timestamp → [2, E] tensor
    edge_type_by_date: dict | None = None    # pd.Timestamp → [E] tensor


def assemble_fold(
    fold: sp.Fold,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    cfg: dict,
    graphs: pd.DataFrame | None = None,
) -> FoldTensors:
    """Build per-fold train/val/test tensors. Scaler fitted on train only.
    If graphs is provided, builds edge_index/edge_type lookup by date."""
    universe = list(cfg["universe"]["tickers"])

    # Feature columns come from config — NEVER inferred by string-
    # matching the parquet columns. A schema change must fail loudly.
    feature_cols = list(cfg["features"]["columns"])
    missing = [c for c in feature_cols if c not in features.columns]
    assert not missing, (
        f"configs.yaml lists feature columns absent from features.parquet: "
        f"{missing}. Available: {sorted(features.columns)}"
    )
    assert len(feature_cols) == 8, (
        f"Expected 8 features per spec §11, got {len(feature_cols)}"
    )

    lookback = int(cfg["features"]["lookback_days"])

    # Wide pivots
    feat_wide = features.pivot(index="date", columns="ticker",
                               values=feature_cols).sort_index()
    tgt_wide = targets.pivot(index="date", columns="ticker",
                             values="excess_return").sort_index()

    def build(date_set: pd.DatetimeIndex, with_labels: bool):
        Xs, ys, kept_dates = [], [], []
        for d in date_set:
            if d not in feat_wide.index:
                continue
            if with_labels and d not in tgt_wide.index:
                continue
            loc = feat_wide.index.get_loc(d)
            # Window is [t-lookback+1 .. t] INCLUSIVE of the prediction
            # date (spec §4). The label describes t+1.
            start = loc - lookback + 1
            if start < 0:
                continue  # not enough history — drop, do NOT zero-pad
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
        X = np.stack(Xs, axis=0) if Xs else np.empty((0, len(universe), lookback, len(feature_cols)))
        y = np.stack(ys, axis=0) if ys else np.empty((0, len(universe)))
        return X, y, np.array(kept_dates)

    X_train, y_train, train_dates_kept = build(fold.train_dates, with_labels=True)
    X_val,   y_val,   val_dates_kept   = build(fold.val_dates,   with_labels=True)
    X_test,  y_test,  test_dates_kept  = build(fold.test_dates,  with_labels=True)

    # ── hard asserts (per spec §7.2) ───────────────────────────────
    # No test date appears in the training set; train/test windows are
    # chronologically disjoint (this catches interleaved folds).
    train_idx = pd.DatetimeIndex(train_dates_kept)
    val_idx   = pd.DatetimeIndex(val_dates_kept)
    test_idx  = pd.DatetimeIndex(test_dates_kept)

    if len(train_idx) > 0 and len(test_idx) > 0:
        overlap = train_idx.intersection(test_idx)
        assert len(overlap) == 0, (
            f"{len(overlap)} test dates appear in the training set: "
            f"{list(overlap[:5])}"
        )
        assert train_idx.max() < test_idx.min(), (
            f"Chronological order violated: last train date "
            f"{train_idx.max().date()} >= first test date "
            f"{test_idx.min().date()}"
        )

    # ── fit scaler on training dates only (§1.1) ───────────────────
    # Pool across days, stocks, and lags — fit ONE mean/std PER FEATURE,
    # not per (lag, feature) cell. The reshape is [N, n_features] not
    # [N, lookback * n_features].
    if len(X_train) > 0:
        n_feat = X_train.shape[3]
        flat = X_train.reshape(-1, n_feat)
        scaler = StandardScaler().fit(flat)
        assert scaler.mean_.shape == (n_feat,), (
            f"Scaler fitted {scaler.mean_.shape[0]} params, expected {n_feat}. "
            f"The reshape is wrong — pooling should give one mean per feature."
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

    # ── input shape assert ─────────────────────────────────────────
    for arr_name, arr in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
        if len(arr) > 0:
            assert arr.shape[2] == lookback, (
                f"{arr_name} has lookback dim {arr.shape[2]}, expected {lookback}"
            )
            assert arr.shape[3] == len(feature_cols), (
                f"{arr_name} has feature dim {arr.shape[3]}, expected {len(feature_cols)}"
            )
            assert arr.shape[1] == len(universe), (
                f"{arr_name} has n_stocks dim {arr.shape[1]}, expected {len(universe)}"
            )

    # ── graph data ──────────────────────────────────────────────────
    edge_index_by_date = None
    edge_type_by_date = None
    if graphs is not None:
        # Build a lookup: date → (edge_index, edge_type)
        # The graph uses integer node indices 0..n_stocks-1 in the same
        # order as `universe`. This is the critical alignment — a mismatch
        # here silently scrambles which stock is which (§7.2).
        edge_index_by_date = {}
        edge_type_by_date = {}
        for d in graphs["date"].unique():
            day = graphs[graphs["date"] == d]
            ei = torch.tensor(
                day[["src", "dst"]].values.T, dtype=torch.long
            )  # [2, E]
            et = torch.tensor(day["edge_type"].values, dtype=torch.long)  # [E]
            edge_index_by_date[pd.Timestamp(d)] = ei
            edge_type_by_date[pd.Timestamp(d)] = et

        # Assert date-range intersection: every date in the training set
        # must have a graph entry (§3.5).
        graph_dates = set(edge_index_by_date.keys())
        all_used = (set(train_dates_kept) | set(val_dates_kept) |
                    set(test_dates_kept))
        missing = all_used - graph_dates
        if missing:
            missing_sorted = sorted(missing)[:5]
            raise RuntimeError(
                f"{len(missing)} dates have no graph entry. "
                f"First 5: {[m.date() for m in missing_sorted]}. "
                f"Re-run graph.py to regenerate the graph cache."
            )

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



def set_seed(seed: int) -> None:
    """Pin every RNG we use. torch + numpy + (best-effort) cuDNN."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _param_fingerprint(model: torch.nn.Module) -> str:
    """Exact byte-level fingerprint of a model's trainable parameters.

    Used to assert no weights were loaded before training (Rule 15).

    IMPORTANT — why bytes and not sums. An earlier version returned
    ``[float(p.sum().item()) for p in ...]``. That is not comparable
    across devices: CUDA reduces with a parallel tree while CPU sums
    sequentially, and float addition is not associative, so identical
    weights produce sums differing in the last bits. Comparing those
    with ``==`` fails spuriously.

    ``.cpu()`` is a pure memory copy that preserves the bit pattern
    exactly, so hashing the raw bytes is stable on any device.
    """
    h = hashlib.sha256()
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def make_model(model_id: str, n_features: int = 8) -> torch.nn.Module:
    """Factory: return the right model for a canonical model ID.
    Model IDs are defined in configs.yaml models.ids — never hand-typed."""
    if model_id == "transformer_nograph":
        return model_mod.TransformerEncoder(n_features=n_features)
    elif model_id == "lstm":
        return model_mod.LSTMBaseline(n_features=n_features)
    elif model_id in ("dmrgt_all3", "dmrgt_pos_only", "dmrgt_no_sector"):
        return model_mod.RGCNDMRGT(n_features=n_features)
    else:
        raise ValueError(
            f"Unknown model_id: {model_id!r}. "
            f"Must be one of configs.yaml models.ids"
        )


def _filter_edges(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    model_id: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply ablation filtering BEFORE edges reach the model (Rule: filter upstream).
    - dmrgt_pos_only: keep only type 0 (+corr)
    - dmrgt_no_sector: keep types 0 and 1 (drop sector)
    - dmrgt_all3: keep all
    """
    if model_id == "dmrgt_pos_only":
        mask = (edge_type == 0)
    elif model_id == "dmrgt_no_sector":
        mask = (edge_type != 2)
    else:
        return edge_index, edge_type
    return edge_index[:, mask], edge_type[mask]


def train_one_seed(
    tensors: FoldTensors,
    cfg: dict,
    seed: int,
    device: torch.device,
    model_id: str,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Train one seed on one fold. Returns (metrics_df, test_pred, best_state)."""
    set_seed(seed)

    epochs = int(cfg["training"]["epochs"])
    batch_size = int(cfg["training"]["batch_size"])
    lr = float(cfg["training"]["learning_rate"])
    huber_delta = float(cfg["training"]["huber_delta"])
    patience = int(cfg["training"]["early_stopping_patience"])

    n_stocks = tensors.X_train.shape[1] if len(tensors.X_train) > 0 else len(cfg["universe"]["tickers"])
    n_features = tensors.X_train.shape[3] if len(tensors.X_train) > 0 else 8
    is_graph_model = model_id.startswith("dmrgt")

    # ── Rule 15: random-init assertion ───────────────────────────────
    # Construct the model twice under the same seed; fingerprints must
    # match. Then construct under a different seed; fingerprints must
    # differ. Together these show the weights come from seeded random
    # initialisation rather than a loaded state dict.
    #
    # All three are built and compared ON CPU. Comparing a CUDA model
    # against a CPU model is invalid — see _param_fingerprint. The
    # training model moves to the device only after the checks pass.
    set_seed(seed)
    net = make_model(model_id, n_features=n_features)      # CPU
    fp1 = _param_fingerprint(net)

    set_seed(seed)
    net2 = make_model(model_id, n_features=n_features)     # CPU
    fp2 = _param_fingerprint(net2)
    assert fp1 == fp2, (
        f"Random-init assertion failed: same seed produced different params. "
        f"Seed={seed}, model={model_id}. Seeding is not deterministic."
    )

    set_seed(seed + 9999)
    net3 = make_model(model_id, n_features=n_features)     # CPU
    fp3 = _param_fingerprint(net3)
    assert fp1 != fp3, (
        f"Random-init assertion failed: different seeds produced identical "
        f"params. Seed={seed} vs seed={seed + 9999}, model={model_id}. "
        f"The model may be ignoring the seed, or weights were loaded."
    )
    del net2, net3, fp2, fp3

    # Checks passed — move the training model to the device.
    net = net.to(device)
    # Restore the RNG state consumed by the two extra constructions, so
    # dropout and batch shuffling start from the same point they would
    # have without the assertion.
    set_seed(seed)

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.HuberLoss(delta=huber_delta)

    X_train_t = torch.from_numpy(tensors.X_train).float().to(device)
    y_train_t = torch.from_numpy(tensors.y_train).float().to(device)
    X_val_t = torch.from_numpy(tensors.X_val).float().to(device)
    y_val_t = torch.from_numpy(tensors.y_val).float().to(device)

    # ── pre-compute edge tensors for graph models ─────────────────────
    train_edge_index = None
    train_edge_type = None
    val_edge_index = None
    val_edge_type = None
    if is_graph_model and tensors.edge_index_by_date is not None:
        # Build per-date edge tensors aligned with X_train/X_val
        def _edge_for_dates(dates, edge_by_date, edge_type_by_date):
            ei_list, et_list = [], []
            for d in dates:
                d_ts = pd.Timestamp(d)
                ei = edge_by_date[d_ts].to(device)
                et = edge_type_by_date[d_ts].to(device)
                # Ablation filtering happens HERE, before the model sees edges
                ei, et = _filter_edges(ei, et, model_id)
                ei_list.append(ei)
                et_list.append(et)
            return ei_list, et_list

        train_edge_index, train_edge_type = _edge_for_dates(
            tensors.train_dates, tensors.edge_index_by_date,
            tensors.edge_type_by_date)
        val_edge_index, val_edge_type = _edge_for_dates(
            tensors.val_dates, tensors.edge_index_by_date,
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
            if is_graph_model:
                # For graph models, process one day at a time (each day
                # has its own edge list). Batch size > 1 requires careful
                # handling — we process each day in the batch separately.
                yp_list = []
                for j in range(len(Xb)):
                    Xb_j = Xb[j:j+1]  # [1, n_stocks, L, F]
                    ei = train_edge_index[idx[j].item()]
                    et = train_edge_type[idx[j].item()]
                    yp_j = net(Xb_j, ei, et)
                    yp_list.append(yp_j)
                yp = torch.cat(yp_list, dim=0)  # [B, n_stocks]
            else:
                yp = net(Xb)  # [B, n_stocks]
            loss = loss_fn(yp, yb)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        train_loss = epoch_loss / max(1, n_batches)

        # Validation
        net.eval()
        with torch.no_grad():
            if len(X_val_t) > 0:
                val_loss = 0.0
                n_vb = 0
                for i in range(0, len(X_val_t), batch_size):
                    Xb = X_val_t[i: i + batch_size]
                    yb = y_val_t[i: i + batch_size]
                    if is_graph_model:
                        yp_list = []
                        for j in range(len(Xb)):
                            Xb_j = Xb[j:j+1]
                            ei = val_edge_index[i + j]
                            et = val_edge_type[i + j]
                            yp_j = net(Xb_j, ei, et)
                            yp_list.append(yp_j)
                        yp = torch.cat(yp_list, dim=0)
                    else:
                        yp = net(Xb)
                    val_loss += float(loss_fn(yp, yb).item())
                    n_vb += 1
                val_loss /= max(1, n_vb)
            else:
                val_loss = float("nan")

        metrics.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    # Restore best
    if best_state is not None:
        net.load_state_dict(best_state)

    # Predict on test
    net.eval()
    preds = []
    with torch.no_grad():
        if len(tensors.X_test) > 0:
            X_test_t = torch.from_numpy(tensors.X_test).float().to(device)
            for i in range(0, len(X_test_t), batch_size):
                Xb = X_test_t[i: i + batch_size]
                if is_graph_model and tensors.edge_index_by_date is not None:
                    yp_list = []
                    for j in range(len(Xb)):
                        Xb_j = Xb[j:j+1]
                        d = pd.Timestamp(tensors.test_dates[i + j])
                        ei = tensors.edge_index_by_date[d].to(device)
                        et = tensors.edge_type_by_date[d].to(device)
                        ei, et = _filter_edges(ei, et, model_id)
                        yp_j = net(Xb_j, ei, et)
                        yp_list.append(yp_j)
                    yp = torch.cat(yp_list, dim=0).cpu().numpy()
                else:
                    yp = net(Xb).cpu().numpy()
                preds.append(yp)
    test_pred = np.concatenate(preds, axis=0) if preds else np.empty((0, n_stocks))

    return pd.DataFrame(metrics), test_pred, best_state



def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def build_run_manifest(
    cfg: dict, hash_check: dict[str, str], fold_results: list[dict],
    model_id: str = "transformer_nograph",
) -> dict:
    return {
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
        "model_id": model_id,
        "init": "random",
        "config_snapshot": cfg,
        "cache_hashes": hash_check,
        "fold_results": fold_results,
    }



def run_all(cfg: dict, verify_hashes: bool = True,
            hashes: dict[str, str] | None = None,
            model_id: str = "transformer_nograph") -> None:
    """Run all folds × all seeds for one model, write outputs to OUTPUT_DIR.
    Pass hashes= from a prior verify_cache_hashes() call when verify_hashes=False."""
    paths.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_dir = paths.CHECKPOINTS_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if verify_hashes:
        print("[train] verifying cache hashes…")
        hash_check = verify_cache_hashes(stop_on_mismatch=True)
    elif hashes is not None:
        hash_check = dict(hashes)  # defensive copy
        print("[train] using pre-verified hashes "
              f"({len(hash_check)} files)")
    else:
        raise ValueError(
            "verify_hashes=False requires hashes= to be passed. "
            "Call verify_cache_hashes() first and pass the result "
            "as hashes= so the run manifest records which data "
            "this run trained on."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # Load caches
    print("[train] loading features & targets…")
    features = pd.read_parquet(paths.FEATURES)
    targets = pd.read_parquet(paths.TARGETS)

    # Load graphs for graph models
    is_graph_model = model_id.startswith("dmrgt")
    graphs = None
    if is_graph_model:
        if not paths.GRAPHS.exists():
            raise FileNotFoundError(
                f"{paths.GRAPHS} missing — run 'python graph.py' first"
            )
        graphs = pd.read_parquet(paths.GRAPHS)
        print(f"[train] loaded graphs: {len(graphs):,} rows, "
              f"{graphs['date'].nunique()} dates")

    folds = sp.make_folds(cfg, available_dates=features["date"].unique())

    all_predictions = []
    all_metrics = {}
    fold_results = []
    seeds = list(cfg["training"]["seeds"])

    for fold in folds:
        print(f"\n[train] === fold {fold.fold_idx} {fold.name!r} ===")
        print(f"  train dates: {fold.train_dates[0].date()} → "
              f"{fold.train_dates[-1].date()}  (n={len(fold.train_dates)})")
        print(f"  val   dates: {fold.val_dates[0].date()} → "
              f"{fold.val_dates[-1].date()}  (n={len(fold.val_dates)})")
        print(f"  test  dates: {fold.test_dates[0].date()} → "
              f"{fold.test_dates[-1].date()}  (n={len(fold.test_dates)})")

        tensors = assemble_fold(fold, features, targets, cfg, graphs)
        print(f"  X_train: {tensors.X_train.shape}  X_val: {tensors.X_val.shape}  "
              f"X_test: {tensors.X_test.shape}")
        if is_graph_model and tensors.edge_index_by_date is not None:
            n_graph_dates = len(tensors.edge_index_by_date)
            print(f"  graph dates available: {n_graph_dates}")

        # Scaler-was-fitted-before-test assertion
        assert tensors.fitted_scaler_before_test, (
            f"fold {fold.name}: scaler was not fitted before test data was touched"
        )

        fold_pred_rows = []
        for seed in seeds:
            print(f"  --- seed {seed} ---")
            t0 = time.time()
            metrics_df, test_pred, best_state = train_one_seed(
                tensors, cfg, seed, device, model_id)
            dt = time.time() - t0
            print(f"      epochs: {len(metrics_df)}  "
                  f"final val_loss: {metrics_df['val_loss'].iloc[-1]:.6f}  "
                  f"elapsed: {dt:.1f}s")
            all_metrics[f"{fold.name}/seed_{seed}"] = metrics_df.to_dict(orient="records")
            fold_pred_rows.append((seed, test_pred))

            # Save checkpoint
            ckpt_path = ckpt_dir / f"{model_id}__{fold.name}__seed{seed}.pt"
            torch.save({
                "model_id": model_id,
                "fold": fold.name,
                "seed": seed,
                "model_state_dict": best_state,
                "fold_meta": {
                    "name":         tensors.name,
                    "fold_idx":     tensors.fold_idx,
                    "n_train_days": int(len(tensors.X_train)),
                    "n_val_days":   int(len(tensors.X_val)),
                    "n_test_days":  int(len(tensors.X_test)),
                    "test_tickers": tensors.test_tickers,
                },
                "scaler_mean":  tensors.scaler_mean,
                "scaler_scale": tensors.scaler_scale,
                "feature_cols": tensors.feature_cols,
            }, ckpt_path)

        # Stack per-seed test predictions for this fold
        n_test = len(tensors.test_dates)
        n_stocks = len(tensors.test_tickers)
        for s_idx, (seed, preds) in enumerate(fold_pred_rows):
            assert preds.shape == (n_test, n_stocks), (
                f"prediction shape mismatch: {preds.shape} vs "
                f"({n_test}, {n_stocks})"
            )
            for i, d in enumerate(tensors.test_dates):
                for j, t in enumerate(tensors.test_tickers):
                    all_predictions.append({
                        "model": model_id,
                        "fold": fold.name,
                        "seed": int(seed),
                        "date": pd.Timestamp(d),
                        "ticker": t,
                        "y_pred": float(preds[i, j]),
                        "y_true": float(tensors.y_test[i, j]),
                    })

        fold_results.append({
            "fold":         fold.name,
            "n_train_days": int(len(tensors.X_train)),
            "n_val_days":   int(len(tensors.X_val)),
            "n_test_days":  int(n_test),
            "train_start":  (str(tensors.train_dates[0].date())
                             if len(tensors.train_dates) else None),
            "train_end":    (str(tensors.train_dates[-1].date())
                             if len(tensors.train_dates) else None),
            "val_start":    (str(tensors.val_dates[0].date())
                             if len(tensors.val_dates) else None),
            "val_end":      (str(tensors.val_dates[-1].date())
                             if len(tensors.val_dates) else None),
            "test_start":   (str(tensors.test_dates[0].date())
                             if len(tensors.test_dates) else None),
            "test_end":     (str(tensors.test_dates[-1].date())
                             if len(tensors.test_dates) else None),
            "feature_cols": tensors.feature_cols,
            "scaler_mean":  tensors.scaler_mean.tolist(),
            "scaler_scale": tensors.scaler_scale.tolist(),
            "seeds":        seeds,
        })

    # ── write outputs ───────────────────────────────────────────────
    pred_df = pd.DataFrame(all_predictions)
    pred_path = paths.OUTPUT_DIR / f"predictions_{model_id}.parquet"
    pred_df.to_parquet(pred_path, index=False)

    # ── Per-model row-count assertion ────────────────────────────────
    # Test-day counts DIFFER PER FOLD — the standard four folds give
    # 252 / 253 / 252 / 251 trading days, because calendar years contain
    # different numbers of trading days and holidays fall differently.
    #
    # The expectation must therefore SUM the actual per-fold test-day
    # counts. Multiplying n_folds by any single fold's count (e.g. the
    # last one) is wrong and fails a run that completed correctly.
    n_folds = len(folds)
    n_seeds = len(seeds)
    total_test_days = sum(int(fr["n_test_days"]) for fr in fold_results)
    expected_rows = total_test_days * n_seeds * n_stocks
    actual_rows = len(pred_df)
    per_fold = " + ".join(str(int(fr["n_test_days"])) for fr in fold_results)
    assert actual_rows == expected_rows, (
        f"Row count mismatch for {model_id}: expected {expected_rows:,} "
        f"(test days {per_fold} = {total_test_days} across {n_folds} folds "
        f"x {n_seeds} seeds x {n_stocks} stocks), got {actual_rows:,}. "
        f"A fold or seed may have silently failed to produce predictions."
    )
    print(f"\n[train] wrote {pred_path}  rows={actual_rows:,}  "
          f"(expected {expected_rows:,} = [{per_fold}] x {n_seeds} seeds "
          f"x {n_stocks} stocks)")

    metrics_path = paths.TRAIN_METRICS
    metrics_path.write_text(json.dumps(all_metrics, indent=2))
    print(f"[train] wrote {metrics_path}")

    manifest = build_run_manifest(cfg, hash_check, fold_results, model_id)
    manifest_path = paths.RUN_MANIFEST
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"[train] wrote {manifest_path}")

    # ── summary table ───────────────────────────────────────────────
    print("\n[train] === SUMMARY ===")
    summary_rows = []
    for fr in fold_results:
        fold_name = fr["fold"]
        for seed in fr["seeds"]:
            m = all_metrics[f"{fold_name}/seed_{seed}"]
            summary_rows.append({
                "model": model_id,
                "fold": fold_name,
                "seed": seed,
                "n_epochs": len(m),
                "final_train_loss": m[-1]["train_loss"],
                "final_val_loss":   m[-1]["val_loss"],
            })
    summary_df = pd.DataFrame(summary_rows)
    with pd.option_context("display.width", 120, "display.float_format",
                           "{:,.6f}".format):
        print(summary_df.to_string(index=False))



def main():
    parser = argparse.ArgumentParser(description="DMRGT Phase 2 multi-model trainer")
    parser.add_argument("--config", default=None,
                        help="Path to configs.yaml (default: paths.CONFIG)")
    parser.add_argument("--model", default="transformer_nograph",
                        choices=["lstm", "transformer_nograph", "dmrgt_pos_only",
                                 "dmrgt_no_sector", "dmrgt_all3"],
                        help="Which model to train (default: transformer_nograph)")
    parser.add_argument("--no-verify-hashes", action="store_true",
                        help="Skip cache hash verification (NOT recommended)")
    parser.add_argument("--folds", type=int, nargs="*", default=None,
                        help="Restrict to a subset of fold indices (0-based)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.folds is not None:
        cfg = dict(cfg)
        cfg["folds"] = [cfg["folds"][i] for i in args.folds
                        if 0 <= i < len(cfg["folds"])]
    if args.no_verify_hashes:
        hashes = verify_cache_hashes(stop_on_mismatch=False)
        run_all(cfg, verify_hashes=False, hashes=hashes, model_id=args.model)
    else:
        run_all(cfg, verify_hashes=True, model_id=args.model)


if __name__ == "__main__":
    main()
