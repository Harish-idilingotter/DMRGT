"""DMRGT Phase 1 — Transformer trainer (§7.2).

Kaggle-side training: load cached features/targets, train the model,
emit predictions.parquet. Does NOT run backtests or compute rank IC —
those belong to eval.py running locally.
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


def assemble_fold(
    fold: sp.Fold,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    cfg: dict,
) -> FoldTensors:
    """Build per-fold train/val/test tensors. Scaler fitted on train only."""
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
    )



def set_seed(seed: int) -> None:
    """Pin every RNG we use. torch + numpy + (best-effort) cuDNN."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_seed(
    tensors: FoldTensors,
    cfg: dict,
    seed: int,
    device: torch.device,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Train one seed on one fold. Returns (metrics_df, test_pred, best_state)."""
    set_seed(seed)

    epochs = int(cfg["training"]["epochs"])
    batch_size = int(cfg["training"]["batch_size"])
    lr = float(cfg["training"]["learning_rate"])
    huber_delta = float(cfg["training"]["huber_delta"])
    patience = int(cfg["training"]["early_stopping_patience"])

    n_stocks = tensors.X_train.shape[1] if len(tensors.X_train) > 0 else 20
    n_features = tensors.X_train.shape[3] if len(tensors.X_train) > 0 else 8
    net = model_mod.DMRGTPhase1(n_features=n_features).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.HuberLoss(delta=huber_delta)

    X_train_t = torch.from_numpy(tensors.X_train).float().to(device)
    y_train_t = torch.from_numpy(tensors.y_train).float().to(device)
    X_val_t = torch.from_numpy(tensors.X_val).float().to(device)
    y_val_t = torch.from_numpy(tensors.y_val).float().to(device)

    metrics = []
    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    n_train = len(X_train_t)

    for epoch in range(epochs):
        net.train()
        # Shuffle training days
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n_train, batch_size):
            idx = perm[i: i + batch_size]
            Xb = X_train_t[idx]
            yb = y_train_t[idx]
            opt.zero_grad()
            yp = net(Xb)              # [B, n_stocks]
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
                # Process val in mini-batches to avoid memory issues
                val_loss = 0.0
                n_vb = 0
                for i in range(0, len(X_val_t), batch_size):
                    Xb = X_val_t[i: i + batch_size]
                    yb = y_val_t[i: i + batch_size]
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
    cfg: dict, hash_check: dict[str, str], fold_results: list[dict]
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
        "config_snapshot": cfg,
        "cache_hashes": hash_check,
        "fold_results": fold_results,
    }



def run_all(cfg: dict, verify_hashes: bool = True,
            hashes: dict[str, str] | None = None) -> None:
    """Run all folds × all seeds, write outputs to OUTPUT_DIR.
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

        tensors = assemble_fold(fold, features, targets, cfg)
        print(f"  X_train: {tensors.X_train.shape}  X_val: {tensors.X_val.shape}  "
              f"X_test: {tensors.X_test.shape}")
        # Scaler-was-fitted-before-test assertion
        assert tensors.fitted_scaler_before_test, (
            f"fold {fold.name}: scaler was not fitted before test data was touched"
        )

        fold_pred_rows = []
        for seed in seeds:
            print(f"  --- seed {seed} ---")
            t0 = time.time()
            metrics_df, test_pred, best_state = train_one_seed(tensors, cfg, seed, device)
            dt = time.time() - t0
            print(f"      epochs: {len(metrics_df)}  "
                  f"final val_loss: {metrics_df['val_loss'].iloc[-1]:.6f}  "
                  f"elapsed: {dt:.1f}s")
            all_metrics[f"{fold.name}/seed_{seed}"] = metrics_df.to_dict(orient="records")
            fold_pred_rows.append((seed, test_pred))

            # Save checkpoint — explicit metadata, no deep-copy of
            # tensors via dataclasses.asdict.
            ckpt_path = ckpt_dir / f"{fold.name}__seed{seed}.pt"
            torch.save({
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
                        "fold": fold.name,
                        "seed": int(seed),
                        "date": pd.Timestamp(d),
                        "ticker": t,
                        "y_pred": float(preds[i, j]),
                        "y_true": float(tensors.y_test[i, j]),
                    })

        # Date boundaries + scaler stats land in the manifest so
        # verify_env.py can confirm the two environments built
        # identical folds.
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
    pred_path = paths.PREDICTIONS
    pred_df.to_parquet(pred_path, index=False)
    print(f"\n[train] wrote {pred_path}  rows={len(pred_df):,}")

    metrics_path = paths.TRAIN_METRICS
    metrics_path.write_text(json.dumps(all_metrics, indent=2))
    print(f"[train] wrote {metrics_path}")

    manifest = build_run_manifest(cfg, hash_check, fold_results)
    manifest_path = paths.RUN_MANIFEST
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"[train] wrote {manifest_path}")

    # ── summary table ───────────────────────────────────────────────
    print("\n[train] === SUMMARY ===")
    summary_rows = []
    for fr in fold_results:
        fold = fr["fold"]
        for seed in fr["seeds"]:
            m = all_metrics[f"{fold}/seed_{seed}"]
            summary_rows.append({
                "fold": fold,
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
    parser = argparse.ArgumentParser(description="DMRGT Phase 1 Transformer trainer")
    parser.add_argument("--config", default=None,
                        help="Path to configs.yaml (default: paths.CONFIG)")
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
        # Still collect hashes for the manifest, but don't halt on mismatch.
        hashes = verify_cache_hashes(stop_on_mismatch=False)
        run_all(cfg, verify_hashes=False, hashes=hashes)
    else:
        run_all(cfg, verify_hashes=True)


if __name__ == "__main__":
    main()
