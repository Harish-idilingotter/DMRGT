"""DMRGT Phase 4 — training (spec §6.5).

train_one(arm, fold, seed, batch_size): seeded CPU construction + byte
fingerprint (Phase 3 Rule 15 assertion ported), AdamW + linear warmup +
ReduceLROnPlateau on validation rank IC, gradient clipping, early stopping
on validation mean rank IC, optional adaptive Huber delta (arm D) from the
epoch's TRAINING residuals. Only the train and val splits are ever
requested — no test data is touched here.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import d4_data
import d4_eval as ev
import d4_graph
import d4_models


# ─── seeding + init fingerprint (ported from Phase 3 train.py) ─────────────

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def param_fingerprint(model: torch.nn.Module) -> str:
    """Byte-level fingerprint of a model's trainable parameters (device-stable)."""
    h = hashlib.sha256()
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def arm_cfg(arm: str) -> dict:
    return d4_data.load_snapshot()["CONFIG"]["arms"][arm]


def build_model(arm: str, snapshot: dict | None = None):
    snapshot = snapshot or d4_data.load_snapshot()
    a = snapshot["CONFIG"]["arms"][arm]
    return d4_models.make_model(a["encoder"], snapshot["CONFIG"]["mamba"])


def ckpt_paths(out_dir: Path, arm: str, fold: str, seed: int) -> dict:
    tag = f"{arm}_{fold}_{seed}"
    return {"ckpt": out_dir / f"ckpt_{tag}.pt", "valpred": out_dir / f"valpred_{tag}.parquet",
            "log": out_dir / f"log_{tag}.csv", "done": out_dir / f"done_{tag}.json"}


def score_val(pred: np.ndarray, y: np.ndarray, dates, raw_wide: pd.DataFrame,
              universe: list, bt_cfg: dict) -> tuple[float, float]:
    """Validation (mean rank IC, Sharpe after costs) through the ported harness."""
    ics = [ev.per_day_rank_ic(pred[i], y[i]) for i in range(len(pred))]
    mean_ic = float(np.nanmean(ics)) if np.isfinite(ics).any() else float("nan")
    wide_pred = pd.DataFrame(pred, index=pd.DatetimeIndex(dates), columns=universe)
    bt_raw = raw_wide.reindex(wide_pred.index).reindex(columns=universe)
    _, st = ev.backtest_daily_deciles(
        wide_pred, bt_raw, universe, bt_cfg["commission_bps"], bt_cfg["slippage_bps"],
        float(bt_cfg["decile_fraction"]))
    return mean_ic, float(st["sharpe"])


def train_one(arm: str, fold: str, seed: int, batch_size: int, max_epochs: int | None = None,
              out_dir: Path | None = None, shuffle_targets: bool = False,
              write: bool = True) -> dict:
    snap = d4_data.load_snapshot()
    C = snap["CONFIG"]
    tc, a = C["training"], C["arms"][arm]
    out_dir = Path(out_dir) if out_dir is not None else d4_data.WORK
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    t_start = time.time()
    lookback = int(a["lookback"])
    cfg = d4_data.load_cfg()
    universe = list(cfg["universe"]["tickers"])
    sd_tr = d4_graph.get_split_g(fold, "train", lookback)
    sd_va = d4_graph.get_split_g(fold, "val", lookback)
    edges = d4_graph.load_daily_edges()
    y_train = sd_tr.y
    if shuffle_targets:                       # leakage gate: same days, labels scrambled across dates
        y_train = y_train[np.random.default_rng(12345).permutation(len(y_train))]

    # ── Rule 15: random-init assertion, on CPU ────────────────────────
    set_seed(seed)
    net = build_model(arm, snap)
    fp1 = param_fingerprint(net)
    set_seed(seed)
    net2 = build_model(arm, snap)
    assert fp1 == param_fingerprint(net2), (
        f"Random-init assertion failed: same seed produced different params (seed={seed}).")
    set_seed(seed + 9999)
    net3 = build_model(arm, snap)
    assert fp1 != param_fingerprint(net3), (
        f"Random-init assertion failed: different seeds produced identical params (seed={seed}).")
    del net2, net3
    net = net.to(device)
    set_seed(seed)                            # restart the RNG at the seeded point
    n_params = int(sum(p.numel() for p in net.parameters()))

    X_tr = torch.from_numpy(sd_tr.X).float().to(device)
    y_tr = torch.from_numpy(y_train).float().to(device)
    X_va = torch.from_numpy(sd_va.X).float().to(device)
    tr_ei = [edges[pd.Timestamp(d)][0].to(device) for d in sd_tr.dates]
    tr_et = [edges[pd.Timestamp(d)][1].to(device) for d in sd_tr.dates]
    va_ei = [edges[pd.Timestamp(d)][0].to(device) for d in sd_va.dates]
    va_et = [edges[pd.Timestamp(d)][1].to(device) for d in sd_va.dates]
    raw_wide = ev.raw_returns_from_targets(d4_data.tables()["targets"], universe)

    opt = torch.optim.AdamW(net.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    pl = tc["plateau"]
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode=pl["mode"], factor=pl["factor"], patience=pl["patience"], min_lr=pl["min_lr"])
    warm, lr0, lr1 = int(tc["warmup_epochs"]), tc["warmup_start_lr"], tc["lr"]
    epochs = int(max_epochs if max_epochs is not None else tc["max_epochs"])
    patience = int(tc["early_stop_patience"])
    adaptive = bool(a["adaptive_huber"])
    ah = tc["adaptive_huber"]
    huber_delta = float(tc["huber_delta_fixed"])

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    log, best_ic, best_state, best_epoch, bad = [], -float("inf"), None, -1, 0
    n_train = len(X_tr)

    for epoch in range(epochs):
        t_ep = time.time()
        if epoch < warm:                      # linear warmup 1e-4 → 1e-3
            lr = lr0 + (lr1 - lr0) * epoch / max(1, warm - 1)
            for g in opt.param_groups:
                g["lr"] = lr
        lr_now = float(opt.param_groups[0]["lr"])

        net.train()
        perm = torch.randperm(n_train, device=device)
        ep_loss, n_b, resid = 0.0, 0, []
        for i in range(0, n_train, batch_size):
            idx = perm[i: i + batch_size]
            rows = idx.tolist()
            opt.zero_grad()
            yp = net.forward_with_edge_lists(X_tr[idx], [tr_ei[j] for j in rows], [tr_et[j] for j in rows])
            yb = y_tr[idx]
            loss = F.huber_loss(yp, yb, delta=huber_delta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), tc["grad_clip_norm"])
            opt.step()
            ep_loss += float(loss.item())
            n_b += 1
            if adaptive:
                resid.append((yp.detach() - yb).reshape(-1))
        train_loss = ep_loss / max(1, n_b)

        net.eval()
        val_pred = predict(net, X_va, va_ei, va_et, batch_size)
        val_ic, val_sh = score_val(val_pred, sd_va.y, sd_va.dates, raw_wide, universe, C["backtest"])
        log.append({"epoch": epoch, "lr": lr_now, "huber_delta": huber_delta,
                    "train_loss": train_loss, "val_rank_ic": val_ic, "val_sharpe": val_sh,
                    "epoch_seconds": time.time() - t_ep})

        if adaptive:                          # next epoch's delta, from THIS epoch's training residuals
            r = torch.cat(resid)
            mad = torch.median(torch.abs(r - torch.median(r)))
            huber_delta = float(torch.clamp(ah["c"] * 1.4826 * mad, min=ah["min"], max=ah["max"]))

        score = val_ic if np.isfinite(val_ic) else -float("inf")
        if score > best_ic:
            best_ic, best_epoch, bad = score, epoch, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
        if epoch >= warm:                     # plateau scheduler is not stepped during warmup
            sched.step(score if np.isfinite(score) else -1.0)
        if bad >= patience:
            break

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    val_pred = predict(net, X_va, va_ei, va_et, batch_size)
    assert val_pred.shape == (len(sd_va.dates), len(universe))
    peak = (torch.cuda.max_memory_allocated() / 2**30) if device.type == "cuda" else 0.0
    best_row = log[best_epoch] if best_epoch >= 0 else log[-1]
    result = {
        "arm": arm, "fold": fold, "seed": seed, "lookback": lookback, "batch_size": batch_size,
        "fingerprint": fp1, "n_params": n_params,
        "epochs_run": len(log), "best_epoch": best_epoch,
        "best_val_rank_ic": float(best_row["val_rank_ic"]), "best_val_sharpe": float(best_row["val_sharpe"]),
        "n_train_days": int(n_train), "n_val_days": int(len(sd_va.dates)),
        "mean_epoch_seconds": float(np.mean([r["epoch_seconds"] for r in log])),
        "last_epoch_seconds": float(log[-1]["epoch_seconds"]),
        "peak_gpu_gb": float(peak), "total_seconds": time.time() - t_start,
        "shuffled_targets": bool(shuffle_targets),
    }
    if write:
        p = ckpt_paths(out_dir, arm, fold, seed)
        torch.save({"arm": arm, "fold": fold, "seed": seed, "lookback": lookback,
                    "encoder": a["encoder"], "model_state_dict": best_state,
                    "scaler_mean": d4_data.get_builder(fold, lookback).scaler_mean,
                    "scaler_scale": d4_data.get_builder(fold, lookback).scaler_scale},
                   p["ckpt"])
        ev.make_pred_frame(arm, fold, seed, "val", sd_va.dates, universe, val_pred, sd_va.y) \
            .to_parquet(p["valpred"], index=False)
        pd.DataFrame(log).to_csv(p["log"], index=False)
        p["done"].write_text(json.dumps(result, indent=2))      # marker written last
    result["val_pred"] = val_pred
    return result


def predict(net, X, ei, et, batch_size: int) -> np.ndarray:
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            n_b = min(batch_size, len(X) - i)
            out.append(net.forward_with_edge_lists(X[i: i + n_b], ei[i: i + n_b], et[i: i + n_b])
                       .cpu().numpy())
    return np.concatenate(out, 0)


def probe_memory(arm: str, fold: str, batch_size: int) -> float:
    """One forward+backward at `batch_size`; returns peak GB (0 on CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = arm_cfg(arm)
    sd = d4_graph.get_split_g(fold, "train", int(a["lookback"]))
    edges = d4_graph.load_daily_edges()
    set_seed(0)
    net = build_model(arm).to(device)
    n = min(batch_size, len(sd.X))
    X = torch.from_numpy(sd.X[:n]).float().to(device)
    y = torch.from_numpy(sd.y[:n]).float().to(device)
    ei = [edges[pd.Timestamp(d)][0].to(device) for d in sd.dates[:n]]
    et = [edges[pd.Timestamp(d)][1].to(device) for d in sd.dates[:n]]
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    net.train()
    loss = F.huber_loss(net.forward_with_edge_lists(X, ei, et), y)
    loss.backward()
    peak = (torch.cuda.max_memory_allocated() / 2**30) if device.type == "cuda" else 0.0
    del net, X, y, loss
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return float(peak)
