"""DMRGT Phase 5 — training (Phase 4 spec §6.5 + Phase 4.1 F1/F2 + Phase 5 arms).

train_one(arm, fold, seed, batch_size, graph, tag): seeded CPU construction + byte
fingerprint (Phase 3 Rule 15 assertion ported), then one of two schedules:

  'adaptive' (arms A, B, C): two-group AdamW (no weight decay on A_log, D, dt_proj.bias,
      biases, norm weights), linear warmup, ReduceLROnPlateau, gradient clipping, early
      stopping — all driven ONLY by a 3-epoch rolling mean of val_es mean rank IC;
      the best checkpoint can only come from epochs >= warmup (Phase 4.1 F2.2).
  'phase3'   (arm P): the Phase 3 schedule — Adam, fixed LR 1e-3, Huber delta 1.0, no
      clipping, early stopping on the validation Huber loss (patience 10) — with the
      loss measured on val_es only (Phase 5 spec §4).

val_sel is logged for the record and is never read by any training logic. Only train and
validation splits are ever requested — no test data is touched here.
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


def ckpt_paths(out_dir: Path, tag: str, fold: str, seed: int) -> dict:
    t = f"{tag}_{fold}_{seed}"
    return {"ckpt": out_dir / f"ckpt_{t}.pt", "valpred": out_dir / f"valpred_{t}.parquet",
            "log": out_dir / f"log_{t}.csv", "done": out_dir / f"done_{t}.json"}


# ─── F2.2: training control sees ONLY val_es metrics ───────────────────────

class ControlMetrics:
    """The only object the stopping rule and LR scheduler receive. Construction asserts that
    every metric handed in is a val_es metric, so val_sel cannot reach training control."""

    def __init__(self, **metrics):
        assert all(k.startswith("val_es") for k in metrics), \
            f"training control may only use val_es metrics, got {sorted(metrics)}"
        self.m = dict(metrics)


class AdaptiveController:
    """Rolling-3 val_es rank IC; best checkpoint only from epochs >= warmup; patience counts
    post-warmup epochs without improvement; LR plateau steps on the same rolling metric."""

    def __init__(self, warmup: int, patience: int):
        self.warmup, self.patience = warmup, patience
        self.hist, self.best, self.best_epoch, self.bad = [], -float("inf"), -1, 0

    def update(self, epoch: int, cm: ControlMetrics) -> dict:
        ic = cm.m["val_es_rank_ic"]
        self.hist.append(ic)
        last = [x for x in self.hist[-3:] if np.isfinite(x)]
        roll = float(np.mean(last)) if last else float("nan")
        improved = stop = False
        if epoch >= self.warmup:                       # warmup epochs are never eligible
            score = roll if np.isfinite(roll) else -float("inf")
            if score > self.best:
                self.best, self.best_epoch, self.bad, improved = score, epoch, 0, True
            else:
                self.bad += 1
            stop = self.bad >= self.patience
        return {"roll3": roll, "improved": improved, "stop": stop}


class Phase3Controller:
    """Phase 3 rule: best = lowest validation Huber loss (strict <), patience on epochs
    without improvement, counted from epoch 0 — evaluated on val_es only."""

    def __init__(self, patience: int):
        self.patience, self.best, self.best_epoch, self.bad = patience, float("inf"), -1, 0

    def update(self, epoch: int, cm: ControlMetrics) -> dict:
        loss = cm.m["val_es_loss"]
        improved = loss < self.best
        if improved:
            self.best, self.best_epoch, self.bad = loss, epoch, 0
        else:
            self.bad += 1
        return {"roll3": float("nan"), "improved": improved, "stop": self.bad >= self.patience}


def phase3_val_loss(pred: np.ndarray, y: np.ndarray, batch_size: int) -> float:
    """Phase 3 train_one_seed: mean over validation batches of the batch-mean Huber loss (delta 1.0)."""
    tot, n = 0.0, 0
    for i in range(0, len(pred), batch_size):
        tot += float(F.huber_loss(torch.from_numpy(pred[i: i + batch_size]).float(),
                                  torch.from_numpy(y[i: i + batch_size]).float(), delta=1.0))
        n += 1
    return tot / max(1, n)


def mean_rank_ic(pred: np.ndarray, y: np.ndarray) -> float:
    ics = [ev.per_day_rank_ic(pred[i], y[i]) for i in range(len(pred))]
    return float(np.nanmean(ics)) if np.isfinite(ics).any() else float("nan")


def train_one(arm: str, fold: str, seed: int, batch_size: int, graph: str = "baseline",
              tag: str | None = None, max_epochs: int | None = None, out_dir: Path | None = None,
              shuffle_targets: bool = False, write: bool = True) -> dict:
    snap = d4_data.load_snapshot()
    C = snap["CONFIG"]
    tc, a = C["training"], C["arms"][arm]
    tag = tag or arm
    out_dir = Path(out_dir) if out_dir is not None else d4_data.WORK
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    t_start = time.time()
    lookback = int(a["lookback"])
    phase3 = a["schedule"] == "phase3"
    cfg = d4_data.load_cfg(graph)
    universe = list(cfg["universe"]["tickers"])
    sd_tr = d4_graph.get_split_g(fold, "train", lookback, graph)
    sd_va = d4_graph.get_split_g(fold, "val", lookback, graph)
    sd_es = d4_graph.get_split_g(fold, "val_es", lookback, graph)
    sd_sel = d4_graph.get_split_g(fold, "val_sel", lookback, graph)
    bld = d4_data.get_builder(fold, lookback, graph_key=graph)
    scaler_mean, scaler_scale = bld.scaler_mean.copy(), bld.scaler_scale.copy()
    edges = d4_graph.load_daily_edges(graph)
    y_train = sd_tr.y
    if shuffle_targets:                       # leakage gate: same days, labels scrambled across dates
        y_train = y_train[np.random.default_rng(12345).permutation(len(y_train))]
    n_es = len(sd_es.dates)
    assert len(sd_va.dates) == n_es + len(sd_sel.dates)
    assert (sd_va.dates[:n_es] == sd_es.dates).all() and (sd_va.dates[n_es:] == sd_sel.dates).all()

    # ── Rule 15: random-init assertion, on CPU ────────────────────────
    set_seed(seed)
    net = build_model(arm, snap)              # F1: Mamba dt-init assertion runs inside make_model
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
    dt_mean = d4_models.assert_dt_init(net) if a["encoder"] == "mamba" else float("nan")
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

    huber_delta = float(tc["huber_delta_fixed"])
    epochs = int(max_epochs if max_epochs is not None else tc["max_epochs"])
    patience = int(tc["early_stop_patience"])
    warm = int(tc["warmup_epochs"])
    if phase3:
        opt = torch.optim.Adam(net.parameters(), lr=tc["lr"])         # Phase 3: Adam, fixed LR, no wd
        sched, ctl, clip = None, Phase3Controller(patience), None
    else:
        opt = torch.optim.AdamW(d4_models.param_groups(net, tc["weight_decay"]), lr=tc["lr"])
        pl = tc["plateau"]
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode=pl["mode"], factor=pl["factor"], patience=pl["patience"], min_lr=pl["min_lr"])
        ctl, clip = AdaptiveController(warm, patience), tc["grad_clip_norm"]
    lr0, lr1 = tc["warmup_start_lr"], tc["lr"]

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    log, best_state = [], None
    n_train = len(X_tr)

    for epoch in range(epochs):
        t_ep = time.time()
        if not phase3 and epoch < warm:       # linear warmup 1e-4 → 1e-3
            lr = lr0 + (lr1 - lr0) * epoch / max(1, warm - 1)
            for g in opt.param_groups:
                g["lr"] = lr
        lr_now = float(opt.param_groups[0]["lr"])

        net.train()
        perm = torch.randperm(n_train, device=device)
        ep_loss, n_b = 0.0, 0
        for i in range(0, n_train, batch_size):
            idx = perm[i: i + batch_size]
            rows = idx.tolist()
            opt.zero_grad()
            yp = net.forward_with_edge_lists(X_tr[idx], [tr_ei[j] for j in rows], [tr_et[j] for j in rows])
            loss = F.huber_loss(yp, y_tr[idx], delta=huber_delta)
            loss.backward()
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()
            ep_loss += float(loss.item())
            n_b += 1
        train_loss = ep_loss / max(1, n_b)

        net.eval()
        val_pred = predict(net, X_va, va_ei, va_et, batch_size)
        es_ic = mean_rank_ic(val_pred[:n_es], sd_es.y)
        es_loss = phase3_val_loss(val_pred[:n_es], sd_es.y, batch_size)
        sel_ic = mean_rank_ic(val_pred[n_es:], sd_sel.y)         # logged for the record ONLY
        cm = ControlMetrics(val_es_rank_ic=es_ic, val_es_loss=es_loss)   # val_sel is not passed on
        res = ctl.update(epoch, cm)
        log.append({"epoch": epoch, "lr": lr_now, "train_loss": train_loss, "val_es_loss": es_loss,
                    "val_es_rank_ic": es_ic, "val_es_rank_ic_roll3": res["roll3"],
                    "val_sel_rank_ic": sel_ic, "epoch_seconds": time.time() - t_ep})
        if res["improved"]:
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        if sched is not None and epoch >= warm:                  # plateau scheduler: not during warmup
            r3 = res["roll3"]
            sched.step(r3 if np.isfinite(r3) else -1.0)
        if res["stop"]:
            break

    fell_back = best_state is None
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    val_pred = predict(net, X_va, va_ei, va_et, batch_size)
    assert val_pred.shape == (len(sd_va.dates), len(universe))
    peak = (torch.cuda.max_memory_allocated() / 2**30) if device.type == "cuda" else 0.0
    best_row = log[ctl.best_epoch] if ctl.best_epoch >= 0 else log[-1]
    result = {
        "arm": arm, "tag": tag, "fold": fold, "seed": seed, "graph": graph, "lookback": lookback,
        "schedule": a["schedule"], "batch_size": batch_size,
        "fingerprint": fp1, "n_params": n_params, "dt_init_mean": dt_mean,
        "epochs_run": len(log), "best_epoch": ctl.best_epoch, "no_eligible_epoch_fallback_to_last": bool(fell_back),
        "best_val_es_rank_ic": float(best_row["val_es_rank_ic"]), "best_val_sel_rank_ic": float(best_row["val_sel_rank_ic"]),
        "n_train_days": int(n_train), "n_val_days": int(len(sd_va.dates)), "n_val_es_days": int(n_es),
        "mean_epoch_seconds": float(np.mean([r["epoch_seconds"] for r in log])),
        "last_epoch_seconds": float(log[-1]["epoch_seconds"]),
        "peak_gpu_gb": float(peak), "total_seconds": time.time() - t_start,
        "shuffled_targets": bool(shuffle_targets),
    }
    if write:
        p = ckpt_paths(out_dir, tag, fold, seed)
        torch.save({"arm": arm, "tag": tag, "fold": fold, "seed": seed, "lookback": lookback, "graph": graph,
                    "encoder": a["encoder"], "model_state_dict": best_state if best_state is not None else net.state_dict(),
                    "scaler_mean": scaler_mean, "scaler_scale": scaler_scale}, p["ckpt"])
        vf = ev.make_pred_frame(tag, fold, seed, "val", sd_va.dates, universe, val_pred, sd_va.y)
        vf["val_half"] = np.where(np.arange(len(vf)) < n_es * len(universe), "es", "sel")
        assert (vf["val_half"].to_numpy() == ev.val_half_labels(fold, vf["date"])).all()
        vf.to_parquet(p["valpred"], index=False)
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


def probe_memory(arm: str, fold: str, batch_size: int, graph: str = "baseline") -> float:
    """One forward+backward at `batch_size`; returns peak GB (0 on CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = arm_cfg(arm)
    sd = d4_graph.get_split_g(fold, "train", int(a["lookback"]), graph)
    edges = d4_graph.load_daily_edges(graph)
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
