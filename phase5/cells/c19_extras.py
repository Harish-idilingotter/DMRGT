# ─── Cell 19: efficiency, Δ diagnostics (Phase 4.1 F1), extended compute benchmark (Phase 4.1 C3) ───
import torch, d4_graph, d4_models
from scipy.stats import spearmanr

_t0 = time.time()

# ── model efficiency: every arm, same GPU and batch size ──
def _done(arm):
    fs = FOLD_NAMES
    return [json.loads(d4_train.ckpt_paths(WORK, tag_for(arm, f), f, s)["done"].read_text()) for f in fs for s in CONFIG["seeds"]]

def _latency_ms(arm, n_passes=200):
    """Median wall time of one single-day inference pass (encode + propagate), synchronised."""
    a = CONFIG["arms"][arm]
    d4_train.set_seed(0)
    net = d4_train.build_model(arm).to(_DEV).eval()
    sd = d4_graph.get_split_g(FOLD_NAMES[-1], "val", a["lookback"], GRAPH_WINNER)
    x = torch.from_numpy(sd.X[:1]).float().to(_DEV)
    ei, et = (t.to(_DEV) for t in _edges_w[pd.Timestamp(sd.dates[0])])
    times = []
    with torch.no_grad():
        for i in range(n_passes + 10):
            if _DEV.type == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            net(x, ei, et)
            if _DEV.type == "cuda":
                torch.cuda.synchronize()
            if i >= 10:
                times.append((time.perf_counter() - t) * 1000)
    return float(np.median(times))

_rows = []
for _arm in ARMS_FOLDS14:
    _d = _done(_arm)
    _rows.append({"arm": _arm, "encoder": CONFIG["arms"][_arm]["encoder"], "lookback": CONFIG["arms"][_arm]["lookback"],
                  "schedule": CONFIG["arms"][_arm]["schedule"], "parameters": _d[0]["n_params"],
                  "mean_seconds_per_epoch": float(np.mean([x["mean_epoch_seconds"] for x in _d])),
                  "mean_epochs_to_early_stop": float(np.mean([x["epochs_run"] for x in _d])),
                  "mean_total_train_seconds_per_run": float(np.mean([x["total_seconds"] for x in _d])),
                  "inference_latency_ms_per_day_median": _latency_ms(_arm),
                  "peak_gpu_gb_max": float(np.max([x["peak_gpu_gb"] for x in _d])), "batch_size": BATCH_SIZE})
EFF = pd.DataFrame(_rows)
EFF.to_csv(WORK / "phase5_efficiency.csv", index=False)
print(EFF.to_string(index=False))

# ── Δ diagnostics (reported, not a gate): the winner (if Mamba) and arm B regardless ──
_vix = pd.read_parquet(CACHE / "prices.parquet").query("ticker == '^VIX'").set_index("date")["close"].sort_index()

def _delta_diag(arm, folds, label):
    per_day, per_chan, a_mins = {}, {}, []
    for f in folds:
        mis = [(INF.get((f, s)) if (arm == WINNER and f in FOLD_NAMES) else
                (INFH.get((f, s)) if (arm == WINNER and f in HOLDOUT_NAMES) else None)) or
               d4_graph.ModelInference(d4_train.ckpt_paths(WORK, tag_for(arm, f), f, s)["ckpt"], _DEV, SNAPSHOT)
               for s in seeds_for(f)]
        ch = np.mean([m.delta_channels("test") for m in mis], axis=0)                 # [days, d_inner], mean over seeds
        a_mins += [m.model.encoder.mamba.a_min() for m in mis]
        dates = [pd.Timestamp(d) for d in mis[0].split_data("test").dates]
        for d, row in zip(dates, ch):
            per_day[d], per_chan[d] = float(row.mean()), row
    s = pd.Series(per_day).sort_index()
    v = _vix.reindex(s.index)
    out = {"label": label, "n_days": int(len(s)), "mean_delta": float(s.mean()), "std_delta": float(s.std()),
           "cv_daily_mean_delta": float(s.std() / s.mean()),
           "spearman_with_vix": float(spearmanr(s.values, v.values).correlation)}
    covid = s[(s.index >= "2020-03-01") & (s.index <= "2020-04-30")]
    y19 = s[(s.index >= "2019-01-01") & (s.index <= "2019-12-31")]
    out["ratio_mean_delta_covid_2020_03_04_vs_2019"] = float(covid.mean() / y19.mean()) if len(covid) and len(y19) else None
    a_min = float(np.mean(a_mins))
    med = pd.Series({d: float(np.median(r)) for d, r in per_chan.items()}).sort_index()   # median channel's Δ per day
    mem = np.log(2.0) / (med * a_min)                                                   # days until exp(-Δ·A_min)^k < 0.5
    calm, vol = v <= v.quantile(0.25), v >= v.quantile(0.75)
    out["effective_memory_days_median_channel"] = {"a_min": a_min, "calm_days_vix_bottom_quartile": float(mem[calm].mean()),
                                                   "volatile_days_vix_top_quartile": float(mem[vol].mean()),
                                                   "all_days": float(mem.mean())}
    out["flat_delta"] = bool(abs(out["spearman_with_vix"]) < 0.1 and out["cv_daily_mean_delta"] < 0.05)
    return out, s

DELTA = {}
_series = {}
_targets_d = ([("winner", WINNER)] if WENC == "mamba" else []) + ([("arm_B", "B")] if "B" in ARMS_FOLDS14 and WINNER != "B" else [])
if WENC == "mamba" and WINNER == "B":
    _targets_d = [("winner_and_arm_B", "B")]
for _lab, _arm in _targets_d:
    DELTA[f"{_lab}_folds_1_4"], _series[_lab] = _delta_diag(_arm, FOLD_NAMES, f"{_arm} test folds 1-4")
    if _arm == WINNER:
        DELTA[f"{_lab}_holdout_5_6"], _ = _delta_diag(_arm, HOLDOUT_NAMES, f"{_arm} holdout folds 5-6")
DELTA["note"] = ("Reported as it comes out. 'flat_delta' = |Spearman with VIX| < 0.1 and CV < 0.05: Δ does not vary with regime. "
                 "Calm / volatile = bottom / top VIX quartile of the diagnostic dates (this definition is the notebook's, not pre-registered by the spec). "
                 "Arm B on the holdout exists only when B is the winner (S4 trains winner, A, P).")
(WORK / "phase5_delta_diagnostics.json").write_text(json.dumps(DELTA, indent=2, default=float))
if _series:
    pd.DataFrame({k: v for k, v in _series.items()}).rename_axis("date").to_csv(WORK / "phase5_delta_by_date.csv")
print(json.dumps(DELTA, indent=2, default=float)[:2500])

# ── real-data graph-construction cost (CPU, float64), winner graph window ──
_build = d4_graph.make_graph_builder(cfg_w)
_fold4 = _folds[FOLD_NAMES[-1]]
_idx = RETS.index
_pos = [_idx.get_loc(d) for d in _fold4.test_dates if d in _idx][:300]
_rc = d4_graph.warm_rolling_corr(RETS, _idx[_pos[0] - 1], _W)
_rr = RETS.to_numpy(dtype=np.float64)
_t_full, _t_inc, _t_edges = [], [], []
for _i in _pos:
    t = time.perf_counter(); RETS.iloc[_i - _W + 1: _i + 1].corr().values; _t_full.append(time.perf_counter() - t)
    t = time.perf_counter(); _rc.push(_rr[_i]); _Ci = _rc.corr(); _t_inc.append(time.perf_counter() - t)
    t = time.perf_counter(); _build(_Ci); _t_edges.append(time.perf_counter() - t)
_rate_sel = SIG["phase4b"]["test_rebuild_rate"][f"{_selN}|folds_1_4"]
_nt = len(_fold4.test_dates)
REAL = {"n_stocks": len(UNIVERSE), "window": _W, "n_updates_timed": len(_pos),
        "full_recompute_ms_median": float(np.median(_t_full) * 1000), "incremental_update_ms_median": float(np.median(_t_inc) * 1000),
        "edge_build_ms_median": float(np.median(_t_edges) * 1000), "selected_policy_test_rebuild_rate": _rate_sel,
        "test_period_days_fold4": _nt,
        "test_period_total_s_full_recompute_daily": _nt * (np.median(_t_full) + np.median(_t_edges)),
        "test_period_total_s_incremental_daily": _nt * (np.median(_t_inc) + np.median(_t_edges)),
        "test_period_total_s_incremental_selected": _nt * (np.median(_t_inc) + np.median(_t_edges) * _rate_sel),
        "policy_run_seconds_mean_per_run": SIG["phase4b"]["graph_build_seconds_mean_per_run"]}
(WORK / "phase5_real_graph_cost.json").write_text(json.dumps(REAL, indent=2, default=float))

# ── extended compute benchmark (Phase 4.1 C3): where, if anywhere, does the mechanism matter? ──
_Pb = CONFIG["phase4b"]
_lb = CONFIG["arms"][WINNER]["lookback"]

def _sync():
    if _DEV.type == "cuda":
        torch.cuda.synchronize()

def _bench(N, window, T, seed=0, n_sample=20):
    rng = np.random.default_rng(seed)
    nf = max(3, N // 10)
    L1, L2 = rng.normal(size=(nf, N)), rng.normal(size=(nf, N))
    Fm = rng.normal(size=(T + window, nf))
    R = np.where((np.arange(T + window) < (T + window) // 2)[:, None], Fm @ L1, Fm @ L2) * 0.01 + rng.normal(size=(T + window, N)) * 0.01
    sec = rng.integers(0, 11, size=N)
    grp = [np.flatnonzero(sec == k) for k in range(11)]
    src, dst = [], []
    for g in grp:                                               # intra-sector directed edges (both ways)
        s_, d_ = np.meshgrid(g, g, indexing="ij"); m = s_ != d_
        src.append(s_[m]); dst.append(d_[m])
    sec_src, sec_dst = np.concatenate(src).tolist(), np.concatenate(dst).tolist()
    bld = lambda C: d4_graph.edges_from_corr(C, cfg_w, sec_src, sec_dst)
    days = np.linspace(window, T + window - 1, n_sample).astype(int)
    t_full, t_edges, t_xfer = [], [], []
    for d in days:
        t = time.perf_counter(); C = np.corrcoef(R[d - window + 1: d + 1].T); t_full.append(time.perf_counter() - t)
        t = time.perf_counter(); ei, et = bld(C); t_edges.append(time.perf_counter() - t)
        _sync(); t = time.perf_counter(); ei.to(_DEV); et.to(_DEV); _sync(); t_xfer.append(time.perf_counter() - t)
    rc = d4_graph.RollingCorr(window, N)
    for r in R[:window]:
        rc.push(r)
    t_inc = []
    for r in R[window: window + min(T, 120)]:                   # per-update cost is flat in t: time 120 updates
        t = time.perf_counter(); rc.push(r); rc.corr(); t_inc.append(time.perf_counter() - t)
    # model inference for this N (winner architecture, untrained weights, random inputs), timed on the device
    net = d4_models.make_model(WENC, CONFIG["mamba"]).to(_DEV).eval()
    try:
        x = torch.randn(1, N, _lb, 8, device=_DEV)
        e1, e2 = ei.to(_DEV), et.to(_DEV)
        with torch.no_grad():
            for _ in range(3):
                net(x, e1, e2)
            t_model = []
            for _ in range(10):
                _sync(); t = time.perf_counter(); net(x, e1, e2); _sync(); t_model.append(time.perf_counter() - t)
        t_model = float(np.median(t_model))
    except torch.cuda.OutOfMemoryError:
        t_model = float("nan"); torch.cuda.empty_cache()
    del net
    m = lambda a: float(np.median(a))
    tf, te, tx, ti = m(t_full), m(t_edges), m(t_xfer), m(t_inc)
    r = _rate_sel
    g_full = tf + te + tx                                       # full recompute every day
    g_inc = ti + te + tx                                        # incremental, rebuild every day
    g_trig = ti + r * (te + tx)                                 # incremental + selected trigger rate
    return {"n_stocks": N, "window": window, "days": T, "ms_full_recompute": tf * 1e3, "ms_incremental_update": ti * 1e3,
            "ms_edge_build": te * 1e3, "ms_edge_transfer": tx * 1e3, "ms_model_inference": t_model * 1e3,
            "trigger_rebuild_rate": r,
            "graph_ms_per_day_full_daily": g_full * 1e3, "graph_ms_per_day_incremental_daily": g_inc * 1e3,
            "graph_ms_per_day_incremental_trigger": g_trig * 1e3,
            "graph_share_full_daily": g_full / (g_full + t_model), "graph_share_incremental_daily": g_inc / (g_inc + t_model),
            "graph_share_incremental_trigger": g_trig / (g_trig + t_model),
            "total_s_full_daily": (g_full + t_model) * T, "total_s_incremental_trigger": (g_trig + t_model) * T}

_bench_rows = []
for _N in _Pb["scale_benchmark_n_stocks"]:
    for _w in _Pb["scale_benchmark_windows"]:
        _bench_rows.append(_bench(_N, _w, _Pb["scale_benchmark_days"]))
        print(f"benchmark N={_N} window={_w} done ({time.time() - _t0:.0f}s)", flush=True)
BENCH = pd.DataFrame(_bench_rows)
BENCH.to_csv(WORK / "phase5b_compute_benchmark.csv", index=False)
print(BENCH[["n_stocks", "window", "ms_full_recompute", "ms_incremental_update", "ms_edge_build", "ms_model_inference",
             "graph_share_full_daily", "graph_share_incremental_trigger"]].to_string(index=False))
EXTRAS_SECONDS = time.time() - _t0
print(f"extras done in {EXTRAS_SECONDS / 60:.1f} min")
