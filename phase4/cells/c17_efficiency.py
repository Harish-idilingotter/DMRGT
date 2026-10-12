# ─── Cell 17: efficiency + technical-effect measurements ───
import torch, d4_graph, d4_models

_t0 = time.time()
_fold_names = [f["name"] for f in STATIC["folds"]]
_seeds = CONFIG["seeds"]

# 14.1 Model efficiency — same GPU, same batch size, every completed arm (A vs winner is the headline pair)
def _done(arm):
    return [json.loads(d4_train.ckpt_paths(WORK, arm, f, s)["done"].read_text()) for f in _fold_names for s in _seeds]

def _latency_ms(arm, n_passes=200):
    """Median wall time of one single-day inference pass (encode + propagate), synchronised."""
    a = CONFIG["arms"][arm]
    d4_train.set_seed(0)
    net = d4_train.build_model(arm).to(_DEV).eval()
    sd = d4_graph.get_split_g(_fold_names[-1], "val", a["lookback"])
    x = torch.from_numpy(sd.X[:1]).float().to(_DEV)
    ei, et = (t.to(_DEV) for t in _edges[pd.Timestamp(sd.dates[0])])
    times = []
    with torch.no_grad():
        for i in range(n_passes + 10):
            if _DEV.type == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            net(x, ei, et)
            if _DEV.type == "cuda":
                torch.cuda.synchronize()
            if i >= 10:                                  # discard warm-up passes
                times.append((time.perf_counter() - t) * 1000)
    return float(np.median(times))

_rows = []
for _arm in PLAN["arms_run"]:
    if not COMPLETE[_arm]:
        continue
    _d = _done(_arm)
    _rows.append({"arm": _arm, "encoder": CONFIG["arms"][_arm]["encoder"], "lookback": CONFIG["arms"][_arm]["lookback"],
                  "parameters": _d[0]["n_params"],
                  "mean_seconds_per_epoch": float(np.mean([x["mean_epoch_seconds"] for x in _d])),
                  "mean_epochs_to_early_stop": float(np.mean([x["epochs_run"] for x in _d])),
                  "mean_total_train_seconds_per_run": float(np.mean([x["total_seconds"] for x in _d])),
                  "inference_latency_ms_per_day_median": _latency_ms(_arm),
                  "peak_gpu_gb_max": float(np.max([x["peak_gpu_gb"] for x in _d])),
                  "batch_size": BATCH_SIZE})
EFF = pd.DataFrame(_rows)
EFF.to_csv(WORK / "phase4_efficiency.csv", index=False)
print(EFF.to_string(index=False))

# 14.2 Graph-construction cost (CPU, float64) — real 40-stock data
_build = d4_graph.make_graph_builder(cfg)
_sec_src, _sec_dst = d4_graph._sector_edges(cfg)
_fold4 = [f for f in d4_data.tables()["folds"] if f.name == _fold_names[-1]][0]
_idx = RETS.index
_pos_test = [_idx.get_loc(d) for d in _fold4.test_dates if d in _idx][:300]
_rc = d4_graph.warm_rolling_corr(RETS, _idx[_pos_test[0] - 1], _W)
_rows_ret = RETS.to_numpy(dtype=np.float64)
_t_full, _t_inc, _t_edges = [], [], []
for _i in _pos_test:
    t = time.perf_counter(); _C_full = RETS.iloc[_i - _W + 1: _i + 1].corr().values; _t_full.append(time.perf_counter() - t)
    t = time.perf_counter(); _rc.push(_rows_ret[_i]); _C_inc = _rc.corr(); _t_inc.append(time.perf_counter() - t)
    t = time.perf_counter(); _build(_C_inc); _t_edges.append(time.perf_counter() - t)
REAL_COST = {"n_updates_timed": len(_pos_test),
             "full_recompute_ms_median": float(np.median(_t_full) * 1000),
             "incremental_update_ms_median": float(np.median(_t_inc) * 1000),
             "edge_build_ms_median": float(np.median(_t_edges) * 1000),
             "policy_run_seconds_mean_per_run": SIG["phase4b"]["graph_build_seconds_mean_per_run"],
             "test_rebuild_rate": SIG["phase4b"]["test_rebuild_rate"]}
_n_td = len(_fold4.test_dates)
REAL_COST["test_period_total_seconds_full_recompute_daily"] = _n_td * (np.median(_t_full) + np.median(_t_edges))
REAL_COST["test_period_total_seconds_incremental_daily"] = _n_td * (np.median(_t_inc) + np.median(_t_edges))
REAL_COST["test_period_total_seconds_incremental_selected"] = _n_td * (
    np.median(_t_inc) + np.median(_t_edges) * SIG["phase4b"]["test_rebuild_rate"][f"{WINNER}_selected"])
(WORK / "phase4_real_graph_cost.json").write_text(json.dumps(REAL_COST, indent=2, default=float))
print(json.dumps(REAL_COST, indent=2, default=float))

# Synthetic scaling benchmark: random returns with a mid-sample regime switch, window 60, 750 days
_target_rate = float(SIG["phase4b"]["test_rebuild_rate"][f"{WINNER}_selected"])
_sec_none = ([], [])
def _bench(N, T=750, seed=0):
    rng = np.random.default_rng(seed)
    nf = max(3, N // 10)
    L1, L2 = rng.normal(size=(nf, N)), rng.normal(size=(nf, N))
    F = rng.normal(size=(T, nf))
    R = np.where((np.arange(T) < T // 2)[:, None], F @ L1, F @ L2) * 0.01 + rng.normal(size=(T, N)) * 0.01
    days = list(range(_W, T))
    bld = lambda C: d4_graph.edges_from_corr(C, cfg, [], [])
    t = time.perf_counter()
    for d in days:
        bld(np.corrcoef(R[d - _W + 1: d + 1].T))
    full = time.perf_counter() - t
    def run(rule, stale):
        rc = d4_graph.RollingCorr(_W, N)
        for r in R[:_W]:
            rc.push(r)
        t = time.perf_counter()
        _, rb = d4_graph.run_policy(days, lambda d: R[d], bld, lambda d: 0.0, rule, stale, rc)
        return time.perf_counter() - t, len(rb) / len(days)
    inc, _ = run(d4_graph.rule_daily, float("inf"))
    # corr-drift threshold bisected so the realised rate matches the selected policy's test rebuild rate
    pc_steps = [None] * (len(days) + 1)
    rc0 = d4_graph.RollingCorr(_W, N)
    for r in R[:_W]:
        rc0.push(r)
    pc_steps[0] = rc0.corr()
    for k, d in enumerate(days):
        rc0.push(R[d]); pc_steps[k + 1] = rc0.corr()
    def rate_of(th):
        return d4_graph.simulate_rate(days, pc_steps, None, d4_graph.make_rule("corr_drift", th, None), _MAXSTALE)
    th = d4_graph._bisect(rate_of, 0.0, 5.0, _target_rate)
    trig, real = run(d4_graph.make_rule("corr_drift", th, None), _MAXSTALE)
    return {"n_stocks": N, "days": len(days), "full_recompute_daily_s": full, "incremental_daily_s": inc,
            "incremental_trigger_s": trig, "trigger_target_rate": _target_rate, "trigger_realised_rate": real,
            "speedup_incremental_vs_full": full / inc, "speedup_trigger_vs_full": full / trig}

BENCH = pd.DataFrame([_bench(N) for N in CONFIG["phase4b"]["scale_benchmark_n_stocks"]])
BENCH.to_csv(WORK / "phase4b_compute_benchmark.csv", index=False)
print(BENCH.to_string(index=False))

# 14.3 Explainability hook: mean Δ per trading day (winner, averaged over seeds), 2019–2022. Read-only.
_drows = []
if _encoder == "mamba":
    for _fn in _fold_names:
        _per_seed = [pd.Series(INF[(_fn, s)].delta_by_day(["test"])) for s in _seeds]
        _dfm = pd.concat(_per_seed, axis=1)
        _drows.append(pd.DataFrame({"date": _dfm.index, "fold": _fn, "mean_delta": _dfm.mean(axis=1).values,
                                    "std_delta_across_seeds": _dfm.std(axis=1).values}))
DELTA = pd.concat(_drows, ignore_index=True) if _drows else pd.DataFrame(columns=["date", "fold", "mean_delta"])
DELTA.to_csv(WORK / "phase4_delta_by_date.csv", index=False)
if len(DELTA):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _fig, _ax = plt.subplots(figsize=(11, 3.2))
    _d = DELTA.sort_values("date")
    _ax.plot(pd.to_datetime(_d["date"]), _d["mean_delta"], lw=0.8)
    _ax.set_ylabel("mean Δ (memory-reset strength)"); _ax.set_title("Mamba Δ by trading day (winner, mean over seeds)")
    _fig.tight_layout(); _fig.savefig(WORK / "phase4_delta_by_date.png", dpi=130); plt.close(_fig)
EFF_SECONDS = time.time() - _t0
print(f"efficiency + technical-effect measurements done in {EFF_SECONDS / 60:.1f} min")
