# ─── Cell 16: ONE-SHOT test scoring (runs only because both lock files now exist) ───
import hashlib, d4_stats
from d4_eval import random_model, persistence_model

if (WORK / "phase4_TEST_SCORED.json").exists():
    raise RuntimeError("phase4_TEST_SCORED.json already exists — no double scoring within a run.")
d4_data.assert_test_unlocked()
_t0 = time.time()
_folds = {f.name: f for f in d4_data.tables()["folds"]}
_fold_names = [f["name"] for f in STATIC["folds"]]
_seeds = CONFIG["seeds"]
_edges = d4_graph.load_daily_edges()
_thr = json.loads((WORK / "phase4b_policy.json").read_text())["thresholds"]

_pred_hashes, TEST = {}, {}

def _save_testpred(name, fold, seed, df):
    p = WORK / f"testpred_{name}_{fold}_{seed}.parquet"
    df.to_parquet(p, index=False)
    _pred_hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()

# ── 4A: every completed arm, all 20 checkpoints, the cached daily graph ──
for _arm in PLAN["arms_run"]:
    if not COMPLETE[_arm]:
        continue
    _fr = []
    for _fn in _fold_names:
        for _s in _seeds:
            _mi = INF.get((_fn, _s)) if _arm == WINNER else None
            if _mi is None:
                _mi = d4_graph.ModelInference(d4_train.ckpt_paths(WORK, _arm, _fn, _s)["ckpt"], _DEV, SNAPSHOT)
            _df = _mi.frame(_arm, "test", _mi.predict_with_graphs("test", _edges))
            _save_testpred(_arm, _fn, _s, _df)
            _fr.append(_df)
    TEST[_arm] = pd.concat(_fr, ignore_index=True)

# ── 4B: the winner's checkpoints under selected / daily / frozen_at_start (continuous from the last training day) ──
_sel = json.loads((WORK / "phase4b_policy.json").read_text())["policy"]
_pols = {f"{WINNER}_selected": _sel, f"{WINNER}_daily": DAILY, f"{WINNER}_frozen_at_start": FROZEN}
POLICY_STATS = {n: [] for n in _pols}
for _fn in _fold_names:
    _fold = _folds[_fn]
    _test_set = set(pd.Timestamp(d) for d in d4_graph.get_split_g(_fn, "test", CONFIG["arms"][WINNER]["lookback"]).dates)
    for _s in _seeds:
        _mi = INF[(_fn, _s)]
        _dby = _mi.delta_by_day(["val", "test"]) if _encoder == "mamba" else None
        for _n, _p in _pols.items():
            _g, _rb, _st = d4_graph.run_named_policy(_p, _thr[f"{_fn}|{_s}"] if _p["kind"] == "trigger" else None,
                                                     _fold, RETS, _fold.test_dates[-1], _dby, cfg, _W, _MAXSTALE)
            _df = _mi.frame(_n, "test", _mi.predict_with_graphs("test", _g))
            _save_testpred(_n, _fn, _s, _df)
            _n_test_days = len([d for d in d4_graph.policy_days(RETS, _fold, _fold.test_dates[-1]) if d in _test_set])
            POLICY_STATS[_n].append({"fold": _fn, "seed": _s, "test_rebuilds": sum(1 for d in _rb if d in _test_set),
                                     "test_days": _n_test_days, "graph_build_seconds": _st["graph_build_seconds"],
                                     "policy_seconds": _st["policy_seconds"]})
            TEST.setdefault(_n, []).append(_df)
for _n in _pols:
    TEST[_n] = pd.concat(TEST[_n], ignore_index=True)

# ── metrics: every model through the same harness, mean over its 20 (fold, seed) runs ──
_rows = []
for _name, _df in TEST.items():
    _res = d4_eval.evaluate_predictions(_df, TARGETS, cfg, split="test")
    _m = pd.DataFrame([vars(r.fold_metrics) for r in _res])
    _rows.append({"model": _name, "n_runs": len(_m), **{k: float(_m[k].mean()) for k in
                  ("mean_rank_ic", "directional_acc", "sharpe_after_costs", "max_drawdown", "turnover", "cum_return")}})
# floors: random + persistence through the same harness (same 4 folds × 5 seeds)
_feats, _tgts = d4_data.tables()["features"], d4_data.tables()["targets"]
for _name, _fn_ in (("random", lambda s: (lambda X: random_model(X, seed=s))),
                    ("persistence", lambda s: (lambda X: persistence_model(X)))):
    _ms = [d4_eval.evaluate_fold(_name, _fn_(s), _folds[f], _feats, _tgts, cfg, seed=s).fold_metrics
           for f in _fold_names for s in _seeds]
    _m = pd.DataFrame([vars(x) for x in _ms])
    _rows.append({"model": _name, "n_runs": len(_m), **{k: float(_m[k].mean()) for k in
                  ("mean_rank_ic", "directional_acc", "sharpe_after_costs", "max_drawdown", "turnover", "cum_return")}})
TEST_METRICS = pd.DataFrame(_rows)
TEST_METRICS.to_csv(WORK / "phase4_test_metrics.csv", index=False)
print(TEST_METRICS.to_string(index=False))

# ── significance ──
_alpha = CONFIG["significance"]["alpha"]
SIG = {"phase4a": {}, "phase4b": {}}
SIG["phase4a"]["primary"] = {"comparison": f"{WINNER} vs A", **d4_stats.compare_arms(TEST[WINNER], TEST["A"], cfg)}
_others = [a for a in ("B", "C", "D") if a in TEST and a != WINNER]
_sec = {a: d4_stats.compare_arms(TEST[a], TEST["A"], cfg) for a in _others}
SIG["phase4a"]["secondary"] = {f"{a} vs A": v for a, v in _sec.items()}
if _sec:
    SIG["phase4a"]["secondary_holm_wilcoxon"] = d4_stats.holm_correction(
        {f"{a} vs A": v["wilcoxon"]["p_value"] for a, v in _sec.items()}, _alpha)
    SIG["phase4a"]["secondary_holm_bootstrap_two_sided"] = d4_stats.holm_correction(
        {f"{a} vs A": v["bootstrap_sharpe"]["p_two_sided"] for a, v in _sec.items()}, _alpha)

_selN, _dailyN, _frozN = f"{WINNER}_selected", f"{WINNER}_daily", f"{WINNER}_frozen_at_start"
try:
    _vs_daily = d4_stats.compare_arms(TEST[_selN], TEST[_dailyN], cfg)
except ValueError as _e:                    # identical per-run ICs → Wilcoxon undefined
    _a, _b = d4_stats._test_rows(TEST[_selN]), d4_stats._test_rows(TEST[_dailyN])
    _vs_daily = {"wilcoxon": {"note": f"Wilcoxon undefined: {_e}", "p_value": 1.0},
                 "bootstrap_sharpe": d4_stats.bootstrap_sharpe_test(
                     d4_stats.daily_returns(_a, cfg), d4_stats.daily_returns(_b, cfg), cfg)}
_margin = CONFIG["phase4b"]["test_noninferiority_margin_sharpe"]
_lo = _vs_daily["bootstrap_sharpe"]["ci_95"][0]
_vs_daily["noninferiority"] = {"margin": _margin, "ci95_lower_bound": _lo, "non_inferior": bool(_lo > -_margin)}
SIG["phase4b"]["selected_vs_daily"] = _vs_daily
SIG["phase4b"]["selected_vs_frozen_at_start"] = d4_stats.compare_arms(TEST[_selN], TEST[_frozN], cfg)
_ps = pd.DataFrame([dict(policy=n, **r) for n, rs in POLICY_STATS.items() for r in rs])
SIG["phase4b"]["test_rebuild_rate"] = {n: float(g["test_rebuilds"].sum() / g["test_days"].sum())
                                       for n, g in _ps.groupby("policy")}
SIG["phase4b"]["graph_build_seconds_mean_per_run"] = {n: float(g["graph_build_seconds"].mean())
                                                      for n, g in _ps.groupby("policy")}
SIG["winner_arm"], SIG["selected_policy"] = WINNER, _sel["name"]
(WORK / "phase4_significance_results.json").write_text(json.dumps(SIG, indent=2, default=float))

(WORK / "phase4_TEST_SCORED.json").write_text(json.dumps({
    "winner_arm": WINNER, "policy": _sel, "config_sha256": CONFIG_SHA256,
    "prediction_file_sha256": _pred_hashes,
    "scored_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2))
TEST_SECONDS = time.time() - _t0
print(f"test scoring done in {TEST_SECONDS / 60:.1f} min")
print(json.dumps({"primary": {k: SIG["phase4a"]["primary"][k] for k in ("comparison",)},
                  "wilcoxon_p": SIG["phase4a"]["primary"]["wilcoxon"]["p_value"],
                  "bootstrap_ci95": SIG["phase4a"]["primary"]["bootstrap_sharpe"]["ci_95"],
                  "4B_noninferior_vs_daily": _vs_daily["noninferiority"],
                  "test_rebuild_rate": SIG["phase4b"]["test_rebuild_rate"]}, indent=2, default=float))
