# ─── Cell 18: ONE-SHOT test scoring — folds 1-4 and holdout folds 5-6 (first time any test row is read) ───
import hashlib, d4_stats
from d4_eval import random_model, persistence_model

if (WORK / "phase5_TEST_SCORED.json").exists():
    raise RuntimeError("phase5_TEST_SCORED.json already exists — no double scoring within a run.")
d4_data.assert_test_unlocked()
_t0 = time.time()
_edges_w = d4_graph.load_daily_edges(GRAPH_WINNER)
_folds = {f.name: f for f in d4_data.tables()["folds"]}
_tb = d4_data.tables()
_BT, _QIN, _FEE = CONFIG["backtest"], CONFIG["postproc"]["q_in"], (CONFIG["backtest"]["commission_bps"] + CONFIG["backtest"]["slippage_bps"]) / 1e4
_pol_json = json.loads((WORK / "phase5b_policy.json").read_text())
ALL_EVAL = FOLD_NAMES + HOLDOUT_NAMES
_pred_hashes = {}
PRED = {}                                                      # model → fold → val+test rows, all seeds

def _save_test(name, fold, seed, df_test):
    p = WORK / f"testpred_{name}_{fold}_{seed}.parquet"
    df_test.to_parquet(p, index=False)
    _pred_hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()

def _val_rows(arm, fold, seed):
    return pd.read_parquet(d4_train.ckpt_paths(WORK, tag_for(arm, fold), fold, seed)["valpred"]).assign(model=arm)

# ── arms: every checkpoint, the cached daily graph of the winning variant ──
INFH = {}                                                      # holdout winner models
_arm_sets = {f: ["P", "A"] + S2_ARMS[1:] for f in FOLD_NAMES}
_arm_sets.update({f: [WINNER, "A", "P"] for f in HOLDOUT_NAMES})
for _fn in ALL_EVAL:
    for _arm in _arm_sets[_fn]:
        _fr = []
        for _s in seeds_for(_fn):
            _mi = INF.get((_fn, _s)) if (_arm == WINNER and _fn in FOLD_NAMES) else None
            if _mi is None:
                _mi = d4_graph.ModelInference(d4_train.ckpt_paths(WORK, tag_for(_arm, _fn), _fn, _s)["ckpt"], _DEV, SNAPSHOT)
                if _arm == WINNER and _fn in HOLDOUT_NAMES:
                    INFH[(_fn, _s)] = _mi
            _dt = _mi.frame(_arm, "test", _mi.predict_with_graphs("test", _edges_w))
            _save_test(_arm, _fn, _s, _dt)
            _fr.append(pd.concat([_val_rows(_arm, _fn, _s), _dt], ignore_index=True))
        PRED.setdefault(_arm, {})[_fn] = pd.concat(_fr, ignore_index=True)
    print(f"arms scored on {_fn} ({time.time() - _t0:.0f}s)", flush=True)

# ── Phase 4B: the winner under selected / daily / frozen_at_start, continuous from the last training day ──
_selN, _dailyN, _frozN = f"{WINNER}_selected", f"{WINNER}_daily", f"{WINNER}_frozen_at_start"
_pols = {_selN: _pol_json["policy"], _dailyN: DAILY, _frozN: FROZEN}
POLICY_STATS, HOLD_THRESH = [], {}
for _fn in ALL_EVAL:
    _fold = _folds[_fn]
    _test_set = set(pd.Timestamp(d) for d in d4_graph.get_split_g(_fn, "test", CONFIG["arms"][WINNER]["lookback"], GRAPH_WINNER).dates)
    for _s in seeds_for(_fn):
        _mi = INF[(_fn, _s)] if _fn in FOLD_NAMES else INFH[(_fn, _s)]
        if _fn in FOLD_NAMES:
            _th = _pol_json["thresholds"][f"{_fn}|{_s}"]
        else:                                                  # holdout: calibrate the selected policy on this fold's training days
            _th = calibrate_policy(_mi, _fold, _pol_json["policy"])
            HOLD_THRESH[f"{_fn}|{_s}"] = _th
        _dby = _mi.delta_by_day(["val", "test"]) if WENC == "mamba" else None
        for _n, _p in _pols.items():
            _g, _rb, _st = d4_graph.run_named_policy(_p, _th if _p["kind"] == "trigger" else None, _fold, RETS,
                                                     _fold.test_dates[-1], _dby, cfg_w, _W, _MAXSTALE)
            _dt = _mi.frame(_n, "test", _mi.predict_with_graphs("test", _g))
            _dv = _mi.frame(_n, "val", _mi.predict_with_graphs("val", _g))
            _save_test(_n, _fn, _s, _dt)
            PRED.setdefault(_n, {}).setdefault(_fn, []).append(pd.concat([_dv, _dt], ignore_index=True))
            _nd = len([d for d in d4_graph.policy_days(RETS, _fold, _fold.test_dates[-1]) if d in _test_set])
            POLICY_STATS.append({"policy": _n, "fold": _fn, "seed": _s, "test_rebuilds": sum(1 for d in _rb if d in _test_set),
                                 "test_days": _nd, "graph_build_seconds": _st["graph_build_seconds"],
                                 "policy_seconds": _st["policy_seconds"]})
for _n in _pols:
    for _fn in ALL_EVAL:
        PRED[_n][_fn] = pd.concat(PRED[_n][_fn], ignore_index=True)

# ── floors: random (fresh draw per seed and day, F3) and persistence, same days as the models ──
_f1 = _tb["feat_wide"]["f1_logret"][UNIVERSE]
for _fn in ALL_EVAL:
    _sv = d4_graph.get_split_g(_fn, "val", 30, GRAPH_WINNER)
    _st_ = d4_graph.get_split_g(_fn, "test", 30, GRAPH_WINNER)
    _dates = list(_sv.dates) + list(_st_.dates)
    _y = np.concatenate([_sv.y, _st_.y])
    _split = np.array(["val"] * len(_sv.dates) + ["test"] * len(_st_.dates))
    _Xr = [np.zeros((len(UNIVERSE), 1, 8)) for _ in _dates]
    for _i, _d in enumerate(_dates):
        _Xr[_i][:, 0, 0] = _f1.loc[pd.Timestamp(_d)].to_numpy()
    _per = {"persistence": [(0, np.stack([persistence_model(x) for x in _Xr]))],
            "random": [(s, np.stack([random_model(x, s, i) for i, x in enumerate(_Xr)])) for s in seeds_for(_fn)]}
    for _name, _lst in _per.items():
        _fr = []
        for _s, _p in _lst:
            _d = d4_eval.make_pred_frame(_name, _fn, _s, "x", _dates, UNIVERSE, _p, _y)
            _d["split"] = np.repeat(_split, len(UNIVERSE))
            _d["val_half"] = np.where(_d["split"] == "val", d4_eval.val_half_labels(_fn, _d["date"]), "")
            _fr.append(_d)
        PRED.setdefault(_name, {})[_fn] = pd.concat(_fr, ignore_index=True)

# ── helpers: raw (per run, Phase 3 construction) and strategy (ensemble + chosen post-processing) levels ──
def rows(name, folds, split="test"):
    return pd.concat([PRED[name][f] for f in folds], ignore_index=True).query("split == @split")

def raw_summary(name, folds):
    res = d4_eval.evaluate_predictions(rows(name, folds), TARGETS, cfg, split="test")
    m = pd.DataFrame([vars(r.fold_metrics) for r in res])
    return {"model": name, "level": "raw", "n_runs": len(m), "n_days": int(m["n_days"].sum()),
            **{k: float(m[k].mean()) for k in ("mean_rank_ic", "directional_acc", "sharpe_after_costs", "max_drawdown", "turnover", "cum_return")}}

def strat(name, folds, hl=None, q_out=None):
    """Pooled (date order) test-day strategy returns and one-way turnover; EMA/buffer state runs on from validation."""
    hl = CHOSEN_HL if hl is None else hl
    q_out = CHOSEN_QOUT if q_out is None else q_out
    rs, ts = [], []
    for f in folds:
        df = PRED[name][f]
        ret, turn = d4_eval.strategy_series(df, RAW_WIDE, UNIVERSE, _BT, hl, _QIN, q_out)
        td = pd.DatetimeIndex(sorted(df.loc[df["split"] == "test", "date"].unique()))
        rs.append(ret.reindex(td)); ts.append(turn.reindex(td))
    return pd.concat(rs).sort_index(), pd.concat(ts).sort_index()

def strat_summary(name, folds, hl=None, q_out=None):
    r, t = strat(name, folds, hl, q_out)
    s = d4_eval.pooled_stats(r, t)
    return {"model": name, "level": "strategy", "n_runs": int(rows(name, folds)["seed"].nunique() * len(folds)),
            "n_days": s["n_days"], "mean_rank_ic": float("nan"), "directional_acc": float("nan"),
            "sharpe_after_costs": s["sharpe"], "max_drawdown": s["max_drawdown"], "turnover": s["turnover"], "cum_return": s["cum_return"]}

def table(models, folds):
    out = []
    for m in models:
        out += [raw_summary(m, folds), strat_summary(m, folds)]
    return pd.DataFrame(out)

ARMS_FOLDS14 = ["P", "A"] + S2_ARMS[1:]
ARMS_HOLDOUT = [WINNER, "A", "P"]
_pol_models = [_selN, _dailyN, _frozN]
_floors = ["random", "persistence"]
TEST_METRICS = table(ARMS_FOLDS14 + _pol_models + _floors, FOLD_NAMES)
HOLD_METRICS = table(ARMS_HOLDOUT + _pol_models + _floors, HOLDOUT_NAMES)
TEST_METRICS.to_csv(WORK / "phase5_test_metrics.csv", index=False)
HOLD_METRICS.to_csv(WORK / "phase5_holdout_metrics.csv", index=False)
for _tab in (TEST_METRICS, HOLD_METRICS):
    _rt = _tab[(_tab["model"] == "random") & (_tab["level"] == "raw")]["turnover"].iloc[0]
    assert _rt > 0.5, f"random baseline turnover {_rt:.3f} <= 0.5 — the random floor is not drawing fresh noise per day"
print("TEST folds 1-4\n", TEST_METRICS.to_string(index=False))
print("HOLDOUT folds 5-6\n", HOLD_METRICS.to_string(index=False))

# ── descriptive: contribution of each post-processing component (folds 1-4) ──
_comp = []
for _m in ARMS_FOLDS14:
    _raw = raw_summary(_m, FOLD_NAMES)
    _comp.append({"model": _m, "component": "raw (Phase 3 construction, per seed)", "sharpe": _raw["sharpe_after_costs"],
                  "turnover": _raw["turnover"], "cost_drag_annual": _FEE * _raw["turnover"] * 252})
    for _lab, _hl, _q in (("ensemble only", 0, _QIN), ("+ smoothing", CHOSEN_HL, _QIN), ("+ buffer (chosen)", CHOSEN_HL, CHOSEN_QOUT)):
        _r, _t = strat(_m, FOLD_NAMES, _hl, _q)
        _s = d4_eval.pooled_stats(_r, _t)
        _comp.append({"model": _m, "component": _lab, "sharpe": _s["sharpe"], "turnover": _s["turnover"],
                      "cost_drag_annual": _FEE * _s["turnover"] * 252})
COMPONENTS = pd.DataFrame(_comp)
COMPONENTS.to_csv(WORK / "phase5_postproc_components.csv", index=False)

# ── significance ──
_alpha = CONFIG["significance"]["alpha"]

def verdict(ci_low, ci_high, name_a, name_b):
    if ci_low > 0:
        return f"{name_a} is significantly better than {name_b}"
    if ci_high < 0:
        return f"{name_a} is significantly worse than {name_b}"
    return f"no significant difference between {name_a} and {name_b}"

def wil(a, b, folds):
    ia = d4_stats.per_fold_seed_ic(rows(a, folds), cfg)
    ib = d4_stats.per_fold_seed_ic(rows(b, folds), cfg)
    try:
        return d4_stats.wilcoxon_test(ia, ib, cfg)
    except ValueError as e:                                   # identical per-run ICs → undefined
        return {"note": f"Wilcoxon undefined: {e}", "p_value": 1.0, "n_pairs": len(ia), "wins": 0}

def boot_strategy(a, b, folds):
    ra, _ = strat(a, folds); rb, _ = strat(b, folds)
    assert ra.index.equals(rb.index), "pooled strategy series are not on the same days"
    out = d4_stats.bootstrap_sharpe_test(ra, rb, cfg)
    out["verdict"] = verdict(out["ci_95"][0], out["ci_95"][1], a, b)
    return out

def compare(a, b, folds):
    return {"model_level_wilcoxon_raw_rank_ic": wil(a, b, folds), "strategy_level_bootstrap_sharpe": boot_strategy(a, b, folds)}

SIG = {"note": "model level = Wilcoxon on paired (fold, seed) raw mean rank IC; strategy level = stationary bootstrap on the "
               "after-cost Sharpe difference of pooled daily strategy returns (ensemble + chosen post-processing)",
       "winner_arm": WINNER, "graph": GRAPH_WINNER, "postproc": {"ema_half_life": CHOSEN_HL, "q_in": _QIN, "q_out": CHOSEN_QOUT},
       "selected_policy": _pol_json["policy"]["name"], "primary_holdout": {}, "secondary_folds_1_4": {}, "phase4b": {}}
# primary (confirmatory): holdout folds 5-6
SIG["primary_holdout"][f"{WINNER} vs A"] = compare(WINNER, "A", HOLDOUT_NAMES)
SIG["primary_holdout"][f"{WINNER} vs P"] = compare(WINNER, "P", HOLDOUT_NAMES)
SIG["primary_holdout"]["note"] = "10 paired (fold, seed) observations per Wilcoxon test: low power."
# secondary (Holm-corrected together): folds 1-4
_sec, _pv = {}, {}
_other = [a for a in S2_ARMS[1:] if a != WINNER]
for _a, _b in [(WINNER, "A")] + [(o, "A") for o in _other] + [("A", "P")]:
    _sec[f"{_a} vs {_b}"] = compare(_a, _b, FOLD_NAMES)
    _pv[f"{_a} vs {_b} (model, Wilcoxon)"] = _sec[f"{_a} vs {_b}"]["model_level_wilcoxon_raw_rank_ic"]["p_value"]
    _pv[f"{_a} vs {_b} (strategy, bootstrap two-sided)"] = _sec[f"{_a} vs {_b}"]["strategy_level_bootstrap_sharpe"]["p_two_sided"]
for _m in ARMS_FOLDS14:                           # post-processing effect: strategy vs raw, per arm
    _raw_r = d4_stats.daily_returns(rows(_m, FOLD_NAMES), cfg)
    _st_r, _ = strat(_m, FOLD_NAMES)
    _common = _raw_r.index.intersection(_st_r.index)
    assert len(_common) == len(_raw_r) == len(_st_r), "raw and strategy series cover different days"
    _b = d4_stats.bootstrap_sharpe_test(_st_r.reindex(_raw_r.index), _raw_r, cfg)
    _b["verdict"] = verdict(_b["ci_95"][0], _b["ci_95"][1], f"{_m} strategy", f"{_m} raw")
    _sec[f"post-processing effect {_m}"] = _b
    _pv[f"post-processing effect {_m} (bootstrap two-sided)"] = _b["p_two_sided"]
SIG["secondary_folds_1_4"]["comparisons"] = _sec
SIG["secondary_folds_1_4"]["holm"] = d4_stats.holm_correction(_pv, _alpha)
# Phase 4B (own criteria, not in the Holm family): non-inferiority vs daily, superiority vs frozen_at_start; folds 1-4 and holdout
_m4 = CONFIG["phase4b"]["test_noninferiority_margin_sharpe"]
for _lab, _fs in (("folds_1_4", FOLD_NAMES), ("holdout_5_6", HOLDOUT_NAMES)):
    _vd = compare(_selN, _dailyN, _fs)
    _lo = _vd["strategy_level_bootstrap_sharpe"]["ci_95"][0]
    _vd["noninferiority"] = {"margin": _m4, "ci95_lower_bound": _lo, "non_inferior": bool(_lo > -_m4)}
    SIG["phase4b"][f"{_lab}: selected vs daily"] = _vd
    SIG["phase4b"][f"{_lab}: selected vs frozen_at_start"] = compare(_selN, _frozN, _fs)
_ps = pd.DataFrame(POLICY_STATS)
_ps["is_holdout"] = _ps["fold"].isin(HOLDOUT_NAMES)
SIG["phase4b"]["test_rebuild_rate"] = {f"{n}|{'holdout' if h else 'folds_1_4'}": float(g["test_rebuilds"].sum() / g["test_days"].sum())
                                       for (n, h), g in _ps.groupby(["policy", "is_holdout"])}
SIG["phase4b"]["graph_build_seconds_mean_per_run"] = {n: float(g["graph_build_seconds"].mean()) for n, g in _ps.groupby("policy")}
SIG["phase4b"]["holdout_calibrated_thresholds"] = HOLD_THRESH
(WORK / "phase5_significance_results.json").write_text(json.dumps(SIG, indent=2, default=float))

(WORK / "phase5_TEST_SCORED.json").write_text(json.dumps({
    "winner_arm": WINNER, "graph": GRAPH_WINNER, "postproc": {"ema_half_life": CHOSEN_HL, "q_in": _QIN, "q_out": CHOSEN_QOUT},
    "policy": _pol_json["policy"], "config_sha256": CONFIG_SHA256, "universe_sha256": UNIVERSE_SHA256,
    "prediction_file_sha256": _pred_hashes, "scored_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2))
TEST_SECONDS = time.time() - _t0
print(f"test scoring done in {TEST_SECONDS / 60:.1f} min")
for _k, _v in SIG["primary_holdout"].items():
    if isinstance(_v, dict):
        print(f"HOLDOUT {_k}: {_v['strategy_level_bootstrap_sharpe']['verdict']}; "
              f"Wilcoxon p = {_v['model_level_wilcoxon_raw_rank_ic']['p_value']:.4f}")
