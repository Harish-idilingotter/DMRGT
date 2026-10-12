# ─── Cell 16 (4B): trigger calibration (training period) + policy selection (val_sel strategy returns) ───
# Inference only: the winner's 20 checkpoints; nothing is trained. Test is still locked until the file is written.
import d4_graph, d4_eval

_t0 = time.time()
_P = CONFIG["phase4b"]
cfg_w = d4_data.load_cfg()                                  # graph parameters of the S1 winner
_W = int(cfg_w["graph"]["corr_window"])                     # the policy rebuilds the graph the models were trained on
_MAXSTALE = _P["max_staleness_days"]
_folds = {f.name: f for f in d4_data.tables()["folds"]}
_DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RETS = d4_graph.universe_returns(cfg_w)
CORR = d4_graph.corr_by_date(RETS, _W)
_sigs = [s for s in _P["signals"] if WENC == "mamba" or s == "corr_drift"]    # Δ needs a Mamba encoder

TRIGGERS = [{"name": f"{s}@{int(round(r * 100))}", "kind": "trigger", "signal": s, "target_rate": r}
            for s in _sigs for r in _P["target_rebuild_rates"]]
DAILY = {"name": "daily", "kind": "daily"}
FROZEN = {"name": "frozen_at_start", "kind": "frozen"}
ALL_POL = TRIGGERS + [DAILY, FROZEN]
SNAPSHOT = json.loads((WORK / "config_snapshot.json").read_text())
CORR_CACHE = {}                                             # per-fold corr-threshold memo (model-independent)

def calibrate_policy(mi, fold, policy):
    """Thresholds for one trigger policy, calibrated on the fold's TRAINING days only."""
    days = [pd.Timestamp(d) for d in mi.split_data("train").dates]
    pos = RETS.index.get_indexer(days)
    assert (np.diff(pos) == 1).all(), "training sample dates are not contiguous in the return calendar"
    steps = [CORR[RETS.index[pos[0] - 1]]] + [CORR[d] for d in days]
    dtrain = mi.delta_by_day(["train"]) if WENC == "mamba" else None
    return d4_graph.calibrate_thresholds(policy["signal"], policy["target_rate"], days, steps, dtrain, _MAXSTALE,
                                         corr_cache=CORR_CACHE.setdefault((mi.fold, mi.graph), {}))

INF = {}                                                    # (fold, seed) → ModelInference, reused later
THRESH = {}                                                 # "fold|seed" → {policy: thresholds}
_frames = {p["name"]: [] for p in ALL_POL}
_rate = {p["name"]: [0, 0] for p in ALL_POL}                # rebuilds / days on val_sel only
for _fn in FOLD_NAMES:
    _fold = _folds[_fn]
    _sel_set = set(pd.Timestamp(d) for d in d4_graph.get_split_g(_fn, "val_sel", CONFIG["arms"][WINNER]["lookback"], GRAPH_WINNER).dates)
    for _seed in CONFIG["seeds"]:
        _mi = d4_graph.ModelInference(d4_train.ckpt_paths(WORK, WINNER, _fn, _seed)["ckpt"], _DEV, SNAPSHOT)
        INF[(_fn, _seed)] = _mi
        _dval = _mi.delta_by_day(["val"]) if WENC == "mamba" else None
        THRESH[f"{_fn}|{_seed}"] = {p["name"]: calibrate_policy(_mi, _fold, p) for p in TRIGGERS}
        for _p in ALL_POL:                                  # continuous from the last training day through both validation halves
            _g, _rb, _st = d4_graph.run_named_policy(_p, THRESH[f"{_fn}|{_seed}"].get(_p["name"]), _fold, RETS,
                                                     _fold.val_dates[-1], _dval, cfg_w, _W, _MAXSTALE)
            _frames[_p["name"]].append(_mi.frame(_p["name"], "val", _mi.predict_with_graphs("val", _g)))
            _rate[_p["name"]][0] += sum(1 for d in _rb if d in _sel_set)
            _rate[_p["name"]][1] += len(_sel_set)
    print(f"fold {_fn} done ({time.time() - _t0:.0f}s)", flush=True)

_rows = []
for _p in ALL_POL:
    _df = pd.concat(_frames[_p["name"]], ignore_index=True)
    rets, turns = [], []
    for _fn in FOLD_NAMES:
        _d = _df[_df["fold"] == _fn]
        ret, turn = d4_eval.strategy_series(_d, RAW_WIDE, UNIVERSE, CONFIG["backtest"], CHOSEN_HL, CONFIG["postproc"]["q_in"], CHOSEN_QOUT)
        sel = pd.DatetimeIndex(sorted(_d.loc[_d["val_half"] == "sel", "date"].unique()))
        rets.append(ret.reindex(sel)); turns.append(turn.reindex(sel))
    _s = d4_eval.pooled_stats(pd.concat(rets), pd.concat(turns))
    _raw = d4_eval.evaluate_predictions(_df[_df["val_half"] == "sel"], TARGETS, cfg, split="val")
    _rows.append({"policy": _p["name"], "kind": _p["kind"], "signal": _p.get("signal"), "target_rate": _p.get("target_rate"),
                  "val_sel_rebuild_rate": _rate[_p["name"]][0] / _rate[_p["name"]][1],
                  "val_sel_strategy_sharpe": _s["sharpe"], "val_sel_strategy_turnover": _s["turnover"],
                  "val_sel_raw_mean_rank_ic": float(np.mean([r.fold_metrics.mean_rank_ic for r in _raw])),
                  "n_runs": len(_raw)})
VAL4B = pd.DataFrame(_rows)
VAL4B.to_csv(WORK / "phase5b_validation_summary.csv", index=False)
print(VAL4B.to_string(index=False))

# integration check: the incremental `daily` policy must reproduce the winner arm's own val_sel strategy Sharpe
_daily_sh = float(VAL4B.loc[VAL4B["policy"] == "daily", "val_sel_strategy_sharpe"].iloc[0])
_arm_sh = float(strategy_val_sel(WINNER, CHOSEN_HL, CHOSEN_QOUT)["sharpe"])
assert abs(_daily_sh - _arm_sh) < 0.05, f"daily-policy val_sel strategy Sharpe {_daily_sh:.4f} != winner's {_arm_sh:.4f}"

# selection rule (pre-registered): lowest realised val_sel rebuild rate among trigger policies whose val_sel
# STRATEGY Sharpe >= daily's − margin; ties → higher val_sel rank IC; none qualifies → highest Sharpe (flagged)
_margin = _P["val_noninferiority_margin_sharpe"]
_trig = VAL4B[VAL4B["kind"] == "trigger"]
_ok = _trig[_trig["val_sel_strategy_sharpe"] >= _daily_sh - _margin]
if len(_ok):
    _best = _ok.sort_values(["val_sel_rebuild_rate", "val_sel_raw_mean_rank_ic"], ascending=[True, False]).iloc[0]
    _noninf = True
else:
    _best = _trig.sort_values("val_sel_strategy_sharpe", ascending=False).iloc[0]
    _noninf = False
SELECTED = next(p for p in TRIGGERS if p["name"] == _best["policy"])
print(f"Phase 4B selected policy: {SELECTED['name']} (val_sel rebuild rate {_best['val_sel_rebuild_rate']:.3f}, "
      f"strategy Sharpe {_best['val_sel_strategy_sharpe']:.3f} vs daily {_daily_sh:.3f}; non-inferiority met: {_noninf})")

(WORK / "phase5b_policy.json").write_text(json.dumps({
    "policy": SELECTED, "winner_arm": WINNER, "graph": GRAPH_WINNER, "max_staleness_days": _MAXSTALE, "corr_window": _W,
    "thresholds": {k: v[SELECTED["name"]] for k, v in THRESH.items()}, "all_thresholds": THRESH,
    "realised_val_sel_rebuild_rate": float(_best["val_sel_rebuild_rate"]),
    "val_sel_strategy_sharpe_selected": float(_best["val_sel_strategy_sharpe"]), "val_sel_strategy_sharpe_daily": _daily_sh,
    "val_margin": _margin, "validation_noninferiority_met": _noninf,
    "daily_vs_winner_val_sel_strategy_sharpe_diff": _daily_sh - _arm_sh,
    "validation_table": VAL4B.to_dict(orient="records"), "config_sha256": CONFIG_SHA256,
    "selected_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2))
PHASE4B_SECONDS = time.time() - _t0
assert d4_data.test_unlocked(), "all four lock files should exist now"
print(f"wrote phase5b_policy.json — all four locks exist; test split unlocked ({PHASE4B_SECONDS / 60:.1f} min)")
