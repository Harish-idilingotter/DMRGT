# ─── Cell 15: Phase 4B — trigger calibration (train) + policy selection (validation) ───
# Inference only: the winner's 20 checkpoints; nothing is trained. Test is still locked.
import d4_graph, d4_eval

_t0 = time.time()
_P = CONFIG["phase4b"]
_W, _MAXSTALE = _P["corr_window"], _P["max_staleness_days"]
_folds = {f.name: f for f in d4_data.tables()["folds"]}
_fold_names = [f["name"] for f in STATIC["folds"]]
_DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_encoder = CONFIG["arms"][WINNER]["encoder"]
RETS = d4_graph.universe_returns(cfg)
CORR = d4_graph.corr_by_date(RETS, _W)
_sigs = [s for s in _P["signals"] if _encoder == "mamba" or s == "corr_drift"]   # Δ needs a Mamba encoder

TRIGGERS = [{"name": f"{s}@{int(round(r * 100))}", "kind": "trigger", "signal": s, "target_rate": r}
            for s in _sigs for r in _P["target_rebuild_rates"]]
DAILY = {"name": "daily", "kind": "daily"}
FROZEN = {"name": "frozen_at_start", "kind": "frozen"}
_all_pol = TRIGGERS + [DAILY, FROZEN]

INF = {}                                    # (fold, seed) → ModelInference, reused by Cell 16 / 17
THRESH = {}                                 # "fold|seed" → {policy: thresholds}
_frames = {p["name"]: [] for p in _all_pol}
_rates = {p["name"]: [] for p in _all_pol}  # (n_rebuilds, n_days) per run
_corr_cache = {}                            # per-fold corr-threshold memo (state-independent of the model)

for _fn in _fold_names:
    _fold = _folds[_fn]
    for _seed in CONFIG["seeds"]:
        _mi = d4_graph.ModelInference(d4_train.ckpt_paths(WORK, WINNER, _fn, _seed)["ckpt"], _DEV, SNAPSHOT)
        INF[(_fn, _seed)] = _mi
        _dtrain = _mi.delta_by_day(["train"]) if _encoder == "mamba" else None
        _dval = _mi.delta_by_day(["val"]) if _encoder == "mamba" else None
        # calibration sim over the fold's TRAINING days (leakage-free)
        _days = [pd.Timestamp(d) for d in _mi.split_data("train").dates]
        _pos = RETS.index.get_indexer(_days)
        assert (np.diff(_pos) == 1).all(), "training sample dates are not contiguous in the return calendar"
        _steps = [CORR[RETS.index[_pos[0] - 1]]] + [CORR[d] for d in _days]
        THRESH[f"{_fn}|{_seed}"] = {}
        for _p in TRIGGERS:
            _th = d4_graph.calibrate_thresholds(_p["signal"], _p["target_rate"], _days, _steps, _dtrain,
                                                _MAXSTALE, corr_cache=_corr_cache.setdefault(_fn, {}))
            THRESH[f"{_fn}|{_seed}"][_p["name"]] = _th
        # validation: each policy runs continuously from the last training day through validation
        for _p in _all_pol:
            _g, _rb, _st = d4_graph.run_named_policy(
                _p, THRESH[f"{_fn}|{_seed}"].get(_p["name"]), _fold, RETS, _fold.val_dates[-1],
                _dval, cfg, _W, _MAXSTALE)
            _pred = _mi.predict_with_graphs("val", _g)
            _frames[_p["name"]].append(_mi.frame(_p["name"], "val", _pred))
            _rates[_p["name"]].append((_st["n_rebuilds"], _st["n_days"]))
    print(f"fold {_fn} done ({time.time() - _t0:.0f}s)", flush=True)

_targets = TARGETS
_rows = []
for _p in _all_pol:
    _df = pd.concat(_frames[_p["name"]], ignore_index=True)
    _bt = d4_eval.backtest(_df, cfg, split="val", targets=_targets)
    _nr, _nd = np.sum([r[0] for r in _rates[_p["name"]]]), np.sum([r[1] for r in _rates[_p["name"]]])
    _rows.append({"policy": _p["name"], "kind": _p["kind"], "signal": _p.get("signal"),
                  "target_rate": _p.get("target_rate"), "val_rebuild_rate": float(_nr / _nd),
                  "val_sharpe": _bt["sharpe"], "val_mean_rank_ic": _bt["mean_rank_ic"],
                  "val_turnover": _bt["turnover"], "n_runs": _bt["n_runs"]})
VAL4B = pd.DataFrame(_rows)
VAL4B.to_csv(WORK / "phase4b_validation_summary.csv", index=False)
print(VAL4B.to_string(index=False))

# integration check: the incremental `daily` policy must reproduce the arm's own validation Sharpe
_daily_sh = float(VAL4B.loc[VAL4B["policy"] == "daily", "val_sharpe"].iloc[0])
_arm_sh = float(VAL4A.loc[VAL4A["arm"] == WINNER, "val_sharpe"].iloc[0])
assert abs(_daily_sh - _arm_sh) < 0.05, (
    f"daily-policy val Sharpe {_daily_sh:.4f} != winner's Phase 4A val Sharpe {_arm_sh:.4f}")

# selection rule (pre-registered): lowest realised val rebuild rate among the trigger policies whose
# mean val Sharpe >= daily's − margin; ties → higher val rank IC; none qualifies → highest val Sharpe.
_margin = _P["val_noninferiority_margin_sharpe"]
_trig = VAL4B[VAL4B["kind"] == "trigger"]
_ok = _trig[_trig["val_sharpe"] >= _daily_sh - _margin]
if len(_ok):
    _best = _ok.sort_values(["val_rebuild_rate", "val_mean_rank_ic"], ascending=[True, False]).iloc[0]
    _noninf = True
else:
    _best = _trig.sort_values("val_sharpe", ascending=False).iloc[0]
    _noninf = False
SELECTED = next(p for p in TRIGGERS if p["name"] == _best["policy"])
print(f"Phase 4B selected policy: {SELECTED['name']} (val rebuild rate {_best['val_rebuild_rate']:.3f}, "
      f"val Sharpe {_best['val_sharpe']:.3f} vs daily {_daily_sh:.3f}; non-inferiority met: {_noninf})")

(WORK / "phase4b_policy.json").write_text(json.dumps({
    "policy": SELECTED, "winner_arm": WINNER, "max_staleness_days": _MAXSTALE, "corr_window": _W,
    "thresholds": {k: v[SELECTED["name"]] for k, v in THRESH.items()},
    "all_thresholds": THRESH,
    "realised_val_rebuild_rate": float(_best["val_rebuild_rate"]),
    "val_sharpe_selected": float(_best["val_sharpe"]), "val_sharpe_daily": _daily_sh,
    "val_margin": _margin, "validation_noninferiority_met": _noninf,
    "daily_vs_phase4A_val_sharpe_diff": _daily_sh - _arm_sh,
    "validation_table": VAL4B.to_dict(orient="records"),
    "config_sha256": CONFIG_SHA256,
    "selected_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2))
PHASE4B_SECONDS = time.time() - _t0
print(f"wrote phase4b_policy.json — test split is now unlocked ({PHASE4B_SECONDS / 60:.1f} min)")
