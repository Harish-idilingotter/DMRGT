# ─── Cell 15 (S3): post-processing selection + winner arm — validation val_sel only ───
import hashlib, d4_eval

_t0 = time.time()
_PP = CONFIG["postproc"]
RAW_WIDE = d4_eval.raw_returns_from_targets(TARGETS, UNIVERSE)
ARMS4 = ["P", "A"] + S2_ARMS[1:]                                   # P, A, B (, C)

def val_frame(arm, fold):
    """Validation predictions (es + sel halves, all seeds) of an arm on a fold."""
    return pd.concat([pd.read_parquet(d4_train.ckpt_paths(WORK, tag_for(arm, fold), fold, s)["valpred"])
                      for s in seeds_for(fold)], ignore_index=True).assign(model=arm)

VAL = {(a, f): val_frame(a, f) for a in ARMS4 for f in FOLD_NAMES}

def strategy_val_sel(arm, hl, q_out):
    """Pooled (folds 1-4, date order) val_sel strategy stats of an arm under one post-processing config."""
    rets, turns = [], []
    for f in FOLD_NAMES:
        df = VAL[(arm, f)]
        ret, turn = d4_eval.strategy_series(df, RAW_WIDE, UNIVERSE, CONFIG["backtest"], hl, _PP["q_in"], q_out)
        sel = pd.DatetimeIndex(sorted(df.loc[df["val_half"] == "sel", "date"].unique()))
        rets.append(ret.reindex(sel)); turns.append(turn.reindex(sel))
    return d4_eval.pooled_stats(pd.concat(rets), pd.concat(turns))

_rows = []
for _a in ARMS4:
    for _hl in _PP["ema_half_life_days"]:
        for _q in _PP["q_out"]:
            _s = strategy_val_sel(_a, _hl, _q)
            _rows.append({"arm": _a, "ema_half_life": _hl, "q_out": _q, "val_sel_strategy_sharpe": _s["sharpe"],
                          "val_sel_turnover": _s["turnover"], "val_sel_max_drawdown": _s["max_drawdown"]})
PP_TABLE = pd.DataFrame(_rows)
_cfg_tab = (PP_TABLE.groupby(["ema_half_life", "q_out"])[["val_sel_strategy_sharpe", "val_sel_turnover"]].mean()
            .reset_index().sort_values("val_sel_strategy_sharpe", ascending=False).reset_index(drop=True))
print(_cfg_tab.to_string(index=False))
_best = _cfg_tab.loc[0, "val_sel_strategy_sharpe"]
_near = _cfg_tab[_cfg_tab["val_sel_strategy_sharpe"] >= _best - _PP["tie_threshold"]]
_pick = _near.sort_values("val_sel_turnover").iloc[0] if len(_near) > 1 else _cfg_tab.iloc[0]
CHOSEN_HL, CHOSEN_QOUT = int(_pick["ema_half_life"]), float(_pick["q_out"])
print(f"Chosen post-processing: EMA half-life {CHOSEN_HL}, q_out {CHOSEN_QOUT} (q_in {_PP['q_in']}); "
      f"tie rule applied: {len(_near) > 1}")

# raw (Phase 3 construction, per seed) and strategy numbers for every arm, val_sel only
_sum = []
for _a in ARMS4:
    _df = pd.concat([VAL[(_a, f)] for f in FOLD_NAMES])
    _res = d4_eval.evaluate_predictions(_df[_df["val_half"] == "sel"], TARGETS, cfg, split="val")
    _st = strategy_val_sel(_a, CHOSEN_HL, CHOSEN_QOUT)
    _sum.append({"arm": _a, "encoder": CONFIG["arms"][_a]["encoder"], "lookback": CONFIG["arms"][_a]["lookback"],
                 "schedule": CONFIG["arms"][_a]["schedule"], "n_runs": len(_res),
                 "val_sel_raw_sharpe": float(np.mean([r.fold_metrics.sharpe_after_costs for r in _res])),
                 "val_sel_raw_mean_rank_ic": float(np.mean([r.fold_metrics.mean_rank_ic for r in _res])),
                 "val_sel_raw_turnover": float(np.mean([r.fold_metrics.turnover for r in _res])),
                 "val_sel_strategy_sharpe": _st["sharpe"], "val_sel_strategy_turnover": _st["turnover"]})
VAL5 = pd.DataFrame(_sum)
VAL5.to_csv(WORK / "phase5_validation_summary.csv", index=False)
print(VAL5.to_string(index=False))

(WORK / "phase5_postproc.json").write_text(json.dumps({
    "chosen": {"ema_half_life_days": CHOSEN_HL, "q_in": _PP["q_in"], "q_out": CHOSEN_QOUT,
               "ensemble": _PP["ensemble"]},
    "tie_rule_applied": bool(len(_near) > 1), "selection": {"metric": _PP["selection_metric"], "tiebreak": _PP["tiebreak"],
                                                               "tie_threshold": _PP["tie_threshold"]},
    "config_means": _cfg_tab.to_dict(orient="records"), "table_9x4": PP_TABLE.to_dict(orient="records"),
    "config_sha256": CONFIG_SHA256, "selected_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2))

# winner arm: Mamba B vs C under the chosen configuration (val_sel strategy Sharpe)
_sel = CONFIG["selection_4A"]
_cand = VAL5[VAL5["arm"].isin(SELECTION_CANDIDATES)].sort_values("val_sel_strategy_sharpe", ascending=False).reset_index(drop=True)
WINNER, _wtie = _cand.loc[0, "arm"], False
if len(_cand) > 1 and abs(_cand.loc[0, "val_sel_strategy_sharpe"] - _cand.loc[1, "val_sel_strategy_sharpe"]) < _sel["tie_threshold"]:
    _wtie = True
    if _cand.loc[1, "val_sel_raw_mean_rank_ic"] > _cand.loc[0, "val_sel_raw_mean_rank_ic"]:
        WINNER = _cand.loc[1, "arm"]
WENC = CONFIG["arms"][WINNER]["encoder"]
print(f"Phase 5 winner: {WINNER} (tiebreak applied: {_wtie}; candidates {SELECTION_CANDIDATES})")
_ckh = {f"{f}|{s}": hashlib.sha256(d4_train.ckpt_paths(WORK, WINNER, f, s)["ckpt"].read_bytes()).hexdigest()
        for f in FOLD_NAMES for s in CONFIG["seeds"]}
(WORK / "phase5_winner.json").write_text(json.dumps({
    "winner_arm": WINNER, "arm_config": CONFIG["arms"][WINNER], "tiebreak_applied": _wtie, "selection_rule": _sel,
    "candidates": SELECTION_CANDIDATES, "validation_summary": VAL5.to_dict(orient="records"),
    "checkpoint_sha256": _ckh, "config_sha256": CONFIG_SHA256,
    "selected_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2))
S3_SECONDS = time.time() - _t0
print(f"wrote phase5_postproc.json and phase5_winner.json (test still locked until phase5b_policy.json) — {S3_SECONDS / 60:.1f} min")
