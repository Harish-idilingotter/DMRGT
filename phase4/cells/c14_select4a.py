# ─── Cell 14: Phase 4A selection — validation only ───
import hashlib, d4_eval

TARGETS = pd.read_parquet(CACHE / "targets.parquet")
_fold_names = [f["name"] for f in STATIC["folds"]]
_seeds = CONFIG["seeds"]

def load_arm_preds(arm):
    frames = [pd.read_parquet(d4_train.ckpt_paths(WORK, arm, f, s)["valpred"])
              for f in _fold_names for s in _seeds]
    return pd.concat(frames, ignore_index=True)

_rows, _val_days = [], None
for _arm in PLAN["arms_run"]:
    if not COMPLETE[_arm]:
        continue
    _df = load_arm_preds(_arm)
    assert set(_df["split"].unique()) == {"val"}, "valpred files must hold validation rows only"
    # row count: SUM of per-fold day counts (not a product) × seeds × stocks
    _n_days = {f: _df[_df["fold"] == f]["date"].nunique() for f in _fold_names}
    assert len(_df) == sum(_n_days.values()) * len(_seeds) * len(UNIVERSE), f"{_arm}: valpred row count mismatch"
    _res = d4_eval.evaluate_predictions(_df, TARGETS, cfg, split="val")
    assert len(_res) == len(_fold_names) * len(_seeds)
    _rows.append({"arm": _arm, "encoder": CONFIG["arms"][_arm]["encoder"], "lookback": CONFIG["arms"][_arm]["lookback"],
                  "adaptive_huber": CONFIG["arms"][_arm]["adaptive_huber"], "n_runs": len(_res),
                  "val_sharpe": float(np.mean([r.fold_metrics.sharpe_after_costs for r in _res])),
                  "val_mean_rank_ic": float(np.mean([r.fold_metrics.mean_rank_ic for r in _res])),
                  "val_dir_acc": float(np.mean([r.fold_metrics.directional_acc for r in _res])),
                  "val_turnover": float(np.mean([r.fold_metrics.turnover for r in _res]))})
VAL4A = pd.DataFrame(_rows)
VAL4A.to_csv(WORK / "phase4_validation_summary.csv", index=False)
print(VAL4A.to_string(index=False))

_sel = CONFIG["selection_4A"]
_cand = VAL4A[VAL4A["arm"].isin(_sel["candidates"])].sort_values("val_sharpe", ascending=False).reset_index(drop=True)
assert len(_cand) >= 2, "need at least B and C complete to select a winner"
WINNER = _cand.loc[0, "arm"]
_tie = False
if abs(_cand.loc[0, "val_sharpe"] - _cand.loc[1, "val_sharpe"]) < _sel["tie_threshold"]:
    _tie = True
    if _cand.loc[1, "val_mean_rank_ic"] > _cand.loc[0, "val_mean_rank_ic"]:
        WINNER = _cand.loc[1, "arm"]
print(f"Phase 4A winner: {WINNER} (tiebreak applied: {_tie})")

_ckpt_hashes = {f"{f}|{s}": hashlib.sha256(d4_train.ckpt_paths(WORK, WINNER, f, s)["ckpt"].read_bytes()).hexdigest()
                for f in _fold_names for s in _seeds}
(WORK / "phase4_winner.json").write_text(json.dumps({
    "winner_arm": WINNER, "arm_config": CONFIG["arms"][WINNER], "tiebreak_applied": _tie,
    "selection_rule": _sel, "validation_summary": VAL4A.to_dict(orient="records"),
    "checkpoint_sha256": _ckpt_hashes, "config_sha256": CONFIG_SHA256,
    "selected_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2))
print("wrote phase4_winner.json (test still locked until phase4b_policy.json exists)")
