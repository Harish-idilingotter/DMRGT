# ─── Cell 13 (S1): graph grid with arm A's setup → phase5_graph_winner.json (validation val_sel only) ───
import d4_eval

TARGETS = pd.read_parquet(CACHE / "targets.parquet")
_t0 = time.time()
launch_stage("S1", s1_jobs(GRID_SEEDS))

_rows = []
for _v, _params in CONFIG["graph_grid"]["variants"].items():
    _frames = [pd.read_parquet(d4_train.ckpt_paths(WORK, grid_tag(_v), f, s)["valpred"])
               for f in FOLD_NAMES for s in GRID_SEEDS[_v]]
    _df = pd.concat(_frames, ignore_index=True)
    _df = _df[_df["val_half"] == "sel"]                    # selection reads val_sel ONLY
    assert set(_df["val_half"].unique()) == {"sel"} and set(_df["split"].unique()) == {"val"}
    _res = d4_eval.evaluate_predictions(_df, TARGETS, cfg, split="val")
    assert len(_res) == len(FOLD_NAMES) * len(GRID_SEEDS[_v])
    _rows.append({"variant": _v, **_params, "n_runs": len(_res), "seeds": len(GRID_SEEDS[_v]),
                  "val_sel_sharpe_raw": float(np.mean([r.fold_metrics.sharpe_after_costs for r in _res])),
                  "val_sel_mean_rank_ic": float(np.mean([r.fold_metrics.mean_rank_ic for r in _res])),
                  "val_sel_dir_acc": float(np.mean([r.fold_metrics.directional_acc for r in _res])),
                  "val_sel_turnover": float(np.mean([r.fold_metrics.turnover for r in _res])),
                  "val_sel_cum_return": float(np.mean([r.fold_metrics.cum_return for r in _res]))})
GRID = pd.DataFrame(_rows).sort_values("val_sel_sharpe_raw", ascending=False).reset_index(drop=True)
GRID.to_csv(WORK / "phase5_graph_grid.csv", index=False)
print(GRID.to_string(index=False))

_g = CONFIG["graph_grid"]
GRAPH_WINNER = GRID.loc[0, "variant"]
_tie = GRID[GRID["val_sel_sharpe_raw"] >= GRID.loc[0, "val_sel_sharpe_raw"] - _g["tie_threshold"]]
if len(_tie) > 1:                                          # within 0.02 of the best → higher val_sel rank IC
    GRAPH_WINNER = _tie.sort_values("val_sel_mean_rank_ic", ascending=False).iloc[0]["variant"]
A_TAG = grid_tag(GRAPH_WINNER)
GRAPH_WINNER_PARAMS = CONFIG["graph_grid"]["variants"][GRAPH_WINNER]
print(f"Graph winner: {GRAPH_WINNER} {GRAPH_WINNER_PARAMS} (tiebreak applied: {len(_tie) > 1})")
update_snapshot(GRAPH_WINNER={"name": GRAPH_WINNER, **GRAPH_WINNER_PARAMS})
(WORK / "phase5_graph_winner.json").write_text(json.dumps({
    "winner": GRAPH_WINNER, "params": GRAPH_WINNER_PARAMS, "tiebreak_applied": bool(len(_tie) > 1),
    "selection_metric": _g["selection_metric"], "tie_threshold": _g["tie_threshold"],
    "grid": GRID.to_dict(orient="records"), "trained_with_arm": "A", "grid_seeds": GRID_SEEDS,
    "config_sha256": CONFIG_SHA256, "selected_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2))
S1_SECONDS = time.time() - _t0
print(f"S1 done in {S1_SECONDS / 3600:.2f} h")
