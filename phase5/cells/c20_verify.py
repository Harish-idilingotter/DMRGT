# ─── Cell 20: output verification + auto-generated PHASE5_RESULTS_DRAFT.md (hard stop on failure) ───
import glob, hashlib

_required = ["config_snapshot.json", "run_manifest.json", "budget_plan.json", "phase5_unit_tests.json",
             "phase5_leakage_gate.json", "phase5_graph_grid.csv", "phase5_graph_winner.json", "phase5_postproc.json",
             "phase5_validation_summary.csv", "phase5_winner.json", "phase5b_validation_summary.csv", "phase5b_policy.json",
             "phase5_test_metrics.csv", "phase5_holdout_metrics.csv", "phase5_significance_results.json",
             "phase5_TEST_SCORED.json", "phase5_efficiency.csv", "phase5_delta_diagnostics.json",
             "phase5b_compute_benchmark.csv"]
_missing = [f for f in _required if not (WORK / f).exists()]
assert not _missing, f"missing required outputs: {_missing}"

_n_runs = len(FOLD_NAMES) * len(CONFIG["seeds"])
for _a in ["P", "B"] + (["C"] if "C" in S2_ARMS else []):
    for _pat in ("ckpt_{}_*.pt", "log_{}_*.csv", "valpred_{}_*.parquet", "done_{}_*.json"):
        _n = len(glob.glob(str(WORK / _pat.format(f"{_a}_fold_[1-4]*"))))
        assert _n == _n_runs, f"arm {_a}: expected {_n_runs} files for {_pat}, found {_n}"
for _pat in ("ckpt_{}_*.pt", "valpred_{}_*.parquet"):
    assert len(glob.glob(str(WORK / _pat.format(f"{A_TAG}_fold_*")))) == _n_runs, f"arm A ({A_TAG}) incomplete"
for _v in CONFIG["graph_grid"]["variants"]:
    assert len(glob.glob(str(WORK / f"ckpt_{grid_tag(_v)}_fold_*.pt"))) == len(FOLD_NAMES) * len(GRID_SEEDS[_v]), f"grid variant {_v} incomplete"
for _a in (WINNER, "A", "P"):
    assert len(glob.glob(str(WORK / f"ckpt_{_a}_fold_[56]*.pt"))) == len(HOLDOUT_NAMES) * len(HOLDOUT_SEEDS), f"holdout arm {_a} incomplete"
for _p in glob.glob(str(WORK / "valpred_*.parquet"))[:20]:
    assert "val_half" in pd.read_parquet(_p, columns=["val_half"]).columns, f"{_p} lacks val_half"
assert len(glob.glob(str(WORK / "testpred_*.parquet"))) > 0
assert len(glob.glob(str(WORK / "dmrgt4" / "*.py"))) >= 7, "dmrgt4/*.py (the code that produced the results) missing"
_scored = json.loads((WORK / "phase5_TEST_SCORED.json").read_text())
for _name, _h in _scored["prediction_file_sha256"].items():
    assert hashlib.sha256((WORK / _name).read_bytes()).hexdigest() == _h, f"{_name} changed after scoring"
print("all required outputs present and consistent")

# ── draft results, every number read from the files ──
def _md(df, fmt="{:.4f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r.values) + " |")
    return "\n".join(out)

_J = lambda n: json.loads((WORK / n).read_text())
_gw, _pp, _win, _pol, _sig = (_J("phase5_graph_winner.json"), _J("phase5_postproc.json"), _J("phase5_winner.json"),
                              _J("phase5b_policy.json"), _J("phase5_significance_results.json"))
_gate, _plan, _real, _dd = _J("phase5_leakage_gate.json"), _J("budget_plan.json"), _J("phase5_real_graph_cost.json"), _J("phase5_delta_diagnostics.json")
_grid, _val5, _val4b = (pd.read_csv(WORK / "phase5_graph_grid.csv"), pd.read_csv(WORK / "phase5_validation_summary.csv"),
                        pd.read_csv(WORK / "phase5b_validation_summary.csv"))
_tm, _hm, _eff, _bench = (pd.read_csv(WORK / "phase5_test_metrics.csv"), pd.read_csv(WORK / "phase5_holdout_metrics.csv"),
                          pd.read_csv(WORK / "phase5_efficiency.csv"), pd.read_csv(WORK / "phase5b_compute_benchmark.csv"))
_comp = pd.read_csv(WORK / "phase5_postproc_components.csv")
_U = pd.read_csv(CACHE / "universe.csv")
_bm = json.loads((CACHE / "build_manifest.json").read_text()) if (CACHE / "build_manifest.json").exists() else {}
_alpha = CONFIG["significance"]["alpha"]
_rates = _sig["phase4b"]["test_rebuild_rate"]

L = ["# DMRGT — Phase 5 Results: S&P 100, Mamba fixes, graph retuning, cost-aware portfolio construction", "",
     "*Auto-generated from the run's JSON/CSV files. Every number is read from a file.*", "",
     "**Scope note.** Phase 4.1 was not run separately on the 40-stock universe, so there is no clean 40-stock answer on the fixed Mamba. "
     "All comparisons here are made *within* the S&P 100 run against in-universe baselines P and A. Phase 4 run 1 remains the result of record for 40 stocks.", "",
     "## 1. Universe", "",
     f"- N = {len(UNIVERSE)} stocks (universe.csv SHA-256 `{UNIVERSE_SHA256}`); excluded: " +
     ", ".join(f"{r.ticker} ({r.exclusion_reason})" for r in _U[_U['included'].astype(str) != 'True'].itertuples()) + ".",
     "- **Survivorship bias:** " + (_bm.get("limitations", ["see build_manifest.json"])[0]), "",
     "## 2. Holdout 2023–2024 (headline; never seen by any earlier phase, scored once)", "", _md(_hm), ""]
for _k, _v in _sig["primary_holdout"].items():
    if isinstance(_v, dict):
        _w, _b = _v["model_level_wilcoxon_raw_rank_ic"], _v["strategy_level_bootstrap_sharpe"]
        L += [f"- **{_k}** — strategy level: {_b['verdict']} (Sharpe diff {_b['observed_sharpe_difference']:+.3f}, "
              f"95% CI [{_b['ci_95'][0]:+.3f}, {_b['ci_95'][1]:+.3f}]); model level (raw rank IC): Wilcoxon p = {_w['p_value']:.4f}, "
              f"wins {_w.get('wins', 'n/a')}/{_w.get('n_pairs', 'n/a')} (low power: {_w.get('n_pairs', 'n/a')} pairs)."]
L += ["", "## 3. Test folds 1–4 (2019–2022)", "", _md(_tm), "", "### 3.1 Secondary comparisons (Holm-corrected together)", ""]
for _k, _v in _sig["secondary_folds_1_4"]["comparisons"].items():
    if "strategy_level_bootstrap_sharpe" in _v:
        L.append(f"- {_k}: model Wilcoxon p = {_v['model_level_wilcoxon_raw_rank_ic']['p_value']:.4f}; strategy — {_v['strategy_level_bootstrap_sharpe']['verdict']}.")
    else:
        L.append(f"- {_k}: {_v['verdict']} (Sharpe diff {_v['observed_sharpe_difference']:+.3f}).")
L += ["- Holm-adjusted: " + "; ".join(f"{n}: p_adj = {v['p_adj']:.4f} (reject: {v['reject']})" for n, v in _sig["secondary_folds_1_4"]["holm"].items()), "",
      "## 4. Graph retuning (S1, validation val_sel, arm A's setup)", "", _md(_grid), "",
      f"**Winner: `{_gw['winner']}`** {_gw['params']} (tiebreak applied: {_gw['tiebreak_applied']}).", "",
      "## 5. Cost-aware portfolio construction (S3)", "",
      f"Chosen on validation (val_sel): EMA half-life {_pp['chosen']['ema_half_life_days']} days, q_in {_pp['chosen']['q_in']}, q_out {_pp['chosen']['q_out']}; "
      f"seed ensemble = {_pp['chosen']['ensemble']}.", "", "### 5.1 Validation (val_sel) summary — raw vs strategy", "", _md(_val5), "",
      "### 5.2 Contribution of each component (test folds 1–4)", "", _md(_comp), "",
      f"**Phase 4A winner (B vs C): {_win['winner_arm']}** (tiebreak applied: {_win['tiebreak_applied']}).", "",
      "## 6. Phase 4B — regime-triggered graph refresh", "",
      f"- Selected policy `{_pol['policy']['name']}`: val_sel rebuild rate {_pol['realised_val_sel_rebuild_rate']:.3f}, strategy Sharpe "
      f"{_pol['val_sel_strategy_sharpe_selected']:.3f} vs daily {_pol['val_sel_strategy_sharpe_daily']:.3f} (margin {_pol['val_margin']}; met: {_pol['validation_noninferiority_met']}).",
      "- Test rebuild rates: " + ", ".join(f"{k} = {v:.3f}" for k, v in _rates.items()) + "."]
for _lab in ("folds_1_4", "holdout_5_6"):
    _nd = _sig["phase4b"][f"{_lab}: selected vs daily"]
    _fz = _sig["phase4b"][f"{_lab}: selected vs frozen_at_start"]
    L += [f"- {_lab}: non-inferiority vs daily — CI lower bound {_nd['noninferiority']['ci95_lower_bound']:+.3f} vs −{_nd['noninferiority']['margin']} → "
          f"non-inferior: {_nd['noninferiority']['non_inferior']}; vs frozen_at_start — {_fz['strategy_level_bootstrap_sharpe']['verdict']}."]
L += ["", "### 6.1 Validation table (val_sel, strategy returns)", "", _md(_val4b), "",
      "## 7. Efficiency and technical effect", "", "### 7.1 Model efficiency", "", _md(_eff), "",
      "### 7.2 Δ diagnostics (reported, not a gate)", ""]
for _k, _v in _dd.items():
    if isinstance(_v, dict):
        _m = _v["effective_memory_days_median_channel"]
        L.append(f"- {_v['label']}: mean Δ {_v['mean_delta']:.4f}, CV {_v['cv_daily_mean_delta']:.3f}, Spearman with VIX {_v['spearman_with_vix']:+.3f}, "
                 f"2020 Mar–Apr / 2019 ratio {_v['ratio_mean_delta_covid_2020_03_04_vs_2019']}, effective memory (median channel) "
                 f"calm {_m['calm_days_vix_bottom_quartile']:.1f} d vs volatile {_m['volatile_days_vix_top_quartile']:.1f} d"
                 + ("; **Δ does not vary with regime**" if _v["flat_delta"] else "") + ".")
_max_share = float(_bench["graph_share_incremental_daily"].max())
L += ["", "### 7.3 Graph-construction cost", "",
      f"- Real data (N = {_real['n_stocks']}, window {_real['window']}): full recompute {_real['full_recompute_ms_median']:.3f} ms vs incremental "
      f"{_real['incremental_update_ms_median']:.3f} ms per update; edge build {_real['edge_build_ms_median']:.3f} ms.", "", _md(_bench[
          ["n_stocks", "window", "ms_full_recompute", "ms_incremental_update", "ms_edge_build", "ms_model_inference",
           "graph_share_full_daily", "graph_share_incremental_daily", "graph_share_incremental_trigger"]]), "",
      ("**Graph construction stays below 5% of per-day pipeline time at every benchmarked size; no technical-effect claim is made from this data.**"
       if _max_share < 0.05 else
       f"Graph construction reaches {_max_share:.1%} of per-day pipeline time at the largest benchmarked size under daily incremental rebuilds; see the share columns for where the trigger rate helps."), "",
      "## 8. Reproducibility record", "",
      f"- Pre-registered config SHA-256: `{CONFIG_SHA256}`; changes from Phase 4 listed in `config_snapshot.json`.",
      f"- Leakage gate passed: {_gate['passed']} (random-weights {_gate['random_weights']['abs_ic_in_se']:.2f} SE, shuffled-target {_gate['shuffled_targets']['abs_ic_in_se']:.2f} SE).",
      f"- Budget: batch size {_plan['batch_size']}; drops applied: {_plan['drops_applied'] or 'none'}; final projection {_plan['final_projection']['projected_total_h']:.2f} h.",
      "- Locks written in order: `phase5_graph_winner.json`, `phase5_postproc.json`, `phase5_winner.json`, `phase5b_policy.json`; test scored once (`phase5_TEST_SCORED.json`).",
      "- Code that produced the results: `dmrgt4/*.py` in this run's output.", ""]
(WORK / "PHASE5_RESULTS_DRAFT.md").write_text("\n".join(L))
print("wrote PHASE5_RESULTS_DRAFT.md")
print(f"total run time {(time.time() - RUN_T0) / 3600:.2f} h")
