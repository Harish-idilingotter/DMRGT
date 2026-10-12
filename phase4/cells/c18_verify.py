# ─── Cell 18: output verification + auto-generated PHASE4_RESULTS_DRAFT.md (hard stop on failure) ───
import glob, hashlib

_fold_names = [f["name"] for f in STATIC["folds"]]
_seeds = CONFIG["seeds"]
_n_runs = len(_fold_names) * len(_seeds)
_required = ["config_snapshot.json", "run_manifest.json", "budget_plan.json", "phase4_leakage_gate.json",
             "phase4_validation_summary.csv", "phase4_winner.json",
             "phase4b_validation_summary.csv", "phase4b_policy.json",
             "phase4_test_metrics.csv", "phase4_significance_results.json", "phase4_TEST_SCORED.json",
             "phase4_efficiency.csv", "phase4b_compute_benchmark.csv", "phase4_delta_by_date.csv"]
_missing = [f for f in _required if not (WORK / f).exists()]
assert not _missing, f"missing required outputs: {_missing}"
_done_arms = [a for a in PLAN["arms_run"] if COMPLETE[a]]
for _a in _done_arms:
    assert len(glob.glob(str(WORK / f"ckpt_{_a}_*.pt"))) == _n_runs, f"arm {_a}: expected {_n_runs} checkpoints"
    assert len(glob.glob(str(WORK / f"log_{_a}_*.csv"))) == _n_runs, f"arm {_a}: expected {_n_runs} logs"
    assert len(glob.glob(str(WORK / f"valpred_{_a}_*.parquet"))) == _n_runs
    assert len(glob.glob(str(WORK / f"testpred_{_a}_fold_*.parquet"))) == _n_runs
assert len(glob.glob(str(WORK / "testpred_*.parquet"))) >= len(_done_arms) * _n_runs + 3 * _n_runs
assert len(glob.glob(str(WORK / "dmrgt4" / "*.py"))) >= 7, "dmrgt4/*.py (the code that produced the results) missing"
# the scored predictions on disk must still hash to what phase4_TEST_SCORED.json recorded
_scored = json.loads((WORK / "phase4_TEST_SCORED.json").read_text())
for _name, _h in _scored["prediction_file_sha256"].items():
    assert hashlib.sha256((WORK / _name).read_bytes()).hexdigest() == _h, f"{_name} changed after scoring"
print("all required outputs present and consistent")

# ── draft results, every number read from the files ──
def _md(df, floatfmt="{:.4f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                                      for v in r.values) + " |")
    return "\n".join(out)

_J = lambda n: json.loads((WORK / n).read_text())
_winner, _pol, _sig = _J("phase4_winner.json"), _J("phase4b_policy.json"), _J("phase4_significance_results.json")
_gate, _plan, _real = _J("phase4_leakage_gate.json"), _J("budget_plan.json"), _J("phase4_real_graph_cost.json")
_val4a, _val4b = pd.read_csv(WORK / "phase4_validation_summary.csv"), pd.read_csv(WORK / "phase4b_validation_summary.csv")
_tm, _eff, _bench = pd.read_csv(WORK / "phase4_test_metrics.csv"), pd.read_csv(WORK / "phase4_efficiency.csv"), \
    pd.read_csv(WORK / "phase4b_compute_benchmark.csv")
_alpha = CONFIG["significance"]["alpha"]
_p = _sig["phase4a"]["primary"]
_w, _b = _p["wilcoxon"], _p["bootstrap_sharpe"]
_nd, _fz = _sig["phase4b"]["selected_vs_daily"], _sig["phase4b"]["selected_vs_frozen_at_start"]
_rates = _sig["phase4b"]["test_rebuild_rate"]
_selN = f"{_winner['winner_arm']}_selected"

_4a_concl = ("is significantly better than" if (_w["p_value"] < _alpha and _b["ci_95"][0] > 0) else
             "is not shown to be significantly better than")
_4b_concl = ("non-inferior to daily rebuilding" if _nd["noninferiority"]["non_inferior"]
             else "NOT shown to be non-inferior to daily rebuilding")
L = []
L += ["# DMRGT — Phase 4 Results: Mamba encoder, adaptive training, regime-triggered graph refresh", "",
      "*Auto-generated from the run's JSON/CSV files. Every number below is read from a file, none typed by hand.*", "",
      "## 1. Objective", "",
      "Phase 4A: does a selective state-space (Mamba) encoder with adaptive training beat the Phase 3 Transformer "
      "encoder inside the same notebook, schedule and graph? Phase 4B: can the daily graph rebuild be replaced by "
      "an inference-time refresh triggered by a regime signal (correlation drift and/or the Mamba memory-reset "
      "signal Δ) with incremental correlation updates, without losing risk-adjusted performance and at lower "
      "graph-construction cost?", "",
      "## 2. Methodology Summary", "",
      f"- 4 walk-forward folds × {len(_seeds)} seeds = {_n_runs} paired runs per arm; identical folds to Phase 3.",
      f"- Arms: " + "; ".join(f"{a} = {CONFIG['arms'][a]['encoder']}, lookback {CONFIG['arms'][a]['lookback']}"
                              f"{', adaptive Huber' if CONFIG['arms'][a]['adaptive_huber'] else ''}" for a in _done_arms) + ".",
      f"- Batch size {_plan['batch_size']} for all arms; D skipped: {_plan.get('D_skipped_reason', 'no')}.",
      "- Selection (4A and 4B) on validation only; test split locked until both lock files existed; scored once.",
      f"- Config snapshot SHA-256: `{CONFIG_SHA256}`.",
      f"- Leakage gate: random-weights |IC| = {_gate['random_weights']['abs_ic_in_se']:.2f} SE, "
      f"shuffled-target |IC| = {_gate['shuffled_targets']['abs_ic_in_se']:.2f} SE (limit {_gate['max_abs_ic_in_standard_errors']} SE); "
      f"passed: {_gate['passed']}.", "",
      "## 3. Phase 4A — Validation Summary (selection)", "", _md(_val4a), "",
      f"**Winner: arm {_winner['winner_arm']}** (tiebreak applied: {_winner['tiebreak_applied']}).", "",
      "## 4. Test Results (mean over the 20 runs per model)", "", _md(_tm), "",
      "## 5. Significance Tests", "",
      f"### 5.1 Phase 4A primary — {_p['comparison']}", "",
      f"- Wilcoxon (paired rank IC): wins {_w['wins']}/{_w['n_pairs']}, p = {_w['p_value']:.4f}, "
      f"mean paired difference {_w['mean_paired_difference']:+.5f}.",
      f"- Stationary bootstrap (Sharpe difference): observed {_b['observed_sharpe_difference']:+.3f}, "
      f"95% CI [{_b['ci_95'][0]:+.3f}, {_b['ci_95'][1]:+.3f}], one-sided p = {_b['bootstrap_p_value']:.4f}.", ""]
for _k, _v in _sig["phase4a"].get("secondary", {}).items():
    L.append(f"- Secondary {_k}: Wilcoxon p = {_v['wilcoxon']['p_value']:.4f}; bootstrap CI "
             f"[{_v['bootstrap_sharpe']['ci_95'][0]:+.3f}, {_v['bootstrap_sharpe']['ci_95'][1]:+.3f}].")
for _k in ("secondary_holm_wilcoxon", "secondary_holm_bootstrap_two_sided"):
    if _k in _sig["phase4a"]:
        L.append(f"- Holm-adjusted ({_k}): " + "; ".join(f"{n}: p_adj = {v['p_adj']:.4f} (reject: {v['reject']})"
                                                         for n, v in _sig["phase4a"][_k].items()))
L += ["", f"### 5.2 Phase 4B — selected policy `{_pol['policy']['name']}`", "",
      f"- Validation: realised rebuild rate {_pol['realised_val_rebuild_rate']:.3f}, Sharpe "
      f"{_pol['val_sharpe_selected']:.3f} vs daily {_pol['val_sharpe_daily']:.3f} "
      f"(non-inferiority margin {_pol['val_margin']}; met: {_pol['validation_noninferiority_met']}).",
      f"- Test rebuild rates: " + ", ".join(f"{k} = {v:.3f}" for k, v in _rates.items()) + ".",
      f"- Non-inferiority vs daily: bootstrap 95% CI lower bound {_nd['noninferiority']['ci95_lower_bound']:+.3f} "
      f"vs −{_nd['noninferiority']['margin']} → non-inferior: {_nd['noninferiority']['non_inferior']}; "
      f"Wilcoxon on rank IC p = {_nd['wilcoxon']['p_value']:.4f}.",
      f"- Superiority vs frozen_at_start: Sharpe difference {_fz['bootstrap_sharpe']['observed_sharpe_difference']:+.3f}, "
      f"one-sided p = {_fz['bootstrap_sharpe']['bootstrap_p_value']:.4f}.", "",
      "### 5.3 Phase 4B validation table", "", _md(_val4b), "",
      "## 6. Efficiency and Technical Effect", "", "### 6.1 Model efficiency", "", _md(_eff), "",
      "### 6.2 Graph-construction cost (real 40-stock data, CPU, float64)", "",
      f"- Per update: full recompute {_real['full_recompute_ms_median']:.3f} ms vs incremental "
      f"{_real['incremental_update_ms_median']:.3f} ms; edge build {_real['edge_build_ms_median']:.3f} ms.",
      f"- Test-period total (fold 4): full-recompute daily {_real['test_period_total_seconds_full_recompute_daily']:.3f} s, "
      f"incremental daily {_real['test_period_total_seconds_incremental_daily']:.3f} s, "
      f"incremental + selected policy {_real['test_period_total_seconds_incremental_selected']:.3f} s.", "",
      "### 6.3 Synthetic scaling benchmark", "", _md(_bench), "",
      "## 7. Conclusion (rule-derived)", "",
      f"- Phase 4A: winner {_winner['winner_arm']} {_4a_concl} the Transformer arm A (alpha = {_alpha}).",
      f"- Phase 4B: selected policy `{_pol['policy']['name']}` is {_4b_concl} at margin "
      f"{_nd['noninferiority']['margin']} Sharpe, rebuilding on {_rates[_selN]:.1%} of test days.", "",
      "## 8. Notes on Interpretation", "",
      "- The model is trained on daily graphs; only the inference-time graph refresh changes in 4B (no per-policy retraining).",
      "- The backtest, cost model and Sharpe construction are the Phase 3 harness, unchanged.",
      "- Comparison with the Phase 3 Transformer is done locally after the run (spec §16).", "",
      "## 9. Reproducibility Record", "",
      f"- Run manifest: `run_manifest.json`; budget plan: `budget_plan.json` "
      f"(projected total {_plan['projection']['with_D' if _plan['run_D'] else 'without_D']['projected_total_h']:.2f} h).",
      f"- Winner checkpoint hashes: `phase4_winner.json`; test prediction hashes: `phase4_TEST_SCORED.json`.",
      f"- Code that produced the results: `dmrgt4/*.py` in this run's output.", ""]
(WORK / "PHASE4_RESULTS_DRAFT.md").write_text("\n".join(L))
print("wrote PHASE4_RESULTS_DRAFT.md")
print(f"total run time {(time.time() - RUN_T0) / 3600:.2f} h")
