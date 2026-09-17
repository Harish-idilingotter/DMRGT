"""DMRGT Phase 3 — one-shot final test scoring + comparison report
(§8 of the single-session rework).

Everything that needed the GPU happened in ONE committed Kaggle session:
the 140-run grid, the val-only winner selection, the in-session frozen
graph, the 20-run dmrgt_frozen training, and the significance tests.
THIS script is the local epilogue that runs on the downloaded outputs:

  1. dynamic arm = the WINNING grid variant's already-existing test rows
     (predictions_dmrgt_all3_grid_{winner}.parquet) — dmrgt_all3 is
     never retrained (§1); these rows are read here for the first time,
     strictly after selection was locked in-session.
  2. frozen arm = predictions_dmrgt_frozen.parquet (test rows).
  3. both scored through the SAME evaluation pipeline with the
     DECLARED-IN-ADVANCE identity post-processing (Rule 19). The
     post-processing tuning stage was dropped from the single-session
     design (§0): the only configuration search is the graph grid, and
     the post-processing is the config default, fixed before any test
     row was read.
  4. p-values recomputed locally (stats.run_tests_from_predictions) and
     CROSS-CHECKED against the in-session
     phase3_significance_results.json — a mismatch halts, because it
     would mean the downloaded artifacts are not from the run that
     produced the in-session numbers.
  5. random/persistence floors through the same harness, then the
     verdict.

One-shot enforcement (Rule 18): the prediction files' hashes, the
winner, and the post-processing config are recorded in
artifacts/phase3_TEST_SCORED.json at scoring time. Re-running with the
SAME files is a deterministic no-op; re-running with changed prediction
files, a different winner, or different post-processing raises — test
rows are scored once, ever, and nothing is re-decided after test
numbers exist.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import baselines
import data
import eval as ev
import paths
import splits
import stats


# ─── one-shot guard (Rule 18) ─────────────────────────────────────────────

def _one_shot_guard(pred_paths: dict[str, Path], winner: str,
                    postprocessing: dict) -> None:
    """Refuse to score test rows again under any changed condition: the
    prediction files, the winning variant, or the post-processing."""
    marker = paths.OUTPUT_DIR / "phase3_TEST_SCORED.json"
    current = {name: paths.sha256_of_file(p)
               for name, p in pred_paths.items()}
    record = {
        "winner": winner,
        "postprocessing": postprocessing,
        "prediction_hashes": current,
    }
    if not marker.exists():
        from datetime import datetime, timezone
        marker.write_text(json.dumps({
            "scored_at": datetime.now(timezone.utc).isoformat(),
            **record,
        }, indent=2))
        return
    recorded = json.loads(marker.read_text())
    if recorded.get("winner") != winner:
        raise RuntimeError(
            f"HALT: the winner recorded at first scoring "
            f"({recorded.get('winner')!r}) differs from the current one "
            f"({winner!r}). Rule 18: once test rows are scored the "
            f"decision chain is final — a different winner means a "
            f"different test result was observed. Delete {marker} ONLY "
            f"if you are consciously restarting the entire Phase 3 "
            f"final stage.")
    if recorded.get("postprocessing") != postprocessing:
        raise RuntimeError(
            f"HALT: post-processing config changed since the first "
            f"scoring ({recorded.get('postprocessing')} → "
            f"{postprocessing}). Rule 18/19: test rows are scored once, "
            f"under the post-processing declared in advance.")
    for name, h in current.items():
        if name in recorded.get("prediction_hashes", {}) \
                and recorded["prediction_hashes"][name] != h:
            raise RuntimeError(
                f"HALT: {name} prediction file changed since test rows "
                f"were first scored (hash "
                f"{recorded['prediction_hashes'][name][:12]}… → "
                f"{h[:12]}…). Rule 18: the test split is scored once, on "
                f"the trained predictions — never re-scored from "
                f"re-trained or edited files.")
    print("[final] test rows were scored before with these exact "
          "prediction files, winner, and post-processing — re-scoring "
          "is deterministic (identical numbers).")


# ─── in-session significance cross-check ──────────────────────────────────

def _cross_check_in_session(local_sig: dict) -> dict:
    """Recomputed p-values must agree with the in-session
    phase3_significance_results.json (notebook Cell 6). A mismatch means
    the downloaded artifacts are not from the run that produced the
    in-session numbers — halt rather than report two answers.

    Tolerances: counts exact; the observed Sharpe difference is pure
    arithmetic on the prediction files (1e-6); p-values get 0.01 because
    scipy/arch versions can differ between Kaggle and local (the
    quantisation of a 5,000-replicate bootstrap is coarser than that
    anyway) — tight enough to catch a wrong winner file, loose enough
    not to false-alarm on library drift.
    """
    p = paths.OUTPUT_DIR / "phase3_significance_results.json"
    if not p.exists():
        print("[final] WARNING: in-session phase3_significance_results.json "
              "was not downloaded — local p-values stand uncross-checked.")
        return {"checked": False, "reason": "in-session json missing"}
    in_session = json.loads(p.read_text())
    wil_l, wil_s = local_sig["wilcoxon"], in_session["wilcoxon"]
    boot_l, boot_s = local_sig["bootstrap_sharpe"], in_session["bootstrap_sharpe"]
    mismatches = []
    for label, a, b, tol in [
        ("wilcoxon.n_pairs", wil_l["n_pairs"], wil_s["n_pairs"], 0),
        ("wilcoxon.wins", wil_l["wins"], wil_s.get("wins"), 0),
        ("wilcoxon.p_value", wil_l["p_value"], wil_s["p_value"], 0.01),
        ("bootstrap.bootstrap_p_value", boot_l["bootstrap_p_value"],
         boot_s["bootstrap_p_value"], 0.01),
        ("bootstrap.observed_sharpe_difference",
         boot_l["observed_sharpe_difference"],
         boot_s["observed_sharpe_difference"], 1e-6),
    ]:
        if b is None or abs(float(a) - float(b)) > tol:
            mismatches.append(f"{label}: local={a} vs in-session={b}")
    if mismatches:
        raise RuntimeError(
            "HALT: the locally recomputed significance tests disagree "
            "with the in-session phase3_significance_results.json:\n  "
            + "\n  ".join(mismatches)
            + "\nThe downloaded prediction files are not from the run "
            "that produced the in-session numbers (wrong winner variant? "
            "stale download?). Do not report these numbers.")
    print("[final] cross-check vs in-session significance results: "
          "p-values agree ✓")
    return {"checked": True,
            "in_session": {"wilcoxon_p_value": wil_s["p_value"],
                           "bootstrap_p_value": boot_s["bootstrap_p_value"]}}


# ─── reference floors (same harness, no special casing) ───────────────────

def reference_floors(cfg: dict, folds) -> dict[str, dict]:
    """random + persistence through the same per-day harness on the test
    windows — the floor the trained arms must beat. Deterministic
    (seed 1234 + fold index), exactly as the leakage gate ran them."""
    if not (paths.FEATURES.exists() and paths.TARGETS.exists()):
        return {}
    features = pd.read_parquet(paths.FEATURES)
    targets = pd.read_parquet(paths.TARGETS)
    out = {}
    for model_name in ["random", "persistence"]:
        results = []
        for fold in folds:
            if model_name == "random":
                # Per-call counter (same as the leakage gate): noise must
                # differ day to day or the book never turns over and the
                # floor's costs are never charged.
                call = {"i": 0}

                def fn(X, *, fold_idx=fold.fold_idx, call=call):
                    call["i"] += 1
                    return baselines.random_model(
                        X, seed=1234 + fold_idx * 1000 + call["i"])
            else:
                fn = baselines.persistence_model
            results.append(ev.evaluate_fold(
                model_name=model_name,
                model_fn=fn,
                fold=fold,
                features=features,
                targets=targets,
                cfg=cfg,
                seed=1234 + fold.fold_idx,
            ))
        out[model_name] = {
            "mean_rank_ic": float(np.mean(
                [r.fold_metrics.mean_rank_ic for r in results])),
            "sharpe_after_costs": float(np.mean(
                [r.fold_metrics.sharpe_after_costs for r in results])),
            "directional_acc": float(np.mean(
                [r.fold_metrics.directional_acc for r in results])),
            "n_folds": len(results),
        }
    return out


# ─── main ─────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = data.load_config(str(paths.CONFIG))

    # ── §12 report order: sensitivity table FIRST ─────────────────────
    grid_csv = paths.OUTPUT_DIR / "phase3_sensitivity_grid.csv"
    config_json = paths.OUTPUT_DIR / "phase3_frozen_graph_config.json"
    if not config_json.exists():
        raise FileNotFoundError(
            f"{config_json} missing — winner selection happens IN-SESSION "
            f"(notebook Cell 3) and its outputs are downloaded; the local "
            f"chain starts from that locked winner. Re-download the "
            f"session outputs.")
    rec = json.loads(config_json.read_text())
    winner = rec["winner"]

    print("=" * 78)
    print("PHASE 3 — FINAL COMPARISON REPORT")
    print("=" * 78)
    print("\n[1] Graph sensitivity grid (validation-only, §6b):")
    if grid_csv.exists():
        print(pd.read_csv(grid_csv).to_string(index=False))
    print(f"\n  winner: {winner} "
          f"(k_pos={rec['winning_graph_params']['k_pos']}, "
          f"k_neg={rec['winning_graph_params']['k_neg']}, "
          f"corr_window={rec['winning_graph_params']['corr_window']}) "
          f"— {'CHANGED from baseline' if rec['changed_from_baseline'] else 'baseline config holds'}")

    # ── declared-in-advance post-processing (Rule 19) ──────────────────
    pp = dict(ev.DEFAULT_POSTPROCESSING)
    print(f"\n[2] Post-processing (identity, declared in advance — the "
          f"single-session design has no tuning stage):")
    print(f"  band={pp['no_trade_band']:.4f}  "
          f"decile_fraction={pp['decile_fraction']} "
          f"(None → cfg default {cfg['backtest']['decile_fraction']})  "
          f"smoothing_days={pp['smoothing_days']}")

    # ── ONE-SHOT test scoring (the only place test rows are scored) ────
    print(f"\n[3] One-shot final TEST scoring (identical post-processing "
          f"on both arms):")
    pred_paths = {
        "dynamic": paths.OUTPUT_DIR
        / f"predictions_dmrgt_all3_grid_{winner}.parquet",
        "frozen": paths.OUTPUT_DIR / "predictions_dmrgt_frozen.parquet",
    }
    for label, p in pred_paths.items():
        if not p.exists():
            raise FileNotFoundError(
                f"{p} missing — download the single session's outputs "
                f"into {paths.OUTPUT_DIR}/ first.")
    _one_shot_guard(pred_paths, winner, pp)

    targets = pd.read_parquet(paths.TARGETS)
    arm_results = {}
    for label, p in pred_paths.items():
        res = ev.evaluate_predictions(p, targets, cfg, split="test",
                                      postprocessing=pp)
        expected = (f"dmrgt_all3_grid_{winner}" if label == "dynamic"
                    else "dmrgt_frozen")
        assert all(r.model == expected for r in res), (
            f"{p.name} holds {sorted({r.model for r in res})}, expected "
            f"{expected!r} — wrong winner file?")
        arm_results[label] = res
        agg_ic = float(np.mean([r.fold_metrics.mean_rank_ic for r in res]))
        agg_sh = float(np.mean([r.fold_metrics.sharpe_after_costs for r in res]))
        print(f"  {label:8s} ({expected})  {len(res)} runs  "
              f"mean rank IC {agg_ic:+.5f}  Sharpe {agg_sh:+.3f}")

    # ── per-run table (fold, seed granularity — the honest view) ──────
    rows = []
    for label in ["dynamic", "frozen"]:
        for r in arm_results[label]:
            m = r.fold_metrics
            rows.append({
                "arm": label, "model": m.model, "fold": m.fold,
                "seed": m.seed, "split": m.split, "n_days": m.n_days,
                "mean_rank_ic": m.mean_rank_ic,
                "std_rank_ic": m.std_rank_ic,
                "directional_acc": m.directional_acc,
                "sharpe_after_costs": m.sharpe_after_costs,
                "max_drawdown": m.max_drawdown,
                "turnover": m.turnover,
                "cum_return": m.cum_return,
            })
    per_run = pd.DataFrame(rows)
    per_run_csv = paths.OUTPUT_DIR / "phase3_final_metrics.csv"
    per_run.to_csv(per_run_csv, index=False)

    print(f"\n[4] Per-run test metrics ({len(per_run)} rows → {per_run_csv}):")
    with pd.option_context("display.width", 140, "display.float_format",
                           "{:+.4f}".format):
        print(per_run.groupby("arm")[
            ["mean_rank_ic", "directional_acc", "sharpe_after_costs",
             "max_drawdown", "turnover", "cum_return"]
        ].mean().to_string())

    # ── significance (§6 of the rework) — recomputed locally ───────────
    print(f"\n[5] Significance tests (recomputed from the downloaded "
          f"prediction files):")
    sig = stats.run_tests_from_predictions(
        pred_paths["dynamic"], pred_paths["frozen"], cfg,
        winning_graph_config=rec["winning_graph_params"])
    wil, boot = sig["wilcoxon"], sig["bootstrap_sharpe"]
    print(f"  Wilcoxon signed-rank (paired rank IC, n={wil['n_pairs']}):")
    print(f"    dynamic wins {wil['wins']}/{wil['n_pairs']}  "
          f"statistic={wil['statistic']:.4f}  p={wil['p_value']:.4f}  "
          f"alpha={wil['alpha']}  significant={wil['significant']}")
    print(f"    dynamic IC {wil['dynamic_mean_rank_ic']:+.5f} vs frozen "
          f"{wil['frozen_mean_rank_ic']:+.5f} "
          f"(paired diff {wil['mean_paired_difference']:+.5f})")
    print(f"  Stationary bootstrap on Sharpe difference:")
    print(f"    dyn {boot['observed_sharpe_dynamic']:+.3f} vs fro "
          f"{boot['observed_sharpe_frozen']:+.3f} "
          f"(diff {boot['observed_sharpe_difference']:+.3f}, "
          f"95% CI [{boot['ci_95'][0]:+.3f}, {boot['ci_95'][1]:+.3f}])")
    print(f"    one-sided p={boot['bootstrap_p_value']:.4f}  "
          f"two-sided p={boot['p_two_sided']:.4f}")

    cross = _cross_check_in_session(sig)

    # ── reference floors ──────────────────────────────────────────────
    feats_dates = pd.read_parquet(paths.FEATURES, columns=["date"])
    folds = splits.make_folds(cfg, available_dates=pd.DatetimeIndex(
        feats_dates["date"].unique()))
    floors = reference_floors(cfg, folds)
    if floors:
        print(f"\n[6] Reference floors (same harness, no special casing):")
        for name, f in floors.items():
            print(f"  {name:12s}  mean rank IC {f['mean_rank_ic']:+.5f}  "
                  f"Sharpe {f['sharpe_after_costs']:+.3f}  "
                  f"dir acc {f['directional_acc']:.4f}")

    # ── verdict ──────────────────────────────────────────────────────
    print(f"\n[7] CONCLUSION:")
    print(f"  {sig['conclusion']}")
    dyn_sh = boot["observed_sharpe_dynamic"]
    fro_sh = boot["observed_sharpe_frozen"]
    floor_sh = floors.get("persistence", {}).get("sharpe_after_costs", float("nan"))
    for label, sh in [("dmrgt_all3", dyn_sh), ("dmrgt_frozen", fro_sh)]:
        vs = ("ABOVE" if sh > floor_sh else "BELOW")
        print(f"  {label}: Sharpe {sh:+.3f} — {vs} the persistence "
              f"floor ({floor_sh:+.3f})")

    # ── persist everything ─────────────────────────────────────────────
    aggregate = {}
    for label, model_prefix in [("dmrgt_all3", "dmrgt_all3"),
                                ("dmrgt_frozen", "dmrgt_frozen")]:
        sub = per_run[per_run["model"].str.startswith(model_prefix)]
        aggregate[label] = {
            "n_runs": int(len(sub)),
            **{k: float(sub[k].mean()) for k in
               ["mean_rank_ic", "directional_acc", "sharpe_after_costs",
                "max_drawdown", "turnover", "cum_return"]},
        }
    final = {
        "winner_record": rec,
        "postprocessing": pp,
        "per_run": rows,
        "aggregate": aggregate,
        "significance_local": sig,
        "significance_cross_check": cross,
        "reference_floors": floors,
    }
    out_json = paths.OUTPUT_DIR / "phase3_final_results.json"
    out_json.write_text(json.dumps(final, indent=2, default=str))
    print(f"\n[final] wrote {per_run_csv}")
    print(f"[final] wrote {out_json}")
    print("[final] phase3_significance_results.json stays as the in-session "
          "run wrote it — compare.py cross-checks it, never rewrites it.")


if __name__ == "__main__":
    main()