"""DMRGT Phase 1 — Step 6 comparison driver (§8).

Runs random, persistence, and Transformer through the same eval.py path.
Usage: python compare.py [--skip-tx | --tx-only]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

import baselines
import data
import eval as ev
import paths
import splits as sp


def run_baselines(cfg: dict, features: pd.DataFrame, targets: pd.DataFrame,
                  folds: list) -> list[ev.FoldResult]:
    """Run random + persistence on every fold."""
    results = []
    for fold in folds:
        # Random — 1 seed (the spec only requires the gate, not multi-seed)
        r = ev.evaluate_fold("random", baselines.random_model,
                             fold, features, targets, cfg, seed=0)
        results.append(r)
        # Persistence — deterministic, single run
        r = ev.evaluate_fold("persistence", baselines.persistence_model,
                             fold, features, targets, cfg)
        results.append(r)
    return results


def run_transformer(cfg: dict, targets: pd.DataFrame,
                    predictions_path: Path) -> list[ev.FoldResult]:
    """Re-evaluate downloaded predictions.parquet."""
    if not predictions_path.exists():
        print(f"[compare] predictions file not found: {predictions_path}",
              file=sys.stderr)
        print("[compare] run the Kaggle training notebook first "
              "(see kaggle/KAGGLE.md), then download predictions.parquet "
              "into artifacts/.",
              file=sys.stderr)
        return []
    return ev.evaluate_predictions(predictions_path, targets, cfg)


def aggregate_to_table(results: list[ev.FoldResult]) -> pd.DataFrame:
    """Flatten FoldResults to a DataFrame."""
    rows = []
    for r in results:
        d = r.fold_metrics.__dict__.copy()
        rows.append(d)
    return pd.DataFrame(rows)


def per_model_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Group by model, report mean ± std across folds and seeds."""
    metric_cols = ["mean_rank_ic", "directional_acc", "sharpe_after_costs",
                   "max_drawdown", "turnover", "cum_return"]
    grouped = table.groupby("model")[metric_cols].agg(["mean", "std"])
    # Flatten the column MultiIndex
    grouped.columns = [f"{m}_{stat}" for m, stat in grouped.columns]
    return grouped.reset_index()


def per_fold_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Transformer per-fold results, aggregated across seeds."""
    metric_cols = ["mean_rank_ic", "directional_acc", "sharpe_after_costs",
                   "max_drawdown", "turnover", "cum_return"]
    tx = table[table["model"] == "transformer"]
    if tx.empty:
        return tx
    agg = tx.groupby("fold")[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    return agg.reset_index()


def main():
    parser = argparse.ArgumentParser(description="DMRGT Phase 1 Step 6 comparison")
    parser.add_argument("--config", default="configs.yaml")
    parser.add_argument("--predictions", default=str(paths.PREDICTIONS))
    parser.add_argument("--skip-tx", action="store_true",
                        help="Skip the Transformer evaluation")
    parser.add_argument("--tx-only", action="store_true",
                        help="Only run Transformer evaluation")
    args = parser.parse_args()

    cfg = data.load_config(args.config)

    # Load caches
    if not paths.FEATURES.exists() or not paths.TARGETS.exists():
        print("[compare] cache missing — run `python data.py --features` first",
              file=sys.stderr)
        sys.exit(1)
    features = pd.read_parquet(paths.FEATURES)
    targets = pd.read_parquet(paths.TARGETS)
    folds = sp.make_folds(cfg, available_dates=features["date"].unique())

    results = []
    if not args.tx_only:
        print("[compare] running baselines (random + persistence) on all folds…")
        results += run_baselines(cfg, features, targets, folds)

    if not args.skip_tx:
        print("[compare] evaluating Transformer predictions…")
        results += run_transformer(cfg, targets, Path(args.predictions))

    if not results:
        print("[compare] nothing to report", file=sys.stderr)
        return

    table = aggregate_to_table(results)
    summary = per_model_summary(table)
    fold_summary = per_fold_summary(table)

    print("\n" + "=" * 80)
    print("DMRGT Phase 1 — Step 6 comparison table (§8)")
    print("=" * 80)
    with pd.option_context("display.width", 160, "display.float_format",
                           "{:+.4f}".format):
        print("\n## Per-model summary (mean ± std across folds and seeds)")
        print(summary.to_string(index=False))
        if not fold_summary.empty:
            print("\n## Per-fold Transformer results (across seeds)")
            print(fold_summary.to_string(index=False))

    # The exact format the spec asks for in §8:
    print("\n## Spec table — Model | Mean rank IC | Dir acc | Sharpe | MDD | Turnover")
    print("-" * 80)
    for _, row in summary.iterrows():
        print(f"  {row['model']:12s}  "
              f"IC={row['mean_rank_ic_mean']:+.4f}±{row['mean_rank_ic_std']:.4f}  "
              f"dir={row['directional_acc_mean']:.3f}  "
              f"sharpe={row['sharpe_after_costs_mean']:+.2f}  "
              f"mdd={row['max_drawdown_mean']:.3f}  "
              f"turnover={row['turnover_mean']:.2f}")

    # Also dump the raw per-(fold, seed) table to a CSV for the report
    out_csv = paths.OUTPUT_DIR / "comparison_table.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)
    print(f"\n[compare] wrote {out_csv}")

    # ── MLflow logging (§2.5) ──────────────────────────────────────
    # Log every (model, fold, seed) row as a separate run under the
    # "dmrgt-phase1" experiment. The full audit trail lives in
    # artifacts/mlruns/ and can be browsed with `mlflow ui` from
    # the project root.
    n_logged = 0
    for r in results:
        if ev.log_fold_to_mlflow(r.fold_metrics, cfg,
                                 run_name=f"{r.fold_metrics.model}_"
                                          f"{r.fold_metrics.fold}_"
                                          f"seed{r.fold_metrics.seed}"):
            n_logged += 1
    if n_logged:
        print(f"[compare] logged {n_logged}/{len(results)} runs to MLflow "
              f"at {paths.OUTPUT_DIR / 'mlflow.db'}")
        print(f"[compare] browse with: mlflow ui "
              f"--backend-store-uri sqlite:///{paths.OUTPUT_DIR / 'mlflow.db'}")
    else:
        print("[compare] MLflow logging skipped (disabled or unavailable). "
              "Set DMRGT_DISABLE_MLFLOW=0 to enable.")


if __name__ == "__main__":
    main()
