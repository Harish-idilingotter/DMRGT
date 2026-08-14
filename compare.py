"""DMRGT Phase 2 — multi-model comparison driver (§8).

Runs random, persistence, and all trained models through the same eval.py
path. Produces the full §8.1 comparison table.

Usage: python compare.py [--predictions PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

import baselines
import data as data_mod
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


def _collect_predictions(artifacts_dir: Path, cfg: dict) -> Path | None:
    """Find per-model predictions_{model_id}.parquet files, concatenate into
    a single predictions.parquet, and return its path. Returns None if no
    files are found."""
    combined_path = artifacts_dir / "predictions.parquet"

    # If a combined file already exists and is newer than the per-model
    # files, use it directly.
    per_model_files = sorted(artifacts_dir.glob("predictions_*.parquet"))
    if not per_model_files:
        if combined_path.exists():
            return combined_path
        return None

    # Check if combined is fresh enough
    if combined_path.exists():
        combined_mtime = combined_path.stat().st_mtime
        if all(p.stat().st_mtime <= combined_mtime for p in per_model_files):
            print(f"[compare] using cached {combined_path}")
            return combined_path

    # Concatenate per-model files
    print(f"[compare] concatenating {len(per_model_files)} per-model files…")
    pieces = []
    for p in per_model_files:
        model_id = p.stem.replace("predictions_", "")
        df = pd.read_parquet(p)
        # The model column is already in the parquet from train.py
        actual_models = sorted(df["model"].unique())
        if len(actual_models) != 1 or actual_models[0] != model_id:
            print(f"[compare] WARNING: {p.name} has model(s) {actual_models}, "
                  f"expected [{model_id}]")
        print(f"  {model_id}: {len(df):,} rows")
        pieces.append(df)

    combined = pd.concat(pieces, ignore_index=True)
    combined.to_parquet(combined_path, index=False)
    print(f"[compare] wrote {combined_path}: {len(combined):,} rows, "
          f"{combined['model'].nunique()} models")
    return combined_path


def run_trained_models(cfg: dict, targets: pd.DataFrame,
                       predictions_path: Path) -> list[ev.FoldResult]:
    """Re-evaluate downloaded predictions (Phase 2 schema with model column).
    predictions_path can be a single combined file or a directory containing
    per-model predictions_{model_id}.parquet files."""
    path = Path(predictions_path)
    if path.is_dir():
        resolved = _collect_predictions(path, cfg)
        if resolved is None:
            print(f"[compare] no predictions_*.parquet files found in {path}",
                  file=sys.stderr)
            return []
    elif path.exists():
        resolved = path
    else:
        # Try the artifacts directory as fallback
        resolved = _collect_predictions(paths.OUTPUT_DIR, cfg)
        if resolved is None:
            print(f"[compare] predictions file not found: {predictions_path}",
                  file=sys.stderr)
            return []
    return ev.evaluate_predictions(resolved, targets, cfg)


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
    grouped.columns = [f"{m}_{stat}" for m, stat in grouped.columns]
    return grouped.reset_index()


def per_fold_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Per-fold results, aggregated across seeds, for each model."""
    metric_cols = ["mean_rank_ic", "directional_acc", "sharpe_after_costs",
                   "max_drawdown", "turnover", "cum_return"]
    agg = table.groupby(["model", "fold"])[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    return agg.reset_index()


def main():
    parser = argparse.ArgumentParser(description="DMRGT Phase 2 multi-model comparison")
    parser.add_argument("--config", default="configs.yaml")
    parser.add_argument("--predictions", default=str(paths.PREDICTIONS))
    parser.add_argument("--skip-tx", action="store_true",
                        help="Skip trained-model evaluation")
    parser.add_argument("--tx-only", action="store_true",
                        help="Only run trained-model evaluation")
    args = parser.parse_args()

    cfg = data_mod.load_config(args.config)

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
        print("[compare] evaluating trained model predictions…")
        tx_results = run_trained_models(cfg, targets, Path(args.predictions))
        results += tx_results

        # Row-count assertion: ~1M rows for 5 models
        if tx_results:
            pred_df = pd.read_parquet(args.predictions)
            n_rows = len(pred_df)
            n_models = pred_df["model"].nunique()
            model_ids = sorted(pred_df["model"].unique())
            expected_ids = set(cfg["models"]["ids"])
            missing_ids = expected_ids - set(model_ids)
            if missing_ids:
                print(f"[compare] WARNING: missing model IDs in predictions: {missing_ids}")
            print(f"[compare] predictions: {n_rows:,} rows, "
                  f"{n_models} models: {model_ids}")
            # Rough check: 5 models × 4 folds × 5 seeds × ~250 days × 40 stocks ≈ 1M
            expected_approx = n_models * 4 * 5 * 250 * 40
            if n_rows < expected_approx * 0.5 or n_rows > expected_approx * 1.5:
                print(f"[compare] WARNING: row count {n_rows:,} far from "
                      f"expected ~{expected_approx:,} — a model or fold may have "
                      f"silently failed")

    if not results:
        print("[compare] nothing to report", file=sys.stderr)
        return

    table = aggregate_to_table(results)
    summary = per_model_summary(table)
    fold_summary = per_fold_summary(table)

    print("\n" + "=" * 80)
    print("DMRGT Phase 2 — Multi-model comparison table (§8.1)")
    print("=" * 80)

    # ── Spec table (§8.1) ─────────────────────────────────────────────
    print("\n## Spec table — Model | Mean rank IC | Dir acc | Sharpe | MDD | Turnover")
    print("-" * 80)
    for _, row in summary.iterrows():
        print(f"  {row['model']:25s}  "
              f"IC={row['mean_rank_ic_mean']:+.4f}±{row['mean_rank_ic_std']:.4f}  "
              f"dir={row['directional_acc_mean']:.3f}  "
              f"sharpe={row['sharpe_after_costs_mean']:+.2f}  "
              f"mdd={row['max_drawdown_mean']:.3f}  "
              f"turnover={row['turnover_mean']:.2f}")

    # ── Per-model summary ─────────────────────────────────────────────
    with pd.option_context("display.width", 160, "display.float_format",
                           "{:+.4f}".format):
        print("\n## Per-model summary (mean ± std across folds and seeds)")
        print(summary.to_string(index=False))

    # ── Per-fold breakdown ───────────────────────────────────────────
    if not fold_summary.empty:
        print("\n## Per-fold results (mean across seeds)")
        with pd.option_context("display.width", 200, "display.float_format",
                               "{:+.4f}".format):
            print(fold_summary.to_string(index=False))

    # ── Completion criterion check ────────────────────────────────────
    print("\n## Completion criterion: DMRGT all-3 beats BOTH no-graph Transformer AND LSTM?")
    dmrgt_row = summary[summary["model"] == "dmrgt_all3"]
    tx_row = summary[summary["model"] == "transformer_nograph"]
    lstm_row = summary[summary["model"] == "lstm"]

    if not dmrgt_row.empty and not tx_row.empty and not lstm_row.empty:
        dmrgt_ic = dmrgt_row["mean_rank_ic_mean"].iloc[0]
        tx_ic = tx_row["mean_rank_ic_mean"].iloc[0]
        lstm_ic = lstm_row["mean_rank_ic_mean"].iloc[0]
        beats_tx = dmrgt_ic > tx_ic
        beats_lstm = dmrgt_ic > lstm_ic
        print(f"  DMRGT all-3 IC: {dmrgt_ic:+.4f}")
        print(f"  Transformer IC:  {tx_ic:+.4f}  → beats: {beats_tx}")
        print(f"  LSTM IC:         {lstm_ic:+.4f}  → beats: {beats_lstm}")
        if beats_tx and beats_lstm:
            print("  ✓ COMPLETION CRITERION MET")
        else:
            print("  ✗ COMPLETION CRITERION NOT MET — see §8.3 for diagnosis")
    else:
        print("  (not all models present — train all 5 models first)")

    # ── Dump to CSV ──────────────────────────────────────────────────
    out_csv = paths.OUTPUT_DIR / "comparison_table.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)
    print(f"\n[compare] wrote {out_csv}")

    # ── MLflow logging ───────────────────────────────────────────────
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


if __name__ == "__main__":
    main()
