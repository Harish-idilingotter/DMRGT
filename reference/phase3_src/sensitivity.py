"""DMRGT Phase 3 — graph sensitivity grid (§6b).

Two CLI modes, split across the Kaggle timeline:

  --build (LOCAL, before Kaggle session A)
      Build the 7 graph variants — baseline (k_pos 8, k_neg 4, window 60)
      plus one-at-a-time sweeps k_pos ∈ {6,10}, k_neg ∈ {3,5},
      corr_window ∈ {45,90} — into cache/graphs_<name>.parquet (the cache
      ROOT, alongside graphs.parquet — not a `graph_variants/`
      subdirectory, which `kaggle datasets version` would silently drop;
      see paths.py).
      Asserts the baseline variant is IDENTICAL to cache/graphs.parquet
      (proving the rebuild reproduces Phase 2's dynamic graph exactly),
      then folds their hashes into hashes.json. The user uploads the
      cache dataset (one version bump) so Kaggle can hash-verify every
      variant before training.

  --select-winner (notebook Cell 3 in-session; also runnable locally on
      the downloaded outputs)
      Score each variant's VAL predictions (never test — §1 of the
      single-session rework, Rule 20) through the cost-aware backtest,
      pick the winner on the pre-declared selection metric (val Sharpe),
      and write:
        OUTPUT_DIR/phase3_sensitivity_grid.csv       — the full table
        OUTPUT_DIR/phase3_frozen_graph_config.json   — the winning config
      The winner's ALREADY-EXISTING grid test rows become the final
      dmrgt_all3 result, read for the first time only after this
      function returns and the winner is locked — dmrgt_all3 is never
      retrained. If the winner is not the baseline, the in-session
      frozen-graph cell builds graphs_frozen.parquet at the winning k
      directly from the attached prices.parquet: no cache rebuild, no
      re-upload, no restart.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import pandas as pd

import data
import eval as ev
import graph
import paths


GRID_BASELINE = {"k_pos": 8, "k_neg": 4, "corr_window": 60}

# configs.yaml sweep key → (graph param, variant name prefix). The sweep
# lists in configs.yaml include the baseline value (they document the full
# swept range); the baseline itself is the separate "baseline" variant, so
# equal-to-baseline values are excluded here to avoid duplicate builds.
_SWEEP_KEYS = {
    "k_pos_sweep": ("k_pos", "kpos"),
    "k_neg_sweep": ("k_neg", "kneg"),
    "corr_window_sweep": ("corr_window", "win"),
}


def variant_configs(cfg: dict) -> dict[str, dict]:
    """The 7 grid variants as {name: cfg}, one dimension changed at a
    time from the baseline. Names are stable identifiers used in file
    names on both sides of the Kaggle upload — the notebook's session-A
    list must stay in sync with these."""
    ggrid = cfg.get("graph_grid", {})
    missing = [k for k in list(_SWEEP_KEYS) + ["baseline"]
               if k not in ggrid]
    if missing:
        raise KeyError(
            f"configs.yaml [graph_grid] missing keys: {missing} — the grid "
            f"is declared there (Rule: config is the single source of "
            f"truth), never hardcoded in sensitivity.py."
        )
    baseline = {k: int(v) for k, v in ggrid["baseline"].items()}
    # The config's own graph block must BE the grid baseline — the
    # "baseline" variant is a deepcopy of cfg, and the identity test
    # (baseline variant == cache/graphs.parquet) is only meaningful if
    # the two blocks agree.
    for param, v in baseline.items():
        if int(cfg["graph"][param]) != v:
            raise ValueError(
                f"configs.yaml drift: [graph] {param}="
                f"{cfg['graph'][param]} but [graph_grid].baseline "
                f"{param}={v}. The cache was built at the graph block's "
                f"values; the grid baseline must match."
            )
    out: dict[str, dict] = {"baseline": copy.deepcopy(cfg)}
    for yaml_key, (param, prefix) in _SWEEP_KEYS.items():
        for v in ggrid[yaml_key]:
            v = int(v)
            if v == baseline[param]:
                continue  # the baseline itself — already variant "baseline"
            c = copy.deepcopy(cfg)
            c["graph"][param] = v
            out[f"{prefix}{v}"] = c
    return out


def variant_graph_params(cfg_variant: dict) -> dict:
    g = cfg_variant["graph"]
    return {"k_pos": int(g["k_pos"]), "k_neg": int(g["k_neg"]),
            "corr_window": int(g["corr_window"])}


# ─── --build ───────────────────────────────────────────────────────────────

def build_variants(cfg: dict) -> dict[str, Path]:
    if not paths.PRICES.exists():
        print("[grid] prices.parquet missing — nothing to build from.",
              file=sys.stderr)
        sys.exit(1)
    if not paths.GRAPHS.exists():
        print("[grid] cache/graphs.parquet missing — the baseline identity "
              "check needs it. Do not build variants without the cached "
              "dynamic graph.", file=sys.stderr)
        sys.exit(1)

    prices = pd.read_parquet(paths.PRICES)
    cached_dynamic = pd.read_parquet(paths.GRAPHS)

    written: dict[str, Path] = {}
    for name, vcfg in variant_configs(cfg).items():
        print(f"\n[grid] building variant {name!r} "
              f"{variant_graph_params(vcfg)}")
        df = graph.build_daily_graph_all_dates(prices, vcfg)
        if name == "baseline":
            # §6b: the baseline rebuild must equal the cached dynamic graph
            # exactly, or the whole grid is measuring rebuild drift.
            if not df.equals(cached_dynamic):
                raise AssertionError(
                    "Baseline variant (k_pos=8, k_neg=4, window=60) does "
                    "NOT equal cache/graphs.parquet — the Phase 2 graph "
                    "reproduction is broken. Fix graph.py before building "
                    "the grid."
                )
            print("[grid] baseline variant == cache/graphs.parquet ✓")
        out_path = paths.graph_variant_path(name)
        df.to_parquet(out_path, index=False)
        written[name] = out_path
        print(f"[grid] wrote {out_path}  rows={len(df):,}")

    # Fold the variant hashes into hashes.json (merge — the five main
    # cache entries are preserved). update_hashes picks the variants up
    # from the cache root itself, so there is no directory to pass.
    hashes = paths.update_hashes()
    print(f"\n[grid] hashes.json updated: {len(written)} variants + "
          f"{len(paths.CACHE_PARQUETS)} cache files hashed")
    return written


# ─── --select-winner ───────────────────────────────────────────────────────

def select_winner(cfg: dict) -> dict:
    """Score the grid on VAL predictions only and lock the winner (§1 of
    the single-session rework).

    The safeguard, exactly as specified: every grid prediction file is
    read once into ``df``, asserted to hold only val/test splits, sliced
    to val rows, and the full-frame reference is ``del``eted before any
    scoring runs — so no code path below can reach test rows through a
    lingering variable. The winner's already-existing test rows become
    the final dmrgt_all3 result only AFTER this function returns and the
    winner is locked (notebook Cell 6 / compare.py read them then, and
    only then — Rule 18, stricter not looser).

    Writes phase3_sensitivity_grid.csv and phase3_frozen_graph_config.json
    into OUTPUT_DIR and returns the frozen-config record the notebook's
    frozen-graph cell reads.
    """
    metric = cfg.get("graph_grid", {}).get("selection_metric", "sharpe")
    all_variants = variant_configs(cfg)

    # ── §1 safeguard: load once, keep val rows, drop the full reference ──
    val_only: dict[str, pd.DataFrame] = {}
    for name in all_variants:
        pred_path = paths.OUTPUT_DIR / f"predictions_dmrgt_all3_grid_{name}.parquet"
        if not pred_path.exists():
            raise FileNotFoundError(
                f"{pred_path.name} missing from {paths.OUTPUT_DIR} — a "
                f"partial grid must never pick a winner from the variants "
                f"that happened to finish. Complete all 7 variant runs "
                f"first.")
        df = pd.read_parquet(pred_path)
        assert set(df["split"].unique()) <= {"val", "test"}, (
            f"{pred_path.name}: unexpected split value(s) present"
        )
        val_only[name] = df[df["split"] == "val"]  # test rows never touched here
        del df  # drop the reference to the full frame explicitly (§1)

    rows = []
    metrics: dict[str, dict] = {}
    for name in all_variants:
        d = val_only[name]
        agg = ev.backtest(d, cfg)  # val-only frame — Rule 20
        metrics[name] = agg
        params = variant_graph_params(all_variants[name])
        rows.append({
            "variant": name,
            **params,
            f"val_{metric}": agg[metric],
            "val_sharpe": agg["sharpe"],
            "val_mean_rank_ic": agg["mean_rank_ic"],
            "val_directional_acc": agg["directional_acc"],
            "val_turnover": agg["turnover"],
            "val_cum_return": agg["cum_return"],
            "n_runs": agg["n_runs"],
        })
        print(f"[grid] {name:10s} k_pos={params['k_pos']:2d} "
              f"k_neg={params['k_neg']:2d} win={params['corr_window']:2d}  "
              f"val {metric}={agg[metric]:+.4f}  "
              f"val IC={agg['mean_rank_ic']:+.4f}", flush=True)
    del val_only  # selection inputs out of scope before any test row is read

    table = pd.DataFrame(rows).sort_values(f"val_{metric}",
                                           ascending=False).reset_index(drop=True)
    csv_path = paths.OUTPUT_DIR / "phase3_sensitivity_grid.csv"
    table.to_csv(csv_path, index=False)
    print(f"\n[grid] wrote {csv_path}")

    # Winner on the pre-declared metric — the same first-in-order tie-break
    # as the stable descending sort above, so CSV top row and winner agree.
    winner_name = max(metrics, key=lambda v: metrics[v][metric])
    winner = table.iloc[0]
    assert str(winner["variant"]) == winner_name, (
        f"winner disagreement: max() picked {winner_name!r} but the sorted "
        f"table's top row is {winner['variant']!r}"
    )
    baseline_params = variant_graph_params(all_variants["baseline"])
    winner_params = {"k_pos": int(winner["k_pos"]),
                     "k_neg": int(winner["k_neg"]),
                     "corr_window": int(winner["corr_window"])}
    changed = winner_params != baseline_params
    record = {
        "winner": winner_name,
        "winning_graph_params": winner_params,
        "baseline_graph_params": baseline_params,
        "changed_from_baseline": changed,
        "selection_metric": metric,
        "selection_value": float(winner[f"val_{metric}"]),
        "grid": table.to_dict(orient="records"),
        "rule": (
            "Grid searched on dmrgt_all3 VAL predictions only (§1 of the "
            "single-session rework; Rule 20). The winning k is locked "
            "BEFORE the frozen graph is built in-session and before any "
            "test-split row is read; the winner's existing grid test rows "
            "are the final dmrgt_all3 result — it is never retrained."
        ),
    }
    json_path = paths.OUTPUT_DIR / "phase3_frozen_graph_config.json"
    json_path.write_text(json.dumps(record, indent=2))
    print(f"[grid] wrote {json_path}")

    print("\n" + "=" * 70)
    print(f"[grid] WINNER: {record['winner']} "
          f"(k_pos={winner_params['k_pos']}, "
          f"k_neg={winner_params['k_neg']}, "
          f"corr_window={winner_params['corr_window']})  "
          f"val {metric}={record['selection_value']:+.4f}")
    if changed:
        print("[grid] The winning config DIFFERS from the baseline — the "
              "in-session frozen-graph cell builds graphs_frozen.parquet "
              "at the winning k directly from the attached prices.parquet. "
              "No cache rebuild, no re-upload, no restart (§2, §4).")
    else:
        print("[grid] The baseline config WINS — the in-session "
              "frozen-graph cell builds at the baseline k.")
    print("=" * 70, flush=True)
    return record


# ─── CLI ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="DMRGT Phase 3 graph grid")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build", action="store_true",
                      help="Build the 7 graph variants into cache/ "
                           "(as cache/graphs_<name>.parquet)")
    mode.add_argument("--select-winner", action="store_true",
                      help="Score session-A val predictions, pick the winner, "
                           "write the grid CSV + frozen graph config JSON")
    args = parser.parse_args()

    cfg = data.load_config(str(paths.CONFIG))
    if args.build:
        build_variants(cfg)
    else:
        select_winner(cfg)


if __name__ == "__main__":
    main()