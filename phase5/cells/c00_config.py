import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"       # before CUDA initialises (workers set it too)

import hashlib, json, time
from pathlib import Path

WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)
(WORK / "dmrgt4").mkdir(parents=True, exist_ok=True)   # %%writefile cells below do not create folders
RUN_T0 = time.time()

# ─── Pre-registered configuration (Phase 5 spec §3) — fixed BEFORE any data is touched ───
# Phase 4 CONFIG with ONLY the changes listed in CONFIG_CHANGES below (recorded in the snapshot).
CONFIG = {
    "phase": "5",
    "cache_dataset": "dmrgt-sp100-cache",
    "universe": "from universe.csv (included == True); N read at runtime",
    "seeds": [0, 1, 2, 3, 4],
    "folds": "folds 1-4 identical to Phase 3 (selection folds); folds 5-6 = 2023 / 2024 holdout",
    "folds_selection": ["fold_1_2019_calm", "fold_2_2020_covid", "fold_3_2021", "fold_4_2022_rates"],
    "folds_holdout": [
        {"name": "fold_5_2023", "train_start": "2015-01-01", "train_end": "2022-12-31",
         "test_start": "2023-01-01", "test_end": "2023-12-31"},
        {"name": "fold_6_2024", "train_start": "2015-01-01", "train_end": "2023-12-31",
         "test_start": "2024-01-01", "test_end": "2024-12-31"},
    ],
    "graph": {"k_pos": 8, "k_neg": 4, "corr_window": 60, "file": "graphs_baseline.parquet"},
    "features": 8,
    "backtest": {"commission_bps": 10, "slippage_bps": 5, "decile_fraction": 0.2},

    "arms": {
        "P": {"encoder": "transformer", "lookback": 30, "schedule": "phase3",   "selection_candidate": False},
        "A": {"encoder": "transformer", "lookback": 30, "schedule": "adaptive", "selection_candidate": False},
        "B": {"encoder": "mamba",       "lookback": 30, "schedule": "adaptive", "selection_candidate": True},
        "C": {"encoder": "mamba",       "lookback": 60, "schedule": "adaptive", "selection_candidate": True},
    },
    "arm_D": "removed (adaptive Huber showed no effect in Phase 4; removed to fit the larger universe in one run)",

    "training": {
        "max_epochs": 50, "early_stop_patience": 10, "early_stop_metric": "val_es_mean_rank_ic (3-epoch rolling mean)",
        "batch_size_ladder": [32, 16, 8],          # same batch size for ALL arms
        "optimizer": "AdamW (two parameter groups); arm P: Adam as in Phase 3", "lr": 1e-3, "weight_decay": 1e-4,
        "warmup_epochs": 3, "warmup_start_lr": 1e-4,
        "plateau": {"mode": "max", "factor": 0.5, "patience": 4, "min_lr": 1e-5},
        "grad_clip_norm": 1.0,
        "huber_delta_fixed": 1.0,
        "adaptive_huber": {"c": 1.345, "min": 0.05, "max": 2.0},     # unused (arm D removed)
        "retry_failed_job_once": True,
    },

    "mamba": {"d_model": 64, "n_layers": 2, "d_state": 16, "d_conv": 4, "expand": 2, "dropout": 0.1,
              "dt_min": 1e-3, "dt_max": 1e-1, "dt_init_floor": 1e-4},

    "graph_grid": {
        "trained_with_arm": "A",
        "baseline": {"k_pos": 8, "k_neg": 4, "corr_window": 60},
        "variants": {
            "baseline": {"k_pos": 8,  "k_neg": 4, "corr_window": 60},
            "kpos12":   {"k_pos": 12, "k_neg": 4, "corr_window": 60},
            "kpos16":   {"k_pos": 16, "k_neg": 4, "corr_window": 60},
            "kneg6":    {"k_pos": 8,  "k_neg": 6, "corr_window": 60},
            "kneg8":    {"k_pos": 8,  "k_neg": 8, "corr_window": 60},
            "win40":    {"k_pos": 8,  "k_neg": 4, "corr_window": 40},
            "win90":    {"k_pos": 8,  "k_neg": 4, "corr_window": 90},
        },
        "selection_metric": "val_sel_sharpe_raw",
        "tiebreak_metric": "val_sel_mean_rank_ic", "tie_threshold": 0.02,
    },

    "postproc": {
        "ensemble": "cross-sectional z-score per (date, seed), then mean over the 5 seeds",
        "ema_half_life_days": [0, 2, 5],          # 0 = no smoothing
        "q_in": 0.20,                              # equals decile_fraction: leg size = round(0.2 * N)
        "q_out": [0.20, 0.30, 0.40],               # 0.20 = no buffer
        "selection_metric": "mean over arms P,A,B,C of val_sel strategy Sharpe",
        "tiebreak": "lower mean val_sel turnover if Sharpe within 0.05", "tie_threshold": 0.05,
    },

    "selection_4A": {"candidates": ["B", "C"], "metric": "val_sel strategy Sharpe (chosen postproc)",
                     "tiebreak_metric": "val_sel_mean_rank_ic (raw, per run)", "tie_threshold": 0.05},

    "holdout": {"arms": ["winner", "A", "P"], "seeds": [0, 1, 2, 3, 4]},

    "budget_drop_order": [
        "graph grid: non-baseline variants use seeds [0,1,2] only",
        "drop arm C (then winner = B)",
        "holdout seeds reduced to [0,1,2]",
        "fail fast",
    ],

    "phase4b": {
        "corr_window": 60,                         # follows the S1 graph winner's corr_window at runtime
        "target_rebuild_rates": [0.10, 0.20, 0.35, 0.50],
        "signals": ["corr_drift", "mamba_delta", "combined"],
        "max_staleness_days": 20,
        "full_recompute_every": 250,
        "val_noninferiority_margin_sharpe": 0.10,
        "test_noninferiority_margin_sharpe": 0.25,
        "scale_benchmark_n_stocks": [40, 100, 500, 1000, 2000],
        "scale_benchmark_windows": [60, 250],
        "scale_benchmark_days": 690,
    },

    "significance": {"bootstrap_iterations": 5000, "bootstrap_block_length": 10, "alpha": 0.05,
                     "secondary_correction": "holm"},

    "leakage_gate": {"max_abs_ic_in_standard_errors": 3.0},

    "budget": {"session_limit_hours": 12.0, "plan_limit_hours": 10.5,
               "expected_epochs_for_projection": 35, "safety_factor": 1.15,
               "phase4b_estimate_minutes": 45, "scoring_and_bench_estimate_minutes": 20},
}

CONFIG_CHANGES = [
    "phase: 4 -> 5; cache dataset dmrgt-sp100-cache (N read from universe.csv)",
    "arms: P added (Phase 3 schedule), D removed; arms carry a 'schedule' key",
    "folds 5-6 holdout added (never used by any earlier phase)",
    "graph_grid, postproc, holdout, budget_drop_order added; selection_4A rewritten (val_sel strategy Sharpe)",
    "Phase 4.1 F1: Mamba dt init (dt_min/dt_max/dt_init_floor) + two-group AdamW for every arm (P keeps Phase 3 Adam)",
    "Phase 4.1 F2: val split into val_es / val_sel; rolling-3 early stopping on val_es; no warmup checkpoints",
    "Phase 4.1 F3: random baseline draws fresh noise per (seed, day)",
    "Phase 4.1 C2/C3: direction-aware verdicts, A vs P test, extended compute benchmark (sizes/windows added to phase4b)",
]

# Static values from Phase 3 configs.yaml (not in the cache input) plus the holdout folds. Part of the snapshot hash.
STATIC = {
    "feature_columns": [
        "f1_logret",
        "f2_price_ma",
        "f3_rsi",
        "f4_macd",
        "f5_atr",
        "f6_vol_zscore",
        "f7_market_ret",
        "f8_vix"
    ],
    "folds": [
        {
            "name": "fold_1_2019_calm",
            "train_start": "2015-01-01",
            "train_end": "2018-12-31",
            "test_start": "2019-01-01",
            "test_end": "2019-12-31"
        },
        {
            "name": "fold_2_2020_covid",
            "train_start": "2015-01-01",
            "train_end": "2019-12-31",
            "test_start": "2020-01-01",
            "test_end": "2020-12-31"
        },
        {
            "name": "fold_3_2021",
            "train_start": "2015-01-01",
            "train_end": "2020-12-31",
            "test_start": "2021-01-01",
            "test_end": "2021-12-31"
        },
        {
            "name": "fold_4_2022_rates",
            "train_start": "2015-01-01",
            "train_end": "2021-12-31",
            "test_start": "2022-01-01",
            "test_end": "2022-12-31"
        },
        {
            "name": "fold_5_2023",
            "train_start": "2015-01-01",
            "train_end": "2022-12-31",
            "test_start": "2023-01-01",
            "test_end": "2023-12-31"
        },
        {
            "name": "fold_6_2024",
            "train_start": "2015-01-01",
            "train_end": "2023-12-31",
            "test_start": "2024-01-01",
            "test_end": "2024-12-31"
        }
    ],
    "source": "feature columns and folds 1-4 copied unchanged from Phase 3 configs.yaml; folds 5-6 are the Phase 5 holdout. Universe and sectors come from universe.csv in the cache (cell 2)."
}

SNAPSHOT = {"CONFIG": CONFIG, "STATIC": STATIC, "changes_from_phase4": CONFIG_CHANGES}
_snap_text = json.dumps(SNAPSHOT, indent=2, sort_keys=True)
(WORK / "config_snapshot.json").write_text(_snap_text)
CONFIG_SHA256 = hashlib.sha256(_snap_text.encode()).hexdigest()
print("config_snapshot.json sha256 (pre-registered):", CONFIG_SHA256)


def update_snapshot(**kw):
    """Add runtime facts (universe, graph winner, budget plan) to the snapshot FILE. The
    pre-registered hash above is of CONFIG + STATIC + changes only and never changes."""
    snap = json.loads((WORK / "config_snapshot.json").read_text())
    snap.update(kw)
    (WORK / "config_snapshot.json").write_text(json.dumps(snap, indent=2, sort_keys=True))
    return snap
