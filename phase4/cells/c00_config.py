import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"       # before CUDA initialises (workers set it too)

import hashlib, json, time
from pathlib import Path

WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)
(WORK / "dmrgt4").mkdir(parents=True, exist_ok=True)   # %%writefile cells below do not create folders
RUN_T0 = time.time()

# ─── Pre-registered configuration (spec §4) — fixed BEFORE any data is touched ───
CONFIG = {
    "phase": "4",
    "seeds": [0, 1, 2, 3, 4],
    "folds": "identical to Phase 3 (fold_1_2019_calm, fold_2_2020_covid, fold_3_2021, fold_4_2022_rates)",
    "graph": {"k_pos": 8, "k_neg": 4, "corr_window": 60, "file": "graphs_baseline.parquet"},
    "features": 8,
    "backtest": {"commission_bps": 10, "slippage_bps": 5, "decile_fraction": 0.2},

    "arms": {
        "A": {"encoder": "transformer", "lookback": 30, "adaptive_huber": False, "required": True},
        "B": {"encoder": "mamba",       "lookback": 30, "adaptive_huber": False, "required": True},
        "C": {"encoder": "mamba",       "lookback": 60, "adaptive_huber": False, "required": True},
        "D": {"encoder": "mamba",       "lookback": 60, "adaptive_huber": True,  "required": False},
    },

    "training": {
        "max_epochs": 50, "early_stop_patience": 10, "early_stop_metric": "val_mean_rank_ic",
        "batch_size_ladder": [32, 16, 8],          # same batch size for ALL arms
        "optimizer": "AdamW", "lr": 1e-3, "weight_decay": 1e-4,
        "warmup_epochs": 3, "warmup_start_lr": 1e-4,
        "plateau": {"mode": "max", "factor": 0.5, "patience": 4, "min_lr": 1e-5},
        "grad_clip_norm": 1.0,
        "huber_delta_fixed": 1.0,
        "adaptive_huber": {"c": 1.345, "min": 0.05, "max": 2.0},
        "retry_failed_job_once": True,
    },

    "mamba": {"d_model": 64, "n_layers": 2, "d_state": 16, "d_conv": 4, "expand": 2, "dropout": 0.1},

    "selection_4A": {"metric": "val_sharpe", "tiebreak_metric": "val_mean_rank_ic", "tie_threshold": 0.02,
                     "candidates": ["B", "C", "D"]},

    "phase4b": {
        "corr_window": 60,
        "target_rebuild_rates": [0.10, 0.20, 0.35, 0.50],
        "signals": ["corr_drift", "mamba_delta", "combined"],
        "max_staleness_days": 20,
        "full_recompute_every": 250,
        "val_noninferiority_margin_sharpe": 0.10,
        "test_noninferiority_margin_sharpe": 0.25,
        "scale_benchmark_n_stocks": [40, 100, 500],
    },

    "significance": {"bootstrap_iterations": 5000, "bootstrap_block_length": 10, "alpha": 0.05,
                     "secondary_correction": "holm"},

    "leakage_gate": {"max_abs_ic_in_standard_errors": 3.0},

    "budget": {"session_limit_hours": 12.0, "plan_limit_hours": 10.5,
               "expected_epochs_for_projection": 35, "safety_factor": 1.15,
               "phase4b_estimate_minutes": 45, "scoring_and_bench_estimate_minutes": 20},
}

# Static Phase 3 values that live in configs.yaml (not in the cache input), copied
# unchanged: universe, sectors, feature columns, walk-forward folds. Part of the snapshot hash.
STATIC = {
    "universe": [
        "AAPL",
        "MSFT",
        "NVDA",
        "AVGO",
        "CRM",
        "JPM",
        "BRK-B",
        "GS",
        "BAC",
        "V",
        "JNJ",
        "PFE",
        "UNH",
        "ABBV",
        "MRK",
        "XOM",
        "CVX",
        "COP",
        "SLB",
        "EOG",
        "AMZN",
        "HD",
        "NKE",
        "TSLA",
        "SBUX",
        "PG",
        "KO",
        "WMT",
        "COST",
        "PEP",
        "HON",
        "CAT",
        "BA",
        "UPS",
        "UNP",
        "NEE",
        "DUK",
        "SO",
        "AEP",
        "EXC"
    ],
    "sectors": {
        "AAPL": "Information Technology",
        "MSFT": "Information Technology",
        "NVDA": "Information Technology",
        "AVGO": "Information Technology",
        "CRM": "Information Technology",
        "JPM": "Financials",
        "BRK-B": "Financials",
        "GS": "Financials",
        "BAC": "Financials",
        "V": "Financials",
        "JNJ": "Health Care",
        "PFE": "Health Care",
        "UNH": "Health Care",
        "ABBV": "Health Care",
        "MRK": "Health Care",
        "XOM": "Energy",
        "CVX": "Energy",
        "COP": "Energy",
        "SLB": "Energy",
        "EOG": "Energy",
        "AMZN": "Consumer Discretionary",
        "HD": "Consumer Discretionary",
        "NKE": "Consumer Discretionary",
        "TSLA": "Consumer Discretionary",
        "SBUX": "Consumer Discretionary",
        "PG": "Consumer Staples",
        "KO": "Consumer Staples",
        "WMT": "Consumer Staples",
        "COST": "Consumer Staples",
        "PEP": "Consumer Staples",
        "HON": "Industrials",
        "CAT": "Industrials",
        "BA": "Industrials",
        "UPS": "Industrials",
        "UNP": "Industrials",
        "NEE": "Utilities",
        "DUK": "Utilities",
        "SO": "Utilities",
        "AEP": "Utilities",
        "EXC": "Utilities"
    },
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
        }
    ],
    "source": "values copied from Phase 3 configs.yaml (universe, sectors, features.columns, folds); not in the cache input"
}

SNAPSHOT = {"CONFIG": CONFIG, "STATIC": STATIC}
_snap_text = json.dumps(SNAPSHOT, indent=2, sort_keys=True)
(WORK / "config_snapshot.json").write_text(_snap_text)
CONFIG_SHA256 = hashlib.sha256(_snap_text.encode()).hexdigest()
print("config_snapshot.json sha256:", CONFIG_SHA256)
