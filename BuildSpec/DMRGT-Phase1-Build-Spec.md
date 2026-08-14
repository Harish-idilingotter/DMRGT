# DMRGT — Phase 1 Build Specification

> **For the implementing agent:** This is a complete, ordered build brief for Phase 1 of an academic research project. Follow the steps **in the order given** — the ordering is deliberate and protects against a specific failure mode (data leakage). Do not skip ahead to the model. Do not build the graph in this phase.

---

## 0. Context

**Project:** DMRGT (Dynamic Multi-Relational Graph Transformer) — an academic final-year project that predicts which stocks will outperform their peer group on the next trading day.

**Phase 1 goal:** Build a complete, working, honestly-measured pipeline **with no graph in it yet**. The graph, the R-GCN, the 40-stock universe, significance testing and explainability are all Phase 2+ and are explicitly out of scope here.

**Phase 1 completion criterion:**
> The pipeline runs end-to-end with no future leakage, and transaction costs + slippage are active from day one.

**Duration:** ~2 weeks (1–14 July 2026).

**Execution environment:** The developer has no local GPU. Model training runs on **Kaggle's free GPU** (P100/T4, 16 GB, ~30 hrs/week). Everything else — data acquisition, feature engineering, evaluation, backtesting — runs locally on CPU. The pipeline must therefore be built with a clean local/Kaggle split. See §2.5 for the required workflow. This constraint does **not** change any step or its ordering; it changes where two of the steps physically execute.

**Critical framing:** This phase is about building a trustworthy *measuring instrument*, not a good model. A mediocre model measured honestly is a success. A great-looking model with a leak is a total failure that invalidates every later phase.

---

## 1. Non-Negotiable Rules

These override any other consideration. Violating them silently produces beautiful, meaningless results.

1. **Never fit a scaler on the full dataset.** Scalers are fitted on the training window of each fold only, then applied to validation/test. Scaling happens inside `train.py`, never in the cached Parquet.
2. **Never use a random train/test split.** Splits are always chronological — train on older dates, test on strictly later dates.
3. **The target must use `t+1`.** Features come from day `t` and earlier; the label is day `t+1`. Off-by-one here is the single most common way this project could fail.
4. **Correlation and rolling windows must end at day `t`.** No window may include data from `t+1` or later. Use explicit `.shift(1)` where needed and verify manually.
5. **Transaction costs and slippage are active from the first backtest run.** Never report a cost-free number, not even temporarily.
6. **20 stocks in this phase**, not 40. Chosen stratified across GICS sectors.
7. **Pin all random seeds.** Every run must be reproducible.
8. **No module may assume it is running locally.** All paths resolve through a single `paths.py` helper that detects the environment. A module that hardcodes `cache/prices.parquet` will break on Kaggle.
9. **Never download data from a Kaggle training notebook.** Data enters the project once, locally, and is uploaded as a versioned Kaggle Dataset. A notebook that calls `yfinance` breaks reproducibility, since the data could differ between runs.

---

## 2. Environment Setup

```
Python 3.11+
```

**Dependencies:**
```
pandas
numpy
pyarrow
yfinance
pandas-ta
torch
scipy
vectorbt
mlflow
matplotlib
pytest
kaggle          # dataset upload/versioning CLI
```

**Note on Kaggle:** most of these (torch, pandas, numpy, scipy, matplotlib) are preinstalled in Kaggle's GPU image. Only `vectorbt` and `pandas-ta` typically need installing there — and neither is needed in the training notebook, since backtesting runs locally. Keep the notebook's `pip install` list minimal; installs eat session time.

**Repository structure to create:**

```
dmrgt/
├── configs.yaml           # all hyperparameters, tickers, costs, seeds
├── paths.py               # LOCAL/KAGGLE path resolution — import everywhere
├── requirements.txt
├── README.md
├── data.py                # Step 1 + Step 2      [LOCAL]
├── features.py            # Step 4                [LOCAL]
├── splits.py              # walk-forward folds    [shared]
├── baselines.py           # Step 3 — random, persistence  [LOCAL]
├── eval.py                # Step 3 — metrics + cost-aware backtest [LOCAL]
├── model.py               # Step 5 — Transformer encoder  [shared]
├── train.py               # Step 5 — training loop        [KAGGLE]
├── cache/                 # Parquet outputs (gitignored) — uploaded to Kaggle
│   └── .gitkeep
├── artifacts/             # downloaded Kaggle outputs (gitignored)
│   └── .gitkeep
├── notebooks/
│   ├── verification.ipynb    # manual hand-checks           [LOCAL]
│   └── kaggle_train.ipynb    # thin training wrapper        [KAGGLE]
└── tests/
    ├── test_target.py
    ├── test_features.py
    ├── test_leakage.py
    └── test_paths.py         # asserts paths.py resolves correctly in both envs
```

> **AS-BUILT NOTE (added after the build):** The structure above is
> the original brief. The actual layout in the repo at the end of
> Phase 1 differs in three places — all of which are noted here so
> the next agent has an accurate map of the code:
>
> 1. **`features.py` is a top-level module, not folded into `data.py`.**
>    `data.py` exposes a `--features` CLI subcommand that calls
>    `features.build_features()`, but the feature logic itself lives
>    in its own file. This matches the original spec at the top of
>    §6 ("File: `features.py`, called from `data.py`").
> 2. **A `compare.py` driver exists at the top level** for Step 6
>    (the §8 results table). It explicitly routes random, persistence,
>    and Transformer predictions through the same `eval.py` code path
>    so the comparison is honest. Spec put Step 6 inside `eval.py`;
>    in practice a small CLI driver next to `eval.py` reads better.
>    `eval.py` also exposes `evaluate_predictions()` as the entry
>    point for the predictions.parquet → metrics path.
> 3. **A `kaggle/` directory sits at the top level** with
>    `kaggle/KAGGLE.md`, `kaggle/kaggle_train.ipynb` (the thin wrapper
>    notebook), and `kaggle/code/` (the uploadable `dmrgt-code`
>    dataset). The notebook lives inside the repo rather than at
>    `notebooks/kaggle_train.ipynb` so the Kaggle artifacts are
>    grouped together.
>
> An extra `tests/` file exists beyond the four in the spec:
> `test_idempotency.py` (4 tests pinning `data.py` cache-hit
> behaviour) and `test_universe.py` (7 tests pinning the sector
> stratification). Both grew from §9 DoD gaps.

**`configs.yaml` — create this first**, and have every module read from it. Nothing hardcoded in scripts.

```yaml
universe:
  tickers: [...]           # 20, stratified by sector — see §3
  start_date: "2015-01-01"
  end_date: "2024-12-31"

features:
  lookback_days: 30
  rsi_period: 14
  atr_period: 14
  ma_period: 10
  volume_zscore_window: 60

training:
  seeds: [0, 1, 2, 3, 4]
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
  huber_delta: 1.0
  early_stopping_patience: 10

backtest:
  commission_bps: 10       # 0.10% per trade
  slippage_bps: 5          # 0.05%
  decile_fraction: 0.2     # top/bottom 20% for 20 stocks = 4 each

folds:
  # train_end / test_start / test_end per fold — see §7
```

---

## 2.5 Local / Kaggle Execution Split

### What runs where

| Step | Environment | Reason |
|---|---|---|
| 1 — Data acquisition | **Local** | Needs internet for yfinance; runs in seconds on CPU |
| 2 — Target construction | **Local** | Pure pandas; must be hand-verified interactively |
| 3 — Eval harness + random-model gate | **Local** | CPU only; the gate must pass before any GPU time is spent |
| 4 — Feature engineering | **Local** | CPU only; requires plotting and visual inspection |
| 5 — Transformer training | **Kaggle GPU** | The only step that needs a GPU |
| 6 — Final comparison | **Local** | Runs the eval harness over downloaded predictions |

**The key architectural consequence:** training must be decoupled from evaluation. Kaggle produces *predictions*; local code turns predictions into *metrics*. Do not run the backtest on Kaggle.

### `paths.py` — environment detection

Create this before anything else. Every module imports from it; no module builds a path by hand.

```python
import os
from pathlib import Path

def is_kaggle() -> bool:
    return os.path.exists("/kaggle/input")

if is_kaggle():
    CACHE_DIR   = Path("/kaggle/input/dmrgt-cache")   # read-only
    OUTPUT_DIR  = Path("/kaggle/working")             # writable
else:
    ROOT        = Path(__file__).parent
    CACHE_DIR   = ROOT / "cache"
    OUTPUT_DIR  = ROOT / "artifacts"

CACHE_DIR.mkdir(parents=True, exist_ok=True) if not is_kaggle() else None
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRICES   = CACHE_DIR / "prices.parquet"
FEATURES = CACHE_DIR / "features.parquet"
TARGETS  = CACHE_DIR / "targets.parquet"
```

**Critical:** `/kaggle/input/` is **read-only**. Any attempt to write there fails. All Kaggle outputs go to `/kaggle/working/`.

### The handoff cycle

```
LOCAL                          KAGGLE                        LOCAL
─────                          ──────                        ─────
data.py     ──┐
features.py   ├─▶ cache/*.parquet
targets       ─┘        │
                        │ upload as Kaggle Dataset
                        ▼
                   dmrgt-cache  ──▶  train notebook
                                          │
                                     trains 4 folds × 5 seeds
                                          │
                                          ▼
                                   predictions.parquet
                                   + metrics.json
                                          │
                                     download ──────────────▶ eval.py
                                                              backtest
                                                              results table
```

### Uploading the cache to Kaggle

After Step 4 completes locally, create a Kaggle Dataset containing:

```
dmrgt-cache/
├── prices.parquet
├── features.parquet
├── targets.parquet
└── hashes.json          # all three SHA-256 hashes
```

Total size will be under ~10 MB — trivial for Kaggle.

Two ways to upload; document both in the README:

**Option A — Kaggle web UI:** Datasets → New Dataset → drag the three files → name it `dmrgt-cache` → set to Private.

**Option B — Kaggle API (preferred, scriptable):**
```bash
pip install kaggle
# place kaggle.json in ~/.kaggle/ with 600 permissions
kaggle datasets init -p cache/
# edit cache/dataset-metadata.json: set title and id
kaggle datasets create -p cache/
# for later updates:
kaggle datasets version -p cache/ -m "features v2"
```

**Version the dataset rather than overwriting it.** When features change, push a new version and record which version each training run used. This is part of the reproducibility guarantee.

### The training notebook contract

The Kaggle notebook is a thin wrapper. All real logic lives in `model.py` and `train.py`, which are uploaded alongside the data (as a second Kaggle Dataset, `dmrgt-code`, or pasted into a notebook cell).

Notebook responsibilities, in order:

1. Attach the `dmrgt-cache` dataset
2. `pip install` anything missing (most deps are preinstalled)
3. **Verify data integrity** — recompute the SHA-256 of each parquet and assert it matches `hashes.json`. If a hash mismatches, halt. This catches a stale dataset version silently producing wrong results.
4. Assert `torch.cuda.is_available()` and print the GPU name
5. Run all folds × all seeds
6. Write outputs to `/kaggle/working/`
7. Print a compact summary table so results are visible without downloading

**Required outputs written to `/kaggle/working/`:**

| File | Contents |
|---|---|
| `predictions.parquet` | `fold`, `seed`, `date`, `ticker`, `y_pred`, `y_true` — one row per stock per test date per seed |
| `train_metrics.json` | per-epoch train/val loss, per fold and seed |
| `run_manifest.json` | dataset version, git commit, config snapshot, torch version, GPU model, timestamps |
| `checkpoints/` | one `.pt` per fold per seed |

`predictions.parquet` is the critical artifact. Everything downstream — rank IC, backtest, the comparison table — is computed locally from it.

### Kaggle session constraints

| Constraint | Value | Mitigation |
|---|---|---|
| Max session runtime | ~9 hours (12 for some accounts) | Phase 1 needs minutes, not hours — no issue |
| GPU quota | ~30 hrs/week | Ample; a fold × seed should take 1–3 minutes |
| Idle timeout (interactive) | ~20–40 min | Use **Save Version → Save & Run All (Commit)** for real runs, which executes headlessly |
| Session reset | Wipes `/kaggle/working/` | Always commit; download outputs immediately after |
| Internet | Off by default | Keep it off — data must come from the attached Dataset, not a download |

**Always use "Save & Run All (Commit)" for the runs that produce results**, not interactive execution. Interactive sessions die on idle and lose `/kaggle/working/`. Committed runs execute the whole notebook top to bottom and persist their output, which also makes the run reproducible and citable.

### MLflow on Kaggle

MLflow's local file store writes to disk, which vanishes when the session ends. Two workable options:

- **Simplest (recommended for Phase 1):** skip MLflow inside the notebook. Write `train_metrics.json` and `run_manifest.json` instead, download them, and log to local MLflow from `eval.py`. Same audit trail, no server.
- **Alternative:** set `mlflow.set_tracking_uri("file:/kaggle/working/mlruns")` and download the `mlruns/` directory with the other outputs.

Do not attempt to run an MLflow server on Kaggle.

### Verifying the two environments agree

Before trusting any Kaggle result, run one fold with one seed **on CPU locally** (it will be slow, but a 20-stock Phase 1 model is small enough to finish) and compare against the same fold and seed on Kaggle.

Losses will not match bit-for-bit — CUDA and CPU kernels differ, and cuDNN introduces nondeterminism. What must match:

- Identical number of training samples per fold
- Identical train/test date boundaries
- Final validation loss within a few percent
- Rank IC of the same sign and rough magnitude

A large divergence means something environment-dependent leaked into the pipeline — most often a path resolving differently, or a stale dataset version.

For maximum determinism on Kaggle, set:
```python
torch.manual_seed(seed); np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

---

## 3. Step 1 — Data Acquisition and Caching

**File:** `data.py`

**Task:** Download daily OHLCV for 20 tickers plus two global series, align them to a single common trading calendar, cache to Parquet, record a snapshot hash.

**Ticker selection:** Choose 20 liquid S&P 100 constituents **stratified across GICS sectors** — roughly 2–3 per sector across Technology, Financials, Healthcare, Energy, Consumer Discretionary, Consumer Staples, Industrials, Utilities. Do not take the first 20 alphabetically; sector diversity matters because Phase 2 will add sector edges and negative-correlation edges, both of which need cross-sector variety to be meaningful.

Record the chosen tickers and their sectors in `configs.yaml` with a comment explaining the stratification.

**Global series (same value applied to every stock on a given day):**
- `^GSPC` — S&P 500 index, for market return
- `^VIX` — volatility index

**Implementation requirements:**

1. Download each ticker via `yfinance` with `auto_adjust=True`.
2. **Calendar alignment is critical.** Build a master trading-day index (the intersection of dates where all tickers have data). Reindex every ticker to it.
3. **Validate before proceeding.** Assert:
   - All tickers have identical date indices
   - Zero NaNs in the aligned OHLCV data
   - No duplicate dates
   - Date range covers the configured window
   - Row count is plausible (~250 trading days per year)
   - Print a summary table: ticker, sector, first date, last date, row count, NaN count
4. Save to `cache/prices.parquet` in long format:
   `date | ticker | open | high | low | close | volume`
5. Compute a SHA-256 hash of the saved file and write it to `cache/prices.sha256`. Print it.
6. Make the download idempotent — if `cache/prices.parquet` exists and its hash matches, skip the download and load from cache.

**Do not compute any features yet.** This step ends with clean, aligned, cached raw prices and nothing more.

**Deliverable check:** Re-running `python data.py` twice produces the identical hash both times.

---

## 4. Step 2 — Build and Hand-Verify the Target

**File:** `data.py` (continued)

**This step comes before features. Do not reorder it.** Features can be corrected later at low cost; a wrong target invalidates everything downstream.

**Definition:**

```
daily_return[stock, t]  = close[stock, t] / close[stock, t-1] - 1

excess_return[stock, t] = daily_return[stock, t+1]
                          - mean( daily_return[all stocks, t+1] )
```

The excess return stored against date `t` is the **label for a prediction made on day `t`** using data up to and including day `t`. The value itself describes what happened on `t+1`.

**Implementation requirements:**

1. Compute daily returns per ticker.
2. Compute the equal-weight cross-sectional mean return per date.
3. Compute excess return, then shift so the label aligns to the prediction date `t`.
4. The final row (last date) will have no label — drop it explicitly and log that you did.
5. Save to `cache/targets.parquet`: `date | ticker | excess_return`

**Mandatory verification — implement these as tests in `tests/test_target.py`:**

| Check | Expected result |
|---|---|
| Cross-sectional sum | For every date, excess returns across all 20 stocks sum to ≈ 0 (abs < 1e-10) |
| Cross-sectional mean | ≈ 0 for every date |
| Manual spot-check | Pick 3 random (date, ticker) pairs, compute by hand from raw closes, assert match to 1e-9 |
| Alignment | The label at date `t` must equal the excess return realised on the *next* trading day. Verify against explicit index lookups, not by re-running the same shift logic. |
| No lookahead | Label at the final date is dropped, not filled |

**Also produce, in `notebooks/verification.ipynb`:** a printed table showing, for one chosen date, each ticker's `close[t]`, `close[t+1]`, raw return, universe mean, and excess return — so a human can read across the row and confirm the arithmetic by eye.

**Do not proceed to Step 3 until every check above passes.**

---

## 5. Step 3 — Evaluation Harness (before any real model)

**Files:** `splits.py`, `baselines.py`, `eval.py`

This is the step that most protects the project, and the one most likely to be skipped. Build the measuring instrument, then verify it reads zero when fed nothing.

### 5.1 `splits.py` — walk-forward folds

Generate chronological folds with an **expanding** training window:

```
Fold 1: train 2015-01-01 → 2018-12-31 | test 2019 (calm)
Fold 2: train 2015-01-01 → 2019-12-31 | test 2020 (COVID crash)
Fold 3: train 2015-01-01 → 2020-12-31 | test 2021
Fold 4: train 2015-01-01 → 2021-12-31 | test 2022 (rate shock)
```

Requirements:
- Return fold objects with explicit `train_dates`, `test_dates` arrays
- Assert `max(train_dates) < min(test_dates)` for every fold
- Assert zero overlap between train and test sets
- Reserve the final ~15% of each training window as validation for early stopping (also chronological — the latest portion, not random)

### 5.2 `baselines.py` — the random model and persistence

```python
def random_model(X, seed=None):
    """Ignores all input. Returns pure noise.
    Shape in:  [n_stocks, lookback, n_features]
    Shape out: [n_stocks]
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal(X.shape[0])


def persistence_model(X, feature_idx_logret=0):
    """Predicts that tomorrow's excess return equals today's.
    Uses only the most recent day's log-return feature.
    """
    today_return = X[:, -1, feature_idx_logret]
    return today_return - today_return.mean()
```

Both must accept and return the same shapes the real model will, so they drop into the pipeline unchanged.

### 5.3 `eval.py` — metrics and cost-aware backtest

**Prediction metrics** (computed per test day, then aggregated per fold):
- **Rank IC** — Spearman correlation between predicted and actual excess returns across the cross-section. This is the primary metric.
- **Directional accuracy** — fraction of stocks where `sign(pred) == sign(actual)`
- **Huber loss** and **MSE** on the raw values

**Backtest** (via `vectorbt`):
- Each test day: long the top decile, short the bottom decile, equal-weighted within each leg
- With 20 stocks and `decile_fraction: 0.2`, that is 4 long and 4 short
- Apply `commission_bps` and `slippage_bps` from config to every position change
- Report: Sharpe ratio, max drawdown, turnover, cumulative return
- **Turnover must be reported.** It is the metric that exposes whether a daily-rebalancing strategy is quietly being destroyed by costs.

### 5.4 The calibration check — MANDATORY GATE

Run `random_model` through the complete harness across all folds. Assert:

| Metric | Required result |
|---|---|
| Mean rank IC | Within ±0.02 of 0.00 |
| Directional accuracy | Within ±2% of 50% |
| Backtest return after costs | Negative (you are paying fees to trade noise) |

Implement this as `tests/test_leakage.py` and make it part of the standard test run.

**If a random model scores meaningfully above zero, STOP.** Do not build the real model. There is leakage in the pipeline. Investigate in this order:

1. Is the target using `t+1` or accidentally `t`?
2. Does any rolling window (correlation, moving average, z-score) include the prediction date's own future data?
3. Is any scaler or normalisation fitted outside the training window?
4. Do train and test date ranges overlap anywhere?
5. Is the backtest using the close of day `t+1` to *decide* the trade rather than to *settle* it?

**Also run `persistence_model`** through the harness and record its numbers. It becomes your floor — the real model must beat it to be worth anything.

**Deliverable check:** `pytest tests/test_leakage.py` passes, and you have a printed table of random + persistence results across all four folds.

---

## 6. Step 4 — Feature Engineering

**File:** `features.py`, called from `data.py`

Only now build the 8 features. Build them **one at a time**, and after each one, plot its distribution and inspect it before moving to the next.

| # | Feature | Computation | Sanity check |
|---|---|---|---|
| 1 | Log-return | `log(close[t] / close[t-1])` | Centred near 0, fat tails, typically within ±0.15 |
| 2 | Price / 10-day MA | `close[t] / rolling_mean(close, 10)[t]` | Centred near 1.0, range roughly 0.8–1.2 |
| 3 | RSI (14) | `pandas-ta` | Strictly bounded 0–100, mostly 30–70 |
| 4 | MACD | `pandas-ta` | Centred near 0 |
| 5 | ATR (14) | `pandas-ta` | Always positive; scale it by close price to make it comparable across tickers |
| 6 | Volume z-score | `(vol - rolling_mean(vol,60)) / rolling_std(vol,60)` | Centred near 0, right-skewed |
| 7 | Market return | Log-return of `^GSPC` | **Identical value for all 20 tickers on a given date** |
| 8 | VIX | VIX close level | **Identical value for all 20 tickers on a given date** |

**Critical requirements:**

1. **Every rolling window must end at day `t`.** After computing each feature, explicitly verify that the value at date `t` uses no data from `t+1` onward. Write a test that corrupts all data after a chosen date `t` with `NaN` and confirms the feature at `t` is unchanged.
2. **Do not scale here.** The cache stores raw, unscaled features. Scaling is fitted per fold inside `train.py`.
3. **Handle the warm-up period.** The first ~60 rows will have NaNs from rolling windows. Drop them from the usable date range and log how many rows were dropped.
4. Save to `cache/features.parquet`: `date | ticker | f1 ... f8`
5. Record a hash of this file too.

**Plots to produce** in `notebooks/verification.ipynb`: an 8-panel histogram grid, plus a time-series plot of each feature for one ticker, so anomalies are visible.

**Tests in `tests/test_features.py`:**
- RSI bounded [0, 100]
- ATR strictly positive
- Features 7 and 8 identical across all tickers per date
- The NaN-corruption lookahead test described above, run against all 8 features
- No infinities anywhere

---

## 7. Step 5 — Transformer Model and Training

**Files:** `model.py`, `train.py`

**No graph. No R-GCN. Temporal encoder only.**

### 7.1 `model.py`

```
Input:  X = [n_stocks, 30, 8]
  → Linear projection: 8 → 64                     [n_stocks, 30, 64]
  → Add sinusoidal positional encoding            [n_stocks, 30, 64]
  → TransformerEncoder: 2 layers, 4 heads,
    causal mask, dropout 0.1                      [n_stocks, 30, 64]
  → Take final time step                          [n_stocks, 64]
  → MLP head: 64 → 32 → 1                         [n_stocks, 1]
Output: [n_stocks] predicted excess returns
```

Requirements:
- **One shared Transformer** applied across all stocks — same weights for every ticker. Do not build per-stock models.
- **Causal mask is mandatory**, even though only the final time step is used. It is the architectural leakage guard.
- Positional encoding is required — attention is order-blind without it.
- Keep the interface `forward(X)` so `baselines.py` models remain drop-in compatible.

### 7.2 `train.py` — runs on Kaggle GPU

**This module executes on Kaggle.** It must not download data, must not run a backtest, and must not write outside `/kaggle/working/`. Its single job is: load cached data, train, emit predictions.

**Per-fold procedure:**

1. Load cached features and targets for this fold's date range **via `paths.py`** — never a hardcoded path.
2. **Fit the scaler on training dates only** (`StandardScaler` per feature column), then transform train, validation and test with those frozen parameters. Save the fitted mean/std into the run manifest so it can be inspected locally.
3. Assemble samples: **one sample = one trading day** containing all 20 stocks at once. Shape `[20, 30, 8]` with label `[20]`.
4. Move the model and batches to `cuda` if available; the code must still run on CPU so a local verification run is possible.
5. Train with **Huber loss** on excess return. Adam optimiser, learning rate from config.
6. Early stopping on validation loss with configured patience.
7. Repeat for all 5 seeds in config.
8. Append test-set predictions to the running results table.

**Assertions to include in the training loop:**
- Assert no test date appears in the training set
- Assert the scaler was fitted before any test data was touched
- Assert input tensor shape is exactly `[n_stocks, 30, 8]` at every batch
- Assert the loaded parquet hashes match `hashes.json` before training starts

**Output contract — this is what makes the local/Kaggle split work:**

`train.py` writes `predictions.parquet` to `OUTPUT_DIR` with exactly these columns:

```
fold | seed | date | ticker | y_pred | y_true
```

One row per stock, per test date, per seed, per fold. With 20 stocks × ~250 test days × 4 folds × 5 seeds that is roughly 100,000 rows — a small file.

It also writes `train_metrics.json` (per-epoch losses), `run_manifest.json` (dataset version, config snapshot, library versions, GPU model, timestamps) and `checkpoints/`.

**`train.py` computes no rank IC, no Sharpe, and no backtest.** Those all belong to `eval.py`, which runs locally over the downloaded `predictions.parquet`. Keeping this boundary clean means you can re-run the entire evaluation — try different cost assumptions, different decile fractions — without touching the GPU or burning quota.

### 7.3 `notebooks/kaggle_train.ipynb`

A thin wrapper, no logic of its own:

```python
# Cell 1 — environment check
import torch, os
assert os.path.exists("/kaggle/input/dmrgt-cache"), "Attach the dmrgt-cache dataset"
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
assert torch.cuda.is_available(), "Enable GPU: Settings -> Accelerator -> GPU"

# Cell 2 — integrity check (halt on mismatch)
#   recompute SHA-256 of each parquet, assert against hashes.json

# Cell 3 — import model.py / train.py (from dmrgt-code dataset or pasted)

# Cell 4 — run all folds x seeds

# Cell 5 — write outputs to /kaggle/working/, print summary table
```

Run it with **Save Version → Save & Run All (Commit)**, then download the output files from the committed version's Output tab.

---

## 8. Step 6 — Comparison and Phase 1 Sign-Off

**Runs locally.** Download `predictions.parquet` from the committed Kaggle run into `artifacts/`, then run the complete harness across all four folds for:

1. `random_model` (must still score ≈ 0 — re-verify after all the changes)
2. `persistence_model`
3. The trained Transformer, read from `artifacts/predictions.parquet` (mean and std across 5 seeds)

All three go through the *same* `eval.py` code path. The Transformer is not special-cased — it is simply a source of predictions, exactly like the baselines. This is what makes the comparison honest: if the backtest is generous, it is generous to everyone equally.

**Produce a results table:**

| Model | Mean rank IC | Directional acc. | Sharpe (after costs) | Max drawdown | Turnover |
|---|---|---|---|---|---|
| Random | ~0.00 | ~50% | negative | — | — |
| Persistence | | | | | |
| Transformer | | | | | |

Report mean ± std across seeds for the Transformer. Report per-fold results as well as aggregate, since a model that wins only in the calm fold is a different story from one that wins across regimes.

**Do not tune hyperparameters to make this table look better.** Phase 1 is about establishing an honest baseline, not maximising a number. If the Transformer barely beats persistence, that is a legitimate and reportable Phase 1 outcome — the graph is what is expected to add value in Phase 2.

---

## 9. Phase 1 Definition of Done

```
□ configs.yaml drives every module; nothing hardcoded
□ paths.py resolves correctly in BOTH local and Kaggle environments
□ 20 sector-stratified tickers downloaded, aligned, cached, hashed
□ Re-running data.py produces an identical hash
□ Target verified: cross-sectional sum ≈ 0 on every date
□ Target verified: 3 manual hand-computations match to 1e-9
□ Walk-forward folds generated; zero train/test overlap asserted
□ Random model scores rank IC within ±0.02 of zero  ← THE GATE
□ Random model loses money after costs
□ Persistence baseline recorded across all folds
□ All 8 features built, plotted and inspected individually
□ NaN-corruption lookahead test passes for all 8 features
□ Features cached UNSCALED
□ Scaler fitted per fold on training data only
□ cache/ uploaded to Kaggle as a VERSIONED dataset; version recorded
□ Kaggle notebook verifies parquet hashes before training; halts on mismatch
□ Kaggle notebook asserts GPU is available
□ Transformer trains; one shared model across stocks; causal mask present
□ 5 seeds per fold; train.py emits predictions.parquet, not metrics
□ Kaggle run executed via Save & Run All (Commit), not interactively
□ predictions.parquet + run_manifest.json downloaded to artifacts/
□ One fold verified locally on CPU; agrees with Kaggle within tolerance
□ Backtest includes commission AND slippage; turnover reported
□ Results table complete: random vs persistence vs Transformer
□ All tests in tests/ pass
□ README documents how to reproduce with one command
```

---

## 10. Explicitly Out of Scope for Phase 1

Do not build any of these now. They are Phase 2 and later, and building them early makes leakage harder to isolate.

- `graph.py`, edge construction, correlation matrices
- R-GCN, relation-specific weights, the `[H, Z]` concatenation
- Expansion to 40 stocks
- LSTM baseline
- Frozen-graph vs dynamic-graph comparison
- Wilcoxon and stationary-bootstrap significance tests
- `explain.py`, attention maps, Gradient × Input, edge ablation
- Sensitivity grids over k or correlation window

---

## 11. Reference Facts (do not alter)

- **Phase 1 universe:** 20 stocks, sector-stratified from the S&P 100 (expands to ~40 in Phase 2)
- **Input shape:** `[n_stocks, 30, 8]` — one sample is one trading day covering all stocks simultaneously
- **The 8 features:** log-return, price ÷ 10-day MA, RSI(14), MACD, ATR(14), volume z-score, market return, VIX
- **Target:** next-day return minus the equal-weight universe mean
- **Loss:** Huber (MSE retained as an ablation)
- **Transformer:** 2 layers, 4 heads, causal mask, 64-dim output
- **Horizon:** 1 trading day only
- **Retraining:** once per walk-forward fold, never daily
- **Seeds:** ≥ 5 per configuration
- **Costs:** commission and slippage active from the first backtest run

---

*Research project for academic purposes only — not investment advice.*
