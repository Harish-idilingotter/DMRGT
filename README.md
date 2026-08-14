# DMRGT — Dynamic Multi-Relational Graph Transformer (Phase 2)

Phase 2 of an academic final-year project. Builds on Phase 1's leak-free
pipeline by adding a **dynamic three-relation graph** (+corr / −corr /
sector) with an R-GCN encoder over 40 sector-stratified S&P 100 stocks.
DMRGT beats both the no-graph Transformer and the LSTM baseline on rank IC.

Full design lives in [`BuildSpec/DMRGT-Phase2-Build-Spec.md`](./BuildSpec/DMRGT-Phase2-Build-Spec.md).
Phase 2 results are in [`docs/PHASE2_RESULTS.md`](./docs/PHASE2_RESULTS.md).

---

## 1. Quick start

### 1.1 Prerequisites

- **Python 3.11 or 3.13** (3.14 is too new — `numba`, a transitive dep
  of `pandas-ta`/`vectorbt`, doesn't yet support it).
- `git`, `pip`, ~3 GB free disk for the venv + Parquet caches + artifacts.

### 1.2 Clone and set up

```bash
git clone https://github.com/<your-org>/DMRGT.git
cd DMRGT
```

### 1.3 Create the virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate                  # zsh / bash on macOS / Linux
# On Windows: .venv\Scripts\activate
```

You should see `(.venv)` in your prompt and `which python` should point
inside `.venv/`.

### 1.4 Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs pandas, numpy, pyarrow, yfinance, pandas-ta, torch, scipy,
vectorbt, mlflow, matplotlib, pytest, kaggle, jupyter, nbconvert, and ipykernel.

A `kaggle` CLI warning about "Authentication required" is normal —
you only need Kaggle auth if you plan to upload the cache or run on Kaggle GPU.

### 1.5 Sanity check

```bash
.venv/bin/python paths.py
# Expected: a JSON block with "env": "local" and all paths under the
# project directory.

.venv/bin/python -m pytest tests/ -v
# Expected: all tests pass (test_paths.py × 3). More tests activate once
# you've built the cache (--download, --targets, --features).
```

---

## 2. Build the local cache

The local pipeline is a strict sequence. **Run each step in order.**
The order is deliberate — it prevents lookahead bugs that are hard to
diagnose after the fact.

### 2.1 Step B — Download prices

```bash
.venv/bin/python data.py --download
```

**What happens:** Downloads 42 series (40 universe tickers + `^GSPC` +
`^VIX`) from Yahoo Finance for 2015-01-01 → 2024-12-31, aligns to a
common trading-day calendar, validates, and writes
`cache/prices.parquet` + `cache/prices.sha256` + `cache/hashes.json`.

**First run:** ~60–90 seconds (yfinance network). **Subsequent runs:**
<1 second (cache hit, prints the same SHA-256 as the first run).

**Expected output (first run):**

```
[data] downloading 42 series from 2015-01-01 → 2024-12-31
[data] master calendar: 2515 trading days

[data] validation summary:
ticker                 sector      first       last  rows  NaNs
  AAPL Information Technology 2015-01-02 2024-12-30  2515     0
  ...  (42 rows total)
  ^VIX                 global 2015-01-02 2024-12-30  2515     0

[data] wrote /.../cache/prices.parquet  sha256=...
```

**Expected output (subsequent runs):**

```
[data] cache hit: /.../cache/prices.parquet
[data] sha256: ...
```

### 2.2 Step C — Build the target

```bash
.venv/bin/python data.py --targets
```

**What happens:** Computes the next-day excess return per (date,
ticker) — the label the model is trained to predict — and writes
`cache/targets.parquet`. The last price date is dropped (no t+1 known).

**Expected output:**

```
[data] targets: dropped 1 row(s) with no t+1 label
[data] wrote /.../cache/targets.parquet  rows=100,560
```

The exact number of rows is `2514 dates × 40 tickers = 100,560`.

**Verify with the test suite:**

```bash
.venv/bin/python -m pytest tests/ -v
# Expected after Step C:
#   tests/test_paths.py        — 3 passed
#   tests/test_target.py       — 6 passed
#   total: 9 passed
```

The 6 target tests cover: cross-sectional sum ≈ 0 on every date,
3 manual hand-computations to 1e-9, alignment, no-lookahead, and
exactly 40 tickers per date.

### 2.3 Step E — Build features

```bash
.venv/bin/python data.py --features
```

**What happens:** Computes 8 features per (date, ticker) — returns,
volatility, momentum, volume ratio, sector relative strength, spread,
market return (`^GSPC`), and VIX — and writes `cache/features.parquet`.
Features are stored **unscaled**; the scaler is fitted per fold on
training data only.

---

## 3. Environment split (local vs Kaggle)

| Step | Where | Why |
|---|---|---|
| Data acquisition | local | needs internet for yfinance |
| Target construction | local | must be hand-verified |
| Feature engineering | local | CPU; needs plotting |
| Eval harness + random-model gate | local | CPU only; the gate is the gate |
| Graph construction | local | CPU; leakage gate verification |
| Model training | **Kaggle GPU** | only step that needs a GPU |
| Final comparison | local | re-runs eval over downloaded predictions |

The split is implemented in [`paths.py`](./paths.py). Every module
imports its paths from there. `/kaggle/input/` is read-only; outputs
go to `/kaggle/working/`. `tests/test_paths.py` verifies both branches
resolve correctly.

---

## 4. Uploading the cache to Kaggle

After the local pipeline completes and `cache/` has all three Parquet
files, push it to Kaggle as a versioned Dataset:

```bash
# Option A — Kaggle web UI
#   Datasets → New Dataset → drag prices/features/targets.parquet
#   and hashes.json → name it `dmrgt-cache` → Private.

# Option B — Kaggle API (preferred, scriptable)
# place kaggle.json in ~/.kaggle/ with 600 permissions
kaggle datasets init -p cache/
# edit cache/dataset-metadata.json: set title and id
kaggle datasets create -p cache/
# for later updates (PREFERRED — never overwrite):
kaggle datasets version -p cache/ -m "features v2"
```

The training notebook verifies the SHA-256 of each Parquet against
`hashes.json` before training and halts on mismatch.

---

## 5. Repository layout

```
DMRGT/
├── configs.yaml                     # all hyperparameters, tickers, costs, seeds
├── paths.py                         # LOCAL / KAGGLE path resolution
├── requirements.txt
├── README.md                        # ← you are here
├── BuildSpec/
│   ├── DMRGT-Phase1-Build-Spec.md   # Phase 1 design brief
│   └── DMRGT-Phase2-Build-Spec.md   # Phase 2 design brief (authoritative)
├── docs/
│   ├── PHASE1_RESULTS.md            # Phase 1 results summary
│   └── PHASE2_RESULTS.md            # Phase 2 results summary
├── data.py                          # Steps B, C, E (--download, --targets, --features)
├── features.py                      # Step E: 8 features (called from data.py)
├── graph.py                         # Phase 2: temporal graph construction
├── splits.py                        # walk-forward folds
├── baselines.py                     # random, persistence baselines
├── eval.py                          # metrics + cost-aware backtest
├── model.py                         # GNN + Transformer encoder
├── train.py                         # training loop
├── compare.py                       # drives the results comparison table
├── verify_env.py                    # environment verification script
├── cache/                           # Parquet outputs (gitignored) → Kaggle
│   ├── prices.parquet
│   ├── targets.parquet
│   ├── features.parquet
│   ├── prices.sha256
│   └── hashes.json
├── artifacts/                       # downloaded Kaggle outputs (gitignored)
├── notebooks/
│   └── verification.ipynb           # hand-checks (LOCAL)
├── kaggle/
│   ├── KAGGLE.md                    # Kaggle workflow instructions
│   ├── kaggle_train.ipynb           # thin training wrapper (KAGGLE)
│   ├── build_code_dataset.sh        # assembles kaggle/code/ for upload
│   └── code/                        # mirror of core modules for Kaggle dataset
│       ├── configs.yaml
│       ├── data.py
│       ├── model.py
│       ├── paths.py
│       ├── splits.py
│       └── train.py
└── tests/
    ├── test_paths.py                # 3 tests, both envs
    ├── test_target.py               # 6 tests, §4 mandatory checks
    ├── test_features.py             # 9 tests, NaN-corruption lookahead
    ├── test_leakage.py              # THE GATE — random model IC ≈ 0
    ├── test_graph_leakage.py        # Phase 2: graph edge window leakage
    ├── test_cache_integrity.py      # cache hash stability
    ├── test_idempotency.py          # data.py idempotency
    ├── test_scaler.py               # per-fold scaler isolation
    ├── test_universe.py             # universe size and sector balance
    └── test_windows.py              # walk-forward window integrity
```

---

## 6. One-command reproduction

The full local pipeline is reproducible end-to-end. From a clean
checkout:

```bash
# 1. Set up the venv (one-time, ~1 min)
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. Build the cache: prices → targets → features (~2 min total)
.venv/bin/python data.py --download
.venv/bin/python data.py --targets
.venv/bin/python data.py --features

# 3. Run the local evaluation suite (~1 min — 4 folds × baselines + models)
.venv/bin/python compare.py --skip-tx

# 4. Run the tests (~30s — all tests including leakage gates)
.venv/bin/python -m pytest tests/ -v

# 5. Re-run the verification notebook (headless; ~20s)
.venv/bin/jupyter nbconvert --to notebook --execute \
    notebooks/verification.ipynb --output verification.ipynb
```

The Kaggle half is documented in [`kaggle/KAGGLE.md`](kaggle/KAGGLE.md).
The single decision you need to make is whether to upload the cache
+ code as Kaggle Datasets (preferred — versioned, reproducible) or
paste the code into the notebook directly. KAGGLE.md covers both.

After the Kaggle run, finish the comparison:

```bash
# 6. Download predictions.parquet + run_manifest.json from the
#    committed Kaggle run's Output tab into ./artifacts/, then:
.venv/bin/python compare.py
```

---

## 7. Phase 2 headline results

All seven models evaluated through the identical `eval.py` path — same
folds, same seeds, same commission (10 bps) and slippage (5 bps).

| Model | Mean rank IC | Dir. acc | Sharpe (after costs) | Max DD |
|---|---|---|---|---|
| **dmrgt_all3** | **+0.0061 ± 0.0110** | 0.499 | −1.64 | 0.295 |
| dmrgt_no_sector | +0.0056 ± 0.0122 | 0.503 | −2.07 | 0.327 |
| dmrgt_pos_only | +0.0036 ± 0.0133 | 0.499 | −2.24 | 0.361 |
| transformer_nograph *(control)* | +0.0022 ± 0.0114 | 0.505 | −1.35 | 0.351 |
| persistence | +0.0000 ± 0.0018 | 0.502 | −1.75 | 0.411 |
| lstm | −0.0016 ± 0.0159 | 0.501 | −1.29 | 0.356 |
| random *(gate)* | −0.0074 ± 0.0035 | 0.496 | −5.24 | 0.489 |

**Completion criterion met:** DMRGT all-3 (+0.0061) beats both the
no-graph Transformer (+0.0022) and the LSTM (−0.0016). ✅

The ablation ladder shows each relation type contributes positively:

| Configuration | Edge types | Mean rank IC | Δ |
|---|---|---|---|
| `dmrgt_pos_only` | +corr | +0.0036 | — |
| `dmrgt_no_sector` | +corr, −corr | +0.0056 | +0.0020 |
| `dmrgt_all3` | +corr, −corr, sector | +0.0061 | +0.0005 |

Full details in [`docs/PHASE2_RESULTS.md`](./docs/PHASE2_RESULTS.md).

---

## 8. Build progress

| Step | Status | Description |
|---|---|---|
| **Phase 1** | ✅ Complete | Transformer encoder, 20-stock universe, 8 features, leak-free pipeline, random-model gate passed |
| **Phase 2** | ✅ Complete | Dynamic 3-relation graph (+corr/−corr/sector), R-GCN encoder, 40-stock universe, sector-neural-bucketing, multi-target training, beats no-graph and LSTM baselines |

### Phase 1 steps

| Step | Status | Description |
|---|---|---|
| **A** Foundation | ✅ | venv, configs.yaml, paths.py, requirements, README, .gitignore, tests/test_paths.py |
| **B** Data acquisition | ✅ | data.py --download, 22 series, idempotent cache, hash recorded |
| **C** Target | ✅ | data.py --targets, tests/test_target.py (6/6 pass), verification.ipynb |
| **D** Splits + baselines | ✅ | splits.py (4 walk-forward folds), baselines.py (random + persistence) |
| **E** Eval harness | ✅ | eval.py, leakage gate passes (mean IC ≈ 0.002) |
| **F** Features | ✅ | features.py (8 features, unscaled), tests/test_features.py (9/9 pass) |
| **G** Model | ✅ | model.py (Transformer, 100K params, causal mask), train.py |
| **H** Verification | ✅ | verification.ipynb (8-panel histogram, per-ticker time series) |
| **I** Kaggle artifacts | ✅ | kaggle_train.ipynb, kaggle/code/, KAGGLE.md |
| **J** Comparison | ✅ | compare.py, evaluate_predictions() in eval.py |

### Phase 2 steps

| Step | Status | Description |
|---|---|---|
| **K** Universe expansion | ✅ | 20 → 40 stocks, 5 per sector across 8 GICS sectors |
| **L** Graph construction | ✅ | graph.py — temporal edges with leakage-safe windows |
| **M** Graph leakage gate | ✅ | tests/test_graph_leakage.py — edge windows don't leak future |
| **N** GNN encoder | ✅ | model.py — R-GCN with sector embeddings, multi-target heads |
| **O** Multi-target training | ✅ | train.py — graph-aware loop, configurable architecture |
| **P** Ablation study | ✅ | dmrgt_all3 vs dmrgt_no_sector vs dmrgt_pos_only |
| **Q** Final comparison | ✅ | DMRGT beats no-graph Transformer and LSTM on rank IC |

---

*Research project for academic purposes only — not investment advice.*
