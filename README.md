# DMRGT — Dynamic Multi-Relational Graph Transformer (Phase 1)

Phase 1 of an academic final-year project. **No graph yet.** This phase
builds a trustworthy measuring instrument — a Transformer encoder over
20 sector-stratified S&P 100 stocks, trained on a leak-free pipeline with
transaction costs and slippage active from the first backtest run.

Full design lives in [`DMRGT-Phase1-Build-Spec.md`](./DMRGT-Phase1-Build-Spec.md).

---

## 1. Manual setup

### 1.1 Prerequisites

- **Python 3.11 or 3.13** (3.14 is too new — `numba`, a transitive dep
  of `pandas-ta`/`vectorbt`, doesn't yet support it).
- `git`, `pip`, ~2 GB free disk for the venv + Parquet caches.

### 1.2 Clone and enter the project

```bash
cd /your/projects/path
# (this directory already exists — just `cd` into it)
cd DMRGT
```

### 1.3 Create the virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate                  # zsh / bash on macOS / Linux
# On Windows: .venv\Scripts\activate
```

You should now see `(.venv)` in your prompt and `which python` should
point inside `.venv/`. If you forget to activate, every command below
will fall back to the system Python and the imports will fail.

### 1.4 Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs pandas, numpy, pyarrow, yfinance, pandas-ta, torch, scipy,
vectorbt, mlflow, matplotlib, pytest, kaggle, and (for the
verification notebook) jupyter, nbconvert, ipykernel.

A `kaggle` CLI warning about "Authentication required" is normal —
we don't push anything to Kaggle until Step F.

### 1.5 Sanity check

```bash
.venv/bin/python paths.py
# Expected: a JSON block with "env": "local" and all paths under the
# project directory.

.venv/bin/python -m pytest tests/ -v
# Expected: 3 tests pass (test_paths.py × 3). Step C adds 6 more once
# you've run --download and --targets.
```

---

## 2. Build the local cache

The local pipeline is a strict sequence. **Run each step in order.**
The order is deliberate — it prevents a class of bugs (lookahead in the
target) that are hard to diagnose after the fact.

### 2.1 Step B — Download prices

```bash
.venv/bin/python data.py --download
```

**What happens:** Downloads 22 series (20 universe tickers + `^GSPC` +
`^VIX`) from Yahoo Finance for 2015-01-01 → 2024-12-31, aligns to a
common trading-day calendar, validates, and writes
`cache/prices.parquet` + `cache/prices.sha256` + `cache/hashes.json`.

**First run:** ~30–60 seconds (yfinance network). **Subsequent runs:**
<1 second (cache hit, prints the same SHA-256 as the first run).

**Expected output (first run):**

```
[data] downloading 22 series from 2015-01-01 → 2024-12-31
[data] master calendar: 2515 trading days

[data] validation summary:
ticker                 sector      first       last  rows  NaNs
  AAPL Information Technology 2015-01-02 2024-12-30  2515     0
  ...  (22 rows total)
  ^VIX                 global 2015-01-02 2024-12-30  2515     0

[data] wrote /.../cache/prices.parquet  sha256=3507138cb2bf...
```

**Expected output (subsequent runs):**

```
[data] cache hit: /.../cache/prices.parquet
[data] sha256: 3507138cb2bf...
```

**Verify manually:**

```bash
.venv/bin/python data.py --download     # should be a cache hit
.venv/bin/python data.py --download     # same hash, no re-download
ls -la cache/                           # prices.parquet ~2.4 MB
cat cache/hashes.json                   # contains the SHA-256
```

The §3 deliverable is that **two consecutive runs produce the identical
hash**. This proves the cache is stable.

### 2.2 Step C — Build the target

```bash
.venv/bin/python data.py --targets
```

**What happens:** Computes the next-day excess return per (date,
ticker) — the label that the Transformer will be trained to predict —
and writes `cache/targets.parquet`. The last price date is dropped
(no t+1 known for it).

**Expected output:**

```
[data] targets: dropped 1 row(s) with no t+1 label
[data] wrote /.../cache/targets.parquet  rows=50,280
```

The exact number of rows is `2514 dates × 20 tickers = 50,280`. If you
see `rows=55,308` you have the **bug** version where `^GSPC` and `^VIX`
leaked into the universe mean — make sure your `data.py` is the latest
version.

**Verify with the hand-check notebook:**

```bash
.venv/bin/jupyter nbconvert --to notebook --execute \
    notebooks/verification.ipynb --output verification.ipynb
# or open it interactively:
.venv/bin/jupyter lab notebooks/verification.ipynb
```

The notebook has four cells:

1. **§1 Cross-sectional sum** — `max |sum| = 2.22e-16` (floating-point
   noise; well under the 1e-10 threshold from §4).
2. **§2 Hand-check table** — for the chosen date 2018-01-02, every
   stock's `excess[t+1] = raw_ret[t+1] - mean[t+1]`. The diff between
   the hand-computed value and the cached target should be `0.000e+00`.
3. **§3 Single-stock hand-check** — for any (date, ticker) you choose,
   prints every intermediate value and the cached target.
4. **§4 No-lookahead** — confirms the last price date is NOT in the
   targets table, and that every date has exactly 20 tickers.

**Verify with the test suite:**

```bash
.venv/bin/python -m pytest tests/ -v
# Expected after Step C:
#   tests/test_paths.py        — 3 passed
#   tests/test_target.py       — 6 passed
#   total: 9 passed
```

The 6 target tests cover the 5 mandatory §4 checks (cross-sectional
sum, cross-sectional mean, 3 random manual hand-computations to 1e-9,
alignment, no-lookahead) plus a structural check that every date has
exactly 20 tickers (catches the "globals leaked into the universe"
bug).

---

## 3. Environment split (local vs Kaggle)

| Step | Where | Why |
|---|---|---|
| Data acquisition | local | needs internet for yfinance |
| Target construction | local | must be hand-verified |
| Eval harness + random-model gate | local | CPU only; the gate is the gate |
| Feature engineering | local | CPU; needs plotting |
| Transformer training | **Kaggle GPU** | only step that needs a GPU |
| Final comparison | local | re-runs eval over downloaded predictions |

The split is implemented in [`paths.py`](./paths.py). Every module
imports its paths from there. `/kaggle/input/` is read-only; outputs
go to `/kaggle/working/`. `tests/test_paths.py` verifies both branches
resolve correctly.

---

## 4. Uploading the cache to Kaggle (after Step E)

After Step E completes locally and the cache has all three Parquet
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
├── configs.yaml                  # all hyperparameters, tickers, costs, seeds
├── paths.py                      # LOCAL / KAGGLE path resolution
├── requirements.txt
├── README.md                     # ← you are here
├── DMRGT-Phase1-Build-Spec.md    # full design brief (authoritative)
├── data.py                       # Steps B, C, E (--download, --targets, --features)
├── features.py                   # Step E: 8 features (called from data.py)
├── splits.py                     # walk-forward folds         [Step D]
├── baselines.py                  # random, persistence         [Step D]
├── eval.py                       # metrics + cost-aware backtest [Step D]
├── model.py                      # Transformer encoder         [Step F]
├── train.py                      # training loop               [Step F]
├── cache/                        # Parquet outputs (gitignored) → Kaggle
│   ├── prices.parquet
│   ├── targets.parquet
│   ├── features.parquet          [Step E]
│   ├── prices.sha256
│   └── hashes.json
├── artifacts/                    # downloaded Kaggle outputs (gitignored)
├── notebooks/
│   ├── verification.ipynb        # hand-checks (LOCAL)         ✅ done
│   └── kaggle_train.ipynb        # thin training wrapper (KAGGLE) [Step F]
└── tests/
    ├── test_paths.py             # 3 tests, both envs         ✅ done
    ├── test_target.py            # 6 tests, §4 mandatory checks ✅ done
    ├── test_features.py          [Step E]
    └── test_leakage.py           [Step D — THE GATE]
```

---

## 6. Phase 1 Definition of Done

See §9 of the build spec. Updated status as of the current build:

- [x] `paths.py` resolves correctly in BOTH local and Kaggle environments
- [x] 20 sector-stratified tickers downloaded, aligned, cached, hashed
- [x] Re-running `data.py --download` produces an identical hash
- [x] Target verified: cross-sectional sum ≈ 0 on every date (max 2.22e-16)
- [x] Target verified: 3 random manual hand-computations match to 1e-9
- [x] Target verified: exactly 20 tickers per date (no globals in universe)
- [x] Target verified: last price date is not in targets (no lookahead)
- [x] All target tests pass (6/6)
- [ ] Walk-forward folds generated; zero train/test overlap asserted         [Step D]
- [ ] **Random model scores rank IC within ±0.02 of zero** ← THE GATE       [Step D]
- [ ] Random model loses money after costs                                    [Step D]
- [ ] Persistence baseline recorded across all folds                         [Step D]
- [ ] All 8 features built, plotted and inspected individually               [Step E]
- [ ] NaN-corruption lookahead test passes for all 8 features                 [Step E]
- [ ] Features cached UNSCALED; scaler fitted per fold on training data only [Step F]
- [ ] `cache/` uploaded to Kaggle as a VERSIONED dataset; version recorded   [Step F]
- [ ] Kaggle notebook verifies parquet hashes before training; halts on mismatch [Step F]
- [ ] Kaggle notebook asserts GPU is available                               [Step F]
- [ ] Transformer trains; one shared model across stocks; causal mask present [Step F]
- [ ] 5 seeds per fold; train.py emits predictions.parquet, not metrics       [Step F]
- [ ] Kaggle run executed via Save & Run All (Commit), not interactively     [Step F]
- [ ] predictions.parquet + run_manifest.json downloaded to artifacts/        [Step G]
- [ ] One fold verified locally on CPU; agrees with Kaggle within tolerance  [Step G]
- [ ] Backtest includes commission AND slippage; turnover reported           [Step G]
- [ ] Results table complete: random vs persistence vs Transformer           [Step G]
- [x] All tests in `tests/` pass (9/9 as of Step C)

---

## 7. Build progress

| Step | Status | Description |
|---|---|---|
| **A** Foundation | ✅ Complete | venv, configs.yaml, paths.py, requirements, README, .gitignore, tests/test_paths.py |
| **B** Data acquisition | ✅ Complete | data.py --download, 22 series, idempotent cache, hash recorded |
| **C** Target | ✅ Complete | data.py --targets, tests/test_target.py (6/6 pass), verification.ipynb (4 cells, all green) |
| **D** Eval harness | ⏳ Pending | splits.py, baselines.py, eval.py, tests/test_leakage.py — the random-model gate |
| **E** Features | ⏳ Pending | features.py (stub in place) — 8 features built one at a time, plotted |
| **F** Transformer | ⏳ Pending | model.py, train.py, notebooks/kaggle_train.ipynb — Kaggle GPU |
| **G** Comparison | ⏳ Pending | Results table: random vs persistence vs Transformer |

---

*Research project for academic purposes only — not investment advice.*
