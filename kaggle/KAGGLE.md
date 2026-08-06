# DMRGT Phase 1 — Kaggle Training Instructions

This document walks through the exact sequence of steps to (1) upload
the local cache to Kaggle, (2) upload the code as a second Kaggle
Dataset, (3) create and run the training notebook, and (4) download
the predictions and run the local Step 6 comparison.

> **If anything in this doc is unclear, the spec is the source of
> truth: `DMRGT-Phase1-Build-Spec.md` §2.5 and §7.** The spec also
> explains *why* each step exists — this doc focuses on *how*.

---

## 0. Prerequisites

You need:

- The DMRGT repo set up locally (this directory).
- A populated local cache: `cache/prices.parquet`, `cache/targets.parquet`,
  `cache/features.parquet`, `cache/hashes.json`. Build it with:

  ```bash
  python data.py --download
  python data.py --targets
  python data.py --features
  ```

- A Kaggle account with API access (`kaggle.json` in `~/.kaggle/`,
  `chmod 600` on it).

- All 21 local tests passing: `pytest tests/ -v`.

---

## 1. Upload the cache as a Kaggle Dataset (`dmrgt-cache`)

The cache is the ONLY thing the training notebook reads. It contains:

```
cache/
├── prices.parquet
├── features.parquet
├── targets.parquet
└── hashes.json
```

Total size: ~10 MB. The dataset is versioned — **do not overwrite
existing versions**, push a new one and record which version each
training run used.

### Option A — Kaggle web UI (one-time setup)

1. Go to [kaggle.com/datasets/new](https://www.kaggle.com/datasets/new).
2. Drag the four files from `cache/` onto the upload area.
3. **Title:** `dmrgt-cache`
4. **Slug:** `dmrgt-cache` (must match what `paths.py` expects:
   `/kaggle/input/dmrgt-cache`).
5. **Visibility:** Private.
6. Click Create. The dataset is now available to attach to notebooks.

### Option B — Kaggle API (preferred, scriptable)

From the project root:

```bash
# Initialise metadata (one-time, then commit cache/dataset-metadata.json)
kaggle datasets init -p cache/

# Edit cache/dataset-metadata.json — set:
#   "title": "dmrgt-cache"
#   "id": "<your-kaggle-username>/dmrgt-cache"
# Then create:
kaggle datasets create -p cache/
```

**Later updates — version, don't overwrite:**

```bash
kaggle datasets version -p cache/ -m "features v2 — added new feature"
```

Record the version number printed by Kaggle in your run manifest
(`run_manifest.json` will reflect the dataset version once the
notebook runs).

---

## 2. Upload the code as a second Kaggle Dataset (`dmrgt-code`)

The training code in `kaggle/code/` is the second dataset. Only two files
live there permanently — the rest are copied from root by the build script:

```
kaggle/code/
├── data.py              # slimmed down — only load_config() + all_tickers()
├── dataset-metadata.json
├── configs.yaml         # copied from root by build_code_dataset.sh
├── model.py             #   "
├── paths.py             #   "
├── splits.py            #   "
└── train.py             #   "
```

**Before uploading, assemble the dataset:**

```bash
bash kaggle/build_code_dataset.sh
```

This copies the five root files into `kaggle/code/`. Then upload:

```bash
kaggle datasets create -p kaggle/code/
# or for updates:
kaggle datasets version -p kaggle/code/ -m "updated train.py"
```

---

## 3. Create the training notebook

### Option A — Upload `kaggle_train.ipynb` directly

1. On Kaggle, go to [kaggle.com/code/new](https://www.kaggle.com/code/new).
2. Choose **"Upload an existing notebook"** and select
   `kaggle/kaggle_train.ipynb` from this directory.
3. The notebook opens in the editor.

### Option B — Paste the cells manually

1. Go to [kaggle.com/code/new](https://www.kaggle.com/code/new).
2. Create a new Python notebook.
3. For each of the 5 code cells in `kaggle/kaggle_train.ipynb`, paste
   the code into a new cell. The 6 markdown cells can be skipped or
   pasted as Markdown cells for documentation.

### Configure the notebook

- **Accelerator:** Settings → Accelerator → **GPU P100** (or T4).
  The trainer asserts `torch.cuda.is_available()` and will fail loudly
  on a CPU session.
- **Internet:** OFF. Data must come from the attached dataset, not a
  download. A download would break reproducibility (§2.5).
- **Persistence:** Files only (the default).

### Attach the datasets

In the right-hand panel:

1. **Add data** → Your Datasets → `dmrgt-cache` → Add.
   This makes `/kaggle/input/dmrgt-cache` available.
2. **Add data** → Your Datasets → `dmrgt-code` → Add.
   This makes `/kaggle/input/dmrgt-code` available.

---

## 4. Run the notebook

### Interactive run (development only)

You can run the cells in order during development. **The output is
NOT persisted** — if the session dies (idle timeout ≈ 20–40 min) you
lose `/kaggle/working/`. Use this only for iterating.

### Production run — Save & Run All (Commit)

This is the only way the spec wants you to run for results:

1. Click **Save Version** (top right).
2. In the dialog, set **Version type** to **"Save & Run All
   (Commit)"**.
3. Click **Save**.
4. Wait for the run to complete (~10–30 minutes for 4 folds × 5 seeds
   on a P100; the model is tiny at 100K parameters).
5. The committed run's **Output** tab now has the persisted files.

> **Why Commit, not interactive?** A committed run executes the whole
> notebook top to bottom and persists its output to your Output tab —
> so you can download the predictions even after the session ends. An
> interactive run loses `/kaggle/working/` when the session times out.

---

## 5. Download the predictions

From the committed run's **Output** tab, download:

- **`predictions.parquet`** — fold, seed, date, ticker, y_pred, y_true
  (the critical artifact)
- `run_manifest.json` — dataset version, git commit, library versions,
  GPU name, timestamps
- `train_metrics.json` — per-epoch loss curves (optional)
- `checkpoints/` — one .pt per fold×seed (optional, large)

Place them in the local `artifacts/` directory:

```bash
mkdir -p artifacts
mv ~/Downloads/predictions.parquet artifacts/
mv ~/Downloads/run_manifest.json   artifacts/
# Optionally:
# mv ~/Downloads/train_metrics.json  artifacts/
# mv -r ~/Downloads/checkpoints/    artifacts/
```

---

## 6. Run the local comparison

From the project root:

```bash
python compare.py
```

This runs random + persistence on every fold, then re-evaluates
`artifacts/predictions.parquet` through the SAME `eval.py` code path
(no special-casing for the Transformer). It prints the §8 results
table and writes `artifacts/comparison_table.csv`.

Expected output (numbers will differ — the spec is explicit that
"barely beating persistence is a legitimate Phase 1 outcome"):

```
## Spec table — Model | Mean rank IC | Dir acc | Sharpe | MDD | Turnover
  random        IC=-0.00±0.02  dir=0.50  sharpe=-4  mdd=0.54  turnover=1.59
  persistence   IC=-0.01±0.00  dir=0.50  sharpe=-2  mdd=0.40  turnover=1.54
  transformer   IC=...
```

The random and persistence numbers should be stable across runs
(they're deterministic given the seed). The Transformer numbers
should be in the same ballpark as persistence — the graph in Phase 2
is what is expected to push it past the floor.

---

## 7. One fold local-vs-Kaggle cross-check (§2.5)

The purpose of this check is to confirm that the local and Kaggle
environments are reading the **SAME DATA** and constructing the **SAME
FOLDS**. It is not a model-quality comparison.

Run one fold locally with a small number of epochs — the structural
properties below do not depend on how long training runs:

```bash
python verify_env.py --fold 0 --seed 0 --epochs 3
```

Compare the printed values against the same fold in
`artifacts/run_manifest.json` from the Kaggle run.

**These MUST match exactly:**

| Property | Why it matters |
|---|---|
| Number of training samples | A mismatch means a different dataset version |
| Number of validation samples | Same |
| Number of test samples | Same |
| First and last training date | A mismatch means fold logic diverged |
| First and last test date | Same |
| Ticker list and count | A mismatch means a different cache |
| SHA-256 of each parquet | The definitive data-identity check |

**These will NOT match, and are excluded from the comparison:**

- Validation loss — the local run uses fewer epochs, and CUDA and CPU
  kernels differ numerically even at identical epoch counts
- Rank IC magnitude — depends on training duration
- Any per-batch or per-epoch metric

If every structural property matches, the environments agree and the
Kaggle results can be trusted. If any structural property differs, do
not proceed — the most common cause is a stale `dmrgt-cache` version
attached to the notebook. Re-attach the latest version and re-run.

> The hash check in notebook Cell 2 catches the stale-dataset case
> automatically. This manual check exists to catch the cases a hash
> cannot: a path resolving to an unexpected directory, or a config
> value differing between environments.

---

## 8. Kaggle session gotchas (from §2.5 of the spec)

| Constraint | Value | What to do |
|---|---|---|
| Max session runtime | ~9 h | Phase 1 needs minutes; not an issue. |
| GPU quota | ~30 h/week | 1 fold × 1 seed takes 1–3 min. |
| Idle timeout (interactive) | ~20–40 min | **Always use Save & Run All (Commit).** |
| Session reset | Wipes `/kaggle/working/` | Always commit; download outputs immediately. |
| Internet | Off by default | Keep it OFF — data must come from the dataset, not a download. |

---

## 9. End-to-end summary

```
LOCAL                                KAGGLE                       LOCAL
─────                                ──────                       ─────
data.py --download    ─┐
data.py --targets      ├─▶ cache/*.parquet
data.py --features    ─┘        │
                               │  upload as Dataset
                               ▼
                          dmrgt-cache + dmrgt-code
                               │
                               ▼
                          kaggle_train.ipynb
                          (Save & Run All)
                               │
                          predictions.parquet
                          run_manifest.json
                               │
                          download ──────────▶  python compare.py
                                                   results table
                                                   (§8 of the spec)
```

That's the full Phase 1 handoff. The Transformer numbers, the
persistence numbers, and the random numbers all come out of the same
`eval.py` code path — so if the comparison is generous, it is
generous to everyone equally.
