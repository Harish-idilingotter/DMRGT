# DMRGT — Phase 5 (Nasdaq-100)

Branch `phase-5` holds only: `dmrgt_ndx.ipynb` (the whole experiment), `build_cache.py` (the cache builder) and this file.
Earlier phases live on their own branches and tags (`phase-3`, `phase-4`, `phase3-complete`, `phase4-run1`, `archive-before-ndx`).

## 1. Build the cache (local)

```bash
# regression gate: rebuild the legacy 40-stock cache and compare it with the old Kaggle dataset
kaggle datasets download harishsivakumar1912/dmrgt-phase-3-cache -p /tmp/dmrgt-phase-3-cache --unzip
python build_cache.py --universe legacy40 --out /tmp/legacy40_rebuild --compare /tmp/dmrgt-phase-3-cache
# Nasdaq-100 cache (needs pandas, numpy, pyarrow, yfinance, pandas_ta, requests)
python build_cache.py --universe ndx100 --out cache/ --compare /tmp/dmrgt-phase-3-cache \
                      --regression-json /tmp/legacy40_rebuild/regression_test_legacy40.json
kaggle datasets create -p cache      # dataset-metadata.json sets the private id harishsivakumar1912/dmrgt-ndx-cache
```

## 2. Run the experiment (Kaggle)

Private notebook `dmrgt-ndx`: GPU T4 x2, Internet off, the only input is `dmrgt-ndx-cache`. Save Version → Save & Run All.
Check that at least 11 GPU hours of quota remain first. One run only; a rerun needs a documented bug.

## 3. After the run

```bash
mkdir -p results && kaggle kernels output harishsivakumar1912/dmrgt-ndx -p results/run1
# verify TEST_SCORED.json hashes against results/run1, write results/PHASE5_RESULTS.md from PHASE5_RESULTS_DRAFT.md
git add dmrgt_ndx.ipynb build_cache.py .gitignore results/ && git commit -m "Phase 5 (Nasdaq-100): run 1 results"
git tag phase5-ndx-run1
```

Large files (`*.pt`, `*.parquet`) stay gitignored. Keep the repository, dataset and notebook private until the patent decision.
