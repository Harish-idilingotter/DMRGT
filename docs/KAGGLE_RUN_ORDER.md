# DMRGT Phase 3 — single-session Kaggle run order (for the operator)

This is the exact order of operations for Phase 3 **after the single-
session rework**: ONE Kaggle `Save Version → Save & Run All (Commit)`
run does the sensitivity grid, the val-only winner selection, the
in-session frozen-graph build, the `dmrgt_frozen` arm, and the
significance tests. All code is written; everything below is either the
one Kaggle run or the local epilogue. **Nothing may be re-ordered** —
the leakage rules (16–22) depend on this sequence, and in particular
the winner is locked from validation rows only (§1 of the rework) and
test rows are first read by Cell 6, after the winner is locked.

**The old two-session flow (`RUN_GRID` branch, session A → local winner
pick → rebuild → re-upload → session B → `tune.py`) is gone.** Every
step that used to require leaving Kaggle now happens inside the one
session: the winner is selected in-memory by `sensitivity.select_winner`
and the frozen graph is built by `graph.build_all_frozen_graphs` from
the attached `prices.parquet` — no dataset version bump, no re-attach,
no restart, no round-trip (§2 of the rework: "If any step in the
current design requires a local round-trip, that is the thing to
redesign, not accept.").

---

## Datasets — uploaded once, never bumped mid-pipeline

Phase 3 uses two fresh datasets (Phase 2's `dmrgt-cache` / `dmrgt-code`
are **never touched** — they are the record behind the submitted Phase 2
results):

| Dataset | Contents |
|---|---|
| `dmrgt-phase-3-cache` | the **11 parquets** (4 base + 7 graph variants) + `hashes.json`, all flat |
| `dmrgt-phase-3-code` | the **11 staged modules** (`kaggle/code_phase3/`) + metadata |

**No `--dir-mode`.** `kaggle datasets version` defaults to `--dir-mode
skip`, which silently *omits* subdirectories: the push reports success
and the files are simply not there. `cache/` is flat — every parquet at
the top level — so the plain command uploads all of it. (`--dir-mode
zip` would not fix this either: it archives a subdirectory into a
`.zip` and the files stay *inside* the archive.)

```
cache/                              kaggle/code_phase3/
  prices.parquet                      configs.yaml   paths.py
  features.parquet                    data.py        splits.py
  targets.parquet                     model.py       train.py
  graphs.parquet                      eval.py        graph.py
  graphs_baseline.parquet   ┐         sensitivity.py stats.py
  graphs_kpos6.parquet      │         verify_env.py
  graphs_kpos10.parquet     │
  graphs_kneg3.parquet      │ 7 flat variants       dataset-metadata.json
  graphs_kneg5.parquet      │ (sensitivity.py --build)
  graphs_win45.parquet      │
  graphs_win90.parquet      ┘
  hashes.json                ← hashes ALL 11 parquets (never iterate
                               hashes.json's own keys as files — the
                               gate walks paths.CACHE_PARQUETS, §3.2)
```

**`graphs_frozen.parquet` is NOT in the cache.** It is an *output*,
built in-session by Cell 4 to `/kaggle/working/graphs_frozen.parquet`
(`paths.GRAPHS_FROZEN = OUTPUT_DIR / "graphs_frozen.parquet"`, §4) and
downloaded with the rest of the outputs. Building it locally would
require a new cache dataset version mid-pipeline — the exact round-trip
the rework exists to remove (§10 anti-pattern).

---

## The one Kaggle session (§0–§2)

1. Kaggle → create notebook from `kaggle/kaggle_train_phase3.ipynb`.
2. **Settings**: Accelerator **GPU T4 x2** (never P100 — sm_60 has no
   kernels in this torch build and it crashes inside the first kernel
   launch), Internet **OFF**.
3. **Add Input**: `dmrgt-phase-3-cache` and `dmrgt-phase-3-code` (the
   latest versions). **Restart after attaching** — attaching does not
   mount into a running kernel.
4. **Save Version → Save & Run All (Commit)**. The 8 code cells run
   top-to-bottom, in one kernel, no branches:

   | Cell | Does | Guards it carries |
   |---|---|---|
   | 0 | dataset resolution + env check (versions, GPU **capability** sm_70+, bootstrap smoke) + code fingerprint | §3.1, §3.6, §3.7, §1.3 |
   | 1 | cache integrity: all 11 parquets SHA-256 vs `hashes.json` | §3.2 |
   | 2 | grid: 7 variants × 20 runs = **140 runs** (`dmrgt_all3` trained ONCE per variant, val AND test rows) | §1, §3.6, §5 projection halt |
   | 3 | **winner selection — validation rows ONLY** (`sensitivity.select_winner`) | §1 safeguard: val slice, `del df`, split assert |
   | 4 | **frozen graph built in-session** from attached `prices.parquet` → `/kaggle/working/graphs_frozen.parquet` | §2, §4, Rule 16 |
   | 5 | `dmrgt_frozen` at the winning k: **20 runs** | Rule 19 (edges are the ONLY difference) |
   | 6 | significance tests in-session (Wilcoxon + stationary bootstrap) + `phase3_significance_results.json` | §6 — FIRST read of test rows |
   | 7 | output verification: every §7 file + row-count audit | §7 — "Do not consider the run complete until this cell passes." |

   **160 runs total, not 180** — `dmrgt_all3` is never retrained after
   the grid (§1): the winner's already-existing test rows become the
   final dynamic-arm result, read for the first time in Cell 6.

5. When the commit finishes, **download everything** in the session's
   Output tab into local `artifacts/` and move on to the epilogue.

### Runtime — re-measured after every change to the hot path (§5)

The §3.5 fix (assemble_fold drops missing-graph dates instead of
crashing) touched the measured function, so the projection was
re-measured; never carry a timing number across a code change.

**The §5 chain, as it actually happened:**

1. **Re-measured (2026-09-17, post-§3.5-fix)**: one `(fold, seed)` of
   fold 1 — the *smallest* — took **407 s on this CPU** → **~170 s/run
   on the T4** (the established 2.4× ratio; consistent with the
   original T4 measurement). Materially above the spec's ~50 s/run
   assumption, so the projection was re-derived instead of trusted.
2. **Re-derived projection (single GPU)**: fold wall cost grows with
   the train window (776 → 1533 dates across the four folds), so
   ~170 s for fold 1 projects to ≈ **11 h for the 160 runs** — no
   margin under the session cap, and over the 9 h committed-GPU guard.
3. **The lever that fits: the second T4.** `run_all(devices=...)`
   schedules FOLDS across devices — one thread per GPU, folds paired
   longest+shortest (`_assign_folds`, deterministic from train-window
   length). Pure scheduling: every run reseeds itself
   (`train_one_seed` → `set_seed(seed)`), so results are
   bitwise-identical to the serial schedule — pinned by
   `tests/test_device_parallel.py`. Re-derived projection: **≈5.5–6 h**
   for the 160 runs — clears the cap with wide margin.

**First checkpoint — the §5 halt inside Cell 2.** After the first grid
variant finishes, Cell 2 prints the measured s/run and the projection
for all 160 runs, and **halts the session** if the projection exceeds
9 h. If it halts: stop, re-derive the projection, do not burn the
session hoping it speeds up. Also read the per-seed timing line:

```
  --- seed 0 ---
      epochs 50  best/final val_loss ... — elapsed NNN.Ns
```

The historical context, kept because it explains the design: the first
session attempt (2026-09-16 T4) measured **300 s for one `(fold, seed)`**
pre-batching, projecting ≈16 h for a 140-run grid. `DMRGT.forward`
looped the R-GCN once per date; `forward_with_edge_lists` now batches a
whole date batch into one message-passing call (1.78× end-to-end, pinned
arithmetically by `tests/test_batched_forward.py`). The residual cost is
compute, not scheduling (the T4 led the CPU by only 2.4×), so trust the
in-session halt over any table here.

---

## Local epilogue (§8) — after the session completes

```bash
# 1. everything from the session's Output tab → artifacts/
mv ~/Downloads/*.parquet ~/Downloads/*.json ~/Downloads/*.csv artifacts/

# 2. the one-shot final report
python compare.py

# 3. record it
git add -A && git commit -m \
    "Phase 3: single-session run — grid, frozen twin, significance tests complete"
git tag -a phase3-complete \
    -m "Dynamic vs frozen: p=<wilcoxon>, p=<bootstrap>"
```

`compare.py` re-scores test rows locally **only** because the in-session
Cell 6 already did: it cross-checks its numbers against the downloaded
`phase3_significance_results.json` (exact `n_pairs`/`wins`, p-values
within tolerance) and HALTS with a mismatch message if the downloaded
prediction files are not from the run that produced the in-session
numbers. Its one-shot guard (`artifacts/phase3_TEST_SCORED.json`) makes
a re-run with the same files a deterministic no-op and a re-run with
changed files/winner/postprocessing a hard stop (Rule 18).

There is **no `tune.py` step**: post-processing tuning was dropped by
the rework. Both arms are scored under the identity post-processing
declared in advance (`eval.DEFAULT_POSTPROCESSING`: band 0,
decile fraction = config's 0.20, smoothing 1) — identical for both arms
(Rule 19), declared before any test row was read.

---

## Pre-run checklist — run through this before the session

Each line maps to a failure that actually happened (§n refers to the
known-issues doc). None of these raise a clear error when skipped; that
is exactly why they are a checklist and not a memory.

```
[ ] ./kaggle/build_code_dataset_phase3.sh        # §7 — stages root → kaggle/code_phase3/
[ ] python -m pytest tests/ -v                   # incl. test_kaggle_staging_sync.py (§7)
[ ] python verify_env.py --require-variants      # env + 11-parquet hash gate
[ ] cache/ holds exactly 11 parquets + hashes.json — no .DS_Store / .gitkeep junk (§9.3)
[ ] cache/ does NOT hold graphs_frozen.parquet (it is an output, §4)
[ ] both dataset-metadata.json files have title AND id edited; slugs hyphenated (§9.1/9.2)
[ ] both datasets attached under Add Input (§1.1)
[ ] session RESTARTED after attaching / re-attaching (§1.2)
[ ] Accelerator explicitly GPU T4 x2 — never P100 (§3.7 checks sm_70+)
[ ] Internet OFF
[ ] Cell 0 prints the SAME "code fingerprint" as the staging script did (§1.3)
[ ] only then: Save Version → Save & Run All (Commit)
```

**The §1.3 fingerprint check is the one that fails silently.** A
dataset already attached to a notebook stays pinned to the version it
was added at: pushing a new `kaggle datasets version` does not update
the attachment, and the session keeps running the old code with no
error at all. If Cell 0's fingerprint differs from the staging script's
(`a8410c951d5d4622410a57285667cdeceb72c5bf3f73b68beab1e4997c2fad6c` as
of this writing), remove the dataset from Add Input, re-add it,
restart, and compare again — do not start training. (The fingerprint
changes every time `train.py` / `verify_env.py` / any staged module
changes — re-run the staging script after any root edit, push the
dataset version, and re-attach.)

These names are resolved **in code**, so all four places must agree or
the session dies before training starts:

| Where | What |
|---|---|
| `cache/dataset-metadata.json` | `title` + `id` |
| `kaggle/code_phase3/dataset-metadata.json` | `title` + `id` |
| `paths.py` | `_resolve_kaggle_dataset(...)` — raises `FileNotFoundError` at import |
| notebook Cell 0 + `metadata.kaggle` | `find_dir("dmrgt-phase-3-code")`, `dataSources`, `kernelDataSources` |

**Phase 2 datasets are never version-bumped or otherwise touched.** The
rework's §10 makes this explicit for the single-session flow: the cache
dataset is uploaded once (11 parquets + hashes.json) and consumed
read-only by the session; the frozen graph never joins it.

---

## The rules this order enforces

| Rule | Enforced by |
|---|---|
| 16 — frozen graph from train window only | `graph.build_all_frozen_graphs` leakage asserts + tests (in-session Cell 4) |
| 17/18 — test scored once, post-processing frozen in advance | identity postprocessing declared before any test read; `compare.py` hash marker `phase3_TEST_SCORED.json` |
| 19 — same post-processing + same pipeline on both arms | arms differ ONLY in the edge file (Cell 2 vs Cell 5) |
| 20 — grid never re-run after test results | one session; `dmrgt_all3` trained once per variant (§1) |
| 21 — leakage gate before model work | `tests/test_leakage.py` |
| 22 — exactly two models | `models.ids` gate in `train.run_all` (`dmrgt_all3` + `dmrgt_frozen`) |
| §1 — winner from val rows only | `sensitivity.select_winner` safeguard + `tests/test_winner_selection.py` |
| §6 — significance in-session | Cell 6 + `tests/test_significance.py` |

---

*Research project for academic purposes only — not investment advice.*