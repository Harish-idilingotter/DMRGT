# DMRGT Phase 3 — Graph sensitivity and dynamic-vs-frozen graph comparison

> **Status: code complete, awaiting the single Kaggle session.**
> Every number marked `[PENDING …]` is filled in only after the run that
> produces it, in the order given in `docs/KAGGLE_RUN_ORDER.md`. Nothing
> is back-filled out of order — the one-shot rules depend on it.
>
> **Single-session rework.** Phase 3 originally planned two Kaggle
> sessions (grid, then arms) with a local winner pick, a frozen-graph
> rebuild + cache re-upload, and a post-processing tuning stage between
> them. That flow was reworked: ONE committed session now does the grid
> → val-only winner selection → in-session frozen-graph build →
> `dmrgt_frozen` → significance tests, with no mid-run round-trip. The
> rework's spec is committed at the repo root
> (`DMRGT-Phase3-Build-Spec.md`, §0–§10) and the mechanics below are
> updated accordingly.

Phase 3 answers two questions on top of the Phase 2 pipeline:

1. **Is the baseline graph configuration (k_pos=8, k_neg=4, window=60)
   the right one?** A 7-variant sensitivity grid — one dimension at a
   time — scored on validation Sharpe only (§5 below).
2. **Does the dynamic correlation graph actually matter?** The frozen arm
   holds each fold's graph fixed at its train-end snapshot; if the
   dynamic arm does not beat it (paired rank IC + Sharpe difference,
   with p-values), the daily correlation update adds nothing measurable.

Both questions are answered under the same leakage discipline as
Phases 1/2 (walk-forward folds, costs always on, selection strictly on
validation rows, test rows scored exactly once).

---

## 1. What is identical to Phase 2 (and why it must stay that way)

Universe (40 S&P 100 constituents, 5 per GICS sector), the 8 features, the
4 walk-forward folds (test years 2019–2022, expanding train from 2015,
latest 15% of each train window as chronological validation), the
10 bps commission + 5 bps slippage cost model, and the data cache are
all unchanged — `prices.parquet`, `features.parquet`, `targets.parquet`
hash-verify against Phase 2's recorded fingerprints, and the five-file
hash gate in `train.py` halts any run against a stale cache.

`graphs.parquet` (the dynamic daily graph) carries **Phase 2's exact
content**. The Phase 3 rebuild of that file — under pandas 3, which
removed `pct_change`'s `fill_method` keyword — was verified
content-identical (`.equals()` → True over all 1,571,200 rows) against
Phase 2's original file (byte SHA-256 `555560de…`, as recorded in Phase
2's reproducibility record and still held in Phase 2's `dmrgt-cache` Kaggle
dataset), which was downloaded for the comparison. The local cache now
holds the content-identical pandas-3 rebuild (its file bytes differ only
in parquet writer metadata); the content is pinned by
`tests/test_graph_grid.py` (`PHASE2_GRAPHS_CONTENT_SHA256`), so any
drift — including a rebuild at a new graph config — fails the test suite
until it is consciously accepted and documented.

The model is ONE `DMRGT` class (no `model_id` branching): the same
temporal TransformerEncoder (2 layers, 4 heads, causal mask, 8→64)
as Phases 1/2, the same hand-rolled 3-relation message passing, and a
deliberately simpler head (`Linear(128→32) → ReLU → Linear(32→1)`) than
Phase 2's fusion head — Phase 3 compares graphs, so a smaller head
removes a source of variance that contributed nothing.

## 2. The two corrections carried into Phase 3 (decided, not silent)

Two Phase 2 defects were found while porting the evaluation harness.
Both change numbers; both are documented here so nobody re-litigates
them silently.

### 2.1 Leg direction (the Sharpe inversion)

The Phase 1/2 specs define the strategy as **long the top-predicted
decile, short the bottom-predicted decile**. Phase 2's implementation
selected legs on an ascending rank (`ranks <= k`), which inverts this: it
bought the *bottom*-predicted stocks. The Phase 2 results show the
inversion's fingerprint — the models' Sharpe ordering was exactly
inverted relative to their rank-IC ordering:

| Model (Phase 2, leg-inverted) | Rank IC | Sharpe (after costs) |
|---|---|---|
| dmrgt_all3 | **+0.0061** (best) | −1.64 (worst) |
| transformer_nograph | +0.0022 | −1.35 |
| lstm | −0.0016 (worst) | **−1.29** (best) |

The model with the *best* predictions posted the *worst* Sharpe and vice
versa — the monotone inversion across all three models is the leg flip,
not a cost story (costs alone would preserve the IC ordering at reduced
magnitude). See `docs/PHASE2_RESULTS.md` §6 for the original table.

Phase 3 fixes the legs to match the written definition (`argsort(-pred)`,
longs first, shorts last). Consequences:

- rank IC, directional accuracy, turnover and the cost model are
  **unchanged** and remain comparable across phases;
- **Sharpe and cumulative return are NOT comparable to Phase 2's table**
  (Phase 2's were leg-inverted). Phase 3's are internally consistent and
  economically meaningful — which matters because Phase 3 *selects* on
  Sharpe (tuning and the graph grid both rank on it).

### 2.2 Baseline evaluation window (persistence floor shifts slightly)

Phase 2's closed-form baseline harness built each prediction window
ending at `d-1`; the trained models' assembly always ended at `d`
(inclusive, per the feature rule). Phase 3 unifies both paths on the
inclusive window. The random-model gate is unaffected (noise ignores the
input); the persistence floor moves slightly vs Phase 2's recorded
number — the Phase 3 floor supersedes it.

## 3. The frozen arm (Rule 16)

For each walk-forward fold, the graph is computed **once** from the
trailing 60-day correlation window ending at the fold's last fit date
(train end, before the validation window), and that edge list is held
constant across every date the fold touches — the model trains,
early-stops, and is scored against one structure. The leakage guard
asserts the snapshot's correlation window ends at or before train end,
so the frozen graph cannot encode validation- or test-period
relationships. Early *train* dates reuse the train-end snapshot; that is
look-ahead within the training window only — the same category as the
train-fitted scaler — and never touches val/test data.

The folds' date ranges overlap (each train window starts in 2015), so
`graphs_frozen.parquet` keys edges by **(fold, date)** — a date shared by
two folds legitimately carries two different snapshots, and `train.py`
resolves a sample's edges by its own fold first.

Edge persistence (from Phase 2's diagnostics) is 0.93 day-over-day: the
dynamic graph changes ~1 edge in 14 per stock per day, so the arm
comparison has something to measure — the frozen arm is *not* trivially
equal to the dynamic one (asserted structurally in the tests: on a
mid-test date the two arms' edge lists differ on every fold).

## 4. Post-processing tuning — dropped (single-session rework)

The original Phase 3 plan included a 36-config post-processing tuning
grid (no-trade band × decile fraction × smoothing) selected on
validation Sharpe. The single-session rework **dropped it**: the tuning
stage was the reason the old flow needed a local round-trip between
sessions, and the graph comparison does not need a tuned strategy — it
needs both arms scored under identical, pre-declared conditions.

Consequently:

- **Identity post-processing is declared in advance** and applied
  identically to both arms (Rule 19): no-trade band 0, decile fraction
  0.20 (the config default), smoothing 1 — `eval.DEFAULT_POSTPROCESSING`,
  recorded in `compare.py`'s output JSON.
- `tune.py` and `configs.yaml`'s `[tuning]` section were removed.
- Rule 17 (validation-only selection) still governs the one selection
  that remains: the graph grid's winner (§5).

## 5. Graph configuration sensitivity grid (§6b, in-session)

The baseline graph choice (k_pos=8, k_neg=4, window=60) is Phase 2's
inheritance, not a Phase 3 decision. The grid re-tests it: 7 variants,
**one dimension at a time** (k_pos ∈ {6, 10}, k_neg ∈ {3, 5},
window ∈ {45, 90}, plus the baseline), each trained as `dmrgt_all3`
over 4 folds × 5 seeds (140 runs), scored on validation Sharpe only.

In the single-session flow the grid and the selection happen in ONE
Kaggle session:

- Cell 2 trains all 7 variants (`train.run_all(...,
  output_suffix=f"_grid_{name}")`), writing val AND test rows for each
  — `dmrgt_all3` is trained exactly once per variant (§1 of the rework;
  160 runs total including the frozen arm, not 180).
- Cell 3 calls `sensitivity.select_winner(cfg)`, which reads each grid
  prediction file, keeps ONLY `split == "val"` rows, and picks the
  winner on the configured selection metric (val Sharpe). The
  validation-only safeguard is pinned by `tests/test_winner_selection.py`
  (equivalence with test rows deleted, poisoned-test-rows invariance,
  and a source-level check of the spec's exact mechanics).
- The winner's **already-existing test rows become the final
  `dmrgt_all3` result** — read for the first time in Cell 6, after
  selection is locked. The grid is never re-run (Rule 20), and there is
  no config-edit / graph-rebuild / re-upload branch between selection
  and the frozen arm: Cell 4 builds the frozen graph in-session at the
  winning k from the attached prices.

The baseline variant's cached graph is asserted identical to the
dynamic graph — the grid measures k sensitivity, never rebuild drift.

Results: **[PENDING → Cell 3's `artifacts/phase3_sensitivity_grid.csv`
+ `phase3_frozen_graph_config.json`]**

## 6. Final test scoring + significance (Rules 18–20, in-session §6)

Cell 6 is the first and only in-session reader of test-split rows (the
winner is locked by then). It runs the two significance tests directly
on the two arms' prediction files — same evaluation pipeline, identity
post-processing on both:

1. **Wilcoxon signed-rank** on the 20 paired (fold, seed) mean rank ICs
   — the primary predictive-power test (α = 0.05).
2. **Stationary bootstrap** (Politis–Romano, 5,000 iterations, block
   length 10) on the **Sharpe difference**, with seed-averaged daily
   after-cost returns pooled across folds chronologically and the same
   index draws applied to both arms (paired resampling — the replicate
   isolates the arms, not the resample). Hand-rolled in numpy
   (`stats._stationary_bootstrap_*`): `arch` is not in Kaggle's base
   image and the session has no Internet, and Cell 6 runs in-session —
   the same scheme, cross-validated locally against `arch` on a
   decisive case (`tests/test_significance.py`), and smoke-tested
   in-session by `verify_env.check_bootstrap()` in Cell 0.

It writes `phase3_significance_results.json` (keys exactly:
`wilcoxon`, `bootstrap_sharpe`, `winning_graph_config`, `conclusion`).

Locally, `compare.py` re-scores the downloaded prediction files and
cross-checks every number against the in-session result before printing
the full report; its one-shot guard (`phase3_TEST_SCORED.json`) makes a
re-run with the same files a deterministic no-op and a re-run with
changed files/winner/postprocessing a hard stop (Rule 18).

Per-run test metrics (4 folds × 5 seeds per arm — the honest view):

**[PENDING → `python compare.py` →
`artifacts/phase3_final_metrics.csv`]**

Significance verdict (in-session, cross-checked locally):

**[PENDING → `artifacts/phase3_significance_results.json`]**

Reference floors, same harness, no special casing
(`[PENDING compare.py]`):

| Model | mean rank IC | Sharpe (after costs) | dir. acc |
|---|---|---|---|
| random | [PENDING] | [PENDING] | [PENDING] |
| persistence | [PENDING] | [PENDING] | [PENDING] |

## 7. Verification record (what passed before any training)

- **Leakage gate (§5.3, Rule 21)** — 8/8 tests green before `model.py`
  and `train.py` were written: random model through the full harness →
  mean rank IC +0.0042 (gate ±0.02), directional accuracy 0.5000
  (gate 0.48–0.52), mean Sharpe after costs −4.98 (costs charged:
  turnover strictly between 0 and 2 on every fold). The per-day-noise
  fix in the gate fixture is documented in `tests/test_leakage.py`.
- **Frozen graph invariants** — identical edges within every fold
  (sampled + first/mid/last dates), different across folds, per-date
  counts 320/160/160/640 on all 6,310 (fold, date) pairs, NaN-corruption
  raises, snapshot equals the train-end dynamic graph, and differs from
  the dynamic graph on every fold's mid-test date.
- **Node ordering** — graph indices map to the universe order on both
  files (sector edges connect same-GICS tickers; out-degrees exact;
  (fold, date) lookups exact).
- **Random-init determinism (Rule 15)** — same seed → identical
  parameter-byte fingerprints; different seeds → different fingerprints;
  perturbing any single parameter changes the fingerprint; device move
  preserves it.
- **Cache integrity** — all eleven cache parquets (4 base + 7 variants)
  hash-verify; `graphs.parquet` carries Phase 2's exact graph CONTENT
  (provenance pin, §1); `verify_env.py` passes all environment checks
  (pandas 3 no-fill semantics, GPU compute capability sm_70+, cache +
  variant hashes, and the numpy stationary-bootstrap smoke test).
- **Single-session rework verification** (before any training) —
  winner selection pinned validation-only by `tests/test_winner_selection.py`
  (gold equivalence with test rows deleted, poisoned-test-rows
  invariance, source-level check of the spec's mechanics); significance
  API pinned by `tests/test_significance.py` (spec JSON shape + verbatim
  conclusion, pairing, refusal of unpaired/mixed inputs, deterministic
  paired numpy bootstrap); missing-graph dates now DROP in
  `assemble_fold` with a bounded assert (win90's 2 known missing dates
  drop, everything else complete — `tests/test_frozen_graph.py`);
  staged-tree import parity (`tests/test_kaggle_staging_sync.py` — 11
  modules, fingerprint order pinned against `verify_env.STAGED_MODULES`);
  two-device fold scheduling pinned schedule-invariant
  (`tests/test_device_parallel.py` — parallel bitwise-equal to serial,
  deterministic longest+shortest fold pairing, per-device failure
  propagation).
- **Batched message passing ≡ per-sample loop** — the first Session A
  attempt measured **300 s for one `(fold, seed)`**, the smallest fold,
  which extrapolates to ≈16 h for the 140-run grid against a ~12 h Kaggle
  session cap. `DMRGT.forward` called the R-GCN once per date (776 calls
  per epoch). `forward_with_edge_lists` now batches a whole date batch
  into one `index_add` with per-sample node offsets, measured at a
  **1.78×** end-to-end speedup on real fold-0 tensors (10 epochs × 2
  reps, tight variance; forward-only 2.8×). Equivalence is pinned by
  `tests/test_batched_forward.py`: batched output is bitwise identical to
  the per-date loop on CPU, with disjointness tested directly (rewiring
  one sample's edges moves only that sample — an off-by-one offset would
  otherwise be invisible behind a loose tolerance), plus ragged edge
  counts, the shared-edge path, and seeded reproducibility. **No trained
  result existed before this change** — the interrupted baseline run
  wrote no predictions — so nothing scored was invalidated by it.

  The cause is **compute, not scheduling**, and it is worth recording
  because an earlier reading of the same numbers got it wrong: the T4's
  300 s against this CPU's 710 s for the identical run is only a 2.4×
  gap, which rules out a launch-latency-bound profile (a launch-bound
  T4 would have led the CPU by far more). At batch 1 in float32, 40
  stocks × 30 timesteps through a 2-layer/4-head transformer with
  `dim_ff=256` plus the R-GCN is ~100 MFLOP per date — genuine
  arithmetic. **Batching therefore recovers ~1.8×, not the order of
  magnitude an incorrect launch-bound diagnosis predicted.** The residual
  cost is the model's own forward/backward work, so the only remaining
  levers are the number of runs or the model's size — both design
  decisions, not engineering ones.

  **§5 re-measurement (2026-09-17, after the §3.5 assemble_fold fix)**:
  407 s local for one fold-1 `(fold, seed)` → ~170 s/run on the T4
  (2.4× ratio) — materially above the spec's ~50 s/run assumption, so
  the projection was re-derived per §5 rather than trusted. Growing
  train windows (776 → 1533 dates) put the 160-run single-GPU total at
  ≈11 h: no margin. The lever that fit was the second T4 the
  accelerator already provides: `run_all(devices=...)` schedules folds
  across one thread per GPU, paired longest+shortest from a static cost
  proxy (train-window length; `_assign_folds`). Scheduling cannot
  change any number — every run reseeds itself — and
  `tests/test_device_parallel.py` pins the parallel schedule
  bitwise-identical to the serial one. Re-derived projection ≈5.5–6 h,
  clearing the session cap with wide margin.

## 8. Reproducibility record

**Data provenance** (verified against Phase 2's record and Phase 2's
`dmrgt-cache` Kaggle dataset):

```
prices.parquet         438c0ea294d8de54ac8398e2c4ee92d3665f33526d864efc307f08bd9d3add23
features.parquet       b4614fa0424783724c72b9bd821d375db56b17de9562361ffd4e2e6018b51e1f
targets.parquet        204392bae151e145c49df56b99d371da7cd08444b1b9f4920196708fe6685c22
graphs.parquet         [cache/hashes.json — pandas-3 rebuild, content-identical
                       to Phase 2's 555560de… original; pinned by
                       PHASE2_GRAPHS_CONTENT_SHA256 in tests/test_graph_grid.py]
```

`graphs_frozen.parquet` is **not** in the cache: in the single-session
flow it is an OUTPUT built in-session by Cell 4 at the winning k
(§4 of the rework) and downloaded with the other artifacts. The local
`artifacts/graphs_frozen.parquet` copy from the pre-rework baseline-k
build is pinned by `FROZEN_GRAPHS_CONTENT_SHA256`
(`tests/test_graph_grid.py`) — if the grid's winner is not the
baseline, the in-session file supersedes it and the pin is updated in
the same commit with the reason recorded here.

The 7 sensitivity-grid variants (§6b) — flat in the cache root so the
plain `kaggle datasets version -p cache` uploads them (a
`graph_variants/` subdirectory is silently dropped by the CLI's default
`--dir-mode skip`). `graphs_baseline.parquet` carries the same hash as
`graphs.parquet`, which is the §6b identity check holding on disk:

```
graphs_baseline.parquet f011338998242026c1f123a349164236dad208efc5e235a31053fa217c12072d
graphs_kpos6.parquet    900e78b486163bdcda2e9bd26332567cdc3271d762ca4f283da36511c6d3a676
graphs_kpos10.parquet   de70a32145fa9ee9661f6c24a7ee4705f2bea39664b67bed852fcf65892e826c
graphs_kneg3.parquet    0aa2e897ceeb8a64f8e03cda33bc726c6322751dfe7cf16b4ee0c893e0c9f244
graphs_kneg5.parquet    a586648fbf7419a7d637745776d0baf9222d9b4454e907a2b3a2e73c978c0425
graphs_win45.parquet    2975e4802902e478f2cefce4532189d6263932a50c16dd30cfaa71ccc671c703
graphs_win90.parquet    5a5a67121dc782df71fed2fce3cc7b2328daade866dcf204f23ddd708110791a
```

**Environments**: local (macOS, Python 3.14.6, pandas 3.0.5,
torch 2.14.0, CPU) for graph builds, variant builds, scoring and
significance cross-checks; one Kaggle T4 session (the notebook records
its exact versions in the accumulating `run_manifest.json`) for all
training and the in-session selection / frozen-graph build /
significance tests.

**Run manifests**: `run_manifest.json` accumulates one record per
`train.run_all` call (the Phase 2 defect — one file overwritten per run —
is fixed). Every record carries the cache hashes, the config snapshot,
the edge source, GPU, library versions, and `"init": "random"`. The
single session yields 8 records (7 grid variants + `dmrgt_frozen`).

---

*Research project for academic purposes only — not investment advice.*