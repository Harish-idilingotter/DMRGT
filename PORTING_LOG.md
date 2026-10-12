# Phase 4 — Porting Log (spec §0.1 Step 5)

Source of every ported component: `reference/phase3_src/` (read-only; never edited — see
`git diff phase3-complete -- reference/`: file moves only). Destination: a notebook cell of
`kaggle_phase4.ipynb`, generated from `phase4/modules/*.py` (embedded verbatim as `%%writefile` cells).

Diff method: for every function/class below, the source text was extracted with `ast` from both
sides and compared line-by-line (`difflib`). "identical" = byte-for-byte equal source text.

| # | Component | Phase 3 source | Notebook cell / module | Diff result |
|---|---|---|---|---|
| 1 | `Fold`, `make_folds` | `splits.py` | Cell 3 `d4_data.py` | **identical** |
| 2 | Window assembly, label alignment, train-only `StandardScaler`, graph-coverage drop (+ 5% bound), feature-pivot reindex, row asserts | `train.py::assemble_fold` | Cell 3 `d4_data.py::FoldBuilder` | **restructured, logic identical** except the two documented changes: (1) `lookback` is a parameter (30/60) with a bounded-drop assert `dropped/n_train_days < 0.05` measured against Phase 3's 30-day windows; (2) test windows are built lazily and only after both lock files exist (`get_split` raises `PermissionError` otherwise). Train/val are built eagerly. The date-by-date edge dict build moved to `d4_graph.load_daily_edges` (`groupby` instead of per-date boolean filter; same rows, same order). |
| 3 | Config access | `data.py::load_config`, `configs.yaml` | Cell 0 + Cell 3 `d4_data.load_cfg` | **replaced**: `configs.yaml` is not part of the cache input, so universe, sectors, feature columns and folds are copied unchanged into Cell 0 (`STATIC`, included in the snapshot hash) and a Phase-3-shaped `cfg` dict is rebuilt from the snapshot. |
| 4 | `_pct_change_no_fill`, `_universe_returns`, `_sector_edges` | `graph.py` | Cell 4 `d4_graph.py` | **identical** |
| 5 | Correlation → edges (nan_to_num, diagonal ∓inf, argsort, top-k_pos/bottom-k_neg, sector edges) | `graph.py::_edges_for_returns_window` | Cell 4 `d4_graph.edges_from_corr` | **re-expressed** (array output instead of per-row dicts), same arithmetic and rows in the file's (edge_type, src, dst) order. Verified by unit test 4: the always-fire policy reproduces `graphs_baseline.parquet` on 2455/2455 dates (exact edge-set equality). |
| 6 | Cached daily-edge loader | `train.py::assemble_fold` (edge dict) | Cell 4 `d4_graph.load_daily_edges` | restructured (see #2) |
| 7 | `PositionalEncoding`, `TransformerEncoder`, `RelationMessagePassing` (R-GCN), `stack_per_sample_edges` | `model.py` | Cell 5 `d4_models.py` | **identical** |
| 8 | `DMRGT` (concat head, `forward`, `forward_with_edge_lists`) | `model.py::DMRGT` | Cell 5 `d4_models.py::DMRGT` | **changed as documented**: encoder is selectable (`transformer` → arm A, `mamba` → B/C/D) and stored as `self.encoder` (was `self.transformer`); new split forward `encode` / `propagate` (+ `propagate_with_edge_lists`); head, R-GCN, construction order (encoder → rgcn → head) unchanged, so arm A draws Phase 3's initial weights for a seed. |
| 9 | Eval harness | `eval.py` (whole file) | Cell 6 `d4_eval.py` | **identical** function bodies; only changes: `paths.*` → `d4_data.*` (3 call sites), `evaluate_fold` asserts the test lock files exist, and `make_pred_frame` added. The `source=` keyword is already present at both `load_predictions` call sites in the Phase 3 source. |
| 10 | Baselines | `baselines.py` | Cell 6 `d4_eval.py` | **identical** |
| 11 | Significance tests (Wilcoxon, NumPy stationary bootstrap, composite helpers) | `stats.py` | Cell 7 `d4_stats.py` | **identical** function bodies; only `paths.TARGETS` → `d4_data` (2 sites) and imports. CLI `main()` dropped. Added: `holm_correction`, `compare_arms`. |
| 12 | `set_seed`, `_param_fingerprint`, Rule-15 random-init assertion | `train.py` | Cell 8 `d4_train.py` | **ported**; `set_seed` additionally seeds Python `random`. |
| 13 | Training loop | `train.py::train_one_seed` | Cell 8 `d4_train.train_one` | **changed as specified (§6.5)**: AdamW + weight decay, linear warmup, ReduceLROnPlateau on val rank IC, grad clip 1.0, early stopping on val mean rank IC (Phase 3: Adam, val-loss), adaptive Huber for arm D, test never touched, per-job artefacts + done marker. Batched `forward_with_edge_lists` per step is unchanged. |

New in Phase 4 (no Phase 3 counterpart): `MambaEncoder` family (Cell 5), `RollingCorr`,
`run_policy`, trigger rules, threshold calibration and the Phase 4B inference engine (Cell 4),
`d4_worker.py` (Cell 9), Cells 10–18.

## Documented interpretations (where the spec leaves a detail open)

* **Combined-signal calibration (§12.2).** Each signal's threshold is its own single-signal calibration
  at a shared per-signal rate `q`; `q` is bisected so the *union* rule's simulated training rebuild
  rate hits the target.
* **Policy anchor.** "Last training day" = `fold.train_dates[-1]` (Phase 3's definition, i.e. before
  the validation slice), which is also the snapshot day of Phase 3's frozen graph.
* **Lookback-drop bound (§6.1).** Measured as the extra training days lost versus Phase 3's 30-day
  windows (≈ 30 of ≈ 776 in fold 1); the absolute warm-up loss common to both lookbacks is not counted.
* **Unit test 5** additionally checks that targets are the next-day cross-sectionally demeaned excess
  return recomputed independently from `prices.parquet`.

## Verification before the first Kaggle commit (§0.1)

- [ ] Repository confirmed private — **NOT verifiable from the agent; owner must confirm. Nothing has been pushed.**
- [ ] `phase3-complete` tag exists on the remote — created locally at `175768d`; **not pushed** (privacy gate).
- [x] Currently on `phase-4`; `reference/phase3_src/` contains the Phase 3 code.
- [ ] `phase3_locked/` hashes unchanged — **no `phase3_locked/` directory exists in this checkout**; nothing under `artifacts/` was touched.
- [x] Every ported component listed above with its diff result.

## Deviation: input dataset (decided by the owner)

Instead of attaching `dmrgt-phase-3-cache`, the notebook uses `dmrgt-phase-4-cache`: the same four parquet
files the notebook reads (`prices`, `features`, `targets`, `graphs_baseline`), byte-identical (SHA-256
asserted equal to Phase 3's `hashes.json`), with a `hashes.json` restricted to those four. The unused
variant graphs and `graphs_frozen` are dropped. Cell 2 still hash-verifies every parquet listed. Staged in
`phase4_cache_upload/`. The Phase 3 dataset name stays in the candidate list as a fallback.

---

# Phase 5 changes (`kaggle_phase5.ipynb`, sources in `phase5/`)

Built from the Phase 4 notebook sources, the Phase 5 spec and the Phase 4.1 spec (both read in full; the
Phase 4.1 spec's items F1–F3 and C1–C3 are applied as Phase 5 §4 modifies them). Phase 4 files are untouched.
Cell numbers refer to `kaggle_phase5.ipynb` (module cells 3–9 are the `%%writefile` cells).

| Change | Spec | Cell / module |
|---|---|---|
| Cache = `dmrgt-sp100-cache`; universe + GICS sectors read from `universe.csv`; every file hash in `hashes.json` verified; snapshot hash is of CONFIG + STATIC only, universe facts are added to the snapshot file at runtime | P5 §2/§3 | 0, 2, `d4_data` |
| Four lock files (`phase5_graph_winner/postproc/winner.json`, `phase5b_policy.json`) gate every test read | P5-4 | `d4_data` |
| Six folds (1–4 selection, 5–6 holdout 2023/2024); builder cache capped at 2 folds, tensors stored float32 (values identical to Phase 3's `.float()`) | P5 §7 | 0, `d4_data` |
| `split_val`: validation halves `val_es` / `val_sel`; `valpred` carries `val_half` | 4.1 F2.1 | `d4_data`, `d4_train` |
| Mamba dt init (log-uniform [1e-3, 1e-1] via inverse softplus), `_no_reinit` / `_no_weight_decay` tags, init assertion, two-group AdamW | 4.1 F1 | `d4_models`, `d4_train` |
| Training control sees only `val_es` (`ControlMetrics` asserts it), rolling-3 IC, no warmup checkpoints; plateau scheduler on the same rolling metric | 4.1 F2.2 | `d4_train` |
| Arm P: Phase 3 schedule (Adam, fixed LR, Huber 1.0, no clipping, early stopping on validation Huber loss, patience 10) — loss measured on `val_es` only | 4.1 C1 + P5 §4 | `d4_train` |
| Arm D removed; the adaptive-Huber code path was deleted with it | P5 §3 | `d4_train` |
| Random baseline `random_model(X, seed, day_index)`; floors assert random turnover > 0.5 | 4.1 F3 | `d4_eval`, 18 |
| Graph variants built in-notebook from `prices.parquet` (ported correlation → edges rule on pandas windows); baseline cross-checked against the cache | P5 §5 | `d4_graph`, 10 |
| S1 graph grid → S2 → S3 → 4B → S4 → scoring stage flow; per-stage budget projection with the pre-registered drop order | P5 §8/§9 | 12–18 |
| Seed ensemble, causal EMA, buffer zones, `backtest_from_weights` (same accounting as the ported harness) | P5 §6 | `d4_eval` |
| Phase 4B: policy metrics, selection and rebuild rate on `val_sel` strategy returns; policy window follows the S1 graph winner's `corr_window`; calibration repeated for holdout folds | 4.1 F2.3 + P5 §11 | 16, 18 |
| Direction-aware verdicts; A vs P in the secondary family; strategy-level (bootstrap) and model-level (Wilcoxon) tests kept separate | 4.1 C2 + P5 §11 | 18 |
| Δ diagnostics; extended compute benchmark (N up to 2000, windows 60/250, per-component timing incl. model inference) | 4.1 F1/C3 | 19 |

## Assumptions where the specs are silent (check these)

1. **Arm P and the two-group rule.** 4.1 F1 says the two-group rule applies to the Transformer arms (A, P); 4.1 C1 and Phase 5 §4
   define P as the verbatim Phase 3 schedule (Adam, no weight decay). Implemented: P = Phase 3 Adam (no weight decay, so the
   two-group rule is moot); A, B, C use two-group AdamW.
2. **P's early stopping** uses Phase 3's rule (lowest validation Huber loss, batch-mean averaging as in Phase 3) on `val_es` only.
3. **Grid arm A files** are tagged `A@<variant>`; after S1, "arm A" = the winning variant's runs (topped up to 5 seeds if the
   3-seed budget rule applied). Holdout arm A runs are separate (tag `A`).
4. **Strategy-level stats** pool daily returns across folds in date order; EMA and the buffer state run continuously from the start of
   each fold's validation period into test, and returns are read off only on the days of interest.
5. **Floors at strategy level:** random = the 5 seeds' noise ensembled; persistence = one seed. Persistence uses the ported
   `persistence_model` on the raw day-t log return.
6. **Holm family:** all two-sided secondary p-values (model-level Wilcoxon and strategy-level bootstrap for winner/other Mamba vs A and
   A vs P; post-processing effect per arm). The Phase 4B non-inferiority/superiority tests use their own criteria and are not in the family.
7. **Δ diagnostics:** "calm" / "volatile" = bottom / top VIX quartile of the diagnostic dates; effective memory = ln 2 / (Δ_median-channel · min exp(A_log)).
   Arm B's holdout diagnostics exist only if B is the winner.
8. **Budget:** `phase4b_estimate_minutes` (45) and `scoring_and_bench_estimate_minutes` (20) are kept unchanged as the ≈ 1 h serial tail.
9. **Policy window:** the 4B policy rebuilds with the S1 winner's `corr_window` (the graph the models were trained on), not the fixed 60 in `CONFIG["phase4b"]`.
10. **Not done here:** the Phase 4.1 spec's §0 git procedure (preserve run 1 under `results/phase4_run1`, tag `phase4-run1`) and Phase 5 §2 step 0 are
    owner actions; nothing was committed or pushed.
