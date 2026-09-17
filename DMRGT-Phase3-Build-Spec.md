# DMRGT — Phase 3 Single-Session Rework Specification

> **For the implementing agent:** This supersedes the notebook and training-loop structure used so far. The objective is one committed Kaggle session — grid search, winner selection, frozen-graph construction, final training, and significance testing — with zero mid-run round-trips to a local machine. Every item below was a real failure encountered while building this pipeline, not a hypothetical. Read the whole document before touching any file. Do not skip §1 — it changes the run-count math that everything after it depends on.

---

## 0. The Outcome This Must Produce

One `Save Version → Save & Run All (Commit)` run that:
1. Trains `dmrgt_all3` across the 7-config sensitivity grid (validation-only selection)
2. Picks the winning graph configuration from validation predictions alone
3. Builds the frozen-graph-per-fold artifact in-session, from data already attached — no new dataset version, no re-attach, no restart
4. Trains `dmrgt_frozen` at the winning configuration
5. Runs Wilcoxon signed-rank and stationary bootstrap tests
6. Writes every artifact to `/kaggle/working/` for one single download at the end

No stage requires leaving the notebook. If any step in the current design requires a local round-trip, that is the thing to redesign, not accept.

---

## 1. The Run-Count Optimization That Makes Single-Session Feasible

**Do not retrain `dmrgt_all3` after the grid finishes.** This was unnecessary in every prior draft of this pipeline, and removing it is what makes one session comfortably sufficient rather than borderline.

**Why it's unnecessary:** `train.py` already predicts on both the validation and test split in the same training run (§5.3 of the Phase 3 build spec). This means the grid stage's 7×4×5 = 140 runs **already contain test-split predictions for every one of the 7 configurations** — including whichever one validation eventually picks as the winner.

**The rule this must follow exactly, so it does not become a Rule 17/18 violation:**

- Winner selection code loads **only** `split == "val"` rows. It must not load, filter on, or have any code path that could read `split == "test"` rows before or during selection.
- Once the winner is chosen, that configuration's **already-existing test-split rows** (from the grid run that already happened) become the final `dmrgt_all3` result. They are not re-generated, re-scored, or re-selected — they are simply the test rows that happened to belong to the winning config, read for the first time only after selection is final.
- This is stricter adherence to Rule 18, not looser: no training decision is made after test data is observed, because test data for the winner is never observed until after the decision is locked.

**Net effect on run count:**

| Stage | Runs | GPU work |
|---|---|---|
| Grid (`dmrgt_all3`, 7 configs) | 7 × 4 × 5 = 140 | Yes |
| Final `dmrgt_all3` | 0 — reused from the grid | None |
| Final `dmrgt_frozen` | 4 × 5 = 20 | Yes |
| **Total** | **160**, not 180 | |

**Required assertion enforcing the safeguard, placed immediately after winner selection and before any test-split file is opened:**

```python
def select_winner(cfg):
    val_only = {}
    for variant in grid_variants:
        df = pd.read_parquet(f"predictions_dmrgt_all3_grid_{variant}.parquet")
        assert set(df["split"].unique()) <= {"val", "test"}, "unexpected split value"
        val_only[variant] = df[df["split"] == "val"]          # test rows never touched here
        del df                                                 # drop the reference explicitly

    metrics = {v: eval.backtest(d, cfg) for v, d in val_only.items()}
    winner = max(metrics, key=lambda v: metrics[v][cfg["graph_grid"]["selection_metric"]])
    return winner, metrics
```

The `del df` after slicing to `val_only` is deliberate — it removes any lingering reference to the test rows from scope before the selection logic runs, so a later bug can't accidentally read them from a variable that's still holding the full frame.

---

## 2. Notebook Structure — One Continuous Flow, No Split

Replace the `RUN_GRID` boolean branch with a single linear cell sequence. There is no "Session A" / "Session B" distinction anymore — it is one script from top to bottom.

```
Cell 0   Environment check (dataset resolution, GPU capability)
Cell 1   Cache integrity check (hash verification, all files)
Cell 2   Stage 1 — grid training (140 runs, dmrgt_all3 only)
Cell 3   Stage 2 — winner selection (validation-only, in-memory)
Cell 4   Stage 3 — frozen-graph construction (in-session, from attached prices.parquet)
Cell 5   Stage 4 — final training (dmrgt_frozen only, 20 runs)
Cell 6   Stage 5 — significance testing (Wilcoxon + stationary bootstrap)
Cell 7   Output verification — assert every expected file exists before the run is considered done
```

**Cell 4 (frozen-graph construction) is the one that used to require leaving Kaggle.** It doesn't need to. `build_frozen_graph_for_fold()` is pure pandas/numpy operating on `prices.parquet`, which is already sitting in the attached cache dataset. Write its output straight to `/kaggle/working/graphs_frozen.parquet` — there is no reason this ever needed a new dataset version mid-run.

```python
# Cell 4
import graph
winning_config = json.load(open("/kaggle/working/phase3_frozen_graph_config.json"))
prices = pd.read_parquet(paths.PRICES)
folds = splits.make_folds(cfg, ...)

frozen_edges = pd.concat(
    [graph.build_frozen_graph_for_fold(prices, f, {**cfg["graph"], **winning_config})
     for f in folds],
    ignore_index=True
)
frozen_edges.to_parquet("/kaggle/working/graphs_frozen.parquet", index=False)
print(f"[frozen] {len(frozen_edges):,} rows written")
```

Cell 5 then loads this file directly from `/kaggle/working/` when training `dmrgt_frozen` — not from the attached read-only cache dataset. `paths.py` needs a small addition to support this (§4).

---

## 3. Every Prior Bug, Consolidated — Verify Each Is Actually Fixed Before Running

### 3.1 Kaggle mount resolution

`paths.py`'s dataset resolver must try, in order: `/kaggle/input/<slug>`, `/kaggle/input/datasets/<slug>`, then fall back to `root.rglob(slug)` with an ambiguity check. The actual observed layout was three levels deep (`/kaggle/input/datasets/<username>/<slug>`) — hardcoding any single layout breaks the moment Kaggle changes it again.

```python
def _resolve_kaggle_dataset(slug: str) -> Path:
    root = Path("/kaggle/input")
    for c in (root / slug, root / "datasets" / slug):
        if c.is_dir():
            return c
    matches = [p for p in root.rglob(slug) if p.is_dir()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous matches for '{slug}': {matches}")
    listing = sorted(str(p) for p in root.rglob("*"))[:40]
    raise FileNotFoundError(f"'{slug}' not found. Contents:\n  " + "\n  ".join(listing))
```

**Verify:** Cell 0 prints the resolved path for both `dmrgt-phase-3-cache` and `dmrgt-phase-3-code` before proceeding. If either resolution takes the `rglob` fallback branch, that's fine — but it must not silently fail.

### 3.2 `hashes.json` metadata key

`generated_at` is a timestamp, not a filename. Any code iterating `hashes.json`'s keys as if they're all files will crash trying to open `.../generated_at`.

```python
def verify_cache_hashes():
    expected = json.loads(paths.HASHES_JSON.read_text())
    for name in paths.CACHE_PARQUETS:      # explicit list — NOT expected.keys()
        assert name in expected, f"{name} missing from hashes.json"
        got = sha256_of_file(paths.CACHE_DIR / name)
        assert got == expected[name], f"hash mismatch: {name}"
    return True
```

**`paths.CACHE_PARQUETS` must be extended for the flat grid-variant layout** — the variants live directly in `CACHE_DIR`, not a subfolder (this was corrected after the fact — verify the current list reflects it):

```python
GRID_VARIANT_NAMES = ["baseline", "kpos6", "kpos10", "kneg3", "kneg5", "win45", "win90"]
CACHE_PARQUETS = (
    ["prices.parquet", "features.parquet", "targets.parquet", "graphs.parquet"]
    + [f"graphs_{name}.parquet" for name in GRID_VARIANT_NAMES]
)
# graphs_frozen.parquet is NOT in this list — it's built in-session to
# /kaggle/working/, never part of the read-only attached cache dataset.
```

**Verify:** run the count check — `len(CACHE_PARQUETS)` should be 11 (3 base + `graphs.parquet` + 7 variants). If the working code currently expects a `graph_variants/` subfolder anywhere, that's stale — the actual upload was flat.

### 3.3 Random-init fingerprint must hash bytes, never sum floats

CUDA's parallel reduction and CPU's sequential sum produce different results for identical weights, because float addition is not associative. Summing floats to compare "same seed → same params" fails spuriously across devices.

```python
def _param_fingerprint(model) -> str:
    h = hashlib.sha256()
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
```

All three comparison models (same-seed pair + different-seed check) must be built on **CPU**, compared, and only the surviving one moved to GPU afterward. Moving one to GPU before fingerprinting reintroduces the bug.

**Verify:** the random-init assertion in `train.py` builds all three candidates via `make_model(...)` with no `.to(device)` call until after both assertions pass.

### 3.4 Row-count assertion must sum per-fold counts, not multiply by one fold's count

The four folds have different test-day counts (252/253/252/251 — trading days vary by year). Multiplying `n_folds × n_seeds × (any single fold's day count)` gives a false mismatch on a correct run.

```python
total_test_days = sum(int(fr["n_test_days"]) for fr in fold_results)
expected_rows = total_test_days * n_seeds * n_stocks
assert actual_rows == expected_rows, (
    f"expected {expected_rows:,} ({' + '.join(str(fr['n_test_days']) for fr in fold_results)} "
    f"test days x {n_seeds} seeds x {n_stocks} stocks), got {actual_rows:,}"
)
```

**Verify with the new run-count math:** the grid stage produces predictions for `dmrgt_all3` only, across 7 variants. Each variant's expected row count is independent — do not accidentally sum across variants when checking a single variant's output.

### 3.5 Graph-coverage gaps at window boundaries must drop, not crash

A wider `corr_window` (specifically `win90` in the grid) needs more trailing history before its first valid graph exists than `features.parquet`'s own warm-up already covers. This produces a small, expected gap of a handful of dates at the very start of a fold's training window — not a bug, not a leak.

**The fix belongs inside `assemble_fold()`'s `build()` function**, filtering per-date rather than crashing after the fact:

```python
def build(date_set, with_labels):
    Xs, ys, kept_dates, dropped_no_graph = [], [], [], []
    for d in date_set:
        if d not in feat_wide.index:
            continue
        if with_labels and d not in tgt_wide.index:
            continue
        if graph_dates is not None and d not in graph_dates:
            dropped_no_graph.append(d)   # expected boundary effect — log, don't crash
            continue
        # ... existing window construction unchanged ...
    return X, y, np.array(kept_dates), dropped_no_graph
```

Followed by a **bounded** sanity check, not a hard fail on any drop at all:

```python
total_dropped = len(dropped_train) + len(dropped_val) + len(dropped_test)
total_dates = len(fold.train_dates) + len(fold.val_dates) + len(fold.test_dates)
if graphs is not None and total_dropped > 0:
    drop_frac = total_dropped / total_dates
    print(f"[assemble_fold] {fold.name}: dropped {total_dropped} dates "
          f"({drop_frac:.2%}) with no graph entry")
    assert drop_frac < 0.05, (
        f"{total_dropped} dates ({drop_frac:.2%}) missing graph coverage — "
        f"far more than a window-boundary effect. Check which graph file "
        f"this run was pointed at."
    )
```

**Verify:** run the full grid once with this fix in place and confirm `win90` completes without crashing, and that the printed drop count for `win90` is small (single digits) while all other 6 variants print zero drops.

### 3.6 `stdout`/`stderr` interleaving must not be trusted to order correctly

`print()` and `traceback.print_exc()` write to different streams that flush independently — an exception in a loop can appear to reference the wrong iteration's variable.

```python
try:
    train.run_all(cfg, model_id=mid, edge_source=variant_edges)
except Exception:
    print(f"[grid] VARIANT {variant_name} FAILED", flush=True)
    print(traceback.format_exc(), flush=True)   # NOT traceback.print_exc()
    continue
```

**Verify:** every except block in the notebook uses `traceback.format_exc()` piped through `print(..., flush=True)`, never `traceback.print_exc()` on its own.

### 3.7 GPU compute capability, not just availability

`torch.cuda.is_available()` returns `True` on a P100 even though the installed PyTorch build doesn't support its compute capability (sm_60 vs. the required sm_70+).

```python
assert torch.cuda.is_available(), "Enable GPU: Settings -> Accelerator -> GPU T4"
major, minor = torch.cuda.get_device_capability(0)
assert major >= 7, (
    f"{torch.cuda.get_device_name(0)} is sm_{major}{minor}, unsupported. "
    f"Switch Accelerator to GPU T4."
)
```

**Verify:** Cell 0 prints the GPU name and capability explicitly, and this assertion runs before any training cell.

### 3.8 Root and Kaggle staging copies must be byte-identical

`paths.py`, `graph.py`, `sensitivity.py`, `model.py`, `train.py`, `splits.py`, `configs.yaml` exist both at the project root and inside `kaggle/code_phase3/`. Kaggle only ever executes the staged copy — an edit at the root that isn't synced changes nothing about what actually runs.

```bash
cp paths.py graph.py sensitivity.py model.py train.py splits.py configs.yaml kaggle/code_phase3/
```

**Verify before every `kaggle datasets version`:**
```python
for name in ["paths.py", "graph.py", "sensitivity.py", "model.py", "train.py", "splits.py", "configs.yaml"]:
    assert hashlib.sha256(open(name,"rb").read()).hexdigest() == \
           hashlib.sha256(open(f"kaggle/code_phase3/{name}", "rb").read()).hexdigest(), \
           f"{name} out of sync — re-run the copy step"
```

### 3.9 Dataset naming — new datasets, never overriding Phase 2

`dmrgt-phase-3-cache` and `dmrgt-phase-3-code` are pushed via `kaggle datasets create`, not `kaggle datasets version` against the Phase 2 `dmrgt-cache`/`dmrgt-code`. The Phase 2 datasets remain untouched, as the record behind the already-submitted Phase 2 results.

**Verify:**
```bash
kaggle datasets files harishsivakumar1912/dmrgt-cache   # must be unchanged: 4 parquets + hashes.json
```

### 3.10 Dataset re-attachment after any version bump

An attached dataset stays pinned to whatever version it was at when added. Pushing a new version does not update an existing attachment; the kernel keeps running the old copy with no error.

**Verify (only relevant if this pipeline ends up needing more than one commit — the whole point of §0-2 is that it shouldn't):** remove and re-add the dataset in Add Input, then restart the session, before trusting any run that follows a version bump.

---

## 4. `paths.py` — the One Addition Needed for In-Session Frozen Graph

```python
GRAPHS_FROZEN = OUTPUT_DIR / "graphs_frozen.parquet"   # /kaggle/working/, NOT CACHE_DIR
```

`train.py`'s edge-source selection for `dmrgt_frozen` reads from `paths.GRAPHS_FROZEN` (writable working directory), never from `paths.CACHE_DIR` (read-only attached dataset). This is what avoids the mid-run dataset-version-and-reattach cycle entirely.

---

## 5. Timing Verification — Required Before Committing, Using Real Numbers

Do not trust a projection carried over from an earlier, different measurement. Re-derive it from the current 160-run total.

```
Measured: ~50s per fold x seed (post-fix, confirm this still holds after
          the assemble_fold() change in §3.5, which adds a small amount
          of per-date filtering overhead — re-time one run to confirm)

160 runs x 50s = 8,000s ~= 2h 13m core training
+ frozen-graph construction (Cell 4): seconds, pure pandas
+ significance testing (Cell 6): seconds, scipy/arch on already-small arrays
+ session overhead: ~5 min

Realistic total: ~2h 20m - 2h 45m
```

**This must clear Kaggle's ~12h session cap with wide margin** (it does, by roughly 5x) before you commit to running it unattended. If a re-measurement after applying §3.5's fix comes back materially higher than 50s/run, stop and re-derive the projection before committing — do not assume the original number still holds after a code change to the exact function in the hot path.

---

## 6. Statistical Testing — In-Session, Using What's Already in `/kaggle/working/`

No download/upload round-trip needed for this stage either — `stats.py`'s Wilcoxon and bootstrap functions operate on the predictions already sitting in the working directory from Cells 2 and 5.

```python
# Cell 6
dynamic_test = pd.read_parquet(f"predictions_dmrgt_all3_grid_{winner}.parquet")
dynamic_test = dynamic_test[dynamic_test["split"] == "test"]   # the reused winner's test rows
frozen_test = pd.read_parquet("predictions_dmrgt_frozen.parquet")
frozen_test = frozen_test[frozen_test["split"] == "test"]

wilcoxon_result = stats.wilcoxon_test(
    dynamic_ic=per_fold_seed_ic(dynamic_test),
    frozen_ic=per_fold_seed_ic(frozen_test),
    cfg=cfg,
)
bootstrap_result = stats.bootstrap_sharpe_test(
    dynamic_returns=daily_returns(dynamic_test, cfg),
    frozen_returns=daily_returns(frozen_test, cfg),
    cfg=cfg,
)

json.dump({
    "wilcoxon": wilcoxon_result,
    "bootstrap_sharpe": bootstrap_result,
    "winning_graph_config": winning_config,
    "conclusion": (
        f"Dynamic graph rank IC exceeds frozen in {wilcoxon_result['wins']}/"
        f"{wilcoxon_result['n_pairs']} folds (p={wilcoxon_result['p_value']:.3f}); "
        f"Sharpe difference p={bootstrap_result['bootstrap_p_value']:.3f}"
    )
}, open("/kaggle/working/phase3_significance_results.json", "w"), indent=2)
```

---

## 7. Output Verification — Cell 7, Before Ending the Session

```python
required = [
    "phase3_sensitivity_grid.csv",
    "phase3_frozen_graph_config.json",
    "graphs_frozen.parquet",
    "predictions_dmrgt_frozen.parquet",
    "phase3_significance_results.json",
    "run_manifest.json",
] + [f"predictions_dmrgt_all3_grid_{v}.parquet" for v in GRID_VARIANT_NAMES]

missing = [f for f in required if not (Path("/kaggle/working") / f).exists()]
assert not missing, f"Missing outputs before session end: {missing}"
print(f"[verify] all {len(required)} expected files present")
```

**Do not consider the run complete until this cell passes.** A committed session's Output tab only preserves what's actually written to `/kaggle/working/` — confirm everything landed before closing the tab.

---

## 8. Local Steps — After the Single Session Finishes (No Round-Trip Mid-Run)

```bash
mv ~/Downloads/*.parquet ~/Downloads/*.json ~/Downloads/*.csv artifacts/
python compare.py     # prints sensitivity table + dynamic-vs-frozen table + both p-values

git add -A
git commit -m "Phase 3: single-session run — grid, frozen twin, significance tests complete"
git tag -a phase3-complete -m "Dynamic vs frozen: p=<wilcoxon>, p=<bootstrap>"
```

---

## 9. Definition of Done

```
DESIGN
[ ] Single notebook, no RUN_GRID branch -- one linear cell sequence (§2)
[ ] dmrgt_all3 is NOT retrained after the grid -- winner's test rows reused (§1)
[ ] Winner selection code provably never touches split=="test" before deciding (§1)
[ ] Frozen graph built in-session to /kaggle/working/, no mid-run dataset version bump (§2, §4)
[ ] Statistical tests run in-session, using files already in /kaggle/working/ (§6)

BUGS RE-VERIFIED FIXED
[ ] Kaggle mount resolution: candidates + rglob fallback (§3.1)
[ ] hashes.json: CACHE_PARQUETS explicit list, generated_at never iterated as a file (§3.2)
[ ] CACHE_PARQUETS reflects the FLAT grid-variant layout, 11 entries (§3.2)
[ ] Random-init fingerprint hashes bytes, all 3 comparisons built on CPU (§3.3)
[ ] Row-count assertion SUMS per-fold counts (§3.4)
[ ] Graph-coverage gaps DROP with a bounded assert, not a hard crash (§3.5)
[ ] win90 variant completes without crashing; drop count is single digits (§3.5)
[ ] All except blocks use traceback.format_exc() + print(..., flush=True) (§3.6)
[ ] GPU capability checked (sm_70+), not just availability (§3.7)
[ ] Root/staging sync verified via hash comparison before any dataset push (§3.8)
[ ] dmrgt-phase-3-cache / dmrgt-phase-3-code are NEW datasets, Phase 2 untouched (§3.9)

TIMING
[ ] Single fold x seed re-timed AFTER the §3.5 fix, not assumed unchanged
[ ] 160-run projection recalculated from the re-measured number
[ ] Projection clears 12h with real margin before committing

OUTPUTS
[ ] Cell 7's output-verification assertion passes before ending the session
[ ] All required files confirmed present, downloaded once, prefixed and archived locally
[ ] git tagged phase3-complete
```

---

## 10. Explicit Anti-Patterns — Do Not Do Any of These

- Do not retrain `dmrgt_all3` after grid selection -- reuse its test rows (§1)
- Do not let winner-selection code load or filter on `split == "test"` at any point before the winner is locked
- Do not build the frozen graph as a separate local step requiring a new cache dataset version mid-pipeline -- build it in-session (§2, §4)
- Do not sum floats for the random-init fingerprint, and do not compare a CPU-built fingerprint against a GPU-built one
- Do not hard-crash `assemble_fold()` on any missing graph date -- bound the tolerance instead (§3.5)
- Do not assume a timing estimate still holds after changing the exact function it measured
- Do not version-bump or otherwise touch the Phase 2 `dmrgt-cache` / `dmrgt-code` datasets
- Do not trust `traceback.print_exc()` inside a loop with concurrent `print()` calls

---

*Research project for academic purposes only -- not investment advice.* Implement this plan to perfection