# DMRGT — Phase 2 Build Specification

> **For the implementing agent:** This is a complete, ordered build brief for Phase 2. Follow the steps **in the order given**. The ordering protects against two specific failure modes: data leakage through the correlation window, and an invalid baseline comparison caused by changing the universe size.
>
> **Read §0.3 before writing any code.** There is a blocking environment question that must be resolved first, and a subtle comparison trap that invalidates the entire phase if handled wrong.
>
> Phase 1 is complete and tagged `phase1-complete`. Do not modify Phase 1 artifacts.

---

## 0. Context

### 0.1 What Phase 2 delivers

**Goal:** Add the dynamic three-relation graph and the R-GCN encoder, expand to ~40 stocks, and demonstrate that the graph model beats every non-graph baseline.

**Completion criterion (from the project proposal):**
> Dynamic 3-relation graph (+corr / −corr / sector) + R-GCN; full DMRGT on ~40 stocks. **Beats no-graph and LSTM baselines.**

**Duration:** ~4 weeks (15 July – 11 August 2026).

**Execution environment:** unchanged from Phase 1. Data, features, graph construction and all evaluation run locally on CPU. Only model training runs on Kaggle GPU (T4). The `predictions.parquet` handoff contract is unchanged.

### 0.2 What Phase 2 is NOT

Phase 2 builds the graph model and shows it beats non-graph baselines. It does **not** prove the *dynamic* graph is what helps — the dynamic-versus-frozen comparison and its significance tests are Phase 3. Do not build the frozen-graph twin or the statistical tests here. See §11.

### 0.3 Two things that must be resolved before any other work

**BLOCKER A — PyTorch Geometric availability on Kaggle.**

The training notebook runs with **Internet OFF** (Rule 9). If `torch-geometric` is not preinstalled in Kaggle's GPU image, `pip install` will fail and training cannot run.

Resolve this first, in a throwaway notebook:

```python
import torch
print(torch.__version__)
try:
    import torch_geometric
    print("PyG OK:", torch_geometric.__version__)
except ImportError:
    print("PyG MISSING — see mitigation below")
```

- **If present:** verify its version is compatible with the installed torch, and record both in the run manifest.
- **If missing:** download matching wheels locally (`torch-geometric` plus any of `torch-scatter` / `torch-sparse` the chosen layer requires, built against the exact torch and CUDA version Kaggle reports), upload them as a third Kaggle Dataset named `dmrgt-wheels`, and `pip install --no-index --find-links /kaggle/input/.../dmrgt-wheels` in the notebook. Internet stays OFF.
- **Fallback if wheels prove unworkable:** implement relation-specific message passing directly with `torch.index_add_` and dense per-relation weight matrices. With ~40 nodes and ~600 edges this is a small amount of code and carries no dependency risk. Document the choice in the manifest.

Do not proceed to §3 until this is settled. Discovering it mid-phase wastes days.

**A determinism caveat that follows from this choice.** Phase 1 achieved bit-identical reruns across all 20 fold × seed combinations. Message-passing layers rely on scatter-add operations, which on CUDA are **non-deterministic by default** — floating-point addition is not associative, and the order in which parallel threads accumulate into the same node varies between runs.

This means the Phase 1 reproducibility guarantee may not survive into Phase 2 unless handled. Required:

1. Keep `torch.manual_seed`, `np.random.seed`, `cudnn.deterministic = True`, `cudnn.benchmark = False` as in Phase 1.
2. Additionally attempt `torch.use_deterministic_algorithms(True)`. If a scatter operation raises because no deterministic implementation exists, catch it and record the fact.
3. **Empirically verify:** run one fold × one seed twice on Kaggle and compare the resulting `y_pred` values exactly. Record whether they match bit-for-bit.
4. If they do not, state this plainly in `docs/PHASE2_RESULTS.md` — "graph model results are reproducible to N decimal places, limited by non-deterministic CUDA scatter operations" — rather than silently dropping the Phase 1 claim.

This is a known and accepted limitation of GPU graph operations, not a defect. But it must be measured and disclosed, not assumed away.

**BLOCKER B — the universe-change comparison trap.**

Phase 1's baseline (rank IC −0.0010 ± 0.0154) was measured on **20 stocks**. Phase 2 uses **40**.

The prediction target is *return minus the equal-weight universe mean*. Changing the universe changes the target. A 40-stock excess return is a different quantity from a 20-stock excess return, computed against a different benchmark, over a different cross-section.

> **The Phase 1 number is NOT a valid control for Phase 2 results. Comparing them directly is invalid and would invalidate the phase.**

**Required mitigation:** §4 re-runs the no-graph Transformer on the 40-stock universe under identical folds and seeds. That re-run — not the Phase 1 number — is the control the graph model is measured against.

The Phase 1 result remains valid as a Phase 1 result and stays in `docs/PHASE1_RESULTS.md`. It is simply not comparable across universes.

---

## 1. Non-Negotiable Rules

Rules 1–9 from the Phase 1 spec remain in force **unchanged**. Re-read them. In particular: scalers fitted per fold on training data only, chronological splits, target uses `t+1`, costs active from the first backtest, all paths via `paths.py`, no downloads from the Kaggle notebook.

Phase 2 adds five:

10. **The correlation window must end at day `t`.** The trailing 60-day correlation used to build the graph for prediction date `t` must use returns from `t-59` through `t` inclusive, and must never touch `t+1`. This is the single most likely new leakage path in Phase 2.
11. **The graph for date `t` is built only from data available at `t`.** No forward-looking correlation, no smoothing that reaches ahead, no reindexing that accidentally shifts the window forward.
12. **Edge direction must be an explicit, documented decision.** Top-k selection is not symmetric: stock A's top-8 may include B while B's top-8 excludes A. Choose directed or symmetrised, implement it deliberately, state it in the manifest, and never let it be an accident of implementation.
13. **Only the graph changes between the control and the graph model.** The Transformer encoder, folds, seeds, scaler procedure, loss, costs and evaluation must be byte-for-byte identical between the §4 control and the §7 model. Any other difference confounds the result.
14. **Re-run the random-model gate after every cache regeneration.** The universe expansion produces new features, new targets and new hashes. The gate that passed on 20 stocks says nothing about the 40-stock cache. Re-running it is mandatory, not optional.
15. **Every model in Phase 2 is trained from randomly initialised weights.** No Phase 1 checkpoint may be loaded, warm-started from, fine-tuned, or used to initialise any layer. See §2.5 for why.

---

## 2. What carries forward from Phase 1

### 2.1 Unchanged — do not modify

| Component | Why it is untouched |
|---|---|
| `splits.py` | Same four regime-spanning walk-forward folds |
| `eval.py` | Same rank IC, directional accuracy, cost-aware backtest, turnover |
| `baselines.py` | Random and persistence remain permanent baselines |
| `paths.py` | Same local/Kaggle resolution (extend only for new artifacts) |
| `compare.py` | Same comparison driver; extended with new model rows |
| Kaggle handoff | Same `predictions.parquet` contract: `fold, seed, date, ticker, y_pred, y_true` |
| Loss | Huber on next-day excess return |
| Costs | `commission_bps: 10`, `slippage_bps: 5` |

**This is the payoff for Phase 1.** The measuring instrument does not change when the model does. Every model in Phase 2 flows through the identical `eval.py` path — if the backtest is generous, it is generous to all of them equally.

### 2.2 Extended

| Component | Change |
|---|---|
| `configs.yaml` | 40 tickers; new `graph:` block |
| `data.py` / `features.py` | Same logic, regenerated over 40 tickers |
| `model.py` | R-GCN added alongside the existing Transformer encoder |
| `train.py` | Accepts `edge_index` and `edge_type`; supports model selection |

### 2.3 New

`graph.py`, an LSTM baseline in `baselines.py`, and a graph edge cache.

### 2.4 Knowledge carried forward — record these, they shape Phase 2

1. **The harness is verified leak-free.** Random scored +0.0061 on 20 stocks. It must be re-verified on 40, but the *method* is proven.
2. **Temporal-only models find no cross-sectional signal.** Rank IC −0.0010, p = 0.781. This is the premise the graph is meant to address.
3. **Bit-identical reruns are achievable.** Seed pinning plus `cudnn.deterministic` reproduced all 20 fold × seed rank ICs to four decimal places. Maintain this.
4. **Turnover is the metric that exposes cost destruction.** The Phase 1 Transformer traded at 0.80 turnover; the baselines at ~1.55. A daily-rebuilt graph may churn holdings far more. **Watch turnover closely in Phase 2** — a graph model with a better rank IC but triple the turnover can be worse after costs.
5. **Persistence is slightly anti-predictive** (−0.0086), consistent with short-horizon mean reversion. Useful context when interpreting graph results.
6. **Kaggle mounts datasets at `/kaggle/input/datasets/<username>/<slug>`**, the P100 is unusable (sm_60), and attaching a dataset requires a session restart.

### 2.5 What is NOT carried forward: trained weights (Rule 15)

**The Phase 1 `checkpoints/` directory is a historical artifact. Do not load from it.**

Every model trained in Phase 2 — the no-graph Transformer control, the LSTM, DMRGT, and both ablations — starts from randomly initialised weights, freshly per fold and per seed.

**Why, precisely:**

1. **The prediction target changed.** Excess return is defined as a stock's return minus the equal-weight universe mean. That mean is now computed over 40 stocks, not 20. Weights fitted to predict outperformance against a 20-stock benchmark were fitted to a different quantity. This is not a domain shift to adapt to; it is a different label.

2. **Fold isolation would be violated.** A Phase 1 checkpoint from fold 4 saw training data through 2021. Warm-starting a Phase 2 fold 1 model (which tests on 2019) from those weights would carry 2021 information into a 2019 prediction — leakage through the parameters rather than the data, and invisible to every existing leakage test.

3. **Rule 13 would be broken.** The control and the graph model must differ *only* in the graph. If one is warm-started and the other is not, initialisation becomes a second difference and the comparison is confounded.

4. **Reproducibility.** Phase 1 achieved bit-identical reruns via seed pinning. Warm-starting from a checkpoint makes every result dependent on an opaque binary file rather than on a seed plus the code.

**What IS carried forward from Phase 1 is the architecture definition, not its learned parameters.** The `TransformerEncoder` class in `model.py` is reused verbatim; the numbers inside it are discarded.

**Required assertion in `train.py`:** at the start of each fold × seed, assert that no state dict has been loaded and that model parameters match a freshly seeded initialisation. A simple, effective form is to construct the model twice under the same seed and assert the parameter tensors are equal, then confirm that a different seed produces different parameters.

**Also record in the run manifest** for every training run: `"init": "random"`, the seed used, and the parameter count. A reviewer must be able to confirm from the manifest alone that nothing was warm-started.

---

## 3. Step 1 — Expand the universe and regenerate the cache

**Files:** `configs.yaml`, `data.py`, `features.py` — all **LOCAL**

### 3.1 Ticker selection

Expand from 20 to ~40 liquid S&P 100 constituents, **stratified across GICS sectors**. Keep all 20 existing tickers and add ~20 more, maintaining sector balance — roughly 5 per sector across 8 sectors.

Sector diversity matters more in Phase 2 than it did in Phase 1, for two concrete reasons:

- **Sector edges (type 2) become uninformative if the universe is concentrated.** If most stocks share a GICS label, that edge type connects nearly everything to everything and carries no information.
- **Negative-correlation edges (type 1) require cross-sector variety.** Genuinely anti-correlated pairs — the oil-producer/airline dynamic — arise across sectors, not within them. A homogeneous universe yields "least positively correlated" pairs rather than true rivals, and the negative edge type is one of Phase 2's headline contributions.

Record each ticker's GICS sector in `configs.yaml` alongside a comment stating the stratification, and assert at load time that no sector holds more than 20% of the universe.

### 3.2 Regenerate

Run the full `data.py` pipeline over the expanded universe. The calendar alignment step is more likely to bite here — a newly added ticker may have a shorter history or different halt days. The existing assertions must still pass:

- All tickers share an identical date index
- Zero NaNs in aligned OHLCV
- No duplicate dates
- Row count plausible

**If a new ticker has insufficient history, replace it rather than shortening the common date range.** Losing years of history across all 40 stocks to accommodate one is a bad trade.

### 3.3 Re-verify the target

The target changes because the universe mean changes. Re-run every check from Phase 1 §4:

- Cross-sectional sum of excess returns ≈ 0 on every date (now across 40, not 20)
- Three manual hand-computations matching to 1e-9
- Label at final date dropped, not filled

### 3.4 The new `graph:` config block

Every graph parameter must be config-driven, not hardcoded. Add:

```yaml
graph:
  corr_window: 60           # trailing days of returns per correlation
  corr_method: pearson      # pearson | spearman — pick one, stay consistent
  k_pos: 8                  # top-k positive edges per stock
  k_neg: 4                  # bottom-k negative edges per stock
  symmetrise: false         # Rule 12 — explicit directed/undirected choice
  include_sector: true
  min_history_days: 60      # graph undefined before this many returns exist

models:
  # Canonical model identifiers — referenced by train.py and the
  # `model` column in predictions.parquet. Never hand-typed elsewhere.
  ids: [lstm, transformer_nograph, dmrgt_pos_only, dmrgt_no_sector, dmrgt_all3]
```

`k_pos` and `k_neg` live here rather than in code specifically because Phase 4's sensitivity grid varies them. Hardcoding now means rewriting `graph.py` in September.

### 3.5 Date-range intersection — a silent misalignment risk

Features and graphs have **different warm-up requirements**:

- Features need the longest of their rolling windows (volume z-score at 60 days is the binding constraint)
- Graphs need `corr_window` (60) days of returns

These will not necessarily produce the same first usable date. If `graph.py` and `data.py` each independently drop their own warm-up rows, you can end up with a features table and a graphs table whose date ranges differ by a few days.

**Required:**

1. Compute the usable date range as the **intersection** of dates present in `features.parquet` and `graphs.parquet`.
2. Assert every date in the final training set appears in both.
3. Log how many dates were dropped by the intersection and why.
4. Add a test: `set(features.date) ⊇ set(graphs.date)` for all dates used in any fold.

A missing graph on a training date must **fail loudly**, never fall back to an empty edge list or the previous day's graph. Silently substituting a stale graph is a leakage-adjacent bug that no other test catches.

### 3.6 Regenerate hashes and re-upload

New `hashes.json`. Push a new `dmrgt-cache` dataset version with a message recording the universe change. Record the new version number.

---

## 4. Step 2 — Re-establish baselines on the 40-stock universe

**MANDATORY. This step is what makes Phase 2's comparison valid.** Do not skip it, and do not reuse Phase 1 numbers.

### 4.1 Re-run the random-model gate (Rule 14)

Run `random_model` through the harness on the new 40-stock cache. Required results, unchanged from Phase 1:

| Metric | Required |
|---|---|
| Mean rank IC | within ±0.02 of 0.00 |
| Directional accuracy | within ±2% of 50% |
| Backtest return after costs | negative |

**If this fails, STOP.** The universe expansion introduced leakage. Investigate in the Phase 1 §5.4 order: target alignment, rolling windows, scaler scope, fold overlap, backtest timing.

### 4.2 Re-run persistence

Deterministic. Record for all folds.

### 4.3 Re-run the no-graph Transformer — the control

**Trained from scratch on the 40-stock cache (Rule 15).** Identical architecture class, identical folds, identical five seeds, identical loss and costs. **The only difference from Phase 1 is the universe size.**

To be explicit about what "re-run" means here: this is a full retraining from random initialisation across all 4 folds × 5 seeds on the new cache. It is not an evaluation of the Phase 1 checkpoints against 40-stock data, and it is not a fine-tune of them. Loading `artifacts/phase1_*` checkpoints at any point in Phase 2 is a defect.

This produces the number the graph model must beat. Record it in `docs/PHASE2_RESULTS.md` as `transformer_nograph_40`, clearly distinguished from the Phase 1 20-stock figure.

> Expected outcome: another rank IC near zero. That is fine and probably correct. The point is that it is measured *on the same universe* as the graph model.

### 4.4 Build and run the LSTM baseline

Deferred from Phase 1; required by the Phase 2 completion criterion.

**Place it in `model.py`, not `baselines.py`.** The LSTM has learnable parameters and must be trained per fold per seed on the GPU, exactly like the Transformer. `baselines.py` holds only the two closed-form models (random, persistence) that require no training and run locally. Putting a trained model there would break that separation and the local/Kaggle split.

Architecture:

```
Input:  X = [n_stocks, 30, 8]
  → LSTM: 2 layers, hidden 64, batch_first
  → Take final hidden state              [n_stocks, 64]
  → MLP head: 64 → 32 → 1                [n_stocks]
```

Requirements:
- **One shared LSTM across all stocks**, same as the Transformer — not per-stock models
- Same Huber loss, same folds, same 5 seeds, same scaler procedure
- Parameter count within roughly 2× of the Transformer's, so the comparison is about architecture rather than capacity — record both counts in the manifest

The LSTM is a fair-fight control: a strong sequence model with no graph. If DMRGT beats the Transformer but not the LSTM, the graph is not the source of the improvement.

---

## 5. Step 3 — `graph.py`

**File:** `graph.py` — **LOCAL** (graphs are cached and uploaded, never computed on Kaggle)

### 5.1 What it produces

For **every trading day** in the usable date range, a tagged edge list:

```
edge_index : [2, E]   row 0 = source node, row 1 = destination node
edge_type  : [E]      0 = +corr, 1 = -corr, 2 = sector
```

With 40 stocks the count is determined, not approximate:

| Edge type | Count (directed, before symmetrisation) |
|---|---|
| 0 — positive correlation | 40 × 8 = **320** |
| 1 — negative correlation | 40 × 4 = **160** |
| 2 — sector | Σ over sectors of n(n−1); with 8 sectors of 5 stocks: 8 × 20 = **160** |
| **Total** | **~640 per day** |

The sector count depends on your actual sector distribution — compute it once from `configs.yaml` and **assert the exact expected number**, rather than checking a loose range. Types 0 and 1 are exactly 320 and 160 on every date by construction; if either differs, the top-k selection is broken.

(MDGNN used 62.5M edges on a V100. The size difference is why this trains in minutes on free hardware.)

### 5.2 Construction, per date `t`

1. **Take the trailing 60 days of log-returns ending at `t` inclusive.** Use returns, not prices — two stocks both drifting upward over years look correlated in price even when their daily moves are unrelated.
2. **Compute the 40×40 Spearman or Pearson correlation matrix.** Choose one, state which in the config, and be consistent.
3. **Positive edges (type 0):** for each stock, its **top 8** most positively correlated peers, excluding itself.
4. **Negative edges (type 1):** for each stock, its **bottom 4** most negatively correlated peers.
5. **Sector edges (type 2):** connect every pair sharing a GICS label. Static across all dates.
6. **Concatenate** into one `edge_index` / `edge_type` pair.

### 5.3 Rule 12 — edge direction

Top-k selection is inherently directed. Decide explicitly:

- **Directed:** an edge A→B means "B is among A's top-8." Preserves the asymmetry of the selection.
- **Symmetrised:** add both A→B and B→A whenever either selects the other. Denser, and matches the undirected assumption of many GNN formulations.

Either is defensible. **Implement one deliberately, record the choice in `configs.yaml` and the run manifest, and add a test asserting the chosen property holds.** An implicit choice here is a silent confound.

### 5.4 Leakage guards — the critical tests

Rules 10 and 11 concentrate here. Implement in `tests/test_graph_leakage.py`:

| Test | Method |
|---|---|
| Window ends at `t` | Corrupt all data after date `t` with NaN, rebuild the graph for `t`, assert the edge list is byte-identical |
| Window has correct length | Assert exactly 60 return observations feed each correlation |
| No self-loops | Assert no edge where source == destination |
| Correct edge counts | Assert exactly 8 positive and 4 negative edges per stock per day (before any symmetrisation) |
| Sector edges static | Assert type-2 edges are identical across all dates |
| Every stock connected | Assert no isolated node on any date — this is the property rank-based selection guarantees and THGNN's 0.6 threshold does not |

The first test is the most important in the whole phase. Run it against at least 20 randomly chosen dates, not one.

### 5.5 Caching

Recomputing ~2,400 correlation matrices on every training run wastes hours for an answer that never changes. Compute once, cache to `cache/graphs.parquet` in long format:

```
date | src | dst | edge_type
```

Roughly 600 × 2,400 ≈ 1.4M rows — a few MB. Add its hash to `hashes.json` and include it in the `dmrgt-cache` dataset.

### 5.6 Kaggle-side changes required by the graph cache

The graph cache is a fourth data file, so several Phase 1 components need extending:

| Component | Change |
|---|---|
| `paths.py` | Add `GRAPHS = CACHE_DIR / "graphs.parquet"`; extend `CACHE_PARQUETS` from 3 entries to 4 |
| `hashes.json` | Add the `graphs.parquet` hash; `data.py` must update rather than overwrite |
| Notebook Cell 1 | Add `paths.GRAPHS.exists()` to the existence assertions |
| Notebook Cell 2 | No change if it delegates to `train.verify_cache_hashes` — verify it does |
| `tests/test_cache_integrity.py` | Expect 4 hashes, not 3 |
| `dmrgt-cache` dataset | Re-version to include `graphs.parquet` |

The counting guard in `verify_cache_hashes` (`checked == 3`) will now fail. Update it to read from `paths.CACHE_PARQUETS` rather than a literal, so it never needs manual editing again.

### 5.7 Diagnostics to produce

In `notebooks/graph_verification.ipynb`, so anomalies are visible:

- Edge count per day over time — should be near-constant by construction
- Mean positive and negative correlation strength over time — expect regime shifts, notably a spike in early 2020
- A rendered graph snapshot for one calm date and one crisis date, coloured by edge type
- Edge persistence: what fraction of a stock's top-8 partners remain the same from one day to the next

That last one is worth attention. **If day-to-day edge overlap is near 100%, the graph is barely dynamic and Phase 3's dynamic-versus-frozen comparison may find little difference.** If it is very low, the graph is unstable and turnover will suffer. Record the number — it is genuinely informative about whether the project's central hypothesis has room to be true.

---

## 6. Step 4 — `model.py`: add the R-GCN

**File:** `model.py` — **shared** (runs on both local CPU and Kaggle GPU)

### 6.1 Architecture

```
Input:  X = [n_stocks, 30, 8], edge_index [2, E], edge_type [E]

  Transformer encoder (UNCHANGED from Phase 1)
    → H = [n_stocks, 64]

  R-GCN layer
    input:  H, edge_index, edge_type
    3 relation types, separate learned weight matrix each
    → Z = [n_stocks, 64]

  concat([H, Z])                          [n_stocks, 128]
    → Linear 128 → 64 → ReLU → dropout
    → Linear 64 → 1
Output: [n_stocks] predicted excess returns
```

### 6.2 Requirements

- **The Transformer encoder must be byte-identical to Phase 1's.** Rule 13. If you refactor it, the §4 control must be re-run with the refactored version.
- **Relation-specific weights are the whole point.** A plain GCN with one shared weight matrix averages friends and rivals into meaningless mush. "My partner rose" and "my rival rose" must be able to push the prediction in opposite directions. Assert at construction that the layer holds exactly 3 distinct weight matrices.
- **Keep `H` in the concatenation.** Z alone means the stock's own signal survives only if it passes through the graph intact. Concatenating gives the head a direct line to the stock's own evidence, so a noisy graph on a given day cannot destroy the prediction.
- **Interface:** `forward(X, edge_index, edge_type)`. Non-graph models keep `forward(X)`. `train.py` dispatches on model type — do not force graph arguments into models that ignore them.

### 6.3 Ablation variants required

Both are needed for §8's comparison table:

- `--relations pos_only` — type 0 edges only. Tests whether negative edges add anything (weakness W2-C).
- `--relations no_sector` — types 0 and 1 only.

Implement as config flags filtering the edge list, not as separate model classes.

---

## 7. Step 5 — `train.py`: graph-aware training

**File:** `train.py` — **KAGGLE**

### 7.1 Changes from Phase 1

1. Load `cache/graphs.parquet` alongside features and targets, via `paths.py`.
2. Extend `verify_cache_hashes` to include `graphs.parquet`; update `paths.CACHE_PARQUETS`.
3. For each sample (one trading day), assemble `edge_index` and `edge_type` for that date alongside `X` and `y`.
4. Add `--model {transformer,lstm,dmrgt}` so all Phase 2 models train through one code path.
5. Dispatch: graph models receive edges, non-graph models do not.

### 7.2 New assertions

Alongside all Phase 1 assertions:

- Every sample's edge list date matches its feature window's final date
- `edge_type` contains only values in {0, 1, 2}
- `edge_index` node indices are all `< n_stocks`
- Edge count per day falls in the expected range (fail loudly if a day has zero edges)
- The universe ordering used to build `X` is identical to the node indexing in `edge_index` — **a mismatch here silently scrambles which stock is which, and no assertion elsewhere will catch it**

That last one deserves a dedicated test. It is the kind of bug that produces plausible-looking, entirely meaningless results.

### 7.3 Output contract — CHANGED, and this is not optional

Phase 1 trained one model, so `predictions.parquet` needed no way to say which model produced a row. **Phase 2 trains five** (LSTM, no-graph Transformer, DMRGT pos-only, DMRGT no-sector, DMRGT all-3). The Phase 1 schema cannot represent that.

**Required schema — add a `model` column:**

```
model | fold | seed | date | ticker | y_pred | y_true
```

`model` takes exactly one of: `lstm`, `transformer_nograph`, `dmrgt_pos_only`, `dmrgt_no_sector`, `dmrgt_all3`. Define these as a constant in `configs.yaml`; do not let the strings be typed by hand in five places.

**Row count sanity check:** 5 models × 4 folds × 5 seeds × ~250 test days × 40 stocks ≈ **1,000,000 rows**. That is roughly ten times Phase 1's file, still only a few tens of MB in Parquet. If your row count is materially different, a model or a fold silently failed to run — assert the expected count before writing.

**Consequences to handle:**

- `eval.py` must group by `model` in addition to `fold` and `seed`. This is the one Phase 1 evaluation file that needs touching — keep the change to grouping only; do not alter the rank IC or backtest logic.
- `compare.py` reads the `model` column rather than inferring the model from the filename.
- Phase 1's `phase1_predictions.parquet`, if retained, lacks this column. Do not attempt to load both schemas in one call.

**Alternative if the single-file approach proves awkward on Kaggle:** emit one file per model (`predictions_dmrgt_all3.parquet` etc.) and concatenate locally, adding the `model` column at concatenation time. Either is acceptable — choose one, state it in the manifest, and be consistent.

**Everything else about the contract is unchanged.** `train.py` still computes no rank IC, no Sharpe, no backtest. All evaluation remains local.

Add to `run_manifest.json`: model type, relation configuration, edge-direction choice, graph cache hash, PyG version (or the note that a hand-rolled layer was used), `"init": "random"`, and parameter counts for every model trained.

---

## 8. Step 6 — Comparison and Phase 2 sign-off

**Runs locally.** Download `predictions.parquet` per model and score through the unchanged `eval.py`.

### 8.1 The required table

All models on the **same 40-stock universe**, same folds, same seeds, same costs:

| Model | Mean rank IC | Dir. acc | Sharpe (after costs) | Max DD | Turnover | Params |
|---|---|---|---|---|---|---|
| Random | | | | | | — |
| Persistence | | | | | | — |
| LSTM (no graph) | | | | | | |
| Transformer (no graph) — **the control** | | | | | | |
| DMRGT, +corr only | | | | | | |
| DMRGT, no sector | | | | | | |
| **DMRGT, all 3 relations** | | | | | | |

Report mean ± std across 5 seeds, and per-fold as well as aggregate.

### 8.2 How to read it

**The completion criterion is met if DMRGT (all 3 relations) beats BOTH the no-graph Transformer and the LSTM on rank IC.**

Three things to check before claiming success:

1. **Per-fold consistency.** A model that wins only in the calm 2019 fold is a different story from one that wins across all four. Report both.
2. **Turnover.** If DMRGT's rank IC improves but turnover triples, check whether the Sharpe advantage survives costs. A daily-rebuilt graph can churn holdings hard. This is exactly the failure the surveyed papers hid by ignoring costs — do not repeat it.
3. **Seed spread.** If the std across seeds exceeds the gap to the control, the improvement is not yet distinguishable from noise. Report it plainly; Phase 3 supplies the formal test.

### 8.3 If DMRGT does not beat the baselines

This is a legitimate outcome and must be reported honestly, not tuned away. Before concluding, verify in order:

1. Did the graph leakage tests pass? (A leaking graph usually *inflates* results, so failure here is more likely a real negative.)
2. Is edge persistence so high the graph is effectively static?
3. Is the node ordering aligned with `edge_index`? (§7.2)
4. Do the ablations show *any* relation type contributing?

If the answer is a genuine null result, record it. It is a finding, and Phase 3's comparison still runs — it would then be testing whether dynamism helps a model that itself does not help, which is a narrower but still reportable result.

**Do not tune hyperparameters to manufacture a win.** The sensitivity grid is Phase 4, and it is run over *all* models equally or not at all.

---

## 9. Phase 2 Definition of Done

```
ENVIRONMENT
□ PyG availability on Kaggle resolved (preinstalled / wheels dataset / hand-rolled)
□ Chosen approach recorded in the run manifest

DATA
□ Universe expanded to ~40, stratified; no sector exceeds 20%
□ Calendar alignment assertions pass on the expanded universe
□ Target re-verified: cross-sectional sum ~ 0 on every date (40 stocks)
□ 3 manual hand-computations match to 1e-9
□ hashes.json regenerated; dmrgt-cache re-versioned; version recorded

BASELINES ON 40 STOCKS  ← makes the comparison valid
□ Random-model gate RE-RUN and passing (rank IC within +/-0.02)
□ Persistence re-run and recorded
□ No-graph Transformer re-run on 40 stocks — THE CONTROL
□ Control trained FROM SCRATCH, no Phase 1 checkpoint loaded (Rule 15)
□ train.py asserts random initialisation per fold x seed
□ run_manifest records "init": "random" for every training run
□ LSTM baseline built and run
□ Parameter counts recorded for LSTM and Transformer

GRAPH
□ graph.py builds daily 3-relation edge lists
□ Correlation window ends at t (NaN-corruption test, >=20 dates)
□ Exactly 8 positive + 4 negative edges per stock per day
□ No self-loops; no isolated nodes on any date
□ Sector edges identical across all dates
□ Edge-direction choice made deliberately, documented, and tested
□ graphs.parquet cached and hashed
□ paths.CACHE_PARQUETS extended to 4 files; counting guard reads from it
□ Date-range intersection of features and graphs computed and asserted
□ No fallback to stale or empty graphs on a missing date — fails loudly
□ Sector edge count asserted exactly, not range-checked
□ Edge persistence measured and recorded
□ graph: block in configs.yaml drives every graph parameter

MODEL
□ Transformer encoder byte-identical to Phase 1
□ R-GCN has exactly 3 distinct relation weight matrices (asserted)
□ [H, Z] concatenation retained
□ Ablation flags implemented: pos_only, no_sector

TRAINING
□ Node ordering in X matches edge_index indexing (dedicated test)
□ Edge-date alignment asserted per sample
□ All Phase 1 assertions still present and passing
□ 5 seeds per fold per model (5 models x 4 folds x 5 seeds = 100 runs)
□ predictions.parquet has a `model` column; row count asserted (~1M)
□ eval.py groups by model in addition to fold and seed
□ Determinism empirically checked: one fold x seed run twice, y_pred compared
□ Reproducibility limitation (if any) disclosed in PHASE2_RESULTS.md

RESULTS
□ Full comparison table on the 40-stock universe
□ Per-fold results reported alongside aggregate
□ Turnover reported for every model
□ Completion criterion assessed: DMRGT beats no-graph Transformer AND LSTM
□ docs/PHASE2_RESULTS.md written
□ Artifacts prefixed phase2_; git tagged phase2-complete
```

---

## 9.1 Update `verify_env.py`

The Phase 1 local-vs-Kaggle cross-check must be extended:

- Compare the 4 parquet hashes, not 3
- Report edge count for the checked fold's first test date
- Report the date-range intersection size
- Still exclude validation loss and rank IC from the comparison (structural properties only — this was a documented Phase 1 defect)

---

## 10. Do not do these

- Do not compare Phase 2 results against the Phase 1 20-stock baseline (§0.3, Blocker B)
- Do not modify the Transformer encoder without re-running the §4 control
- Do not load, warm-start from, or fine-tune any Phase 1 checkpoint (Rule 15)
- Do not compute graphs inside the Kaggle notebook — they are cached and uploaded
- Do not tune hyperparameters to make DMRGT win
- Do not skip the random-model re-run because it passed in Phase 1
- Do not overwrite `phase1_*` artifacts
- Do not build the frozen-graph twin or any significance test (Phase 3)

---

## 11. Explicitly out of scope — Phase 3 and later

- Frozen-graph twin and the dynamic-versus-static comparison
- Wilcoxon signed-rank and stationary-bootstrap Sharpe tests
- Sensitivity grid over k, correlation window, lookback (Phase 4)
- `explain.py`, attention maps, Gradient × Input, edge ablation (Phase 5)
- News edges, supply-chain edges, additional global features
- Multi-horizon prediction heads

---

## 12. Reference facts (do not alter)

- **Universe:** ~40 liquid S&P 100 constituents, sector-stratified
- **Input:** `[n_stocks, 30, 8]` — one sample is one trading day, all stocks at once
- **Graph:** rebuilt daily from trailing 60-day return correlations; top-8 positive, bottom-4 negative per stock, plus static same-sector edges; ~500–700 edges per day
- **Edge types:** 0 = +corr, 1 = −corr, 2 = sector
- **Transformer:** 2 layers, 4 heads, causal mask → `H = [n_stocks, 64]`
- **R-GCN:** 3 relation types, separate weights → `Z = [n_stocks, 64]`
- **Head:** concat `[H, Z]` = 128 → MLP → 1 per stock
- **Target:** next-day return minus equal-weight universe mean. **Loss:** Huber
- **Retraining:** once per walk-forward fold, never daily
- **Initialisation:** every Phase 2 model trains from random weights; no Phase 1 checkpoint is reused
- **Training runs in Phase 2:** 5 models (LSTM, no-graph Transformer, DMRGT pos-only, DMRGT no-sector, DMRGT all-3) x 4 folds x 5 seeds = 100 runs
- **Seeds:** ≥ 5 per configuration
- **Costs:** commission 10 bps, slippage 5 bps, active from the first backtest
- **Phase 1 baseline (20 stocks, NOT comparable):** rank IC −0.0010 ± 0.0154, p = 0.781

---

*Research project for academic purposes only — not investment advice.*
