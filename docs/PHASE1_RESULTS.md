# DMRGT Phase 1 — Results and Archival Record

> **Status:** Complete. The leakage gate passed, the no-graph baseline is
> established, and the pipeline is frozen for Phase 2.
>
> **Commit:** `phase1-complete` tag
> **Kaggle run timestamp:** `2026-08-06T17:29:56Z`
> **Date archived:** 2026-08-06

---

## 1. What Phase 1 Set Out to Do

Phase 1's goal was a **trustworthy measuring instrument**, not a performant
model. The completion criterion from the build spec (§0):

> *The pipeline runs end-to-end with no future leakage, and transaction
> costs and slippage are active from day one.*

Every design decision — chronological splits, per-fold scaler fitting,
causal masking, the local/Kaggle execution split, the random-model gate —
serves that single goal. A mediocre model measured honestly is a success.
A great-looking model with a leak is a total failure that invalidates
every later phase.

Phase 1 deliberately excluded the graph, the R-GCN, the 40-stock
expansion, significance testing, and explainability. Those are Phase 2+.

---

## 2. The Leakage Gate

The single most important result in this document.

A model returning pure random numbers, ignoring all inputs, was run
through the complete evaluation harness across all four walk-forward
folds. Transaction costs (10 bps commission + 5 bps slippage) were
applied.

| Metric | Required | Observed | Pass? |
|---|---|---|---|
| Mean rank IC | Within ±0.02 of 0.00 | +0.0061 ± 0.0084 | ✓ |
| Directional accuracy | Within ±2% of 50% | 49.5% | ✓ |
| Backtest return after costs | Negative | Sharpe −4.66 | ✓ |

A random model cannot exploit genuine signal. If it had scored
meaningfully above zero, the only possible explanation would be that the
evaluation pipeline itself was leaking future information. It did not.

**Every subsequent result in this project is measured by an instrument
that has been demonstrated — not assumed — to be incapable of
manufacturing signal from noise.**

The five most common leakage sources were explicitly ruled out:

1. **Target off-by-one.** The label at date `t` is the excess return
   realised on `t+1`. Verified by manual hand-computation of 3 random
   (date, ticker) pairs to 1e-9 precision, and by the cross-sectional
   sum ≈ 0 assertion on every date.
2. **Rolling windows including future data.** Every feature window ends
   at day `t`. Verified by the NaN-corruption lookahead test: corrupting
   all data after `t` with NaN leaves the feature at `t` unchanged.
3. **Scaler fitted outside training window.** `StandardScaler` is fitted
   on training dates only, then applied frozen to validation and test.
   Asserted in `train.assemble_fold()`.
4. **Train/test date overlap.** Walk-forward folds are chronologically
   disjoint. Asserted: `max(train_dates) < min(test_dates)` for every
   fold, and `train_dates ∩ test_dates = ∅`.
5. **Backtest using future close to decide trades.** The backtest uses
   `y_pred` (computed from day `t` data) to form portfolios; `y_true`
   (realised on `t+1`) is only used to *settle* the trade.

---

## 3. The Baseline Result

### Headline table

| Model | Mean rank IC | Directional acc. | Sharpe (after costs) | Max drawdown | Turnover |
|---|---|---|---|---|---|
| Random | +0.0061 ± 0.0084 | 0.495 | −4.66 | 0.541 | 1.60 |
| Persistence | −0.0086 ± 0.0046 | 0.498 | −2.04 | 0.398 | 1.54 |
| Transformer (no graph) | **−0.0010 ± 0.0154** | 0.500 | −1.40 | 0.328 | 0.80 |

Transformer statistics across 20 runs (4 folds × 5 seeds):

```
mean rank IC : -0.00097
std          :  0.01536
one-sample t : t = -0.282, p = 0.781   (not distinguishable from zero)
```

### Per-fold Transformer results

| Fold | Regime | Mean rank IC | Sharpe | Turnover |
|---|---|---|---|---|
| fold_1_2019_calm | Calm | −0.0121 ± 0.0062 | −2.21 | 0.79 |
| fold_2_2020_covid | COVID crash | −0.0057 ± 0.0149 | −0.64 | 0.77 |
| fold_3_2021 | Recovery | +0.0024 ± 0.0131 | −1.19 | 0.85 |
| fold_4_2022_rates | Rate shock | +0.0115 ± 0.0173 | −1.57 | 0.79 |

### Interpretation

A Transformer reading each stock's own 30-day history in isolation — two
encoder layers, four attention heads, causal mask, 64-dim hidden — found
no usable cross-sectional signal. The mean rank IC of −0.0010 is
statistically indistinguishable from zero (p = 0.781, one-sample
t-test).

**The model is not collapsed.** Predictions vary meaningfully across
stocks (std ~0.0005, ranging to ±0.008), and rank IC differs across
seeds rather than being pinned at a single value. The model learned
"excess returns are mostly unpredictable, so guess near zero and hedge."
Under Huber loss on a near-random target, that is the loss-minimising
answer, and it is why the five seeds converged to near-identical
validation losses.

**Turnover is a genuine finding.** The Transformer's turnover (0.80) is
roughly half that of both baselines (1.54–1.60), and its maximum
drawdown (0.328) is correspondingly the lowest of the three. Having
found no exploitable signal, the model learned to predict close to zero
and therefore to rebalance rarely. Its comparatively less-negative
Sharpe reflects restraint in trading, not skill in selection.

**Persistence is independently interesting.** Predicting that tomorrow's
excess return equals today's scored −0.0086 — slightly worse than
random. This is consistent with short-horizon mean reversion in relative
returns: yesterday's outperformers tend to give back a portion of that
outperformance.

---

## 4. Why This Outcome Is the Right One

This result is not a failure of the model; it is the **premise of the
project, measured rather than assumed.**

The problem statement argues that models treating each asset in isolation
cannot capture cross-asset influence — sector momentum, flight-to-safety,
volatility spillover, supply-chain propagation. Phase 1 demonstrates that
empirically on this specific universe, with this specific harness:
temporal patterns alone carry no usable cross-sectional signal.

This establishes the **control condition** against which the dynamic
multi-relational graph is measured in Phases 2 and 3.

The baseline was deliberately not tuned. Phase 3's central claim is a
*difference* — the improvement attributable to the graph. Optimising the
control would compress that difference and make it impossible to
distinguish graph value from baseline under-tuning.

---

## 5. Reproducibility Record

### Data provenance

| File | SHA-256 |
|---|---|
| `prices.parquet` | `3507138cb2bf083af9adc23ea4e4149d33f4590dd4ce4bf7a7da384c92dda9fc` |
| `features.parquet` | `76d1dcc20148c07e235e0407e6c4e32d1dcb7ed3ae72b59a0265de89bcbad67c` |
| `targets.parquet` | `7967f468a3583135dcd35e7d0d22fd796020f3b56a3e4ba8d609522664f2e331` |

### Fold structure

| Fold | Train days | Val days | Test days | Test window |
|---|---|---|---|---|
| 1 | 776 | 142 | 252 | 2019-01-02 → 2019-12-31 |
| 2 | 990 | 180 | 253 | 2020-01-02 → 2020-12-31 |
| 3 | 1,205 | 218 | 252 | 2021-01-04 → 2021-12-31 |
| 4 | 1,419 | 256 | 251 | 2022-01-03 → 2022-12-30 |

### Environment

```
GPU     : Tesla T4 (sm_75)
torch   : 2.10.0+cu128
numpy   : 2.0.2
pandas  : 2.3.3
python  : 3.12.13
```

### Kaggle dataset versions

| Dataset | Version | Notes |
|---|---|---|
| `dmrgt-cache` | v1 (initial) | prices, features, targets, hashes.json |
| `dmrgt-code` | v2 (cache_hashes fix) | train.py, model.py, paths.py, splits.py, data.py, configs.yaml |

The committed Kaggle notebook is at the **Save & Run All (Commit)**
version corresponding to timestamp `2026-08-06T17:29:56Z`.

### Bit-identical rerun

The second Kaggle run (after the `cache_hashes` fix in `train.py`)
produced rank IC values **identical to four decimal places across all 20
fold × seed combinations** compared to the first run. Seed pinning
(`torch.manual_seed`, `np.random.seed`) and `cudnn.deterministic = True`
are confirmed working. This is a stronger reproducibility claim than any
of the three surveyed papers makes.

### Local/Kaggle cross-check

`verify_env.py` confirms the local and Kaggle environments read identical
data and construct identical folds:

```
[verify] COMPARE: PASS — local and Kaggle agree on every structural property.
```

| Property | Local | Kaggle manifest | Match? |
|---|---|---|---|
| n_train_days (fold 1) | 776 | 776 | ✓ |
| n_val_days (fold 1) | 142 | 142 | ✓ |
| n_test_days (fold 1) | 252 | 252 | ✓ |
| train_start | 2015-05-11 | 2015-05-11 | ✓ |
| train_end | 2018-06-07 | 2018-06-07 | ✓ |
| test_start | 2019-01-02 | 2019-01-02 | ✓ |
| test_end | 2019-12-31 | 2019-12-31 | ✓ |
| prices.parquet SHA-256 | match | match | ✓ |
| features.parquet SHA-256 | match | match | ✓ |
| targets.parquet SHA-256 | match | match | ✓ |

### Predictions artifact

- **File:** `artifacts/predictions.parquet` (NOT committed — regenerable)
- **Rows:** 100,800 (20 stocks × ~252 test days × 4 folds × 5 seeds)
- **Columns:** `fold`, `seed`, `date`, `ticker`, `y_pred`, `y_true`
- **y_pred fingerprint:** mean = −0.000031, std = 0.001382, range = [−0.02198, +0.02115]

---

## 6. Known Limitations of Phase 1

These are stated honestly; a reviewer will find them otherwise.

- **Survivorship bias.** Tickers were selected from the current S&P 100,
  so only companies that survived to the present are included. Companies
  that were delisted, bankrupted, or acquired between 2015 and 2024 are
  absent.
- **20 stocks, not 40.** Phase 1 uses a reduced universe for
  debuggability and faster iteration. Phase 2 expands to ~40 stocks
  stratified across GICS sectors.
- **Single hyperparameter configuration.** No tuning was performed,
  deliberately. The sensitivity grid over lookback window, learning
  rate, hidden dimension, and number of layers is Phase 4.
- **One-day horizon only.** Results do not generalise to longer holding
  periods without separate training and validation.
- **No graph.** By design — this is the control condition. The graph,
  R-GCN, and multi-relational edges are Phase 2.

---

## 7. Definition of Done

Reproduced from the build spec (§9). Every item is checked.

```
[x] configs.yaml drives every module; nothing hardcoded
[x] paths.py resolves correctly in BOTH local and Kaggle environments
[x] 20 sector-stratified tickers downloaded, aligned, cached, hashed
[x] Re-running data.py produces an identical hash
[x] Target verified: cross-sectional sum ≈ 0 on every date
[x] Target verified: 3 manual hand-computations match to 1e-9
[x] Walk-forward folds generated; zero train/test overlap asserted
[x] Random model scores rank IC within ±0.02 of zero  ← THE GATE
[x] Random model loses money after costs
[x] Persistence baseline recorded across all folds
[x] All 8 features built, plotted and inspected individually
[x] NaN-corruption lookahead test passes for all 8 features
[x] Features cached UNSCALED
[x] Scaler fitted per fold on training data only
[x] cache/ uploaded to Kaggle as a VERSIONED dataset; version recorded
[x] Kaggle notebook verifies parquet hashes before training; halts on mismatch
[x] Kaggle notebook asserts GPU is available
[x] Transformer trains; one shared model across stocks; causal mask present
[x] 5 seeds per fold; train.py emits predictions.parquet, not metrics
[x] Kaggle run executed via Save & Run All (Commit), not interactively
[x] predictions.parquet + run_manifest.json downloaded to artifacts/
[x] One fold verified locally on CPU; agrees with Kaggle within tolerance
[x] Backtest includes commission AND slippage; turnover reported
[x] Results table complete: random vs persistence vs Transformer
[x] All tests in tests/ pass
[x] README documents how to reproduce with one command
[x] Local/Kaggle cross-check: verify_env.py PASS
```

---

## 8. What Phase 2 Inherits Unchanged

| Component | Status entering Phase 2 |
|---|---|
| `data.py`, `features.py` | Unchanged — cache is regenerated only when the universe expands to 40 |
| `splits.py` | Unchanged — same four regime-spanning folds |
| `eval.py` | Unchanged — same rank IC and cost-aware backtest |
| `baselines.py` | Unchanged — random and persistence remain permanent baselines |
| `paths.py`, Kaggle handoff | Unchanged — same `predictions.parquet` contract |
| The no-graph Transformer | **Becomes a permanent baseline**, not a discarded prototype |
| `model.py` | Extended — R-GCN added alongside the existing encoder |
| `train.py` | Extended — accepts `edge_index` and `edge_type` |
| `configs.yaml` | Extended — 40 tickers, graph parameters |

The no-graph Transformer result recorded in §3 is the number Phase 2
must beat and Phase 3 must beat with statistical significance. It is not
a throwaway.

---

*Research project for academic purposes only — not investment advice.*
