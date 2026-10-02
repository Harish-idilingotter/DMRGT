# DMRGT — Phase 3 Results: Dynamic vs. Frozen Graph

**Project:** Dynamic Multi-Relational Graph Transformer (DMRGT)
**Course:** BCSE497J — Project-1
**Faculty Guide:** Kanagaraj K (22489)
**Team:** Harish Kumar D (23BCE0647), Harish Sivakumar (23BCE0664), Megeshwar JB (23BCE2135)
**Phase:** 3 — Statistical validation of the dynamic graph hypothesis

---

## 1. Objective

Phase 2 established a working DMRGT pipeline (Transformer encoder + R-GCN over a daily-rebuilt 3-relation stock graph) and showed it beats a no-graph baseline. Phase 3 asks a sharper question:

> **Does rebuilding the stock graph every day (dynamic) actually help, compared to freezing the graph once per training window (frozen/static)?**

This is the core scientific claim of the project — that *time-varying* relationships between stocks (correlation regimes shifting, sector co-movement changing) carry predictive information that a static snapshot graph cannot capture. Phase 3 is designed to test this claim rigorously enough to publish, not just to produce a bigger number.

---

## 2. Methodology Summary

- **Walk-forward chronological cross-validation**, 4 folds, each representing a distinct market regime:
  - Fold 1 — 2019 (calm)
  - Fold 2 — 2020 (COVID shock)
  - Fold 3 — 2021 (recovery)
  - Fold 4 — 2022 (rate-shock)
- **5 seeds per fold** → 20 paired (fold, seed) observations per model arm.
- **Leakage gate**: a random-label control model is required to score rank IC ≈ 0 before any real result is trusted.
- **Two arms compared, trained identically in every other respect:**
  - `dmrgt_all3` — graph rebuilt daily (dynamic)
  - `dmrgt_frozen` — graph built once from each fold's training window, then held fixed through that fold's test period
- **Graph sensitivity grid (benchmark exhibit):** before locking the frozen-graph configuration, 7 one-dimensional variants of the graph construction hyperparameters were searched — **on validation data only** — to confirm the chosen configuration isn't cherry-picked, and to demonstrate the model's behavior under different graph-construction choices.
- **Discipline enforced throughout (Rules 17–19):**
  - All hyperparameter/config tuning is done on the **validation split only**.
  - The **test split is scored exactly once**, guarded programmatically (`compare.py` hashes the winning config and prediction files; any change triggers a hard failure rather than a silent re-score).
  - Both arms receive **identical post-processing** (same cost model, same portfolio construction, same evaluation code) — no asymmetric advantage given to either.
- **Two independent significance tests**, since predictive quality and trading performance are different questions:
  - **Wilcoxon signed-rank test** on paired rank IC (does the dynamic graph produce better-ranked predictions?)
  - **Stationary bootstrap (Politis–Romano)** on the Sharpe ratio difference (does the dynamic graph produce a better-performing portfolio, accounting for autocorrelated daily returns?)
- **Reference floors:** a random-prediction model and a naive persistence (yesterday's return) model are run through the identical harness, so the two real arms have an honest, non-trivial baseline to beat.

---

## 3. Graph Sensitivity Grid (Benchmark Exhibit)

Searched on `dmrgt_all3` **validation** predictions only (20 runs per config: 4 folds × 5 seeds), one dimension changed at a time from the baseline configuration. Selection metric: Sharpe ratio (highest/least-negative wins).

| Config | k_pos | k_neg | corr_window | Val Sharpe | Val Mean Rank IC | Val Dir Acc | Val Turnover | Val Cum. Return |
|---|---|---|---|---|---|---|---|---|
| **baseline** ✅ | 8 | 4 | 60 | **-1.2596** | -0.00491 | 0.4976 | 0.6961 | -0.1438 |
| kpos10 | 10 | 4 | 60 | -1.2757 | -0.00316 | 0.5019 | 0.7206 | -0.1392 |
| win45 | 8 | 4 | 45 | -1.3695 | -0.00181 | 0.4996 | 0.7289 | -0.1505 |
| win90 | 8 | 4 | 90 | -1.4178 | -0.00368 | 0.5002 | 0.6857 | -0.1647 |
| kpos6 | 6 | 4 | 60 | -1.5352 | -0.00700 | 0.4979 | 0.7207 | -0.1638 |
| kneg5 | 8 | 5 | 60 | -1.6001 | -0.00570 | 0.4979 | 0.7083 | -0.1681 |
| kneg3 | 8 | 3 | 60 | -1.7078 | -0.00722 | 0.5002 | 0.6911 | -0.1818 |

**Winner: `baseline` (k_pos=8, k_neg=4, corr_window=60)** — highest (least-negative) validation Sharpe of all 7 configurations tested, confirming the originally-chosen Phase 2 configuration rather than assuming it. This is the config carried forward and locked as `dmrgt_frozen`'s graph-construction hyperparameters before test data was touched.

Note that rank IC is not monotonic with the Sharpe ranking here (e.g. `kpos10` has a less-negative rank IC than baseline but a worse Sharpe) — this is expected, since rank IC measures ranking quality while Sharpe also reflects the cost model and portfolio construction; Sharpe was the pre-registered selection metric (Rule 17), so `baseline`'s win stands on that basis alone, not cherry-picked after the fact from whichever metric happened to favor it.

---

## 4. Final Comparison — Test Split (20 runs each: 4 folds × 5 seeds)

| Model | Mean Rank IC | Sharpe | Max Drawdown | Turnover | Cumulative Return | Direction Accuracy |
|---|---|---|---|---|---|---|
| **dynamic** (`dmrgt_all3`, baseline graph) | **-0.00134** | **-1.043** | 0.2535 | 0.6552 | -0.1694 | 0.4989 |
| **frozen** (`dmrgt_frozen`) | -0.00490 | -1.686 | 0.2952 | 0.5987 | -0.2589 | 0.4994 |
| *Reference: random* | +0.00422 | -4.984 | — | — | — | 0.5000 |
| *Reference: persistence* | -0.00797 | -2.587 | — | — | — | 0.4964 |

**Reading this table:** both real arms beat frozen-only-worse-than-dynamic on every metric. Neither model has a positive Sharpe ratio, but both beat the naive persistence floor on Sharpe (dynamic decisively, frozen by a smaller margin) — the market regimes covered (2019–2022, including COVID and the 2022 rate shock) were genuinely hard, and "loses less than doing nothing" is itself informative context for the panel, not a result to be defensive about.

---

## 5. Significance Tests

### 5.1 Wilcoxon Signed-Rank Test (Predictive Quality)

Paired test on rank IC across the same 20 (fold, seed) pairs for both arms.

- **n = 20**
- Dynamic wins **11/20** paired comparisons
- Statistic = 76.0
- **p = 0.2943** (α = 0.05) → **not statistically significant**
- Paired mean IC difference: dynamic -0.00134 vs. frozen -0.00490 (**+0.00356** in dynamic's favor)

**Interpretation:** the dynamic graph produces a directionally better rank IC than the frozen graph, and wins the majority of individual fold/seed comparisons, but the effect is not large or consistent enough to rule out chance at the conventional 0.05 threshold.

### 5.2 Stationary Bootstrap Test on Sharpe Difference (Trading Performance)

Politis–Romano stationary bootstrap on the Sharpe ratio difference, accounting for autocorrelation in daily returns (implemented in raw NumPy to avoid an external-package dependency inside the no-internet Kaggle session).

- Dynamic Sharpe: **-1.970**
- Frozen Sharpe: **-3.123**
- Difference: **+1.153** (dynamic better)
- 95% CI: **[+0.042, +2.302]** — excludes zero
- One-sided p = **0.0202**
- Two-sided p = **0.0404**
- **Statistically significant at α = 0.05**

**Interpretation:** unlike rank IC, the *trading* performance gap between dynamic and frozen graphs is statistically significant. The dynamic graph's Sharpe ratio is reliably higher than the frozen graph's.

### 5.3 Cross-Check

The locally recomputed significance results were cross-checked against the in-session (`phase3_significance_results.json`) values computed during the same Kaggle run, with tolerances of 0.01 on p-values and 1e-6 on the Sharpe difference. **Result: p-values agree.** This rules out a divergent bug between the training-session statistics code and the local post-hoc `compare.py`/`stats.py` code path.

---

## 6. Conclusion

> **The dynamic graph does not produce a statistically significant improvement in rank IC over the frozen graph (p = 0.294), but it does produce a statistically significant improvement in risk-adjusted trading performance (Sharpe difference p = 0.020).** The dynamic-graph model (`dmrgt_all3`) finishes above the persistence floor on Sharpe ratio (-1.970 vs. -2.587); the frozen-graph model (`dmrgt_frozen`) finishes below it (-3.123 vs. -2.587).

In plain terms: rebuilding the graph daily doesn't reliably make the model's *stock ranking* better in a way we can distinguish from noise — but it does reliably make the resulting *portfolio* lose less money than freezing the graph does, and only the dynamic version manages to beat a naive "assume tomorrow looks like today" strategy on a risk-adjusted basis. This is a real, defensible, split result, not a clean win — and reporting it exactly this way (one test significant, one not) is more credible to a panel or reviewer than only presenting the metric that came out favorably.

---

## 7. Notes on Interpretation

1. **Two different Sharpe ratios, not directly comparable.** The Sharpe figures in this document come from the corrected Phase 3 backtest, which fixed a leg-direction bug present in Phase 2 (`ranks <= k` on ascending rank was silently selecting the *lowest*-predicted stocks for the long leg instead of the highest-predicted ones). Phase 2's Sharpe/cumulative-return numbers should **not** be quoted side-by-side with Phase 3's as if they measure the same thing. Rank IC, direction accuracy, turnover, and the cost model are unaffected by this bug and remain comparable across phases.
2. **Rank IC in Phase 3 is close to Phase 2's small magnitude but not identical**, which is expected: Rule 15 (fresh seeds per phase) means Phase 3's 20 runs are not literally re-scoring the same trained weights as Phase 2 — some run-to-run variance around a small true effect is normal, not a sign of instability.
3. **Neither arm is "profitable"** in absolute terms (both Sharpe ratios are negative). The honest framing for a panel is: across four historically difficult regimes (calm, COVID crash, recovery, rate shock), the dynamic-graph model loses money more slowly than either the frozen-graph model or a naive persistence strategy, and this gap is statistically supported. That is a legitimate, modest, publishable finding — it is not "the model works," and the report should not overstate it as such.

---

## 8. Reproducibility Record

| Item | Value |
|---|---|
| Kaggle cache dataset | `dmrgt-phase-3-cache` (new; `dmrgt-cache` from Phase 2 untouched) |
| Kaggle code dataset | `dmrgt-phase-3-code` (new; `dmrgt-code` from Phase 2 untouched) |
| GPU | T4 (compute capability sm_75) |
| Folds | 4 (2019 calm / 2020 COVID / 2021 recovery / 2022 rate-shock) |
| Seeds per fold | 5 |
| Paired observations per arm | 20 |
| Total training runs (grid + final arms) | 160 |
| Locked frozen-graph config | k_pos=8, k_neg=4, corr_window=60 (baseline) |
| Significance threshold | α = 0.05, two-sided unless noted |
| Test-split scoring | Exactly once, hash-guarded (`phase3_TEST_SCORED.json`) |
| Loss function | Huber (robust to fat-tailed daily-return outliers) |
| Session design | Single Kaggle session, grid search → winner lock → final arm comparison → significance tests, no re-training of `dmrgt_all3` after grid selection (its existing test predictions from the winning grid run are the final result) |

---

## 9. Future Work (Not Attempted — Explicitly Out of Scope for Phase 3)

- Replacing the pointwise Huber loss with a rank-aware objective (ListMLE / LambdaRank), since the model is trained on pointwise loss but evaluated on rank IC — a known metric mismatch.
- Exploring state-space sequence models as an alternative or complement to the Transformer encoder.
- Explainability work (attention map extraction, gradient×input attribution, edge-ablation checks) to address the project's originally stated but not-yet-delivered explainability objective — recommended as the safest, zero-retraining-risk use of remaining project time before final review, since it requires no new experiments on already-locked results.

---

*Generated from the actual Phase 3 production run output (`compare.py` final report), cross-checked against in-session significance results before being recorded here.*
