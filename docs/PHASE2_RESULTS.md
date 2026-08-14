# DMRGT Phase 2 — Results

**Phase:** 2 — Add the dynamic three-relation graph
**Completion criterion:** DMRGT beats the no-graph Transformer and the LSTM
**Status:** ✅ **MET** on rank IC
**Kaggle run:** `2026-08-14T14:14:57Z`, Tesla T4
**Universe:** 40 S&P 100 constituents, 5 per sector across 8 GICS sectors

> ⚠️ Items marked **[FILL]** are known only to the implementer — complete them before this document is used for the write-up.

---

## 1. What Phase 2 set out to do

Phase 1 established that a temporal-only Transformer, reading each stock's own 30-day history in isolation, extracts no cross-sectional signal (rank IC −0.0010, p = 0.781, on a 20-stock universe). That is the premise of this project: models treating each asset independently cannot capture cross-asset influence.

Phase 2 tests whether adding a **dynamic, three-relation graph** changes that. The completion criterion, from the project proposal, is:

> Dynamic 3-relation graph (+corr / −corr / sector) + R-GCN; full DMRGT on ~40 stocks. **Beats no-graph and LSTM baselines.**

---

## 2. Headline result

All seven models evaluated through the identical `eval.py` path — same folds, same seeds, same commission (10 bps) and slippage (5 bps).

| Model | Mean rank IC | Dir. acc | Sharpe (after costs) | Max DD | Turnover |
|---|---|---|---|---|---|
| **dmrgt_all3** | **+0.0061 ± 0.0110** | 0.499 | −1.64 | **0.295** | 0.77 |
| dmrgt_no_sector | +0.0056 ± 0.0122 | 0.503 | −2.07 | 0.327 | 0.82 |
| dmrgt_pos_only | +0.0036 ± 0.0133 | 0.499 | −2.24 | 0.361 | 0.83 |
| transformer_nograph *(control)* | +0.0022 ± 0.0114 | 0.505 | −1.35 | 0.351 | 0.64 |
| persistence | +0.0000 ± 0.0018 | 0.502 | −1.75 | 0.411 | 1.54 |
| lstm | −0.0016 ± 0.0159 | 0.501 | −1.29 | 0.356 | 0.49 |
| random *(gate)* | −0.0074 ± 0.0035 | 0.496 | −5.24 | 0.489 | 1.59 |

Statistics are mean ± standard deviation across 20 runs (4 folds × 5 seeds) for trained models; across 4 folds for the deterministic baselines.

**Completion criterion:** DMRGT all-3 (+0.0061) beats both the no-graph Transformer (+0.0022) and the LSTM (−0.0016). ✅

---

## 3. The leakage gate

The random model — which ignores all inputs and returns noise — scored a mean rank IC of **−0.0074**, within the ±0.02 threshold, with directional accuracy of 49.6% and a Sharpe of −5.24 after costs.

Random noise cannot exploit genuine signal. Had it scored meaningfully above zero, the only explanation would be that the evaluation pipeline itself was leaking future information. It did not. Every number in this document was produced by an instrument demonstrated — not assumed — to be incapable of manufacturing signal from noise.

This gate was re-run after the universe expansion from 20 to 40 stocks, as required, because a gate that passed on one cache says nothing about another.

---

## 4. The ablation ladder — evidence on relation types

Each relation type was removed in turn. The ordering is monotonic and matches the design hypothesis:

| Configuration | Edge types | Mean rank IC | Δ from previous |
|---|---|---|---|
| `dmrgt_pos_only` | +corr | +0.0036 | — |
| `dmrgt_no_sector` | +corr, −corr | +0.0056 | **+0.0020** |
| `dmrgt_all3` | +corr, −corr, sector | +0.0061 | **+0.0005** |

**Negative-correlation edges contribute roughly +0.0020 of rank IC** — larger than the contribution of sector edges. This is direct evidence for weakness **W2-C** in the literature survey: THGNN's ablation showed negative relations carry signal, and this project's Rev. 1 design discarded them. Rev. 2 reinstated them, and the reinstatement is measurably justified in a fully reproducible, free-data setting.

Sector edges add a smaller but positive increment. Their contribution may be partly redundant with correlation edges, since same-sector stocks tend to be positively correlated and may already be captured by the top-k selection.

---

## 5. Consistency across market regimes

Per-fold mean rank IC across seeds:

| Model | 2019 calm | 2020 COVID | 2021 | 2022 rates | Folds positive |
|---|---|---|---|---|---|
| **dmrgt_all3** | +0.0072 | +0.0055 | +0.0066 | +0.0051 | **4 / 4** |
| dmrgt_no_sector | +0.0066 | −0.0015 | +0.0100 | +0.0070 | 3 / 4 |
| dmrgt_pos_only | +0.0033 | −0.0007 | +0.0073 | +0.0047 | 3 / 4 |
| transformer_nograph | +0.0041 | −0.0071 | +0.0093 | +0.0025 | 3 / 4 |
| lstm | −0.0062 | −0.0166 | +0.0087 | +0.0078 | 2 / 4 |

**`dmrgt_all3` is the only model positive in every fold.** Every other configuration, including both controls, turns negative somewhere — most often during the 2020 COVID fold, the regime in which cross-asset relationships shift most violently.

This consistency is arguably a stronger result than the mean. A model that wins only in calm conditions is a different proposition from one that holds across a crash and a rate shock.

---

## 6. The cost-adjusted picture — reported plainly

**DMRGT achieves the best rank IC but not the best Sharpe ratio.**

| Model | Rank IC | Turnover | Sharpe (after costs) |
|---|---|---|---|
| dmrgt_all3 | **+0.0061** | 0.77 | −1.64 |
| transformer_nograph | +0.0022 | 0.64 | −1.35 |
| lstm | −0.0016 | 0.49 | **−1.29** |

The graph model trades roughly 20% more than the no-graph Transformer and 57% more than the LSTM. After transaction costs and slippage, that additional turnover consumes its prediction advantage.

This is exactly the failure mode this project's literature survey identifies in the surveyed papers — THGNN states its backtest ignores transaction costs, Fanshawe et al. omit slippage, and MDGNN reports neither. Costs were active in this pipeline from the first backtest run, so the tension surfaced rather than being hidden.

Two qualifications:

- **DMRGT has the lowest maximum drawdown of all seven models (0.295).** The risk picture is mixed, not simply worse.
- **All models have negative Sharpe.** None has sufficient signal to overcome trading costs. This is an honest characterisation of the problem's difficulty, not a defect in any one model.

---

## 7. Statistical status — no claim of significance

Indicative statistics only. Formal significance testing is Phase 3.

**One-sample t-test against zero** (n = 20 runs each):

| Model | Mean IC | t | p |
|---|---|---|---|
| dmrgt_all3 | +0.00608 | +2.466 | **0.023** |
| dmrgt_no_sector | +0.00555 | +2.038 | 0.056 |
| dmrgt_pos_only | +0.00364 | +1.227 | 0.235 |
| transformer_nograph | +0.00218 | +0.853 | 0.404 |
| lstm | −0.00155 | −0.436 | 0.668 |

`dmrgt_all3` is the only model whose rank IC is distinguishable from zero at the 5% level.

**Paired comparison, dmrgt_all3 vs transformer_nograph** (matched on fold and seed):

```
mean difference : +0.00390
wins            : 12 / 20 fold-seed pairs
Wilcoxon        : p = 0.246   (indicative only)
```

> **The gap between DMRGT and the control is NOT statistically significant.** The difference (+0.0039) is smaller than the seed-to-seed standard deviation (~0.011), and the model wins only 12 of 20 paired comparisons. Phase 2's criterion is "beats the baselines," which is met. Whether that improvement is real is the question Phase 3 answers with a properly specified Wilcoxon test on per-window rank IC and a stationary-bootstrap test on the Sharpe difference.
>
> No claim of statistical significance should be made on the basis of this document.

---

## 8. The graph — construction and diagnostics

**Configuration:** trailing 60-day Pearson correlations on log-returns; top-8 positive and bottom-4 negative partners per stock; static same-sector edges. **Directed** (`symmetrise: false`), preserving the asymmetry inherent in top-k selection.

**Edge counts, verified constant across all 2,455 dates:**

| Type | Count | Note |
|---|---|---|
| 0 — positive correlation | 320 | 40 × 8, zero variance across dates |
| 1 — negative correlation | 160 | 40 × 4, zero variance across dates |
| 2 — sector | 160 | 8 sectors × 5 stocks; identical on every date |
| **Total** | **640 / day** | vs MDGNN's 62.5M on a V100 |

**Verified properties:**
- No self-loops
- No isolated nodes on any date
- Node indices strictly within 0–39
- Sector edges byte-identical across all dates (1 distinct edge set)
- Directedness confirmed: 29% of positive edges have no reverse counterpart

### Edge persistence — the dynamism diagnostic

| Horizon | Fraction of a stock's top-8 partners retained |
|---|---|
| 1 trading day | **0.930** (min 0.819, max 0.981) |
| ~200 trading days | ≈ 0.47 |

Roughly one edge in fourteen changes per day. The graph is neither trivially static nor unstable — it drifts steadily, with the relationship map substantially rewritten over a period of months.

**This number matters for Phase 3.** Had persistence been near 1.0, a frozen graph would be nearly indistinguishable from a dynamic one and the dynamic-versus-static comparison would have nothing to measure. At 0.93 there is genuine day-to-day change, though the resulting effect may still be modest.

---

## 9. Implementation notes

**R-GCN:** implemented as a hand-rolled `RelationMessagePassing` layer using `torch.index_add_` with three dense per-relation weight matrices, rather than PyTorch Geometric's `RGCNConv`. Rationale: with 40 nodes and 640 edges the performance difference is negligible, and this avoids `torch-scatter`, a compiled dependency pinned to exact torch and CUDA versions that would be fragile in the Kaggle image. It also guarantees exactly three weight matrices, whereas `RGCNConv` adds a self-loop root transform by default.

**Training:** all five models trained from random initialisation (Rule 15). No Phase 1 checkpoint was loaded, warm-started from, or fine-tuned. The prediction target changed with the universe (excess return is now measured against a 40-stock mean), so Phase 1 weights were fitted to a different quantity.

**Prediction contract:** `predictions_{model_id}.parquet` with columns `model, fold, seed, date, ticker, y_pred, y_true`. 201,600 rows per model — verified as (252 + 253 + 252 + 251) test days × 5 seeds × 40 stocks — and 1,008,000 rows in total across five models.

**Prediction spread by model** (an indicator that no model collapsed to a constant):

| Model | `y_pred` std | `y_true` std |
|---|---|---|
| dmrgt_all3 | 0.00483 | 0.01695 |
| dmrgt_no_sector | 0.00396 | 0.01695 |
| dmrgt_pos_only | 0.00293 | 0.01695 |
| lstm | 0.00160 | 0.01695 |
| transformer_nograph | 0.00144 | 0.01695 |

Prediction spread scales monotonically with graph richness. The graph gives the model more on which to differentiate stocks — it is not merely adding noise.

**Model parameter counts:** **[FILL]** — record for LSTM, Transformer and DMRGT to confirm the comparison is about architecture rather than capacity.

---

## 10. Reproducibility record

**Data provenance** (all four verified against `hashes.json` and the run manifest):

```
prices.parquet    438c0ea294d8de54ac8398e2c4ee92d3665f33526d864efc307f08bd9d3add23
features.parquet  b4614fa0424783724c72b9bd821d375db56b17de9562361ffd4e2e6018b51e1f
targets.parquet   204392bae151e145c49df56b99d371da7cd08444b1b9f4920196708fe6685c22
graphs.parquet    555560de3913ec6231c9626600249a3af660c09dd46784f9825b445dcdef47cf
```

**Fold structure** (verified identical between the local cache and the Kaggle run):

| Fold | Regime | Test days | Test window |
|---|---|---|---|
| 1 | Calm | 252 | 2019-01-02 → 2019-12-31 |
| 2 | COVID crash | 253 | 2020-01-02 → 2020-12-31 |
| 3 | Recovery | 252 | 2021-01-04 → 2021-12-31 |
| 4 | Rate shock | 251 | 2022-01-03 → 2022-12-30 |

**Environment:** Tesla T4 (sm_75), **[FILL: torch version from manifest]**, Python 3.12.

**Kaggle artifacts:** **[FILL]** — `dmrgt-cache` dataset version, `dmrgt-code` dataset version, committed notebook URL. Without these the run cannot be reconstructed even with the code.

**Determinism check:** **[FILL]** — the spec requires running one fold × one seed twice and comparing `y_pred` exactly. Phase 1 achieved bit-identical reruns; message-passing layers use scatter-add, which is non-deterministic on CUDA because floating-point addition is not associative. State whether the Phase 1 guarantee survived, and if not, to how many decimal places results are reproducible. This is a known limitation of GPU graph operations, not a defect — but it must be measured and disclosed rather than assumed away.

**Local cross-check:** **[FILL]** — result of `verify_env.py` comparing one fold on local CPU against the manifest (structural properties only: sample counts, date boundaries, parquet hashes; not validation loss).

**Manifest limitation:** each model overwrote `run_manifest.json`, so the retained manifest records only the last model trained (`dmrgt_all3`). The config snapshot, cache hashes, GPU and fold structure are shared across all five models, so no information is lost — but for Phase 3, `run_all` should accumulate a single manifest covering every model, since the significance tests make "all models trained under identical conditions" a load-bearing claim.

---

## 11. Known limitations

- **Survivorship bias.** Tickers were selected from the current S&P 100; only companies surviving to the present are included.
- **No statistical significance established.** The improvement over the control is smaller than the seed-to-seed variance. Phase 3 addresses this directly.
- **Cost-adjusted performance does not favour the graph model.** DMRGT's higher turnover consumes its prediction advantage under the assumed cost model.
- **All Sharpe ratios are negative.** No model in this comparison is profitable after costs.
- **Single hyperparameter configuration.** No tuning was performed, deliberately. The sensitivity grid over k, correlation window and lookback is Phase 4.
- **One-day horizon only.** Results do not generalise to longer holding periods without separate training and validation.
- **Dynamic-versus-static untested.** Phase 2 shows a graph helps relative to no graph. Whether the *daily rebuilding* is what helps is Phase 3's question.

---

## 12. What Phase 3 inherits

| Component | Status |
|---|---|
| `data.py`, `features.py`, `graph.py` | Unchanged — cache is stable |
| `splits.py`, `eval.py`, `baselines.py` | Unchanged |
| `paths.py`, Kaggle handoff | Unchanged |
| `dmrgt_all3` | **Becomes the dynamic-graph arm of the Phase 3 comparison** |
| `transformer_nograph`, `lstm` | Remain permanent baselines |
| New in Phase 3 | Frozen-graph twin; `stats.py` (Wilcoxon + stationary bootstrap) |

**One consideration to carry forward.** The Phase 3 bootstrap test operates on the **Sharpe difference** between the dynamic and frozen graphs. Section 6 shows turnover is what erodes DMRGT's edge under costs. A frozen graph — whose edges never change — would plausibly generate lower turnover and could therefore win on Sharpe while losing on rank IC. That outcome should be anticipated and reported as a finding, not treated as a failure.

---

*Research project for academic purposes only — not investment advice.*
