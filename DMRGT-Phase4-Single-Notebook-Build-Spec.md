# DMRGT — Phase 4 Build Spec: Single Notebook, Single Run (Mamba + Adaptive Training + Regime-Triggered Graph Refresh)

**This document replaces `DMRGT-Phase4-Mamba-Adaptive-Build-Spec.md` entirely.** It contains everything from that spec, plus Phase 4B (the patentable mechanism), restructured so the whole phase runs as **one Kaggle notebook, one committed run**, with **only the cache as input**.

---

## 0. The Three Hard Requirements

1. **Code lives in the notebook.** No `dmrgt-phase-4-code` dataset. The only attached input is `dmrgt-phase-3-cache` (read-only, unchanged). Every line of model, data, training, evaluation and statistics code is in notebook cells.
2. **One run completes the phase.** A single "Save Version → Save & Run All (Commit)" produces every result: training, selection, Phase 4B tuning, one-shot test scoring, statistics, efficiency measurements and a draft results file. No Session A / Session B split, no manual steps in between.
3. **Built from scratch, but ported, not reinvented.** "From scratch" means a fresh notebook with no imports from the old repo. Where Phase 4 reuses Phase 3 logic (data windows, splits, scaler, R-GCN, Transformer, graph construction, `eval.py`, `stats.py`, baselines), the agent must **copy the Phase 3 code from `reference/phase3_src/` into the notebook**, not rewrite it from memory, then diff it against that reference to confirm the logic is identical (except the documented changes below). Rewriting from memory would silently change the model and break comparability. Repository setup is in §0.1.

**Honest limit on requirement 2:** the notebook is designed and time-budgeted to finish in one run, and every check that could fail runs in the **first ~30 minutes** (fail fast, not at hour 9). But nothing can guarantee against Kaggle-side failures (session preemption, GPU quota running out). Before committing, confirm at least **11 hours of weekly GPU quota** remain.

---

## 0.1 Repository Setup (do this before writing any Phase 4 code)

**Principle:** Phase 4 lives on a new `phase-4` branch created **from** `phase-3`. The Phase 3 code is kept on that branch as a **read-only reference** for porting, and is never deleted before porting is done. The agent never works on the `phase-3` branch or in its files, because the locked Phase 3 result depends on them.

### Step 1 — Check that the repository is private (hard requirement)
Every branch of a public repository is public. Pushing Phase 4 (especially the Phase 4B mechanism) to a public repository counts as public disclosure and can destroy patent novelty.
- If the repository is public: make it private, **or** create a new private repository for Phase 4 and push `phase-4` only there.
- Do not push anything in Steps 2–5 until this is confirmed.

### Step 2 — Freeze Phase 3 permanently

```bash
git checkout phase-3
git status                       # must be clean; commit or stash anything outstanding first
git tag phase3-complete          # skip if this tag already exists
git push origin phase3-complete
```

### Step 3 — Create `phase-4` from `phase-3`

```bash
git checkout -b phase-4
```

### Step 4 — Move the Phase 3 code into a read-only reference folder

```bash
mkdir -p reference/phase3_src
git mv *.py reference/phase3_src/        # adjust paths if the code is in a subfolder
git mv *.ipynb reference/phase3_src/     # only if the old notebook is in the root
git commit -m "phase-4: move Phase 3 code to reference/phase3_src for porting"
```

- Leave the local cache folder where it is (it stays gitignored if it already was). The Kaggle input remains `dmrgt-phase-3-cache`, unchanged.
- Leave `PHASE3_RESULTS.md` and `phase3_locked/` untouched.

### Step 5 — Rules for the agent on `phase-4`
- Build the new notebook at the repo root, e.g. `kaggle_phase4.ipynb`.
- **Read and copy from** `reference/phase3_src/`; **never edit, move, or delete** anything inside it.
- Never modify the cache, `PHASE3_RESULTS.md`, or anything in `phase3_locked/`.
- Never check out or commit to `phase-3`.
- For every ported component, record in `PORTING_LOG.md`: source file and function in `reference/phase3_src/`, destination notebook cell, and the diff result ("identical" or the list of documented intentional changes).

### Step 6 — Removing the reference folder (only at the very end)
Only after Phase 4 is complete, `PORTING_LOG.md` shows every component verified, and `phase4-complete` is tagged:

```bash
git rm -r reference/
git commit -m "phase-4: remove Phase 3 reference copy (preserved on phase-3 branch and phase3-complete tag)"
```

This step is optional. The Phase 3 code stays permanently available on the `phase-3` branch and the `phase3-complete` tag either way.

### Verification before the first Kaggle commit
- [ ] Repository confirmed private
- [ ] `phase3-complete` tag exists on the remote
- [ ] Currently on `phase-4`; `reference/phase3_src/` contains the Phase 3 code
- [ ] `phase3_locked/` hashes unchanged
- [ ] `PORTING_LOG.md` lists every ported component as verified

---

## 1. Why the Code Is Written to Files at Runtime

To use both GPUs of Kaggle's **T4 ×2** accelerator, the notebook launches two worker processes (one per GPU). Worker processes can't use functions defined only inside notebook cells (they aren't importable, so they can't be pickled into a new process). The fix, which keeps all code in the notebook:

- Each code module is a notebook cell starting with `%%writefile /kaggle/working/dmrgt4/<module>.py`.
- Running the cell writes the file; the notebook then does `sys.path.insert(0, "/kaggle/working/dmrgt4")` and imports it.
- The workers are launched with `subprocess.Popen` (separate Python processes, each pinned to one GPU with `CUDA_VISIBLE_DEVICES`). **Processes, not threads**: no shared RNG state, so no seed race (the Phase 3 parallelism concern does not apply).

A bonus: the written `.py` files end up in the run's output, so the exact code that produced the results is saved alongside them.

---

## 2. Kaggle Notebook Settings

| Setting | Value |
|---|---|
| Accelerator | GPU T4 ×2 (if only 1 GPU is detected, the notebook runs sequentially and the budget planner accounts for it) |
| Internet | **Off** (nothing is installed; pure PyTorch, NumPy, pandas, SciPy from the Kaggle image) |
| Input | `dmrgt-phase-3-cache` only |
| Persistence | Off |
| Run mode | Save Version → **Save & Run All (Commit)** |
| Environment | Pin the environment after the first successful commit, so reruns use identical library versions |

---

## 3. Notebook Cell Plan (runs top to bottom, no manual steps)

| Cell | Purpose | Fails fast? |
|---|---|---|
| 0 | Pre-registered CONFIG + snapshot hash | — |
| 1 | Environment + GPU checks | yes |
| 2 | Cache resolution + hash verification | yes |
| 3–9 | `%%writefile` modules (data, graph, models, eval, stats, train, worker) | — |
| 10 | Unit tests (Mamba causality, incremental correlation, graph reproduction, alignment) | yes |
| 11 | Leakage gate (random model + shuffled targets) | yes |
| 12 | Memory + timing probe → budget plan (auto-skip optional arm D if needed) | yes |
| 13 | Launch all training jobs across GPUs, wait, completeness check | — |
| 14 | Phase 4A selection on validation → `phase4_winner.json` | — |
| 15 | Phase 4B trigger calibration + validation selection → `phase4b_policy.json` | — |
| 16 | **One-shot test scoring** (4A + 4B) → `phase4_TEST_SCORED.json` | — |
| 17 | Efficiency + technical-effect measurements + Δ-by-date | — |
| 18 | Output verification + auto-generated `PHASE4_RESULTS_DRAFT.md` | yes |

Everything that can fail on a bug or a leak (cells 1, 2, 10, 11, 12) runs before any long training starts.

---

## 4. Cell 0 — Pre-Registered Configuration

All decisions are fixed here, **before any data is touched**. The cell writes `config_snapshot.json` and prints its SHA-256. The Kaggle version timestamp plus this hash is the pre-registration record.

```python
CONFIG = {
    "phase": "4",
    "seeds": [0, 1, 2, 3, 4],
    "folds": "identical to Phase 3 (fold_1_2019_calm, fold_2_2020_covid, fold_3_2021, fold_4_2022_rates)",
    "graph": {"k_pos": 8, "k_neg": 4, "corr_window": 60, "file": "graphs_baseline.parquet"},
    "features": 8,
    "backtest": {"commission_bps": 10, "slippage_bps": 5, "decile_fraction": 0.2},

    "arms": {
        "A": {"encoder": "transformer", "lookback": 30, "adaptive_huber": False, "required": True},
        "B": {"encoder": "mamba",       "lookback": 30, "adaptive_huber": False, "required": True},
        "C": {"encoder": "mamba",       "lookback": 60, "adaptive_huber": False, "required": True},
        "D": {"encoder": "mamba",       "lookback": 60, "adaptive_huber": True,  "required": False},
    },

    "training": {
        "max_epochs": 50, "early_stop_patience": 10, "early_stop_metric": "val_mean_rank_ic",
        "batch_size_ladder": [32, 16, 8],          # same batch size for ALL arms
        "optimizer": "AdamW", "lr": 1e-3, "weight_decay": 1e-4,
        "warmup_epochs": 3, "warmup_start_lr": 1e-4,
        "plateau": {"mode": "max", "factor": 0.5, "patience": 4, "min_lr": 1e-5},
        "grad_clip_norm": 1.0,
        "huber_delta_fixed": 1.0,
        "adaptive_huber": {"c": 1.345, "min": 0.05, "max": 2.0},
        "retry_failed_job_once": True,
    },

    "mamba": {"d_model": 64, "n_layers": 2, "d_state": 16, "d_conv": 4, "expand": 2, "dropout": 0.1},

    "selection_4A": {"metric": "val_sharpe", "tiebreak_metric": "val_mean_rank_ic", "tie_threshold": 0.02,
                     "candidates": ["B", "C", "D"]},

    "phase4b": {
        "corr_window": 60,
        "target_rebuild_rates": [0.10, 0.20, 0.35, 0.50],
        "signals": ["corr_drift", "mamba_delta", "combined"],
        "max_staleness_days": 20,
        "full_recompute_every": 250,
        "val_noninferiority_margin_sharpe": 0.10,
        "test_noninferiority_margin_sharpe": 0.25,
        "scale_benchmark_n_stocks": [40, 100, 500],
    },

    "significance": {"bootstrap_iterations": 5000, "bootstrap_block_length": 10, "alpha": 0.05,
                     "secondary_correction": "holm"},

    "leakage_gate": {"max_abs_ic_in_standard_errors": 3.0},

    "budget": {"session_limit_hours": 12.0, "plan_limit_hours": 10.5,
               "expected_epochs_for_projection": 35, "safety_factor": 1.15,
               "phase4b_estimate_minutes": 45, "scoring_and_bench_estimate_minutes": 20},
}
```

No value in CONFIG may be changed after the first committed run. If a bug forces a change, it is documented in the results with the reason (see §15).

---

## 5. Cells 1–2 — Environment and Cache

### Cell 1: environment checks (hard stop on failure)
- `torch.cuda.device_count()` ≥ 1; every visible GPU has compute capability ≥ 7.0 (T4 = sm_75). P100 (sm_60) → stop with a clear message.
- Record Python, torch, numpy, pandas, scipy versions, GPU names and count into `run_manifest.json`.
- Set `os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"` before CUDA initialises; the workers set it too.

### Cell 2: cache resolution + hash verification (hard stop on failure)

```python
from pathlib import Path
import hashlib, json

CANDIDATES = [
    Path("/kaggle/input/dmrgt-phase-3-cache"),
    Path("/kaggle/input/datasets/harishsivakumar1912/dmrgt-phase-3-cache"),
]
def resolve_cache():
    for c in CANDIDATES:
        if (c / "features.parquet").exists():
            return c
    hits = list(Path("/kaggle/input").rglob("features.parquet"))
    if len(hits) == 1:
        return hits[0].parent
    raise FileNotFoundError(f"Cache not found or ambiguous: {hits}")

CACHE = resolve_cache()
expected = json.loads((CACHE / "hashes.json").read_text())
for name, h in expected.items():
    if not name.endswith(".parquet"):      # skip metadata keys such as generated_at
        continue
    actual = hashlib.sha256((CACHE / name).read_bytes()).hexdigest()
    assert actual == h, f"Hash mismatch for {name}"
```

Required files: `prices.parquet`, `features.parquet`, `targets.parquet`, `graphs_baseline.parquet`, `hashes.json`.

---

## 6. Cells 3–9 — Modules (`%%writefile`)

### 6.1 `d4_data.py` — ported from Phase 3, plus two changes
Port verbatim: loading, per-fold train/val/test date ranges (including Phase 3's exact validation-split rule), scaler fitted on **train only** per fold, target alignment (features up to day t → target is next-day), row-count assertion (sum, not multiply).

**Change 1 — lookback is a parameter.** Windows are built with `lookback ∈ {30, 60}`. With 60, the earliest training dates lose their windows. Bounded drop: remove those dates and `assert dropped / n_train_days < 0.05` per fold. Log the exact count (expected ≈ 30 of 776 in fold 1 ≈ 3.9%).

**Change 2 — test access gate.** Test rows cannot be loaded until both lock files exist:

```python
LOCKS = [Path("/kaggle/working/phase4_winner.json"), Path("/kaggle/working/phase4b_policy.json")]

def get_split(fold, split, **kw):
    if split == "test" and not all(p.exists() for p in LOCKS):
        raise PermissionError("Test split is locked until phase4_winner.json and phase4b_policy.json exist.")
    ...
```

Training jobs only ever request `"train"` and `"val"`. **No test predictions exist before cell 16**: they are generated from saved checkpoints after the locks. This is stronger than Phase 3, where test predictions were written during training.

### 6.2 `d4_graph.py` — ported loader + new incremental engine + triggers
Port verbatim: loading daily edges from `graphs_baseline.parquet`, and the Phase 3 `graph.py` logic that turns a correlation matrix into edges (top-k positive, top-k negative, sector edges, `symmetrise=False`, `min_history_days=61`, the exact return definition and window alignment).

New: incremental rolling correlation (float64, O(N²) per day instead of O(W·N²)).

```python
import numpy as np
from collections import deque

class RollingCorr:
    def __init__(self, window, n_stocks, full_recompute_every=250):
        self.W = window
        self.buf = deque()
        self.S1 = np.zeros(n_stocks)
        self.S2 = np.zeros((n_stocks, n_stocks))
        self.k = full_recompute_every
        self.n_updates = 0

    def push(self, r):                       # r: (N,) float64 returns for one day
        assert np.isfinite(r).all()
        self.buf.append(r)
        self.S1 += r
        self.S2 += np.outer(r, r)
        if len(self.buf) > self.W:
            old = self.buf.popleft()
            self.S1 -= old
            self.S2 -= np.outer(old, old)
        self.n_updates += 1
        if self.n_updates % self.k == 0:     # remove floating-point drift
            X = np.asarray(self.buf)
            self.S1, self.S2 = X.sum(0), X.T @ X

    def corr(self):
        W = len(self.buf)
        cov = (self.S2 - np.outer(self.S1, self.S1) / W) / (W - 1)
        d = np.sqrt(np.diag(cov))
        return cov / np.outer(d, d)
```

New: trigger policy runner (used only in Phase 4B, inference only):

```python
def run_policy(days, returns_for_day, build_graph, signal_for_day, fire_rule, max_stale, rc):
    """
    For each day: update correlations incrementally, compute signals using only
    information available that day, and rebuild the graph only if the rule fires
    (or the graph is older than max_stale days). Returns the graph used on each day
    and the list of rebuild days.
    """
    C_ref = rc.corr()
    graph = build_graph(C_ref)
    last, graphs, rebuilds = 0, {}, []
    for i, d in enumerate(days):
        rc.push(returns_for_day(d))          # alignment ported exactly from Phase 3 graph.py
        C = rc.corr()
        s_corr = np.linalg.norm(C - C_ref) / np.linalg.norm(C_ref)
        s_delta = signal_for_day(d)          # Mamba Δ for day d, computed before message passing
        if fire_rule(s_corr, s_delta) or (i - last) >= max_stale:
            graph, C_ref, last = build_graph(C), C, i
            rebuilds.append(d)
        graphs[d] = graph
    return graphs, rebuilds
```

`daily` = rule always fires. `frozen_at_start` = rule never fires and `max_stale = ∞`.

### 6.3 `d4_models.py` — ported R-GCN + ported Transformer + new Mamba
- Port verbatim: `RelationMessagePassing` R-GCN (hand-rolled `index_add_`), prediction head, Transformer encoder (for arm A).
- Every model exposes a split forward, so the graph decision in Phase 4B can happen between the two steps:

```python
class DMRGT(nn.Module):
    def encode(self, x):            # x: (N_stocks, L, 8) → (N_stocks, d_hidden)
        return self.encoder(x)
    def propagate(self, h, graph):  # R-GCN + head → (N_stocks,)
        return self.head(self.rgcn(h, graph))
    def forward(self, x, graph):
        return self.propagate(self.encode(x), graph)
```

- Mamba encoder (pure PyTorch, no `mamba-ssm`, no compiled kernels):

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

def selective_scan(u, delta, A, B, C, D):
    # u, delta: (Bt, L, Di); A: (Di, S); B, C: (Bt, L, S); D: (Di,)
    Bt, L, Di = u.shape
    h = torch.zeros(Bt, Di, A.shape[1], device=u.device, dtype=u.dtype)
    ys = []
    for t in range(L):
        dA = torch.exp(delta[:, t].unsqueeze(-1) * A)
        dBu = delta[:, t].unsqueeze(-1) * B[:, t].unsqueeze(1) * u[:, t].unsqueeze(-1)
        h = dA * h + dBu
        ys.append((h * C[:, t].unsqueeze(1)).sum(-1))
    return torch.stack(ys, dim=1) + u * D

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_inner, self.d_state = expand * d_model, d_state
        self.dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                groups=self.d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.last_delta = None                    # read-only, for Phase 4B signal + explainability

    def forward(self, x):
        L = x.shape[1]
        u, z = self.in_proj(x).chunk(2, dim=-1)
        u = F.silu(self.conv1d(u.transpose(1, 2))[..., :L].transpose(1, 2))   # causal conv
        dt, B, C = torch.split(self.x_proj(u), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt))
        self.last_delta = delta.detach()
        y = selective_scan(u, delta, -torch.exp(self.A_log), B, C, self.D)
        return self.out_proj(y * F.silu(z))

class MambaEncoder(nn.Module):
    def __init__(self, n_features=8, d_model=64, n_layers=2, d_state=16, d_conv=4,
                 expand=2, dropout=0.1, d_out=None):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.layers = nn.ModuleList([MambaBlock(d_model, d_state, d_conv, expand) for _ in range(n_layers)])
        self.norms = nn.ModuleList([RMSNorm(d_model) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.final_norm = RMSNorm(d_model)
        self.out = nn.Linear(d_model, d_out) if (d_out and d_out != d_model) else nn.Identity()
        self.return_sequence = False              # True only inside the causality test

    def forward(self, x):
        h = self.input_proj(x)
        for norm, block in zip(self.norms, self.layers):
            h = h + self.dropout(block(norm(h)))
        h = self.final_norm(h)
        return h if self.return_sequence else self.out(h[:, -1])

    def delta_signal(self):
        """Mean memory-reset strength at the final step (last layer), one number per call."""
        return float(self.layers[-1].last_delta[:, -1].mean())
```

`d_out` = the hidden size the ported R-GCN expects from the Transformer.

### 6.4 `d4_eval.py` and `d4_stats.py` — ported verbatim
- `eval.py` **with** the `source=` keyword fix at both call sites (`evaluate_predictions`, `backtest`).
- `stats.py` as-is: Wilcoxon on paired `(fold, seed)` rank IC; NumPy stationary bootstrap on the Sharpe difference (same construction as Phase 3).
- Random and persistence baselines ported.
- Add: Holm correction helper for secondary tests.

### 6.5 `d4_train.py` — training with in-run adaptive hyperparameters
`train_one(arm, fold, seed, batch_size)`:
1. `set_seed(seed)` inside the job (Python, NumPy, torch, CUDA).
2. Build the model **on CPU**, fingerprint by hashing raw parameter bytes (`p.detach().cpu().contiguous().numpy().tobytes()`), then move to GPU.
3. Train on `train`, evaluate on `val` every epoch:
   - **Early stopping on validation mean rank IC**, patience 10, keep best weights in memory, restore at end.
   - **AdamW** (`lr=1e-3, weight_decay=1e-4`); **linear warmup** 1e-4 → 1e-3 over 3 epochs; then **`ReduceLROnPlateau(mode="max", factor=0.5, patience=4, min_lr=1e-5)`** stepped on validation rank IC (not during warmup); gradient clipping at 1.0.
   - **Adaptive Huber delta (arm D only)**, from training residuals of the epoch just finished:

```python
mad = torch.median(torch.abs(train_residuals - torch.median(train_residuals)))
huber_delta = float(torch.clamp(1.345 * 1.4826 * mad, min=0.05, max=2.0))
```

4. Save: best checkpoint (`ckpt_<arm>_<fold>_<seed>.pt`), validation predictions (`valpred_<arm>_<fold>_<seed>.parquet`), epoch log (`log_<arm>_<fold>_<seed>.csv` with `epoch, lr, huber_delta, train_loss, val_rank_ic, val_sharpe, epoch_seconds`), and a done marker (`done_<arm>_<fold>_<seed>.json` with fingerprint, epochs run, best epoch, total seconds).
5. **No test data is touched.**

Determinism: `torch.backends.cudnn.deterministic = True`, `benchmark = False`, `torch.use_deterministic_algorithms(True, warn_only=True)`. `index_add_` on GPU is not bit-deterministic; this is accepted and documented, the same as in Phase 3.

### 6.6 `d4_worker.py` — one process per GPU

```python
# usage: python d4_worker.py --jobs /kaggle/working/jobs_gpu0.json --batch-size 32
for job in jobs:
    if done_marker(job).exists():
        continue                                  # safe if cell 13 is ever re-entered
    for attempt in (1, 2):                        # retry once on failure
        try:
            train_one(**job, batch_size=args.batch_size)
            break
        except Exception:
            write_json(error_file(job, attempt), {"traceback": traceback.format_exc()})
```

Errors never stop the other jobs. Completeness is checked afterwards in cell 13.

---

## 7. Cell 10 — Unit Tests (hard stop on any failure)

1. **Mamba causality (most important).** With `return_sequence=True`: run on random `x`, then replace `x[:, t+1:]` with different random values; outputs at steps `0..t` must match (`torch.allclose(atol=1e-6)`) for several `t`. A failure means future days leak into past states.
2. **Mamba shape, determinism, stability.** `(N, 30, 8)` and `(N, 60, 8)` → `(N, d_out)`; same seed → identical raw-byte hash; 200 random batches with no NaN/Inf in outputs or gradients.
3. **Incremental correlation = full recomputation.** Over the full price history, `RollingCorr.corr()` must match the ported Phase 3 correlation computation to within `1e-9` on every date.
4. **Daily policy reproduces `graphs_baseline.parquet`.** Running `run_policy` with the always-fire rule must produce edge sets identical to `graphs_baseline.parquet` on ≥ 99.5% of dates. Every mismatching date must be a tie at the k-th boundary (correlation gap < 1e-8). This proves the Phase 4B graph builder matches Phase 3 exactly.
5. **Data alignment.** For sampled rows: last feature date < target date; the scaler was fitted only on training dates; `get_split(..., "test")` raises `PermissionError` (locks don't exist yet).

---

## 8. Cell 11 — Leakage Gate (hard stop on failure)

Uses **validation only** (the test split is still locked).

1. **Random-weights model** (Mamba, lookback 30) through the full pipeline on all folds' validation data: |mean rank IC| must be < 3 × its standard error (standard error = std of daily rank IC ÷ √n_days).
2. **Shuffled-target run:** train arm B for fold 1, seed 0 on training targets shuffled across dates (same days, labels scrambled). Validation |mean rank IC| < 3 × standard error. If the model learns from scrambled labels, something is leaking.
3. Unit test 1 (causality) passed.

Write `phase4_leakage_gate.json`. Any failure → `raise`, and the run ends within the first ~30 minutes.

---

## 9. Cell 12 — Memory and Timing Probe → Budget Plan

1. **Memory:** try arm C (Mamba, lookback 60, the heaviest) on fold 4 (the largest) with `batch_size=32` for one forward+backward. On `torch.cuda.OutOfMemoryError` or peak memory > 12 GB → move down the ladder (16, then 8). The chosen batch size applies to **all arms** (so arms stay comparable).
2. **Timing:** for arms A, B and C, train fold 4 / seed 0 for 2 epochs and time epoch 2 (epoch 1 includes warm-up overheads). Probe runs are discarded, never used as results.
3. **Projection:**

```python
per_run_seconds[arm, fold] = sec_per_epoch[arm] * (n_train_days[fold] / n_train_days["fold_4"]) \
                             * CONFIG["budget"]["expected_epochs_for_projection"] * CONFIG["budget"]["safety_factor"]
per_run_seconds["D", fold] = per_run_seconds["C", fold] * 1.05
# Wall time = longest-job-first assignment of all jobs across n_gpus (see §10)
projected_total_h = elapsed_h + makespan_h + phase4b_h + scoring_h
```

4. **Decision (pre-registered, automatic):**
   - `projected_total_h ≤ 10.5` with A–D → run everything.
   - Fits only without D → **skip D** and record `"D_skipped_reason": "budget"` in `budget_plan.json`.
   - A+B+C don't fit → `raise` with the numbers printed (fail fast, before long training starts). Nothing is silently shrunk: seeds, folds and epochs are never reduced automatically.

---

## 10. Cell 13 — Launch All Training, Wait, Check

1. Jobs = required arms (+ D if not skipped) × 4 folds × 5 seeds (80 jobs with D, 60 without).
2. Sort jobs by projected duration, longest first, and assign each to the GPU with the least assigned time so far. Write `jobs_gpu0.json` and `jobs_gpu1.json`.
3. Launch one `subprocess.Popen` per GPU with `CUDA_VISIBLE_DEVICES=<k>`, logging to `logs/gpu<k>.log`.
4. Poll every 60 seconds, printing `done / total`, elapsed time and projected finish. This keeps the notebook output alive and gives a readable progress trail.
5. After both workers exit:
   - Required arms A, B, C must each have **all 20** done markers. Otherwise → print every error traceback and `raise`.
   - Arm D: all 20 → eligible for selection; partial → excluded from selection and reported as such.

---

## 11. Cell 14 — Phase 4A Selection (validation only)

1. For each candidate arm (B, C, and D if complete): mean validation Sharpe and mean validation rank IC over its 20 runs (from `valpred_*` files; validation rows only).
2. Winner = highest validation Sharpe. If the top two differ by < 0.02 → higher validation rank IC wins.
3. Write `phase4_winner.json`: winning arm, config, the full validation summary table, and SHA-256 hashes of the winner's 20 checkpoints. Also write `phase4_validation_summary.csv` (all arms, including A for context).

---

## 12. Cell 15 — Phase 4B: Trigger Calibration and Selection (train + validation only)

Uses the **winner's 20 checkpoints**, inference only. Nothing is trained.

1. **Signals** (all use only information available on day d):
   - `corr_drift`: relative change of the correlation matrix since the last rebuild, ‖C_d − C_ref‖ / ‖C_ref‖.
   - `mamba_delta`: the winner model's mean memory-reset strength Δ on day d (`encoder.delta_signal()` after `encode`, before `propagate`). Arms whose encoder isn't Mamba can't use this signal; if the winner were ever the Transformer, Phase 4B runs with `corr_drift` only.
   - `combined`: rebuild if **either** signal passes its threshold.
2. **Calibration on the training period** (leakage-free): for each fold, seed model, signal and target rate in {10%, 20%, 35%, 50%}, find the threshold whose simulated rebuild rate over that fold's **training** days is closest to the target (bisection, 20 iterations). Max staleness = 20 days in all policies.
3. **Validation evaluation:** run each policy continuously from the last training day through the validation period and generate validation predictions. Also run the two anchors: `daily` (rebuild every day) and `frozen_at_start` (graph from the last training day, never rebuilt).
   - That is 12 trigger policies + 2 anchors, each × 20 models: inference only, a few minutes in total.
4. **Selection rule (pre-registered):** among the 12 trigger policies, choose the one with the **lowest realised validation rebuild rate** whose mean validation Sharpe is **≥ daily's mean validation Sharpe − 0.10**. Ties → higher validation rank IC. If no policy qualifies → choose the policy with the highest validation Sharpe and flag `"validation_noninferiority_met": false` (reported, not hidden).
5. Write `phase4b_policy.json` (policy, per-fold/per-seed thresholds, realised validation rebuild rates, full validation table) and `phase4b_validation_summary.csv`.

Note: the model is trained on daily graphs and only the **inference-time** graph refresh changes. This is a deliberate design choice (no retraining per policy); training under the triggered policy is future work.

---

## 13. Cell 16 — One-Shot Test Scoring

Runs only because both lock files now exist. If `phase4_TEST_SCORED.json` already exists in the working directory → `raise` (no double scoring within a run).

**Generate test predictions from checkpoints (first time any test row is read):**
- Arms A, B, C, (D): all 20 checkpoints each, daily graph.
- Phase 4B: the winner's 20 checkpoints under the selected policy, `daily`, and `frozen_at_start` (each policy runs continuously from the last training day, through validation, into test).
- Random and persistence floors.

**Phase 4A tests:**
- **Primary:** winner vs. A (same notebook, same schedule, different encoder): Wilcoxon on paired `(fold, seed)` rank IC + stationary bootstrap on the Sharpe difference (5000 iterations, block length 10).
- **Secondary (Holm-corrected):** every other completed Mamba arm vs. A.

**Phase 4B tests:**
- **Non-inferiority vs. daily:** the bootstrap 95% CI lower bound of Sharpe(selected) − Sharpe(daily) must be > −0.25 → "non-inferior". Also report the two-sided Wilcoxon on rank IC.
- **Superiority vs. frozen_at_start:** one-sided bootstrap on the Sharpe difference.
- Report the realised **test rebuild rate** and the graph-construction time used.

Write `phase4_significance_results.json`, `phase4_test_metrics.csv`, and `phase4_TEST_SCORED.json` (winner, policy, CONFIG hash, prediction-file hashes, timestamp).

**Comparison with Phase 3's Transformer** is done **locally after the run** (§16), because Phase 3's predictions aren't in the cache.

---

## 14. Cell 17 — Efficiency and Technical-Effect Measurements

### 14.1 Model efficiency (arm A vs. winner, same GPU and batch size)
Parameters; mean seconds per epoch; epochs to early stop; total training time per run; inference latency per trading day (median of 200 timed passes with `torch.cuda.synchronize()` around each); peak GPU memory. Save as `phase4_efficiency.csv`. Report honestly even if Mamba is slower: a pure-PyTorch loop may lose on speed at lookback 30.

### 14.2 Graph-construction cost (the patent-relevant evidence)
On the real 40-stock data, CPU, float64:
- Time per update: full recomputation vs. `RollingCorr.push` + `corr()`.
- Total graph-construction time over the test period under `daily` vs. the selected policy.

Synthetic scaling benchmark (random returns with a mid-sample regime switch, window 60, 750 days):

| N stocks | Full recompute daily | Incremental daily | Incremental + selected trigger rate |
|---|---|---|---|
| 40 | … | … | … |
| 100 | … | … | … |
| 500 | … | … | … |

Save as `phase4b_compute_benchmark.csv`. At 40 stocks the absolute savings are tiny; the scaling table shows where the mechanism matters, and that is what a technical-effect claim would rest on.

### 14.3 Explainability hook
Mean Δ per trading day (winner model, averaged over seeds) for 2019–2022 → `phase4_delta_by_date.csv`, plus a plot. Read-only; never used for training or selection.

---

## 15. Cell 18 — Output Verification and Draft Results

Assert that all of these exist (otherwise `raise`):

```
config_snapshot.json, run_manifest.json, budget_plan.json, phase4_leakage_gate.json,
phase4_validation_summary.csv, phase4_winner.json,
phase4b_validation_summary.csv, phase4b_policy.json,
phase4_test_metrics.csv, phase4_significance_results.json, phase4_TEST_SCORED.json,
phase4_efficiency.csv, phase4b_compute_benchmark.csv, phase4_delta_by_date.csv,
ckpt_*.pt (20 per completed arm), valpred_*.parquet, testpred_*.parquet, log_*.csv,
dmrgt4/*.py (the code that produced the results)
```

Then auto-generate `PHASE4_RESULTS_DRAFT.md` from the JSON/CSV files (same section structure as `PHASE3_RESULTS.md`), with every number filled in from the files, never typed by hand.

**Rule for reruns:** the first committed version that contains `phase4_TEST_SCORED.json` is the result of record. A later version may replace it only to fix a documented bug; both versions' results are then reported, with the reason.

---

## 16. After the Run (local)

1. Download the version's output (Kaggle UI → Output, or `kaggle kernels output <user>/<notebook-slug> -p ./phase4_output`).
2. Verify `phase4_TEST_SCORED.json` hashes against the downloaded prediction files.
3. **Winner vs. Phase 3 Transformer:** run the Phase 3 `stats.py` functions locally on Phase 4 winner test predictions vs. `predictions_dmrgt_all3_grid_baseline.parquet` (paired by `(fold, seed)`). Report this as a secondary comparison: different pipelines, same data and folds.
4. Finalise `PHASE4_RESULTS.md` from the draft.
5. Commit the notebook, outputs, `PORTING_LOG.md` and results to the **private** `phase-4` branch, and tag `phase4-complete`. Verify that the `phase3_locked/` hashes are unchanged and that nothing under `reference/phase3_src/` was modified (`git diff phase3-complete -- reference/` should show only the file moves, no content changes). Optionally remove `reference/` afterwards (§0.1 Step 6).

---

## 17. Patent Notes (not legal advice; confirm with VIT's IPR cell or a patent attorney)

- The candidate claim is the Phase 4B mechanism: **an inference-time relational-graph refresh triggered by a regime signal (including the selective state-space encoder's own memory-reset signal Δ), combined with incremental correlation updates**, shown (a) non-inferior to daily rebuilding on risk-adjusted performance and (b) cheaper in graph-construction compute, with the scaling benchmark as evidence.
- Mamba + adaptive training alone (Phase 4A) is likely obvious; it is the foundation, not the claim.
- Keep the notebook **private** on Kaggle and the repo private. No arXiv, talks, public notebooks or public commits until a filing decision is made. File a provisional application **before** any paper.
- Keep a dated private `INVENTION_LOG.md`.

---

## 18. Anti-Patterns

- Do not attach a code dataset or `pip install` anything: the code is in the notebook, and Internet is off.
- Do not rewrite Phase 3 components from memory: port them from `reference/phase3_src/` and diff them.
- Do not work on the `phase-3` branch, and do not edit, move or delete anything in `reference/phase3_src/` or `phase3_locked/`.
- Do not delete the reference folder before porting is verified in `PORTING_LOG.md`.
- Do not push `phase-4` until the repository is confirmed private.
- Do not use threads for parallelism: use one process per GPU.
- Do not generate or read test predictions before both lock files exist.
- Do not use validation or test residuals for the adaptive Huber delta.
- Do not change CONFIG after the first commit, or shrink seeds/folds/epochs to fit time.
- Do not drop a required arm because it lost, or re-score test after seeing results.
- Do not make the notebook or repo public before the patent decision.
