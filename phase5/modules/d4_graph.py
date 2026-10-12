"""DMRGT Phase 5 — graph module (Phase 4 spec §6.2 + Phase 5 §5).

Ported verbatim from Phase 3 graph.py: _pct_change_no_fill, _universe_returns,
_sector_edges, and the correlation→edges rule (nan_to_num, diagonal ∓inf,
argsort, top-k_pos / bottom-k_neg, sector edges, directed). The daily-edge
loader reads graphs_baseline.parquet.

New: RollingCorr (incremental float64 correlation), run_policy (trigger-
policy runner), trigger rules + threshold calibration, and the Phase 4B
inference engine (checkpoint → per-day encodings/Δ → policy graphs →
predictions). Inference only; nothing here trains.
"""

from __future__ import annotations

import time
from collections import Counter, deque

import numpy as np
import pandas as pd
import torch

import d4_data
import d4_eval
import d4_models


# ─── ported from Phase 3 graph.py ──────────────────────────────────────────

def _pct_change_no_fill(df: pd.DataFrame) -> pd.DataFrame:
    """pct_change with NO filling, across pandas 2.x and 3.x.

    pandas 2.x: pct_change(fill_method=None) — explicit no-fill.
    pandas 3.x: the fill_method keyword was removed BECAUSE no-fill
    became the only behaviour, so plain pct_change() is identical.
    The cache was built under pandas 2.x no-fill semantics; the §6b
    baseline-identity test proves the rebuild reproduces it byte-for-
    byte under either version.
    """
    try:
        return df.pct_change(fill_method=None)
    except TypeError:  # pandas ≥ 3: fill_method removed
        return df.pct_change()


def _universe_returns(prices: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Wide close prices for the 40 universe tickers → daily returns.

    rets.loc[d] is the return from d-1 to d; the first row (NaN) is
    dropped. Same op order as Phase 2 so the skip logic below lands on
    the same dates.
    """
    universe = list(cfg["universe"]["tickers"])
    px_wide = (prices.pivot(index="date", columns="ticker", values="close")
               .sort_index())
    missing = [t for t in universe if t not in px_wide.columns]
    if missing:
        raise ValueError(f"prices.parquet is missing universe tickers: {missing}")
    px_universe = px_wide[universe]
    rets = _pct_change_no_fill(px_universe)
    return rets.iloc[1:]


def _sector_edges(cfg: dict) -> tuple[list[int], list[int]]:
    """Static intra-sector edges (both directions), as index pairs.

    With 5 tickers per sector × 8 sectors: 8 × 20 = 160 directed edges.
    """
    universe = list(cfg["universe"]["tickers"])
    sectors_cfg = cfg.get("sectors", {})
    if not sectors_cfg:
        raise ValueError("configs.yaml missing 'sectors:' block — required for graph.py")
    include_sector = bool(cfg["graph"].get("include_sector", True))
    srcs: list[int] = []
    dsts: list[int] = []
    if not include_sector:
        return srcs, dsts
    n = len(universe)
    for i in range(n):
        for j in range(n):
            if i >= j:
                continue
            if sectors_cfg.get(universe[i]) == sectors_cfg.get(universe[j]):
                srcs.extend([i, j])  # directed both ways
                dsts.extend([j, i])
    sector_counts = Counter(sectors_cfg[t] for t in universe)
    expected = sum(c * (c - 1) for c in sector_counts.values())
    assert len(srcs) == expected, (
        f"Sector edge count mismatch: built {len(srcs)}, expected "
        f"{expected} from configs.yaml (5 per sector × 8 sectors → 160)"
    )
    return srcs, dsts



def edges_from_corr(corr: np.ndarray, cfg: dict, sector_src, sector_dst):
    """Correlation matrix → (edge_index [2, E], edge_type [E]) torch tensors.

    The rule of Phase 3's _edges_for_returns_window, unchanged: nan_to_num →
    diagonal −inf → argsort(−corr) for type 0 (top k_pos) → diagonal +inf →
    argsort(corr) for type 1 (bottom k_neg) → static sector edges (type 2).
    Rows are emitted in the cached file's (edge_type, src, dst) order.
    """
    n_stocks = corr.shape[0]
    k_pos = int(cfg["graph"]["k_pos"])
    k_neg = int(cfg["graph"]["k_neg"])
    corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)   # returns a copy
    src_l, dst_l, typ_l = [], [], []
    np.fill_diagonal(corr, -np.inf)
    pos_order = np.argsort(-corr, axis=1)
    for i in range(n_stocks):
        for rank in range(k_pos):
            src_l.append(i); dst_l.append(int(pos_order[i, rank])); typ_l.append(0)
    np.fill_diagonal(corr, np.inf)
    neg_order = np.argsort(corr, axis=1)
    for i in range(n_stocks):
        for rank in range(k_neg):
            src_l.append(i); dst_l.append(int(neg_order[i, rank])); typ_l.append(1)
    for s, d in zip(sector_src, sector_dst):
        src_l.append(int(s)); dst_l.append(int(d)); typ_l.append(2)
    src_a, dst_a, typ_a = np.array(src_l), np.array(dst_l), np.array(typ_l)
    order = np.lexsort((dst_a, src_a, typ_a))            # (edge_type, src, dst)
    ei = torch.tensor(np.stack([src_a[order], dst_a[order]]), dtype=torch.long)
    et = torch.tensor(typ_a[order], dtype=torch.long)
    return ei, et


def make_graph_builder(cfg: dict):
    sector_src, sector_dst = _sector_edges(cfg)

    def build_graph(C):
        return edges_from_corr(C, cfg, sector_src, sector_dst)
    return build_graph


_EDGES: dict = {}


def build_variant_edges(variant: str) -> dict:
    """The graph of one grid variant, built in-notebook from prices.parquet with the
    ported rule: trailing corr_window Pearson correlation (pandas, as in Phase 3
    graph.py) → edges_from_corr. {Timestamp: (edge_index, edge_type)}."""
    cfgv = d4_data.load_cfg(variant)
    W = int(cfgv["graph"]["corr_window"])
    rets = universe_returns(cfgv)
    build = make_graph_builder(cfgv)
    out = {}
    for idx in range(W - 1, len(rets)):
        window_rets = rets.iloc[idx - W + 1: idx + 1]
        assert len(window_rets) == W
        assert window_rets.isna().sum().sum() == 0, f"NaN returns in correlation window on {rets.index[idx].date()}"
        out[pd.Timestamp(rets.index[idx])] = build(window_rets.corr().values)
    return out


def load_daily_edges(variant: str = "baseline") -> dict:
    """{Timestamp: (edge_index, edge_type)} for a graph variant (cached per process).
    Built in-notebook for EVERY variant, including 'baseline' (which unit test 6
    cross-checks against the cache's graphs_baseline.parquet)."""
    if variant not in _EDGES:
        _EDGES[variant] = build_variant_edges(variant)
    return _EDGES[variant]


def load_cached_baseline_edges() -> dict:
    """graphs_baseline.parquet from the cache (used only for the cross-check)."""
    g = pd.read_parquet(d4_data.resolve_cache() / "graphs_baseline.parquet")
    out = {}
    for d, day in g.groupby("date", sort=True):
        out[pd.Timestamp(d)] = (torch.tensor(day[["src", "dst"]].values.T, dtype=torch.long),
                                torch.tensor(day["edge_type"].values, dtype=torch.long))
    return out


def universe_returns(cfg: dict) -> pd.DataFrame:
    prices = pd.read_parquet(d4_data.resolve_cache() / "prices.parquet")
    return _universe_returns(prices, cfg)


# ─── incremental rolling correlation (new) ─────────────────────────────────

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


def warm_rolling_corr(rets: pd.DataFrame, end_date, window: int, full_recompute_every=250) -> RollingCorr:
    """RollingCorr holding the `window` returns ending at `end_date` inclusive."""
    loc = rets.index.get_loc(pd.Timestamp(end_date))
    assert loc >= window - 1, f"only {loc + 1} returns up to {end_date}"
    rc = RollingCorr(window, rets.shape[1], full_recompute_every)
    for r in rets.iloc[loc - window + 1: loc + 1].to_numpy(dtype=np.float64):
        rc.push(r)
    return rc


class PrecomputedCorr:
    """Replays correlation matrices computed once (calibration sims). Same
    push()/corr() interface as RollingCorr, so run_policy is one code path."""

    def __init__(self, C_by_step):
        self.C = C_by_step          # list: [C before first push, C after push 1, ...]
        self.i = 0

    def push(self, r):
        self.i += 1

    def corr(self):
        return self.C[self.i]


def corr_by_date(rets: pd.DataFrame, window: int) -> dict:
    """{date: corr matrix over the trailing window ending that date} for every
    date with a full window, via one incremental pass."""
    rc = RollingCorr(window, rets.shape[1])
    out = {}
    for i, (d, r) in enumerate(zip(rets.index, rets.to_numpy(dtype=np.float64))):
        rc.push(r)
        if i >= window - 1:
            out[pd.Timestamp(d)] = rc.corr()
    return out


# ─── trigger policy runner (new; inference only) ───────────────────────────

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


# daily = rule always fires; frozen_at_start = rule never fires and max_stale = ∞.
def rule_daily(s_corr, s_delta):
    return True


def rule_never(s_corr, s_delta):
    return False


def make_rule(signal: str, tau_corr: float | None, tau_delta: float | None):
    if signal == "corr_drift":
        return lambda sc, sd: sc > tau_corr
    if signal == "mamba_delta":
        return lambda sc, sd: sd > tau_delta
    if signal == "combined":
        return lambda sc, sd: (sc > tau_corr) or (sd > tau_delta)
    raise ValueError(signal)


# ─── threshold calibration on TRAINING days (leakage-free) ─────────────────

def simulate_rate(sim_days, C_steps, delta_by_day, rule, max_stale) -> float:
    """Realised rebuild rate of `rule` over sim_days, using the SAME
    run_policy as inference (graph building stubbed out)."""
    pc = PrecomputedCorr(C_steps)
    zero = np.zeros(1)
    _, rebuilds = run_policy(
        sim_days, lambda d: zero, lambda C: None,
        lambda d: float(delta_by_day[d]) if delta_by_day is not None else 0.0,
        rule, max_stale, pc)
    return len(rebuilds) / max(1, len(sim_days))


def _bisect(rate_fn, lo, hi, target, iters=20):
    """Threshold whose rate is closest to target (rate is non-increasing in
    the threshold)."""
    best_t, best_err = hi, abs(rate_fn(hi) - target)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        r = rate_fn(mid)
        err = abs(r - target)
        if err < best_err:
            best_t, best_err = mid, err
        if r > target:
            lo = mid
        else:
            hi = mid
    return best_t


def calibrate_thresholds(signal, target_rate, sim_days, C_steps, delta_by_day, max_stale,
                         iters=20, corr_cache=None):
    """{'tau_corr', 'tau_delta', 'sim_rate'} for one (fold, model, signal, target).
    `corr_cache` memoises the corr-drift threshold per rate: it depends only on
    the fold's correlation history, not on the model."""
    corr_cache = {} if corr_cache is None else corr_cache
    d_vals = np.array([delta_by_day[d] for d in sim_days]) if delta_by_day is not None else None

    def corr_tau(rate):
        key = round(float(rate), 9)
        if key not in corr_cache:
            corr_cache[key] = _corr_tau(rate)
        return corr_cache[key]

    def _corr_tau(rate):
        f = lambda t: simulate_rate(sim_days, C_steps, None, make_rule("corr_drift", t, None), max_stale)
        return _bisect(f, 0.0, 5.0, rate, iters)

    def delta_tau(rate):
        f = lambda t: simulate_rate(sim_days, C_steps, delta_by_day, make_rule("mamba_delta", None, t), max_stale)
        return _bisect(f, float(d_vals.min()) - 1e-9, float(d_vals.max()) + 1e-9, rate, iters)

    if signal == "corr_drift":
        tc, td = corr_tau(target_rate), None
        rule = make_rule(signal, tc, None)
    elif signal == "mamba_delta":
        tc, td = None, delta_tau(target_rate)
        rule = make_rule(signal, None, td)
    elif signal == "combined":
        # One shared per-signal rate q; each signal's threshold is its own
        # single-signal calibration at q; bisect q so the UNION hits target.
        cache = {}

        def pair(q):
            if q not in cache:
                cache[q] = (corr_tau(q), delta_tau(q))
            return cache[q]

        def union_rate(q):
            tc_, td_ = pair(q)
            return simulate_rate(sim_days, C_steps, delta_by_day, make_rule("combined", tc_, td_), max_stale)

        # union rate rises with q, _bisect expects a rate that falls with its argument → search s = 1 − q
        s_ = _bisect(lambda x: union_rate(round(1.0 - x, 9)), 0.0, 1.0, target_rate, iters)
        q = round(1.0 - s_, 9)
        tc, td = pair(q)
        rule = make_rule(signal, tc, td)
    else:
        raise ValueError(signal)
    return {"tau_corr": None if tc is None else float(tc),
            "tau_delta": None if td is None else float(td),
            "sim_rate": float(simulate_rate(sim_days, C_steps, delta_by_day, rule, max_stale))}


# ─── Phase 4B inference engine ─────────────────────────────────────────────

def get_split_g(fold_name: str, split: str, lookback: int, graph: str = "baseline"):
    """Split tensors with Phase 3's graph-coverage rule applied against the given graph
    variant's dates — the same call training uses, so datasets match."""
    gd = set(load_daily_edges(graph).keys())
    return d4_data.get_split(fold_name, split, lookback=lookback, graph_dates=gd, graph_key=graph)


_RET_ROWS: dict = {}


def return_rows(rets: pd.DataFrame) -> dict:
    if not _RET_ROWS:
        for d, r in zip(rets.index, rets.to_numpy(dtype=np.float64)):
            _RET_ROWS[pd.Timestamp(d)] = r
    return _RET_ROWS


class ModelInference:
    """A trained checkpoint put to inference use: encodings and Δ per day are
    computed once; any graph policy then only needs the cheap propagate step."""

    def __init__(self, ckpt_path, device, snapshot: dict):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.ck = ck
        self.arm, self.fold, self.seed = ck["arm"], ck["fold"], int(ck["seed"])
        self.lookback, self.encoder_type = int(ck["lookback"]), ck["encoder"]
        self.graph = ck.get("graph", "baseline")
        self.device = device
        self.model = d4_models.make_model(self.encoder_type, snapshot["CONFIG"]["mamba"])
        self.model.load_state_dict(ck["model_state_dict"])
        self.model.to(device).eval()
        self._enc: dict = {}                 # split -> (h [n,N,d], delta [n] | None)

    def encode(self, split: str, batch: int = 64):
        if split in self._enc:
            return self._enc[split]
        sd = get_split_g(self.fold, split, self.lookback, self.graph)
        X = torch.from_numpy(sd.X).float()
        hs, ds = [], []
        with torch.no_grad():
            for i in range(0, len(X), batch):
                xb = X[i: i + batch].to(self.device)
                hs.append(self.model.encode(xb))
                if self.encoder_type == "mamba":
                    # signal read after encode, before propagate (spec §12)
                    ds.append(self.model.encoder.delta_per_day(len(xb)).cpu())
        h = torch.cat(hs, 0)
        delta = torch.cat(ds, 0).numpy().astype(np.float64) if ds else None
        self._enc[split] = (h, delta)
        return self._enc[split]

    def split_data(self, split: str):
        return get_split_g(self.fold, split, self.lookback, self.graph)

    def delta_by_day(self, splits) -> dict | None:
        out = {}
        for s in splits:
            _, delta = self.encode(s)
            if delta is None:
                return None
            for d, v in zip(self.split_data(s).dates, delta):
                out[pd.Timestamp(d)] = float(v)
        return out

    def predict_with_graphs(self, split: str, graphs: dict, batch: int = 64) -> np.ndarray:
        """graphs: {date: (ei, et)} covering the split's sample dates."""
        h, _ = self.encode(split)
        dates = [pd.Timestamp(d) for d in self.split_data(split).dates]
        preds = []
        with torch.no_grad():
            for i in range(0, len(dates), batch):
                dd = dates[i: i + batch]
                ei = [graphs[d][0].to(self.device) for d in dd]
                et = [graphs[d][1].to(self.device) for d in dd]
                preds.append(self.model.propagate_with_edge_lists(h[i: i + len(dd)], ei, et).cpu().numpy())
        return np.concatenate(preds, 0)

    def frame(self, model_name: str, split: str, preds: np.ndarray) -> pd.DataFrame:
        sd = self.split_data(split)
        df = d4_eval.make_pred_frame(model_name, self.fold, self.seed, split, sd.dates,
                                     sd.tickers, preds, sd.y)
        if split == "val":
            df["val_half"] = d4_eval.val_half_labels(self.fold, df["date"])
        return df

    def delta_channels(self, split: str) -> np.ndarray:
        """[n_days, d_inner] per-day, per-channel Δ of the last layer's final step (diagnostics)."""
        sd = self.split_data(split)
        X = torch.from_numpy(sd.X).float()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 64):
                xb = X[i: i + 64].to(self.device)
                self.model.encode(xb)
                out.append(self.model.encoder.delta_channels_per_day(len(xb)).cpu())
        return torch.cat(out, 0).numpy()


def policy_days(rets: pd.DataFrame, fold, end_date) -> list:
    anchor = pd.Timestamp(fold.train_dates[-1])
    idx = rets.index
    return [pd.Timestamp(d) for d in idx[(idx > anchor) & (idx <= pd.Timestamp(end_date))]]


def run_named_policy(policy: dict, thresholds: dict | None, fold, rets: pd.DataFrame,
                     end_date, delta_by_day: dict | None, cfg: dict, window: int,
                     max_stale: float):
    """One policy over (anchor = last training day, end_date]. Returns
    (graphs, rebuild_days, stats). `policy`: {'kind': 'daily'|'frozen'|'trigger',
    'signal': ...}; `thresholds`: {'tau_corr','tau_delta'} for trigger kinds."""
    build_graph = make_graph_builder(cfg)
    build_time = [0.0]

    def timed_build(C):
        t0 = time.perf_counter()
        g = build_graph(C)
        build_time[0] += time.perf_counter() - t0
        return g

    kind = policy["kind"]
    if kind == "daily":
        rule, stale = rule_daily, max_stale
    elif kind == "frozen":
        rule, stale = rule_never, float("inf")
    else:
        rule = make_rule(policy["signal"], thresholds["tau_corr"], thresholds["tau_delta"])
        stale = max_stale
    days = policy_days(rets, fold, end_date)
    rows = return_rows(rets)
    rc = warm_rolling_corr(rets, fold.train_dates[-1], window)
    sig = (lambda d: delta_by_day.get(d, 0.0)) if delta_by_day is not None else (lambda d: 0.0)
    t0 = time.perf_counter()
    graphs, rebuilds = run_policy(days, lambda d: rows[d], timed_build, sig, rule, stale, rc)
    total = time.perf_counter() - t0
    return graphs, rebuilds, {"n_days": len(days), "n_rebuilds": len(rebuilds),
                              "rebuild_rate": len(rebuilds) / max(1, len(days)),
                              "graph_build_seconds": build_time[0], "policy_seconds": total}
