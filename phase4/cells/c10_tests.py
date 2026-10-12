# ─── Cell 10: unit tests (hard stop on any failure) ───
import hashlib
import torch
import d4_models, d4_graph, d4_data

_t0 = time.time()
UT = {}
_mc = CONFIG["mamba"]

# 1. Mamba causality — future days must not leak into past states
torch.manual_seed(0)
_enc = d4_models.MambaEncoder(8, _mc["d_model"], _mc["n_layers"], _mc["d_state"], _mc["d_conv"],
                              _mc["expand"], _mc["dropout"]).eval()
_enc.return_sequence = True
for _L in (30, 60):
    _x = torch.randn(6, _L, 8)
    with torch.no_grad():
        _y1 = _enc(_x)
        for _t in (0, 3, _L // 2, _L - 2):
            _x2 = _x.clone()
            _x2[:, _t + 1:] = torch.randn_like(_x2[:, _t + 1:])
            _y2 = _enc(_x2)
            assert torch.allclose(_y1[:, :_t + 1], _y2[:, :_t + 1], atol=1e-6), \
                f"Mamba CAUSALITY VIOLATED (L={_L}, t={_t}): future days leak into past states"
UT["mamba_causality"] = "pass"

# 2. Mamba shape, determinism, stability
_enc.return_sequence = False
_enc.train()
for _L in (30, 60):
    _out = _enc(torch.randn(40, _L, 8))
    assert tuple(_out.shape) == (40, _mc["d_model"]), _out.shape
def _fp(seed):
    torch.manual_seed(seed)
    return d4_models.MambaEncoder(8, _mc["d_model"], _mc["n_layers"], _mc["d_state"], _mc["d_conv"],
                                  _mc["expand"], _mc["dropout"])
def _hash(m):
    h = hashlib.sha256()
    for p in m.parameters():
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
assert _hash(_fp(7)) == _hash(_fp(7)) and _hash(_fp(7)) != _hash(_fp(8))
_enc2 = _fp(1)
torch.manual_seed(1)
for _ in range(200):
    _o = _enc2(torch.randn(40, 30, 8))
    _o.pow(2).mean().backward()
    assert torch.isfinite(_o).all(), "NaN/Inf in Mamba output"
    assert all(torch.isfinite(p.grad).all() for p in _enc2.parameters() if p.grad is not None), "NaN/Inf grad"
    _enc2.zero_grad()
UT["mamba_shape_determinism_stability"] = "pass"

# 3. Incremental correlation == full recomputation on every date
_rets = d4_graph.universe_returns(cfg)
_W = CONFIG["graph"]["corr_window"]
_rc = d4_graph.RollingCorr(_W, _rets.shape[1], CONFIG["phase4b"]["full_recompute_every"])
_maxdiff, _n = 0.0, 0
for _i, _r in enumerate(_rets.to_numpy(dtype=np.float64)):
    _rc.push(_r)
    if _i >= _W - 1:
        _full = _rets.iloc[_i - _W + 1: _i + 1].corr().values
        _maxdiff = max(_maxdiff, float(np.abs(_rc.corr() - _full).max()))
        _n += 1
assert _maxdiff < 1e-9, f"RollingCorr deviates from full recomputation: max |diff| {_maxdiff:.3e}"
UT["rolling_corr_max_abs_diff"] = _maxdiff
UT["rolling_corr_dates_checked"] = _n

# 4. Daily policy reproduces graphs_baseline.parquet (≥ 99.5% of dates; mismatches = k-th boundary ties)
_edges = d4_graph.load_daily_edges()
_build = d4_graph.make_graph_builder(cfg)
_rows = d4_graph.return_rows(_rets)
_days = [pd.Timestamp(d) for d in _rets.index[_W:]]               # 2nd graph date onward
_rc4 = d4_graph.RollingCorr(_W, _rets.shape[1], CONFIG["phase4b"]["full_recompute_every"])
for _r in _rets.iloc[:_W].to_numpy(dtype=np.float64):
    _rc4.push(_r)
_graphs, _ = d4_graph.run_policy(_days, lambda d: _rows[d], _build, lambda d: 0.0,
                                 d4_graph.rule_daily, float("inf"), _rc4)
def _edge_set(g):
    return set(zip(g[1].tolist(), g[0][0].tolist(), g[0][1].tolist()))
_first = _rets.index[_W - 1]
_checked = [(pd.Timestamp(_first), _build(d4_graph.warm_rolling_corr(_rets, _first, _W).corr()))] + \
           [(d, _graphs[d]) for d in _days]
_bad = []
for _d, _g in _checked:
    _ref = _edges[_d]
    if _edge_set((( _g[0]), _g[1])) != _edge_set(((_ref[0]), _ref[1])):
        _bad.append(_d)
assert len(_bad) / len(_checked) <= 0.005, f"{len(_bad)}/{len(_checked)} dates differ from graphs_baseline.parquet"
_kpos, _kneg = CONFIG["graph"]["k_pos"], CONFIG["graph"]["k_neg"]
for _d in _bad:                                                   # every mismatch must be a boundary tie
    _i = _rets.index.get_loc(_d)
    _C = _rets.iloc[_i - _W + 1: _i + 1].corr().values.copy()
    _ours, _theirs = _edge_set(_graphs.get(_d, _build(_C))), _edge_set(_edges[_d])
    for (_t, _s, _dd) in (_ours ^ _theirs):
        if _t == 2:
            raise AssertionError(f"sector edge mismatch on {_d.date()}")
        _row = np.delete(_C[_s], _s)
        _v = np.sort(_row)[::-1] if _t == 0 else np.sort(_row)
        _k = _kpos if _t == 0 else _kneg
        assert abs(_v[_k - 1] - _v[_k]) < 1e-8, f"non-tie edge mismatch on {_d.date()} (type {_t}, stock {_s})"
UT["daily_policy_vs_baseline"] = {"dates_checked": len(_checked), "mismatching_dates": len(_bad),
                                  "match_fraction": 1 - len(_bad) / len(_checked)}

# 5. Data alignment + test gate
_tb = d4_data.tables()
_prices = pd.read_parquet(d4_data.resolve_cache() / "prices.parquet")
_close = _prices.pivot(index="date", columns="ticker", values="close").sort_index()[UNIVERSE]
_nxt = _close.shift(-1) / _close - 1
_ex = _nxt.sub(_nxt.mean(axis=1), axis=0)
for _fold in [f.name for f in _tb["folds"]]:
    for _lb in (30, 60):
        _sd = d4_graph.get_split_g(_fold, "train", _lb)
        _bld = d4_data.get_builder(_fold, _lb)
        assert _bld.scaler_fit_dates[1] <= _tb["folds"][[f.name for f in _tb["folds"]].index(_fold)].train_dates[-1], \
            "scaler fit saw dates beyond the training window"
        for _j in np.linspace(0, len(_sd.dates) - 1, 5).astype(int):
            _d = pd.Timestamp(_sd.dates[_j])
            _loc = _tb["feat_wide"].index.get_loc(_d)
            _last_feat_date = _tb["feat_wide"].index[_loc]          # window ends at d
            _target_date = _close.index[_close.index.get_loc(_d) + 1]
            assert _last_feat_date < _target_date, "feature window not strictly before the target date"
            assert np.allclose(_sd.y[_j], _ex.loc[_d].values, atol=1e-12), "target is not the next-day excess return"
            assert _sd.X.shape[1:] == (40, _lb, 8)
try:
    d4_data.get_split("fold_1_2019_calm", "test")
    raise AssertionError("test split was readable while locked")
except PermissionError:
    pass
UT["data_alignment_and_test_gate"] = "pass"

(WORK / "phase4_unit_tests.json").write_text(json.dumps(UT, indent=2))
print(json.dumps(UT, indent=2))
print(f"unit tests passed in {time.time() - _t0:.0f}s")
