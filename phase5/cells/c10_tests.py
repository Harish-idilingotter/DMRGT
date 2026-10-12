# ─── Cell 10: unit tests (hard stop on any failure) ───
import hashlib, tempfile
import torch
import d4_models, d4_graph, d4_data, d4_eval

_t0 = time.time()
UT = {}
_mc = CONFIG["mamba"]
_MB = dict(d_model=_mc["d_model"], n_layers=_mc["n_layers"], d_state=_mc["d_state"], d_conv=_mc["d_conv"],
           expand=_mc["expand"], dropout=_mc["dropout"])
TARGETS = pd.read_parquet(CACHE / "targets.parquet")

# 1. Mamba causality — future days must not leak into past states
torch.manual_seed(0)
_enc = d4_models.MambaEncoder(8, **{k: _MB[k] for k in ("d_model", "n_layers", "d_state", "d_conv", "expand", "dropout")}).eval()
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
def _mk(seed):
    torch.manual_seed(seed)
    return d4_models.MambaEncoder(8, **_MB)
def _hash(m):
    h = hashlib.sha256()
    for p in m.parameters():
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
assert _hash(_mk(7)) == _hash(_mk(7)) and _hash(_mk(7)) != _hash(_mk(8))
_enc2 = _mk(1)
torch.manual_seed(1)
for _ in range(200):
    _o = _enc2(torch.randn(40, 30, 8))
    _o.pow(2).mean().backward()
    assert torch.isfinite(_o).all(), "NaN/Inf in Mamba output"
    assert all(torch.isfinite(p.grad).all() for p in _enc2.parameters() if p.grad is not None), "NaN/Inf grad"
    _enc2.zero_grad()
UT["mamba_shape_determinism_stability"] = "pass"

# 2b. F1: Δ at initialisation is in the intended range and spans an order of magnitude
_enc3 = _mk(3).eval()
_dl = []
with torch.no_grad():
    for _ in range(50):
        _enc3(torch.randn(40, 30, 8))
        _dl.append(_enc3.layers[-1].last_delta.flatten())
_dl = torch.cat(_dl).numpy()
_p5, _p95 = np.percentile(_dl, [5, 95])
assert 0.0005 <= _dl.mean() <= 0.2, f"initial mean Δ {_dl.mean():.4f} outside [0.0005, 0.2]"
assert _p95 / _p5 >= 10, f"initial Δ 5th-95th percentile ratio {_p95 / _p5:.1f} < one order of magnitude"
_m = d4_models.make_model("mamba", CONFIG["mamba"])
d4_models.assert_dt_init(_m)
_tagged = [n for n, p in _m.named_parameters() if getattr(p, "_no_weight_decay", False)]
assert any("A_log" in n for n in _tagged) and any("dt_proj.bias" in n for n in _tagged) and any(n.endswith(".D") for n in _tagged)
_g = d4_models.param_groups(_m, 1e-4)
assert _g[0]["weight_decay"] == 1e-4 and _g[1]["weight_decay"] == 0.0
assert sum(p.numel() for p in _g[0]["params"]) + sum(p.numel() for p in _g[1]["params"]) == sum(p.numel() for p in _m.parameters())
UT["f1_dt_init"] = {"mean_delta": float(_dl.mean()), "p5": float(_p5), "p95": float(_p95), "ratio": float(_p95 / _p5)}

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

# 4 + 6. Daily policy and the in-notebook baseline variant both reproduce the cache's graphs_baseline.parquet
_cached = d4_graph.load_cached_baseline_edges()
_build = d4_graph.make_graph_builder(cfg)
_rows = d4_graph.return_rows(_rets)
_kpos, _kneg = CONFIG["graph"]["k_pos"], CONFIG["graph"]["k_neg"]
def _edge_set(g):
    return set(zip(g[1].tolist(), g[0][0].tolist(), g[0][1].tolist()))
def _tie_check(_d, ours, theirs):
    _i = _rets.index.get_loc(_d)
    _C = _rets.iloc[_i - _W + 1: _i + 1].corr().values
    for (_t, _s, _dd) in (ours ^ theirs):
        assert _t != 2, f"sector edge mismatch on {_d.date()}"
        _row = np.delete(_C[_s], _s)
        _v = np.sort(_row)[::-1] if _t == 0 else np.sort(_row)
        _k = _kpos if _t == 0 else _kneg
        assert abs(_v[_k - 1] - _v[_k]) < 1e-8, f"non-tie edge mismatch on {_d.date()} (type {_t}, stock {_s})"
_days = [pd.Timestamp(d) for d in _rets.index[_W:]]
_rc4 = d4_graph.RollingCorr(_W, _rets.shape[1], CONFIG["phase4b"]["full_recompute_every"])
for _r in _rets.iloc[:_W].to_numpy(dtype=np.float64):
    _rc4.push(_r)
_graphs, _ = d4_graph.run_policy(_days, lambda d: _rows[d], _build, lambda d: 0.0,
                                 d4_graph.rule_daily, float("inf"), _rc4)
_first = pd.Timestamp(_rets.index[_W - 1])
_checked = [(_first, _build(d4_graph.warm_rolling_corr(_rets, _first, _W).corr()))] + [(d, _graphs[d]) for d in _days]
_bad = [d for d, g in _checked if _edge_set(g) != _edge_set(_cached[d])]
assert len(_bad) / len(_checked) <= 0.005, f"{len(_bad)}/{len(_checked)} dates differ from graphs_baseline.parquet"
for _d in _bad:
    _tie_check(_d, _edge_set(dict(_checked)[_d]), _edge_set(_cached[_d]))
UT["daily_policy_vs_cache"] = {"dates_checked": len(_checked), "mismatching_dates": len(_bad),
                               "match_fraction": 1 - len(_bad) / len(_checked)}
_var = d4_graph.load_daily_edges("baseline")
assert set(_var) == set(_cached), "baseline variant covers different dates than the cache"
_bad6 = [d for d in _cached if _edge_set(_var[d]) != _edge_set(_cached[d])]
assert len(_bad6) / len(_cached) <= 0.005, f"baseline variant differs from cache on {len(_bad6)}/{len(_cached)} dates"
for _d in _bad6:
    _tie_check(_d, _edge_set(_var[_d]), _edge_set(_cached[_d]))
UT["baseline_variant_vs_cache"] = {"dates": len(_cached), "mismatching_dates": len(_bad6),
                                   "match_fraction": 1 - len(_bad6) / len(_cached)}

# 5. Data alignment + test gate (all six folds, both lookbacks)
_tb = d4_data.tables()
_prices = pd.read_parquet(CACHE / "prices.parquet")
_close = _prices.pivot(index="date", columns="ticker", values="close").sort_index()[UNIVERSE]
_nxt = _close.shift(-1) / _close - 1
_ex = _nxt.sub(_nxt.mean(axis=1), axis=0)
_folds_by_name = {f.name: f for f in _tb["folds"]}
for _fold in CONFIG["folds_selection"] + HOLDOUT_NAMES:
    for _lb in (30, 60):
        _sd = d4_graph.get_split_g(_fold, "train", _lb, "baseline")
        _bld = d4_data.get_builder(_fold, _lb, graph_key="baseline")
        assert _bld.scaler_fit_dates[1] <= _folds_by_name[_fold].train_dates[-1], "scaler fit saw dates beyond the training window"
        for _j in np.linspace(0, len(_sd.dates) - 1, 5).astype(int):
            _d = pd.Timestamp(_sd.dates[_j])
            _target_date = _close.index[_close.index.get_loc(_d) + 1]
            assert _d < _target_date, "feature window not strictly before the target date"
            assert np.allclose(_sd.y[_j], _ex.loc[_d].values, atol=1e-12), "target is not the next-day excess return"
            assert _sd.X.shape[1:] == (len(UNIVERSE), _lb, 8)
try:
    d4_data.get_split("fold_1_2019_calm", "test")
    raise AssertionError("test split was readable while locked")
except PermissionError:
    pass
UT["data_alignment_and_test_gate"] = "pass"

# 7. Lock enforcement: test rows need ALL FOUR lock files
_real_locks = d4_data.LOCKS
with tempfile.TemporaryDirectory() as _td:
    _tl = [Path(_td) / n for n in ("a.json", "b.json", "c.json", "d.json")]
    d4_data.LOCKS = _tl
    try:
        for _k in range(4):
            try:
                d4_data.assert_test_unlocked()
                raise AssertionError(f"test readable with only {_k} of 4 locks")
            except PermissionError:
                pass
            _tl[_k].write_text("{}")
        d4_data.assert_test_unlocked()                    # all four present → allowed
    finally:
        d4_data.LOCKS = _real_locks
assert len(_real_locks) == 4 and not d4_data.test_unlocked()
UT["four_lock_enforcement"] = "pass"

# F2: val_es / val_sel are disjoint, chronological, and cover the validation period; control cannot see val_sel
for _f in _tb["folds"]:
    _es, _sel = d4_data.split_val(_f.val_dates)
    assert len(_es) + len(_sel) == len(_f.val_dates) and _es.max() < _sel.min() and len(set(_es) & set(_sel)) == 0
    _sdv = d4_graph.get_split_g(_f.name, "val", 30, "baseline")
    _sde = d4_graph.get_split_g(_f.name, "val_es", 30, "baseline")
    _sds = d4_graph.get_split_g(_f.name, "val_sel", 30, "baseline")
    assert len(_sde.dates) + len(_sds.dates) == len(_sdv.dates) and _sde.dates.max() < _sds.dates.min()
    assert (d4_eval.val_half_labels(_f, _sdv.dates) == np.r_[["es"] * len(_sde.dates), ["sel"] * len(_sds.dates)]).all()
try:
    d4_train.ControlMetrics(val_es_rank_ic=0.0, val_sel_rank_ic=0.1)
    raise AssertionError("training control accepted a val_sel metric")
except AssertionError as _e:
    assert "val_es" in str(_e)
_ac = d4_train.AdaptiveController(warmup=3, patience=2)
for _e_, _ic in enumerate([0.9, 0.9, 0.9, 0.01, 0.0, 0.0, 0.0]):     # huge IC during warmup must not become "best"
    _r = _ac.update(_e_, d4_train.ControlMetrics(val_es_rank_ic=_ic, val_es_loss=0.0))
assert _ac.best_epoch >= 3, "a warmup epoch became the best checkpoint"
UT["val_es_val_sel_separation"] = "pass"

# F3: random baseline draws fresh noise for every (seed, day)
_X0 = np.zeros((len(UNIVERSE), 30, 8))
_a, _b, _c = (d4_eval.random_model(_X0, 0, 1), d4_eval.random_model(_X0, 0, 2), d4_eval.random_model(_X0, 0, 1))
assert not np.allclose(_a, _b) and np.array_equal(_a, _c) and not np.allclose(_a, d4_eval.random_model(_X0, 1, 1))
UT["f3_random_baseline"] = "pass"

# Phase 5 §10.1: the weights path reproduces backtest() (single seed, half-life 0, q_out == q_in)
_fold = _tb["folds"][0]
_sdv = d4_graph.get_split_g(_fold.name, "val", 30, "baseline")
_rng = np.random.default_rng(5)
_pred = _rng.standard_normal(_sdv.y.shape)
_pf = d4_eval.make_pred_frame("x", _fold.name, 0, "val", _sdv.dates, UNIVERSE, _pred, _sdv.y)
_bt = d4_eval.backtest(_pf, cfg, split="val", targets=TARGETS)
_raw = d4_eval.raw_returns_from_targets(TARGETS, UNIVERSE)
_Sw = pd.DataFrame(_pred, index=pd.DatetimeIndex(_sdv.dates), columns=UNIVERSE)
_Ww = d4_eval.buffered_weights(_Sw, CONFIG["postproc"]["q_in"], CONFIG["postproc"]["q_in"])
_ret, _st = d4_eval.backtest_from_weights(_Ww, _raw, CONFIG["backtest"]["commission_bps"], CONFIG["backtest"]["slippage_bps"])
for _k, _kk in (("sharpe", "sharpe"), ("turnover", "turnover"), ("max_drawdown", "max_drawdown"), ("cum_return", "cum_return")):
    assert abs(_st[_kk] - _bt[_k]) < 1e-10, f"weights path differs from backtest() on {_k}: {_st[_kk]} vs {_bt[_k]}"
_r2, _t2 = d4_eval.strategy_series(_pf, _raw, UNIVERSE, CONFIG["backtest"], 0, 0.2, 0.2)   # via the ensemble (z-score) path
assert abs(d4_eval.pooled_stats(_r2, _t2)["sharpe"] - _bt["sharpe"]) < 1e-10
UT["harness_regression_new_path"] = "pass"

# Phase 5 §10.2: 3-day, 6-stock hand-computed example
_tk = list("ABCDEF")
_sc = pd.DataFrame([[6, 5, 4, 3, 2, 1], [1, 2, 3, 4, 5, 6], [3, 6, 1, 2, 5, 4]], columns=_tk,
                   index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]), dtype=float)
_rt = pd.DataFrame([[0, 0, 0, 0, 0, 0], [0.02, 0.01, 0, -0.01, 0, -0.02], [-0.01, 0.03, 0.02, 0, 0, 0.01]],
                   columns=_tk, index=_sc.index, dtype=float)
# hand-derived: weights d0 long A short F; d1 long F short A; d2 long B short C (k = round(0.2*6) = 1)
# earned: d0 0; d1 = W0.r1 = 0.02-(-0.02) = 0.04; d2 = W1.r2 = 0.01-(-0.01) = 0.02
# turnover |dW| (sum): d0 0 (first row), d1 4, d2 4; fee 15 bps -> cost = 0.0015*4/2 = 0.003 on d1 and d2
# after-cost: 0, 0.037, 0.017 -> mean 0.018, sample sd sqrt(0.000343); one-way turnover mean = (0+4+4)/3/2
_exp = {"sharpe": 0.018 / np.sqrt(0.000343) * np.sqrt(252), "turnover": 8 / 3 / 2, "cum_return": 1.037 * 1.017 - 1,
        "max_drawdown": 0.0}
_r_h, _s_h = d4_eval.backtest_daily_deciles(_sc, _rt, _tk, 10, 5, 0.2)
_r_w, _s_w = d4_eval.backtest_from_weights(d4_eval.buffered_weights(_sc, 0.2, 0.2), _rt, 10, 5)
for _name, _s in (("ported harness", _s_h), ("weights path", _s_w)):
    for _k, _v in _exp.items():
        assert abs(_s[_k if _k != "max_drawdown" else "max_drawdown"] - _v) < 1e-9, f"hand check failed ({_name}, {_k}): {_s[_k]} vs {_v}"
assert np.allclose(_r_h.values, [0, 0.037, 0.017]) and np.allclose(_r_w.values, [0, 0.037, 0.017])
UT["harness_hand_check"] = "pass"

# Phase 5 §10.3-5: EMA causality, buffer logic, ensemble
_S = pd.DataFrame(np.random.default_rng(1).standard_normal((40, 20)), index=pd.date_range("2020-01-01", periods=40, freq="B"))
for _hl in (2, 5):
    _a = d4_eval.ema_smooth(_S, _hl)
    _S2 = _S.copy(); _S2.iloc[21:] = np.random.default_rng(2).standard_normal(_S2.iloc[21:].shape)
    _b = d4_eval.ema_smooth(_S2, _hl)
    assert np.array_equal(_a.iloc[:21].to_numpy(), _b.iloc[:21].to_numpy()), "EMA is not causal"
assert d4_eval.ema_smooth(_S, 0) is _S
_n = 20; _nl = int(round(0.2 * _n))
_Sx = pd.DataFrame(np.random.default_rng(3).standard_normal((300, _n)))
_W0 = d4_eval.buffered_weights(_Sx, 0.2, 0.2)
_plain = np.zeros(_Sx.shape)
for _t_, _row in enumerate(_Sx.to_numpy()):
    _o = np.argsort(-_row, kind="stable"); _plain[_t_, _o[:_nl]] = 1 / _nl; _plain[_t_, _o[-_nl:]] = -1 / _nl
assert np.array_equal(_W0.to_numpy(), _plain), "q_out == q_in must equal plain top/bottom selection"
_Wb = d4_eval.buffered_weights(_Sx, 0.2, 0.4)
assert (_Wb.abs().sum(axis=1) > 0).all() and ((_Wb > 0).sum(axis=1) == _nl).all() and ((_Wb < 0).sum(axis=1) == _nl).all()
assert not ((_Wb > 0) & (_Wb < 0)).any().any()
assert _Wb.diff().abs().sum(axis=1).mean() <= _W0.diff().abs().sum(axis=1).mean(), "buffer increased turnover"
_pz = pd.DataFrame({"date": np.repeat(pd.date_range("2020-01-01", periods=5), 8), "ticker": list("ABCDEFGH") * 5,
                    "seed": 0, "pred": np.random.default_rng(4).standard_normal(40)})
_one = d4_eval.ensemble_scores(_pz)
_three = d4_eval.ensemble_scores(pd.concat([_pz.assign(seed=s) for s in (0, 1, 2)]))
assert np.allclose(_one.to_numpy(), _three.to_numpy()), "ensemble of identical seeds != single-seed z-score"
UT["portfolio_ema_buffer_ensemble"] = "pass"

(WORK / "phase5_unit_tests.json").write_text(json.dumps(UT, indent=2))
print(json.dumps(UT, indent=2))
print(f"unit tests passed in {time.time() - _t0:.0f}s")
