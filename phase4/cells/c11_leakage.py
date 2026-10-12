# ─── Cell 11: leakage gate (validation data only; test is still locked) ───
import torch, d4_train, d4_eval

_t0 = time.time()
_max_se = CONFIG["leakage_gate"]["max_abs_ic_in_standard_errors"]

def _ic_stats(pred, y):
    ics = np.array([d4_eval.per_day_rank_ic(pred[i], y[i]) for i in range(len(pred))])
    ics = ics[np.isfinite(ics)]
    se = float(ics.std(ddof=1) / np.sqrt(len(ics)))
    return float(ics.mean()), se, len(ics), ics

GATE = {"max_abs_ic_in_standard_errors": _max_se}
_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_fold_names = [f["name"] for f in STATIC["folds"]]

# 1. Random-weights model (Mamba, lookback 30) through the full pipeline, all folds' validation data
_edges = d4_graph.load_daily_edges()
_all, _per_fold = [], {}
for _fn in _fold_names:
    d4_train.set_seed(0)
    _net = d4_train.build_model("B").to(_dev).eval()
    _sd = d4_graph.get_split_g(_fn, "val", CONFIG["arms"]["B"]["lookback"])
    _X = torch.from_numpy(_sd.X).float().to(_dev)
    _ei = [_edges[pd.Timestamp(d)][0].to(_dev) for d in _sd.dates]
    _et = [_edges[pd.Timestamp(d)][1].to(_dev) for d in _sd.dates]
    _pred = d4_train.predict(_net, _X, _ei, _et, 32)
    _m, _se, _n, _ics = _ic_stats(_pred, _sd.y)
    _per_fold[_fn] = {"mean_rank_ic": _m, "se": _se, "n_days": _n}
    _all.append(_ics)
_ics = np.concatenate(_all)
_m, _se = float(_ics.mean()), float(_ics.std(ddof=1) / np.sqrt(len(_ics)))
GATE["random_weights"] = {"mean_rank_ic": _m, "se": _se, "n_days": int(len(_ics)),
                          "abs_ic_in_se": abs(_m) / _se, "per_fold": _per_fold,
                          "passed": bool(abs(_m) < _max_se * _se)}

# 2. Shuffled-target run: arm B, fold 1, seed 0, training labels scrambled across dates
_res = d4_train.train_one("B", _fold_names[0], 0, CONFIG["training"]["batch_size_ladder"][0],
                          out_dir=WORK / "leakage_gate", shuffle_targets=True, write=False)
_sd = d4_graph.get_split_g(_fold_names[0], "val", CONFIG["arms"]["B"]["lookback"])
_m, _se, _n, _ = _ic_stats(_res["val_pred"], _sd.y)
GATE["shuffled_targets"] = {"mean_rank_ic": _m, "se": _se, "n_days": _n, "abs_ic_in_se": abs(_m) / _se,
                            "epochs_run": _res["epochs_run"], "passed": bool(abs(_m) < _max_se * _se)}

# 3. Mamba causality (unit test 1)
GATE["mamba_causality_unit_test"] = UT.get("mamba_causality") == "pass"

GATE["passed"] = bool(GATE["random_weights"]["passed"] and GATE["shuffled_targets"]["passed"]
                      and GATE["mamba_causality_unit_test"])
GATE["seconds"] = time.time() - _t0
(WORK / "phase4_leakage_gate.json").write_text(json.dumps(GATE, indent=2))
print(json.dumps(GATE, indent=2))
if not GATE["passed"]:
    raise RuntimeError("LEAKAGE GATE FAILED — see phase4_leakage_gate.json; ending the run before long training.")
