# ─── Cell 12: memory + timing probe → budget plan ───
import torch, d4_train

_B = CONFIG["budget"]
_ladder = CONFIG["training"]["batch_size_ladder"]
_fold_names = [f["name"] for f in STATIC["folds"]]
_big = _fold_names[-1]                                   # fold 4: the largest
PLAN = {"n_gpus": n_gpus, "probe_fold": _big}

# 1. Memory: heaviest arm (C) on the largest fold; same batch size for ALL arms
_mem_log, BATCH_SIZE = [], None
for _bs in _ladder:
    try:
        _peak = d4_train.probe_memory("C", _big, _bs)
        _mem_log.append({"batch_size": _bs, "peak_gb": _peak, "status": "ok"})
        if _peak > 12.0:
            _mem_log[-1]["status"] = "peak > 12 GB"
            continue
        BATCH_SIZE = _bs
        break
    except torch.cuda.OutOfMemoryError:
        _mem_log.append({"batch_size": _bs, "peak_gb": None, "status": "OOM"})
        torch.cuda.empty_cache()
if BATCH_SIZE is None:
    raise RuntimeError(f"No batch size in {_ladder} fits arm C on {_big}: {_mem_log}")
PLAN["memory_probe"], PLAN["batch_size"] = _mem_log, BATCH_SIZE
print("memory probe:", _mem_log, "→ batch size", BATCH_SIZE)

# 2. Timing: arms A, B, C on fold 4 / seed 0, 2 epochs, time epoch 2 (probe runs are discarded)
SEC_PER_EPOCH = {}
for _arm in ("A", "B", "C"):
    _r = d4_train.train_one(_arm, _big, 0, BATCH_SIZE, max_epochs=2, out_dir=WORK / "probe", write=False)
    SEC_PER_EPOCH[_arm] = _r["last_epoch_seconds"]
    print(f"arm {_arm}: {SEC_PER_EPOCH[_arm]:.1f}s per epoch (epoch 2)")
PLAN["sec_per_epoch"] = SEC_PER_EPOCH

# 3. Projection
def _n_train(arm, fold):
    return len(d4_graph.get_split_g(fold, "train", CONFIG["arms"][arm]["lookback"]).dates)

PER_RUN = {}
for _arm in ("A", "B", "C"):
    for _fn in _fold_names:
        PER_RUN[(_arm, _fn)] = (SEC_PER_EPOCH[_arm] * (_n_train(_arm, _fn) / _n_train(_arm, _big))
                                * _B["expected_epochs_for_projection"] * _B["safety_factor"])
for _fn in _fold_names:
    PER_RUN[("D", _fn)] = PER_RUN[("C", _fn)] * 1.05

def lpt_assign(jobs, n_gpus):
    """Longest-job-first onto the GPU with the least assigned time. jobs: [(dur, job)]."""
    loads, assign = [0.0] * n_gpus, [[] for _ in range(n_gpus)]
    for dur, job in sorted(jobs, key=lambda x: -x[0]):
        k = int(np.argmin(loads))
        loads[k] += dur
        assign[k].append(job)
    return assign, loads

def _jobs(arms):
    return [(PER_RUN[(a, f)], {"arm": a, "fold": f, "seed": s})
            for a in arms for f in _fold_names for s in CONFIG["seeds"]]

_elapsed_h = (time.time() - RUN_T0) / 3600
_tail_h = _B["phase4b_estimate_minutes"] / 60 + _B["scoring_and_bench_estimate_minutes"] / 60
PROJ = {}
for _name, _arms in (("with_D", ["A", "B", "C", "D"]), ("without_D", ["A", "B", "C"])):
    _, _loads = lpt_assign(_jobs(_arms), n_gpus)
    PROJ[_name] = {"makespan_h": max(_loads) / 3600, "n_jobs": len(_jobs(_arms)),
                   "projected_total_h": _elapsed_h + max(_loads) / 3600 + _tail_h}
PLAN["projection"] = PROJ
PLAN["per_run_seconds"] = {f"{a}|{f}": v for (a, f), v in PER_RUN.items()}

# 4. Decision (pre-registered, automatic; seeds / folds / epochs are never reduced)
_limit = _B["plan_limit_hours"]
RUN_D = False
if PROJ["with_D"]["projected_total_h"] <= _limit:
    RUN_D = True
elif PROJ["without_D"]["projected_total_h"] <= _limit:
    PLAN["D_skipped_reason"] = "budget"
else:
    (WORK / "budget_plan.json").write_text(json.dumps(PLAN, indent=2))
    print(json.dumps(PROJ, indent=2))
    raise RuntimeError(f"Arms A+B+C do not fit the {_limit} h plan limit: {PROJ['without_D']}")
PLAN["run_D"] = RUN_D
PLAN["arms_run"] = ["A", "B", "C"] + (["D"] if RUN_D else [])
(WORK / "budget_plan.json").write_text(json.dumps(PLAN, indent=2))
print(json.dumps({k: PLAN[k] for k in ("batch_size", "projection", "run_D", "arms_run")}, indent=2))
