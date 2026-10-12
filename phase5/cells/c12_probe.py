# ─── Cell 12: memory + timing probe → per-stage budget plan (+ the shared job launcher) ───
import torch, subprocess, glob, d4_train

_B = CONFIG["budget"]
_ladder = CONFIG["training"]["batch_size_ladder"]
_big = FOLD_NAMES[-1]                                    # fold 4: the largest selection fold
PLAN = {"n_gpus": n_gpus, "probe_fold": _big, "n_stocks": len(UNIVERSE)}

# 1. Memory: heaviest arm (C) on the largest selection fold; same batch size for ALL arms and the grid
_mem_log, BATCH_SIZE = [], None
for _bs in _ladder:
    try:
        _peak = d4_train.probe_memory("C", _big, _bs, "baseline")
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

# 2. Timing: every arm on fold 4 / seed 0, 2 epochs, time epoch 2 (probe runs are discarded)
SEC_PER_EPOCH = {}
for _arm in ("P", "A", "B", "C"):
    _r = d4_train.train_one(_arm, _big, 0, BATCH_SIZE, graph="baseline", max_epochs=2,
                            out_dir=WORK / "probe", write=False)
    SEC_PER_EPOCH[_arm] = _r["last_epoch_seconds"]
    print(f"arm {_arm}: {SEC_PER_EPOCH[_arm]:.1f}s per epoch (epoch 2)")
PLAN["sec_per_epoch"] = SEC_PER_EPOCH

# 3. Projection, stage by stage (stages are sequential: S2 needs the graph winner, S4 needs all locks)
ALL_FOLDS = FOLD_NAMES + HOLDOUT_NAMES
def _n_train(arm, fold):
    return len(d4_graph.get_split_g(fold, "train", CONFIG["arms"][arm]["lookback"], "baseline").dates)

PER_RUN = {}
for _arm in ("P", "A", "B", "C"):
    _base = _n_train(_arm, _big)
    for _fn in ALL_FOLDS:
        PER_RUN[(_arm, _fn)] = (SEC_PER_EPOCH[_arm] * (_n_train(_arm, _fn) / _base)
                                * _B["expected_epochs_for_projection"] * _B["safety_factor"])

def lpt_assign(jobs, n_gpus):
    """Longest-job-first onto the GPU with the least assigned time. jobs: [(dur, job)]."""
    loads, assign = [0.0] * n_gpus, [[] for _ in range(n_gpus)]
    for dur, job in sorted(jobs, key=lambda x: -x[0]):
        k = int(np.argmin(loads))
        loads[k] += dur
        assign[k].append(job)
    return assign, loads

def grid_tag(variant):
    return f"A@{variant}"

def s1_jobs(grid_seeds):
    out = []
    for v in CONFIG["graph_grid"]["variants"]:
        for fn in FOLD_NAMES:
            for s in grid_seeds[v]:
                out.append((PER_RUN[("A", fn)], {"arm": "A", "fold": fn, "seed": s, "graph": v, "tag": grid_tag(v)}))
    return out

def arm_jobs(arms, folds, seeds, graph):
    return [(PER_RUN[(a, fn)], {"arm": a, "fold": fn, "seed": s, "graph": graph, "tag": a})
            for a in arms for fn in folds for s in seeds]

def project(grid_seeds, s2_arms, holdout_seeds):
    worst_winner = "C" if "C" in s2_arms else "B"
    stages = {
        "S1": s1_jobs(grid_seeds),
        "S2": arm_jobs(s2_arms, FOLD_NAMES, CONFIG["seeds"], "baseline"),
        "S4": arm_jobs([worst_winner, "A", "P"], HOLDOUT_NAMES, holdout_seeds, "baseline"),
    }
    spans = {k: max(lpt_assign(v, n_gpus)[1]) / 3600 for k, v in stages.items()}
    _el = (time.time() - RUN_T0) / 3600
    tail = _B["phase4b_estimate_minutes"] / 60 + _B["scoring_and_bench_estimate_minutes"] / 60
    return {"stage_makespan_h": spans, "n_jobs": {k: len(v) for k, v in stages.items()},
            "elapsed_h": _el, "serial_tail_h": tail, "projected_total_h": _el + sum(spans.values()) + tail}

# 4. Decision (pre-registered budget_drop_order; applied in sequence, nothing else is ever shrunk)
_limit = _B["plan_limit_hours"]
_seeds = CONFIG["seeds"]
GRID_SEEDS = {v: list(_seeds) for v in CONFIG["graph_grid"]["variants"]}
S2_ARMS, HOLDOUT_SEEDS = ["P", "B", "C"], list(CONFIG["holdout"]["seeds"])
PLAN["drops_applied"], PLAN["projection_steps"] = [], []
_steps = [("none", None),
          ("graph grid: non-baseline variants use seeds [0,1,2] only", "grid"),
          ("drop arm C (then winner = B)", "dropC"),
          ("holdout seeds reduced to [0,1,2]", "holdout")]
for _name, _act in _steps:
    if _act == "grid":
        GRID_SEEDS = {v: (list(_seeds) if v == "baseline" else [0, 1, 2]) for v in CONFIG["graph_grid"]["variants"]}
    elif _act == "dropC":
        S2_ARMS = ["P", "B"]
    elif _act == "holdout":
        HOLDOUT_SEEDS = [0, 1, 2]
    if _act:
        PLAN["drops_applied"].append(_name)
    PROJ = project(GRID_SEEDS, S2_ARMS, HOLDOUT_SEEDS)
    PLAN["projection_steps"].append({"after": _name, **PROJ})
    print(f"projection after '{_name}': {PROJ['projected_total_h']:.2f} h  {PROJ['stage_makespan_h']}")
    if PROJ["projected_total_h"] <= _limit:
        break
else:
    (WORK / "budget_plan.json").write_text(json.dumps(PLAN, indent=2, default=float))
    raise RuntimeError(f"Even after every pre-registered drop the projection ({PROJ['projected_total_h']:.2f} h) "
                       f"exceeds the {_limit} h plan limit — failing fast. {PROJ}")
SELECTION_CANDIDATES = [a for a in CONFIG["selection_4A"]["candidates"] if a in S2_ARMS]
PLAN.update(grid_seeds=GRID_SEEDS, s2_arms=S2_ARMS, holdout_seeds=HOLDOUT_SEEDS, selection_candidates=SELECTION_CANDIDATES,
            per_run_seconds={f"{a}|{f}": v for (a, f), v in PER_RUN.items()}, final_projection=PROJ)
(WORK / "budget_plan.json").write_text(json.dumps(PLAN, indent=2, default=float))
print(json.dumps({k: PLAN[k] for k in ("batch_size", "drops_applied", "s2_arms", "holdout_seeds", "final_projection")},
                 indent=2, default=float))

# ── shared launcher: one process per GPU, skip-if-done markers, retry once, completeness check ──
def _tag(job):
    return job.get("tag") or job["arm"]

def launch_stage(name, jobs_with_dur, required=True):
    jobs = [j for _, j in jobs_with_dur]
    pending = [(d, j) for d, j in jobs_with_dur
               if not d4_train.ckpt_paths(WORK, _tag(j), j["fold"], j["seed"])["done"].exists()]
    assign, loads = lpt_assign(pending, n_gpus)
    (WORK / "logs").mkdir(exist_ok=True)
    print(f"[{name}] {len(jobs)} jobs ({len(pending)} pending) on {n_gpus} GPU(s); projected makespan "
          f"{max(loads) / 3600 if pending else 0:.2f} h; batch size {BATCH_SIZE}", flush=True)
    procs = []
    for k in range(n_gpus):
        if not assign[k]:
            continue
        (WORK / f"jobs_{name}_gpu{k}.json").write_text(json.dumps(assign[k], indent=2))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(k), CUBLAS_WORKSPACE_CONFIG=":4096:8",
                   DMRGT4_WORK=str(WORK), DMRGT4_CACHE=str(CACHE))
        log = open(WORK / "logs" / f"{name}_gpu{k}.log", "w")
        procs.append(subprocess.Popen(
            [sys.executable, "/kaggle/working/dmrgt4/d4_worker.py", "--jobs", str(WORK / f"jobs_{name}_gpu{k}.json"),
             "--batch-size", str(BATCH_SIZE)], env=env, stdout=log, stderr=subprocess.STDOUT))
    t_launch = time.time()
    def n_done():
        return sum(d4_train.ckpt_paths(WORK, _tag(j), j["fold"], j["seed"])["done"].exists() for j in jobs)
    while any(p.poll() is None for p in procs):
        time.sleep(60)
        d, el = n_done(), time.time() - t_launch
        todo = len(jobs) - d
        base = len(jobs) - len(pending)
        eta = (el / max(1, d - base) * todo) if d > base else float("nan")
        print(f"[{name} {el / 60:6.1f} min] {d}/{len(jobs)} jobs done; projected remaining {eta / 60:.0f} min", flush=True)
    missing = [j for j in jobs if not d4_train.ckpt_paths(WORK, _tag(j), j["fold"], j["seed"])["done"].exists()]
    if missing and required:
        for p_ in sorted(glob.glob(str(WORK / "error_*.json"))):
            print(p_, "\n", json.loads(Path(p_).read_text())["traceback"])
        raise RuntimeError(f"[{name}] {len(missing)} of {len(jobs)} jobs did not complete: {missing[:5]} ...")
    print(f"[{name}] complete: {len(jobs) - len(missing)}/{len(jobs)} ({(time.time() - t_launch) / 3600:.2f} h)", flush=True)
    return time.time() - t_launch
