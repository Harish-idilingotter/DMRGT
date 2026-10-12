# ─── Cell 13: launch all training jobs across GPUs, wait, completeness check ───
import subprocess, sys, glob

_fold_names = [f["name"] for f in STATIC["folds"]]
ARMS_RUN = PLAN["arms_run"]
_jobs_all = _jobs(ARMS_RUN)                                  # 60 (without D) or 80 (with D) jobs
_assign, _loads = lpt_assign(_jobs_all, n_gpus)
(WORK / "logs").mkdir(exist_ok=True)
for _k in range(n_gpus):
    (WORK / f"jobs_gpu{_k}.json").write_text(json.dumps(_assign[_k], indent=2))
print(f"{len(_jobs_all)} jobs on {n_gpus} GPU(s); projected makespan "
      f"{max(_loads) / 3600:.2f} h; batch size {BATCH_SIZE}")

_procs = []
for _k in range(n_gpus):
    _env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(_k), CUBLAS_WORKSPACE_CONFIG=":4096:8",
                DMRGT4_WORK=str(WORK), DMRGT4_CACHE=str(CACHE))
    _log = open(WORK / "logs" / f"gpu{_k}.log", "w")
    _procs.append(subprocess.Popen(
        [sys.executable, "/kaggle/working/dmrgt4/d4_worker.py", "--jobs", str(WORK / f"jobs_gpu{_k}.json"),
         "--batch-size", str(BATCH_SIZE)], env=_env, stdout=_log, stderr=subprocess.STDOUT))

def _n_done():
    return len(glob.glob(str(WORK / "done_*.json")))

_t_launch = time.time()
while any(p.poll() is None for p in _procs):
    time.sleep(60)
    _d, _el = _n_done(), time.time() - _t_launch
    _eta = (_el / _d * (len(_jobs_all) - _d)) if _d else float("nan")
    print(f"[{_el / 60:6.1f} min] {_d}/{len(_jobs_all)} jobs done; projected remaining {_eta / 60:.0f} min", flush=True)
TRAIN_SECONDS = time.time() - _t_launch

# completeness: A, B, C need all 20 markers; D is eligible only if complete
def _arm_done(arm):
    return [(f, s) for f in _fold_names for s in CONFIG["seeds"]
            if d4_train.ckpt_paths(WORK, arm, f, s)["done"].exists()]

_n_expected = len(_fold_names) * len(CONFIG["seeds"])
COMPLETE = {a: len(_arm_done(a)) == _n_expected for a in ARMS_RUN}
if not all(COMPLETE[a] for a in ("A", "B", "C")):
    for _p in sorted(glob.glob(str(WORK / "error_*.json"))):
        print(_p, "\n", json.loads(Path(_p).read_text())["traceback"])
    raise RuntimeError(f"Required arms incomplete: { {a: len(_arm_done(a)) for a in ARMS_RUN} } of {_n_expected}")
if "D" in ARMS_RUN and not COMPLETE["D"]:
    print(f"Arm D incomplete ({len(_arm_done('D'))}/{_n_expected}) — excluded from selection, reported as such.")
print("completeness:", COMPLETE, f"| training wall time {TRAIN_SECONDS / 3600:.2f} h")
