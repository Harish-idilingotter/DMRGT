"""DMRGT Phase 4 — one worker process per GPU (spec §6.6).

usage: python d4_worker.py --jobs /kaggle/working/jobs_gpu0.json --batch-size 32
Processes, not threads: no shared RNG state. Errors never stop the other jobs.
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import d4_data
import d4_train


def done_marker(job):
    return d4_train.ckpt_paths(d4_data.WORK, job["arm"], job["fold"], job["seed"])["done"]


def error_file(job, attempt):
    return d4_data.WORK / f"error_{job['arm']}_{job['fold']}_{job['seed']}_attempt{attempt}.json"


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--batch-size", type=int, required=True)
    args = ap.parse_args()
    jobs = json.loads(Path(args.jobs).read_text())
    for job in jobs:
        if done_marker(job).exists():
            continue                                  # safe if cell 13 is ever re-entered
        for attempt in (1, 2):                        # retry once on failure
            try:
                t0 = time.time()
                r = d4_train.train_one(**job, batch_size=args.batch_size)
                print(f"[worker] done {job} epochs={r['epochs_run']} "
                      f"ic={r['best_val_rank_ic']:+.4f} {time.time() - t0:.0f}s", flush=True)
                break
            except Exception:
                tb = traceback.format_exc()
                print(f"[worker] FAILED {job} attempt {attempt}\n{tb}", flush=True)
                write_json(error_file(job, attempt), {"job": job, "traceback": tb})


if __name__ == "__main__":
    main()
