"""Run a (dev-built) notebook locally, cell by cell, in one namespace — a stand-in
for Kaggle: handles %%writefile, pretends one T4, runs on CPU. Dev tool only.

  python phase4/dev_run.py DIR/dev_phase4.ipynb [--upto N] [--from N]
"""
import argparse, json, os, sys, time, traceback
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("nb"); ap.add_argument("--upto", type=int, default=10**6); ap.add_argument("--only", type=int, nargs="*"); ap.add_argument("--skip", type=int, nargs="*", default=[])
a = ap.parse_args()

import torch
torch.cuda.device_count = lambda: 1
torch.cuda.get_device_capability = lambda i=0: (7, 5)
torch.cuda.get_device_name = lambda i=0: "DEV-FAKE-T4"

nb = json.loads(Path(a.nb).read_text())
ns = {"__name__": "__main__"}
for i, c in enumerate(nb["cells"]):
    if i > a.upto:
        break
    if a.only and i not in a.only:
        continue
    if i in a.skip:
        ns.setdefault("UT", {"mamba_causality": "pass"}); print(f"[{i}] skipped (dev)"); continue
    src = "".join(c["source"])
    title = c["metadata"].get("title", "")
    print(f"\n===== [{i}] {title} =====", flush=True)
    t0 = time.time()
    if src.startswith("%%writefile"):
        first, rest = src.split("\n", 1)
        p = Path(first.split(None, 1)[1].strip()); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(rest); print("wrote", p); continue
    try:
        exec(compile(src, f"<cell {i}>", "exec"), ns)
    except BaseException:
        traceback.print_exc(); print(f"CELL {i} FAILED after {time.time()-t0:.1f}s"); sys.exit(1)
    print(f"[cell {i} ok {time.time()-t0:.1f}s]")
