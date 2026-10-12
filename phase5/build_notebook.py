"""Assemble kaggle_phase5.ipynb from phase5/modules/*.py and phase5/cells/*.py.

The modules are the source of truth; each is embedded verbatim as a
`%%writefile /kaggle/working/dmrgt4/<module>.py` cell. Cells run top to bottom.

  python phase5/build_notebook.py                       # → kaggle_phase5.ipynb (Kaggle paths)
  python phase5/build_notebook.py --local DIR [--smoke] # dev copy for CPU dry-runs (never uploaded)
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CELLS = ROOT / "phase5" / "cells"
MODULES = ROOT / "phase5" / "modules"

PLAN = [
    ("c00_config.py", "Cell 0 — pre-registered CONFIG + snapshot hash"),
    ("c01_env.py", "Cell 1 — environment + GPU checks"),
    ("c02_cache.py", "Cell 2 — cache resolution + hash verification + universe"),
    ("module:d4_data.py", "Cell 3 — d4_data.py"),
    ("module:d4_graph.py", "Cell 4 — d4_graph.py"),
    ("module:d4_models.py", "Cell 5 — d4_models.py"),
    ("module:d4_eval.py", "Cell 6 — d4_eval.py"),
    ("module:d4_stats.py", "Cell 7 — d4_stats.py"),
    ("module:d4_train.py", "Cell 8 — d4_train.py"),
    ("module:d4_worker.py", "Cell 9 — d4_worker.py"),
    ("c09b_import.py", "Cell 9b — import the modules"),
    ("c10_tests.py", "Cell 10 — unit tests"),
    ("c11_leakage.py", "Cell 11 — leakage gate"),
    ("c12_probe.py", "Cell 12 — memory + timing probe → per-stage budget plan"),
    ("c13_s1_graph_grid.py", "Cell 13 — S1 graph grid"),
    ("c14_s2_train.py", "Cell 14 — S2 train P, B, C"),
    ("c15_s3_select.py", "Cell 15 — S3 post-processing + winner selection"),
    ("c16_phase4b.py", "Cell 16 — Phase 4B calibration + selection"),
    ("c17_s4_holdout.py", "Cell 17 — S4 holdout training"),
    ("c18_scoring.py", "Cell 18 — one-shot test scoring"),
    ("c19_extras.py", "Cell 19 — efficiency, Δ diagnostics, compute benchmark"),
    ("c20_verify.py", "Cell 20 — output verification + draft results"),
]

SMOKE = '''
# ── DEV SMOKE ONLY (never in the Kaggle notebook): tiny folds, few seeds/epochs ──
CONFIG["seeds"] = [0, 1]
CONFIG["holdout"]["seeds"] = [0, 1]
CONFIG["training"]["max_epochs"] = 4
CONFIG["training"]["early_stop_patience"] = 2
CONFIG["budget"]["plan_limit_hours"] = 1000.0
CONFIG["budget"]["expected_epochs_for_projection"] = 4
CONFIG["significance"]["bootstrap_iterations"] = 200
CONFIG["phase4b"]["scale_benchmark_n_stocks"] = [40, 100]
CONFIG["phase4b"]["scale_benchmark_windows"] = [60]
CONFIG["phase4b"]["scale_benchmark_days"] = 120
for _f, _s in zip(STATIC["folds"], ["2018-10-15", "2019-10-15", "2020-10-15", "2021-10-15", "2022-10-15", "2023-10-15"]):
    _f["train_start"] = _s
'''

SMOKE_UNIVERSE = '''
# ── DEV SMOKE ONLY: first 20 stocks ──
UNIVERSE_TICKERS = UNIVERSE_TICKERS[:20]
UNIVERSE_SECTORS = {t: UNIVERSE_SECTORS[t] for t in UNIVERSE_TICKERS}
'''


def build(local: str | None, smoke: bool) -> dict:
    cells = []
    for item, title in PLAN:
        if item.startswith("module:"):
            name = item.split(":", 1)[1]
            src = f"%%writefile /kaggle/working/dmrgt4/{name}\n" + (MODULES / name).read_text()
        else:
            src = (CELLS / item).read_text()
            if smoke and item == "c00_config.py":
                src = src.replace("\nSNAPSHOT = ", SMOKE + "\nSNAPSHOT = ", 1)
            if smoke and item == "c02_cache.py":
                src = src.replace("UNIVERSE_SHA256 =", SMOKE_UNIVERSE + "UNIVERSE_SHA256 =", 1)
        if local:
            src = src.replace("/kaggle/working", f"{local}/working")
            src = src.replace("/kaggle/input/dmrgt-sp100-cache", str(ROOT / "data_build" / "sp100" / "dmrgt-sp100-cache"))
        cells.append({"cell_type": "code", "metadata": {"title": title}, "execution_count": None,
                      "outputs": [], "source": src.splitlines(keepends=True)})
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
            "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--local")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    nb = build(a.local, a.smoke)
    out = Path(a.local) / "dev_phase5.ipynb" if a.local else ROOT / "kaggle_phase5.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1))
    print("wrote", out, f"({len(nb['cells'])} cells)")
