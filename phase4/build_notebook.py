"""Assemble kaggle_phase4.ipynb from phase4/modules/*.py and phase4/cells/*.py.

The modules are the source of truth; each is embedded verbatim as a
`%%writefile /kaggle/working/dmrgt4/<module>.py` cell. Cells run top to bottom.

  python phase4/build_notebook.py                       # → kaggle_phase4.ipynb (Kaggle paths)
  python phase4/build_notebook.py --local DIR [--smoke] # dev copy for CPU dry-runs (never uploaded)
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CELLS = ROOT / "phase4" / "cells"
MODULES = ROOT / "phase4" / "modules"

PLAN = [
    ("c00_config.py", "Cell 0 — pre-registered CONFIG + snapshot hash"),
    ("c01_env.py", "Cell 1 — environment + GPU checks"),
    ("c02_cache.py", "Cell 2 — cache resolution + hash verification"),
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
    ("c12_probe.py", "Cell 12 — memory + timing probe → budget plan"),
    ("c13_train.py", "Cell 13 — launch all training jobs"),
    ("c14_select4a.py", "Cell 14 — Phase 4A selection (validation only)"),
    ("c15_phase4b.py", "Cell 15 — Phase 4B calibration + validation selection"),
    ("c16_test.py", "Cell 16 — one-shot test scoring"),
    ("c17_efficiency.py", "Cell 17 — efficiency + technical-effect measurements"),
    ("c18_verify.py", "Cell 18 — output verification + draft results"),
]

SMOKE = '''
# ── DEV SMOKE ONLY (never in the Kaggle notebook): tiny folds, few seeds/epochs ──
CONFIG["seeds"] = [0, 1]
CONFIG["training"]["max_epochs"] = 2
CONFIG["training"]["early_stop_patience"] = 2
CONFIG["budget"]["plan_limit_hours"] = 1000.0
CONFIG["budget"]["expected_epochs_for_projection"] = 2
CONFIG["significance"]["bootstrap_iterations"] = 200
for _f, _s in zip(STATIC["folds"], ["2018-10-01", "2019-10-01", "2020-10-01", "2021-10-01"]):
    _f["train_start"] = _s
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
        if local:
            src = src.replace("/kaggle/working", f"{local}/working")
            src = src.replace("/kaggle/input/dmrgt-phase-3-cache", str(ROOT / "cache"))
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
    out = Path(a.local) / "dev_phase4.ipynb" if a.local else ROOT / "kaggle_phase4.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1))
    print("wrote", out, f"({len(nb['cells'])} cells)")
