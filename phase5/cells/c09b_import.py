# ─── Cell 9b: import the modules written above ───
import sys, importlib
sys.path.insert(0, "/kaggle/working/dmrgt4")
os.environ["DMRGT4_WORK"] = str(WORK)
import d4_data, d4_models, d4_graph, d4_eval, d4_stats, d4_train
for _m in (d4_data, d4_models, d4_graph, d4_eval, d4_stats, d4_train):
    importlib.reload(_m)
import numpy as np, pandas as pd
cfg = d4_data.load_cfg("baseline")
UNIVERSE = list(cfg["universe"]["tickers"])
FOLD_NAMES = CONFIG["folds_selection"]                                   # folds 1-4
HOLDOUT_NAMES = [f["name"] for f in CONFIG["folds_holdout"]]              # folds 5-6
print(f"modules imported; N = {len(UNIVERSE)}; test split locked: {not d4_data.test_unlocked()}")
