# ─── Cell 9b: import the modules written above ───
import sys, importlib
sys.path.insert(0, "/kaggle/working/dmrgt4")
os.environ["DMRGT4_WORK"] = str(WORK)
import d4_data, d4_models, d4_graph, d4_eval, d4_stats, d4_train
for _m in (d4_data, d4_models, d4_graph, d4_eval, d4_stats, d4_train):
    importlib.reload(_m)
import numpy as np, pandas as pd
cfg = d4_data.load_cfg()
UNIVERSE = list(cfg["universe"]["tickers"])
print("modules imported; test split locked:", not d4_data.test_unlocked())
