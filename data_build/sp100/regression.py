"""§3 regression test: compare the rebuilt 40-stock cache with dmrgt-phase-3-cache."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

new_dir, old_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
R = {"tolerances": {"feature_corr_min": 0.999, "feature_median_abs_diff_max_std_mult": 1e-6,
                    "target_corr_min": 0.9999, "target_frac_within_1e-6_min": 0.999,
                    "graph_identical_dates_min": 0.99}}
ok = True

def load(d, n): return pd.read_parquet(d / n)

# dates and tickers
dt = {}
for n in ("prices", "features", "targets"):
    a, b = load(new_dir, f"{n}.parquet"), load(old_dir, f"{n}.parquet")
    same = (set(a["date"]) == set(b["date"])) and (set(a["ticker"]) == set(b["ticker"]))
    dt[n] = {"dates_identical": set(a["date"]) == set(b["date"]), "tickers_identical": set(a["ticker"]) == set(b["ticker"]),
             "rows_new": len(a), "rows_old": len(b)}
    ok &= same
R["dates_and_tickers"] = dt

# features
fa, fb = load(new_dir, "features.parquet"), load(old_dir, "features.parquet")
m = fa.merge(fb, on=["date", "ticker"], suffixes=("_n", "_o"))
assert len(m) == len(fb) == len(fa), "feature rows do not line up"
feat = {}
for c in [c for c in fb.columns if c.startswith("f")]:
    x, y = m[c + "_n"].to_numpy(), m[c + "_o"].to_numpy()
    corr = float(np.corrcoef(x, y)[0, 1]); mad = float(np.median(np.abs(x - y))); sd = float(np.std(y))
    p = corr >= 0.999 and mad <= 1e-6 * sd
    feat[c] = {"pearson": corr, "median_abs_diff": mad, "col_std": sd, "max_abs_diff": float(np.abs(x - y).max()), "pass": bool(p)}
    ok &= p
R["features"] = feat

# targets
ta, tb = load(new_dir, "targets.parquet"), load(old_dir, "targets.parquet")
m = ta.merge(tb, on=["date", "ticker"], suffixes=("_n", "_o"))
assert len(m) == len(tb) == len(ta), "target rows do not line up"
x, y = m["excess_return_n"].to_numpy(), m["excess_return_o"].to_numpy()
corr, frac = float(np.corrcoef(x, y)[0, 1]), float(np.mean(np.abs(x - y) <= 1e-6))
R["targets"] = {"pearson": corr, "frac_within_1e-6": frac, "max_abs_diff": float(np.abs(x - y).max()),
                "pass": bool(corr >= 0.9999 and frac >= 0.999)}
ok &= R["targets"]["pass"]

# graphs: identical edge sets per date
ga, gb = load(new_dir, "graphs_baseline.parquet"), load(old_dir, "graphs_baseline.parquet")
def sets(g): return {d: frozenset(zip(x.edge_type, x.src, x.dst)) for d, x in g.groupby("date")}
sa, sb = sets(ga), sets(gb)
common = sorted(set(sa) & set(sb))
same = sum(sa[d] == sb[d] for d in common)
R["graphs"] = {"dates_new": len(sa), "dates_old": len(sb), "dates_common": len(common),
               "identical_edge_set_dates": same, "fraction": same / max(1, len(common)),
               "pass": bool(len(sa) == len(sb) and same / max(1, len(common)) >= 0.99)}
ok &= R["graphs"]["pass"]

R["passed"] = bool(ok)
out.write_text(json.dumps(R, indent=2))
print(json.dumps(R, indent=2))
