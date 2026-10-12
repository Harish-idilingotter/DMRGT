"""§5 validation checks for the S&P 100 cache. Writes validation_report.json; exits non-zero on any failure."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml

out = Path(sys.argv[1]); ref = Path("../../cache")
cfg = yaml.safe_load(open("configs_sp100.yaml"))
U = pd.read_csv(out / "universe.csv"); inc = U[U["included"]]
tick = cfg["universe"]["tickers"]; N = len(tick)
k_pos, k_neg = cfg["graph"]["k_pos"], cfg["graph"]["k_neg"]
P, F, T, G = (pd.read_parquet(out / f"{n}.parquet") for n in ("prices", "features", "targets", "graphs_baseline"))
fcols = cfg["features"]["columns"]
R, checks = {"n_stocks": N}, {}

def chk(name, passed, **detail):
    checks[name] = {"pass": bool(passed), **detail}

chk("universe_matches_included", set(tick) == set(inc["ticker"]) and set(F["ticker"]) == set(tick) and set(T["ticker"]) == set(tick),
    n_included=len(inc))
# 1 NaN / Inf
chk("no_nan_inf", F[fcols].notna().all().all() and np.isfinite(F[fcols].to_numpy()).all()
    and T["excess_return"].notna().all() and np.isfinite(T["excess_return"].to_numpy()).all())
# 2 ranges
rsi = F["f3_rsi"]; vz = F["f6_vol_zscore"]
per_t_vz = F.groupby("ticker")["f6_vol_zscore"].apply(lambda s: float(((s > -5) & (s < 5)).mean()))
const_cols = [c for c in fcols if F[c].std() == 0]
const_tk = sorted({t for t, g in F.groupby("ticker") if any(g[c].std() == 0 for c in fcols[:6])})
chk("feature_ranges", rsi.between(0, 100).all() and per_t_vz.min() >= 0.97 and not const_cols and not const_tk,
    rsi_min=float(rsi.min()), rsi_max=float(rsi.max()), vol_z_frac_in_pm5_overall=float(((vz > -5) & (vz < 5)).mean()),
    vol_z_frac_in_pm5_worst_ticker=float(per_t_vz.min()), worst_ticker=str(per_t_vz.idxmin()),
    constant_columns=const_cols, constant_tickers=const_tk)
# 3 alignment, 20 random rows
close = P.pivot(index="date", columns="ticker", values="close").sort_index()
nxt = close[tick].shift(-1) / close[tick] - 1
ex = nxt.sub(nxt.mean(axis=1), axis=0)
rng = np.random.default_rng(0)
sample = T.iloc[rng.choice(len(T), 20, replace=False)]
cal = list(close.index); bad = []
for _, r in sample.iterrows():
    d, t = r["date"], r["ticker"]
    tdate = cal[cal.index(d) + 1]                      # the date the target return is realised on
    in_feat = ((F["date"] == d) & (F["ticker"] == t)).any()
    ok = (d < tdate) and np.isclose(r["excess_return"], ex.loc[d, t], atol=1e-12)
    if in_feat:
        fr = F[(F["date"] == d) & (F["ticker"] == t)].iloc[0]
        # feature f1 at d uses closes up to d only: log(close[d]/close[d-1])
        i = cal.index(d)
        ok = ok and np.isclose(fr["f1_logret"], np.log(close.loc[d, t] / close.loc[cal[i - 1], t]), atol=1e-12)
    if not ok: bad.append([str(d.date()), t])
chk("alignment_20_rows", not bad, failures=bad)
# 4 demeaning
dm = T.groupby("date")["excess_return"].mean().abs()
chk("targets_demeaned", dm.max() < 1e-9, max_abs_mean=float(dm.max()))
# 5 graph
cnt = G.groupby(["date", "edge_type"]).size().unstack(fill_value=0)
sec = dict(zip(inc["ticker"], inc["gics_sector"]))
exp_sec = sum(c * (c - 1) for c in pd.Series(list(sec.values())).value_counts())
exp_pairs = {(tick.index(a), tick.index(b)) for a in tick for b in tick if a != b and sec[a] == sec[b]}
g0 = G[G["date"] == G["date"].iloc[0]]; got_pairs = set(zip(g0[g0.edge_type == 2]["src"], g0[g0.edge_type == 2]["dst"]))
sector_const = G[G.edge_type == 2].groupby("date").apply(lambda x: frozenset(zip(x["src"], x["dst"])), include_groups=False).nunique() == 1
chk("graph_structure", (cnt[0] == N * k_pos).all() and (cnt[1] == N * k_neg).all() and (cnt[2] == exp_sec).all()
    and got_pairs == exp_pairs and sector_const and (G["src"] != G["dst"]).all(),
    pos_per_date=int(N * k_pos), neg_per_date=int(N * k_neg), sector_edges=int(exp_sec), dates=int(G["date"].nunique()))
# 6 row counts (sum over tickers, never a product)
per_t_f = F.groupby("ticker").size(); per_t_t = T.groupby("ticker").size()
chk("row_counts", len(F) == int(per_t_f.sum()) and len(T) == int(per_t_t.sum()) and per_t_f.nunique() == 1 and per_t_t.nunique() == 1
    and len(P) == int(P.groupby("ticker").size().sum()), features_rows=len(F), targets_rows=len(T), prices_rows=len(P),
    features_per_ticker=int(per_t_f.iloc[0]), targets_per_ticker=int(per_t_t.iloc[0]))
# 7 calendars identical to the 40-stock cache
cal_ok = {}
for n, df in (("prices", P), ("features", F), ("targets", T), ("graphs_baseline", G)):
    old = pd.read_parquet(ref / f"{n}.parquet", columns=["date"])
    cal_ok[n] = bool(set(df["date"]) == set(old["date"]))
chk("calendar_identical_to_40_stock_cache", all(cal_ok.values()), per_file=cal_ok)

R["checks"] = checks; R["passed"] = all(c["pass"] for c in checks.values())
(out / "validation_report.json").write_text(json.dumps(R, indent=2))
print(json.dumps({k: v["pass"] for k, v in checks.items()}, indent=1), "\nPASSED:", R["passed"])
sys.exit(0 if R["passed"] else 1)
