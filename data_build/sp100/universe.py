"""§2 universe construction for the S&P 100 cache (rules pre-registered in the spec).

1. fetch the current S&P 100 table (Wikipedia) → universe_raw_snapshot.csv
2. one share class per company; dot tickers → Yahoo format
3. download each ticker with the SAME yfinance call as the ported pipeline
4. screen on the 40-stock cache's trading calendar:
     - data must start at the first calendar day (no later listings),
     - no gap of more than 2 consecutive calendar-trading days,
     - at most 1% missing days overall;
   gaps of <= 2 days are forward-filled
5. write universe.csv, configs_sp100.yaml and the screened per-ticker frames that
   build.py feeds to the verbatim download_prices() through a yfinance shim.

Nothing in the rules is tuned after looking at results (spec G5).
"""
import io, json, pickle, re, sys, time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd, requests, yaml
import yfinance as yf

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "out_sp100"); OUT.mkdir(parents=True, exist_ok=True)
RAW, YF = OUT / "raw_yf", OUT / "yf_cache"; RAW.mkdir(exist_ok=True); YF.mkdir(exist_ok=True)
BASE_CFG = yaml.safe_load(open("configs_40.yaml"))
START, END = BASE_CFG["universe"]["start_date"], BASE_CFG["universe"]["end_date"]
REF_CAL = pd.DatetimeIndex(sorted(pd.read_parquet("../../cache/prices.parquet")["date"].unique()))
GLOBALS = list(BASE_CFG["universe"]["global_series"])
MAX_GAP, MAX_MISSING_FRAC = 2, 0.01
URL = "https://en.wikipedia.org/wiki/S%26P_100"

# 1. snapshot ------------------------------------------------------------
r = requests.get(URL, headers={"User-Agent": "Mozilla/5.0 (DMRGT research build)"}, timeout=30)
r.raise_for_status()
tab = [t for t in pd.read_html(io.StringIO(r.text)) if {"Symbol", "Name", "Sector"} <= set(t.columns) and len(t) > 90][0]
snap = tab.copy(); snap["fetched_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"); snap["source_url"] = URL
snap.to_csv(OUT / "universe_raw_snapshot.csv", index=False)

# 2. normalise + one share class per company ----------------------------
tab["yahoo"] = tab["Symbol"].str.strip().str.replace(".", "-", regex=False)
tab["company"] = tab["Name"].str.replace(r"\s*\(Class [A-Z]\)", "", regex=True).str.strip()
tab["is_class_a"] = tab["Name"].str.contains(r"\(Class A\)")
rows = []
for comp, g in tab.groupby("company", sort=False):
    keep = g[g["is_class_a"]].index[0] if g["is_class_a"].any() else g.index[0]
    for i, rw in g.iterrows():
        rows.append({"ticker": rw["yahoo"], "company": rw["Name"], "gics_sector": rw["Sector"], "_order": i,
                     "included": i == keep, "exclusion_reason": "" if i == keep else "second share class of " + comp})
U = pd.DataFrame(rows).sort_values("_order").reset_index(drop=True)

# 3. download ------------------------------------------------------------
def fetch(t):
    p = RAW / f"{t}.pkl"
    if p.exists():
        return pickle.load(open(p, "rb"))
    for attempt in range(4):
        try:
            df = yf.download(t, start=START, end=END, auto_adjust=True, progress=False, actions=False)
            break
        except Exception as e:                    # rate limit etc.
            print(f"  retry {t}: {e}"); time.sleep(5 * (attempt + 1)); df = pd.DataFrame()
    pickle.dump(df, open(p, "wb"))
    return df

def flat(df):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

def runs(missing_mask):
    """lengths of consecutive True runs, plus whether the first/last run touches the ends"""
    out, n, start = [], len(missing_mask), None
    for i, m in enumerate(missing_mask):
        if m and start is None: start = i
        if (not m) and start is not None: out.append((start, i - start)); start = None
    if start is not None: out.append((start, n - start))
    return out

# 4. screen --------------------------------------------------------------
def screen(t):
    raw = fetch(t)
    if raw is None or len(raw) == 0:
        return None, {"first_date": "", "last_date": "", "missing_days": len(REF_CAL)}, "no data returned by yfinance"
    df = flat(raw)
    ok = df.dropna(subset=["open", "high", "low", "close"])
    first, last = ok.index.min(), ok.index.max()
    aligned = ok.reindex(REF_CAL)
    miss = aligned["close"].isna().to_numpy()
    info = {"first_date": str(first.date()), "last_date": str(last.date()), "missing_days": int(miss.sum())}
    rr = runs(miss)
    if miss[0]:
        return None, info, f"no data at start of period (first data {first.date()}; listed after start)"
    if any(l > MAX_GAP for _, l in rr):
        return None, info, f"gap longer than {MAX_GAP} trading days ({max(l for _, l in rr)})"
    if miss.mean() > MAX_MISSING_FRAC:
        return None, info, f"{miss.mean():.2%} missing trading days (> {MAX_MISSING_FRAC:.0%})"
    filled = aligned.ffill()
    assert filled[["open", "high", "low", "close"]].notna().all().all()
    filled["volume"] = filled["volume"].fillna(0)
    return filled, info, ""

screened = {}
for g in GLOBALS:
    f, info, why = screen(g)
    assert f is not None, f"global series {g} fails the rules: {why}"
    screened[g] = f
infos = {}
for i, rw in U[U["included"]].iterrows():
    t = rw["ticker"]
    f, info, why = screen(t)
    infos[t] = info
    if f is None:
        U.loc[i, ["included", "exclusion_reason"]] = [False, why]
    else:
        screened[t] = f
    print(f"{t:6s} {'ok ' if f is not None else 'EXCL'} {why}", flush=True)
for c in ("first_date", "last_date", "missing_days"):
    U[c] = [infos.get(t, {}).get(c, "") for t in U["ticker"]]

# 5. outputs -------------------------------------------------------------
def to_yf_shape(df):
    d = df[["open", "high", "low", "close", "volume"]].rename(columns=str.capitalize)
    d.index.name = "Date"
    return d

for t, f in screened.items():
    pickle.dump(to_yf_shape(f), open(YF / f"{t}.pkl", "wb"))

inc = U[U["included"]]
U.drop(columns="_order")[["ticker", "company", "gics_sector", "included", "exclusion_reason", "first_date", "last_date", "missing_days"]] \
    .to_csv(OUT / "universe.csv", index=False)
cfg = json.loads(json.dumps(BASE_CFG))
cfg["universe"]["tickers"] = inc["ticker"].tolist()
cfg["sectors"] = {**dict(zip(inc["ticker"], inc["gics_sector"])), **{g: "global" for g in GLOBALS}}
yaml.safe_dump(cfg, open(OUT.parent / "configs_sp100.yaml", "w"), sort_keys=False)
print(f"\nincluded {len(inc)} / {len(U)} candidates; sectors: {inc['gics_sector'].value_counts().to_dict()}")
print(U[~U['included']][["ticker", "exclusion_reason"]].to_string(index=False))
