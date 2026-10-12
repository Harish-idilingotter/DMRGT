# ─── Cell 2: cache resolution + hash verification + universe (hard stop on failure) ───
from pathlib import Path
import hashlib, json
import pandas as pd

CANDIDATES = [
    Path("/kaggle/input/dmrgt-sp100-cache"),
    Path("/kaggle/input/datasets/harishsivakumar1912/dmrgt-sp100-cache"),
]
def resolve_cache():
    for c in CANDIDATES:
        if (c / "features.parquet").exists():
            return c
    hits = list(Path("/kaggle/input").rglob("features.parquet"))
    if len(hits) == 1:
        return hits[0].parent
    raise FileNotFoundError(f"Cache not found or ambiguous: {hits}")

CACHE = resolve_cache()
for _req in ["prices.parquet", "features.parquet", "targets.parquet", "graphs_baseline.parquet",
             "universe.csv", "hashes.json"]:
    assert (CACHE / _req).exists(), f"required cache file missing: {_req}"
expected = json.loads((CACHE / "hashes.json").read_text())
for name, h in expected.items():
    if name == "generated_at":                 # metadata key
        continue
    actual = hashlib.sha256((CACHE / name).read_bytes()).hexdigest()
    assert actual == h, f"Hash mismatch for {name}"
os.environ["DMRGT4_CACHE"] = str(CACHE)    # inherited by the worker processes
print("cache OK:", CACHE, "—", len(expected) - 1, "file hashes verified")

# Universe + GICS sectors from universe.csv (the S&P 100 build's pre-registered rules decided membership)
_U = pd.read_csv(CACHE / "universe.csv")
_inc = _U[_U["included"].astype(str) == "True"]
UNIVERSE_TICKERS = _inc["ticker"].tolist()
UNIVERSE_SECTORS = dict(zip(_inc["ticker"], _inc["gics_sector"]))
UNIVERSE_SHA256 = hashlib.sha256((CACHE / "universe.csv").read_bytes()).hexdigest()
update_snapshot(UNIVERSE={"tickers": UNIVERSE_TICKERS, "sectors": UNIVERSE_SECTORS,
                          "n_stocks": len(UNIVERSE_TICKERS), "universe_csv_sha256": UNIVERSE_SHA256})
_BM = json.loads((CACHE / "build_manifest.json").read_text()) if (CACHE / "build_manifest.json").exists() else {}
print(f"universe: N = {len(UNIVERSE_TICKERS)} stocks, {len(set(UNIVERSE_SECTORS.values()))} GICS sectors; "
      f"excluded: {_U[_U['included'].astype(str) != 'True']['ticker'].tolist()}")
print("SURVIVORSHIP BIAS: the universe is today's S&P 100 constituents traced back to 2015 (see build_manifest.json).")
