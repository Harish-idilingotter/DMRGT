# ─── Cell 2: cache resolution + hash verification (hard stop on failure) ───
from pathlib import Path
import hashlib, json

CANDIDATES = [
    Path("/kaggle/input/dmrgt-phase-4-cache"),
    Path("/kaggle/input/datasets/harishsivakumar1912/dmrgt-phase-4-cache"),
    Path("/kaggle/input/dmrgt-phase-3-cache"),
    Path("/kaggle/input/datasets/harishsivakumar1912/dmrgt-phase-3-cache"),
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
for _req in ["prices.parquet", "features.parquet", "targets.parquet", "graphs_baseline.parquet", "hashes.json"]:
    assert (CACHE / _req).exists(), f"required cache file missing: {_req}"
expected = json.loads((CACHE / "hashes.json").read_text())
for name, h in expected.items():
    if not name.endswith(".parquet"):      # skip metadata keys such as generated_at
        continue
    actual = hashlib.sha256((CACHE / name).read_bytes()).hexdigest()
    assert actual == h, f"Hash mismatch for {name}"
os.environ["DMRGT4_CACHE"] = str(CACHE)    # inherited by the worker processes
print("cache OK:", CACHE, "—", sum(k.endswith('.parquet') for k in expected), "parquet hashes verified")
