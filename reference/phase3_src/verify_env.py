"""DMRGT Phase 3 — environment verification (§13, reworked for the
single-session flow).

Run FIRST in every Kaggle session (and locally after any dependency
change). Checks, in order:

  1. Python / torch / pandas versions, plus the NO-FILL pct_change
     semantics the cache depends on (pandas 2.x and 3.x both work; what
     matters is that a price hole yields NaN, not a carried-forward
     return). graph.py is IN the code dataset now — notebook Cell 4
     builds the frozen-graph OUTPUT in-session from the attached
     prices.parquet (§2 of the rework) — so this check runs everywhere.
     The dynamic cache itself is still never rebuilt on Kaggle (Rule 9).
  2. CUDA device: must be a T4-class GPU. The P100 is UNUSABLE for this
     project (compute capability sm_60 — no supported cuDNN kernels for
     this torch build; training crashes at the first conv/matmul).
  3. The stationary-bootstrap implementation stats.py's Cell 6 uses —
     a numpy implementation of the same Politis–Romano scheme, because
     arch is not in Kaggle's base image and the session has no Internet
     to install it with. Smoke-tested HERE, deterministically, so a
     broken significance step fails in minute one, not in hour ~3.
  4. Cache: all eleven parquets present (4 base + 7 flat grid variants),
     SHA-256 hashes verified against hashes.json (via
     train.verify_cache_hashes — halts on mismatch).

Exit code 1 on any failure. On Kaggle, failing fast here saves a
30-minute training slot.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd


def check_versions() -> list[str]:
    import torch
    problems = []
    print(f"[env] python  {sys.version.split()[0]}")
    print(f"[env] torch   {torch.__version__}")
    print(f"[env] pandas  {pd.__version__}")
    print(f"[env] numpy   {np.__version__}")
    # What actually matters is NO-FILL pct_change semantics (the cache
    # was built that way). pandas 2.x needs fill_method=None; pandas 3.x
    # removed the keyword because no-fill became the only behaviour.
    # graph._pct_change_no_fill handles both — verify it here, where a
    # failure is reported as an environment problem, not mid-build.
    #
    # graph.py IS in the code dataset now: notebook Cell 4 builds the
    # frozen-graph OUTPUT in-session (§2 of the single-session rework).
    # The dynamic cache is still never rebuilt on Kaggle (Rule 9) — the
    # in-session builder consumes the attached prices.parquet and writes
    # an output, it does not fetch or regenerate data.
    import graph
    px = pd.DataFrame({"a": [100.0, 101.0, np.nan, 103.0, 104.0],
                       "b": [50.0, 51.0, 52.0, 53.0, 54.0]})
    rets = graph._pct_change_no_fill(px)
    if not np.isnan(rets.loc[2, "a"]) or not np.isnan(rets.loc[3, "a"]):
        problems.append(
            "pct_change no-fill semantics broken: a price hole must "
            "produce NaN returns at and after the hole (got "
            f"{rets['a'].tolist()}). The graph rebuild would silently "
            "differ from the cache."
        )
    else:
        print("[env] pct_change no-fill semantics ✓ "
              f"(pandas {pd.__version__})")
    return problems


def check_gpu() -> list[str]:
    """Check the device's COMPUTE CAPABILITY, not just availability.

    torch.cuda.is_available() answers "is there a CUDA device", not "does
    this torch build ship kernels for it". The P100 is Pascal (sm_60) and
    Kaggle's torch build only covers sm_70+, so it reports available: True
    and then dies inside a kernel launch. Testing the capability directly
    catches any such device, whatever it is called.
    """
    import torch
    problems = []
    if not torch.cuda.is_available():
        print("[env] WARNING: CUDA not available — training will be CPU "
              "(very slow, hours per arm). On Kaggle: Settings → "
              "Accelerator → GPU T4 x2 or GPU P100 → choose T4.")
        return problems
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    n_gpus = torch.cuda.device_count()
    print(f"[env] GPU: {name}  (compute capability sm_{cap[0]}{cap[1]}, "
          f"{n_gpus} device(s))")
    if cap[0] < 7:
        problems.append(
            f"GPU is {name} (sm_{cap[0]}{cap[1]}) — UNSUPPORTED by this "
            f"torch build (needs sm_70+); training will crash inside a "
            f"kernel launch. Switch the Kaggle accelerator to GPU T4 x2 "
            f"and RESTART the session."
        )
    elif "T4" in name:
        print("[env] T4 detected ✓")
    return problems


# ─── §1.3 — did the latest code dataset actually land? ──────────────────

STAGED_MODULES: tuple[str, ...] = (
    "configs.yaml",
    "paths.py",
    "data.py",
    "splits.py",
    "model.py",
    "train.py",
    "eval.py",
    "graph.py",
    "sensitivity.py",
    "stats.py",
    "verify_env.py",
)


def code_fingerprint() -> str:
    """SHA-256 over the staged code modules, in a fixed order.

    Guards §1.3 of the known-issues doc: a dataset already attached to a
    notebook stays pinned to the version it was added at, so pushing a
    new `kaggle datasets version` changes nothing about what actually
    runs — with NO error. This fingerprint is printed by the staging
    script locally and by this checker on Kaggle; if the two differ, the
    session is running stale code. Remove and re-add the dataset from Add
    Input, restart, and compare again.
    """
    import hashlib

    import paths

    h = hashlib.sha256()
    missing = []
    for name in STAGED_MODULES:
        p = paths.CODE_DIR / name
        if not p.exists():
            missing.append(name)
            continue
        h.update(name.encode())
        h.update(p.read_bytes())
    if missing:
        return f"<incomplete: missing {missing}>"
    return h.hexdigest()


def check_bootstrap() -> list[str]:
    """Smoke-test the stationary-bootstrap implementation stats.py's
    Cell 6 uses (§6 of the single-session rework).

    It is a numpy implementation of the Politis–Romano scheme rather
    than arch's StationaryBootstrap: arch is not in Kaggle's base image
    and the session runs with Internet OFF, so an arch import in Cell 6
    would kill the run at hour ~3. Same scheme, no dependency — and
    checked HERE, deterministically, instead of trusted.
    """
    import stats as st
    problems = []
    try:
        rng = np.random.default_rng(0)
        x = rng.standard_normal(300)
        y = rng.standard_normal(300)
        d1 = st._stationary_bootstrap_diffs(x, y, 10, 25, seed=7)
        d2 = st._stationary_bootstrap_diffs(x, y, 10, 25, seed=7)
        assert d1.shape == (25,), f"replicate shape {d1.shape}"
        assert np.isfinite(d1).all()
        assert np.array_equal(d1, d2), "not deterministic under the seed"
        idx = st._stationary_bootstrap_indices(50, 10, 100, seed=1)
        assert idx.shape == (100, 50), f"index shape {idx.shape}"
        assert idx.min() >= 0 and idx.max() < 50, "index out of range"
        print("[env] stationary bootstrap smoke test ✓ "
              "(deterministic, paired draws in range)")
    except Exception as e:
        problems.append(
            f"stationary bootstrap implementation broken: "
            f"{type(e).__name__}: {e}. Fix before the session — Cell 6 "
            f"runs it in-session."
        )
    return problems


def check_cache(require_variants: bool = False) -> list[str]:
    import paths
    import train
    problems = []
    print(f"[env] cache dir: {paths.CACHE_DIR}")
    missing = [n for n in paths.CACHE_PARQUETS
               if not (paths.CACHE_DIR / n).exists()]
    if missing:
        return [f"cache parquets missing: {missing} — attach the latest "
                f"dmrgt-phase-3-cache dataset version and RESTART the session"]
    if paths.HASHES_JSON.exists():
        try:
            hashes = train.verify_cache_hashes(stop_on_mismatch=True)
            print(f"[env] {len(hashes)} cache hashes verified ✓")
        except RuntimeError as e:
            return [str(e)]
    else:
        problems.append("hashes.json missing — cannot verify integrity")
    variants = paths.graph_variant_files()
    print(f"[env] graph variants present: {len(variants)} "
          f"({[p.name for p in variants]})")
    if require_variants and len(variants) < 7:
        problems.append(
            f"grid session needs all 7 graph variants in "
            f"{paths.CACHE_DIR}, found {len(variants)}: "
            f"{[p.name for p in variants]}. Re-attach the dmrgt-phase-3-cache "
            f"version that carries the graphs_<name>.parquet variants "
            f"and RESTART. (If the dataset itself looks complete, the "
            f"upload almost certainly used --dir-mode skip on a "
            f"subdirectory: re-upload from cache/ with no --dir-mode.)"
        )
    return problems


def main(require_variants: bool = False) -> int:
    import json

    import paths

    print(json.dumps(paths.describe(), indent=2))
    print()

    fp = code_fingerprint()
    print(f"[env] code fingerprint: {fp}")
    print("[env]   ^ must equal the fingerprint printed by "
          "kaggle/build_code_dataset_phase3.sh — if not, the session is "
          "running a stale dataset version (§1.3). Remove and re-add the "
          "dataset, restart, compare again.")
    print()

    problems: list[str] = []
    problems += check_versions()
    problems += check_gpu()
    # Always — the significance step runs IN-SESSION now (Cell 6), on
    # Kaggle too, so the bootstrap implementation must hold everywhere.
    problems += check_bootstrap()
    problems += check_cache(require_variants=require_variants)

    print()
    if problems:
        print("[env] FAILED — fix before training:")
        for p in problems:
            print(f"  ✗ {p}")
        return 1
    print("[env] ALL CHECKS PASSED — safe to train.")
    return 0


def _cli() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Phase 3 environment check")
    parser.add_argument("--require-variants", action="store_true",
                        help="Fail unless all 7 grid variants are present "
                             "(Kaggle session A)")
    args = parser.parse_args()
    return main(require_variants=args.require_variants)


if __name__ == "__main__":
    sys.exit(_cli())