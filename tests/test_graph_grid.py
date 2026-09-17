"""DMRGT Phase 3 — sensitivity grid tests (§6b).

Checks on the 7 built variants in cache/ (as graphs_<name>.parquet —
they sit in the cache root so `kaggle datasets version -p cache` uploads
them; a subdirectory would be silently skipped):
  * per-variant per-date counts follow the variant's own
    (k_pos, k_neg): 40·k_pos + 40·k_neg + 160 edges per date;
  * the BASELINE variant is identical to cache/graphs.parquet — the
    grid measures k sensitivity, not rebuild drift;
  * each variant differs from the baseline where its dimension changed
    (a sweep that produced an identical graph would silently shrink
    the grid);
  * NaN-corruption: a NaN inside any variant's correlation window makes
    the builder raise (Rule 4), for all 7 parameter sets;
  * hashes: every variant parquet is recorded in hashes.json so Kaggle
    can verify integrity before training.

The heavy "build the variants" step is the module fixture, so the
whole build happens once for the test run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data
import graph
import paths
import sensitivity


@pytest.fixture(scope="session")
def cfg() -> dict:
    return data.load_config(str(paths.CONFIG))


# Provenance pin (§6b) — CONTENT level, not bytes. Phase 2's original
# graphs.parquet (SHA-256 555560de…, recorded in docs/PHASE2_RESULTS.md
# §10 and still held in Phase 2's `dmrgt-cache` Kaggle dataset) was downloaded on
# 2026-09-09 and verified .equals() → True against this repo's pandas-3
# rebuild, which now serves as cache/graphs.parquet. The FILE bytes differ
# only in parquet writer metadata (pandas 2.3.3 wrote the original,
# 3.0.5 the rebuild), so the pin is over canonicalised content:
# shape, column order, and int64-cast values, hashed in order.
PHASE2_GRAPHS_CONTENT_SHA256 = (
    "1f4ab875b77f2801b04f5b9f2c2807f77fde0c648a93a02d56878f0410f50837")


def _content_fingerprint(df: pd.DataFrame) -> str:
    """Deterministic content hash of a graph frame, independent of the
    parquet writer version: dates as int64 ns, ints as int64, string
    columns as their repr'd list."""
    import hashlib
    h = hashlib.sha256()
    h.update(str(df.shape).encode())
    for col in df.columns:
        h.update(col.encode())
        a = df[col].to_numpy()
        if np.issubdtype(a.dtype, np.datetime64):
            a = a.astype("datetime64[ns]").astype("int64")
            h.update(a.tobytes())
        elif a.dtype.kind in "iu":
            h.update(a.astype("int64").tobytes())
        else:
            h.update(repr(a.astype(str).tolist()).encode())
    return h.hexdigest()


def test_cache_dynamic_graph_is_phase2_content():
    """cache/graphs.parquet must carry Phase 2's graph CONTENT.

    The baseline-identity test proves the Phase 3 REBUILD equals the
    cache; this pin proves the CACHE is content-identical to Phase 2's
    original — together the reproduction proof is non-circular. If the
    grid legitimately selects a non-baseline k and graphs.parquet is
    rebuilt at the winning config, update
    PHASE2_GRAPHS_CONTENT_SHA256 in the SAME commit and record the
    reason in docs/PHASE3_RESULTS.md."""
    assert paths.GRAPHS.exists(), "cache/graphs.parquet missing"
    got = _content_fingerprint(pd.read_parquet(paths.GRAPHS))
    assert got == PHASE2_GRAPHS_CONTENT_SHA256, (
        f"cache/graphs.parquet content fingerprint {got} != Phase 2's "
        f"recorded {PHASE2_GRAPHS_CONTENT_SHA256} — the dynamic graph "
        f"changed. If it was rebuilt deliberately at a winning grid "
        f"config, update the pin constant in the same commit and "
        f"document it; otherwise find what touched the cache before "
        f"any training uses it."
    )


# The frozen arm's edge table, pinned at the same baseline-k build that
# passed the §6 frozen invariants on 2026-09-09 (4,038,400 rows; one
# train-end snapshot per fold stamped to all fold dates, keyed
# (fold, date)).
FROZEN_GRAPHS_CONTENT_SHA256 = (
    "e69842ed19b625aa374b5c22ec0f96926e1ff9bee425ea05f9c99fcd542ab90c")


def test_frozen_graph_content_pinned():
    """graphs_frozen.parquet must be exactly the file the frozen arm's
    design was verified against. Any silent rebuild (e.g. a stray
    `python graph.py` after a configs.yaml edit) fails here before it
    can leak into a training run. A deliberate rebuild at a winning
    grid config updates this constant in the same commit."""
    assert paths.GRAPHS_FROZEN.exists(), "cache/graphs_frozen.parquet missing"
    got = _content_fingerprint(pd.read_parquet(paths.GRAPHS_FROZEN))
    assert got == FROZEN_GRAPHS_CONTENT_SHA256, (
        f"graphs_frozen.parquet content fingerprint {got} != the "
        f"recorded {FROZEN_GRAPHS_CONTENT_SHA256} — the frozen graph "
        f"was rebuilt or edited. If deliberate (winning grid config), "
        f"update the constant in the same commit and re-run the "
        f"frozen-graph invariant tests."
    )


@pytest.fixture(scope="session")
def variants(cfg) -> dict[str, pd.DataFrame]:
    """Build (or read, if already built) the 7 variants. Building here
    means `pytest tests/test_graph_grid.py` is self-contained."""
    names = list(sensitivity.variant_configs(cfg).keys())
    existing = {n: paths.graph_variant_path(n) for n in names
                if paths.graph_variant_path(n).exists()}
    if len(existing) == len(names):
        return {n: pd.read_parquet(p) for n, p in existing.items()}
    print("\n[grid-test] variants not fully built — building now "
          "(one-time, ~minutes)...")
    sensitivity.build_variants(cfg)
    return {n: pd.read_parquet(paths.graph_variant_path(n))
            for n in names}


def _expected_per_date(cfg_variant: dict) -> int:
    g = cfg_variant["graph"]
    n = len(cfg_variant["universe"]["tickers"])
    return n * int(g["k_pos"]) + n * int(g["k_neg"]) + 160


def test_seven_variants_built(variants):
    assert set(variants.keys()) == {
        "baseline", "kpos6", "kpos10", "kneg3", "kneg5", "win45", "win90"}
    for name, df in variants.items():
        assert len(df) > 0, f"variant {name} is empty"


def test_baseline_variant_identical_to_cached_dynamic(variants, cfg):
    """§6b: the baseline rebuild must equal cache/graphs.parquet
    EXACTLY. Same dates, same edges, same order, same dtypes."""
    # If a non-baseline sweep won, KAGGLE_RUN_ORDER steps 7–9 rebuild
    # graphs.parquet at the winning k while the variant parquets stay
    # frozen at the k session A trained on (Rule 22). The two are then
    # SUPPOSED to differ — skip with the reason, rather than report
    # reproduction drift that is not there.
    live = {k: int(cfg["graph"][k]) for k in ("k_pos", "k_neg", "corr_window")}
    grid = {k: int(v) for k, v in cfg["graph_grid"]["baseline"].items()}
    if live != grid:
        pytest.skip(
            f"cache/graphs.parquet was rebuilt at the winning grid config "
            f"{live}, so it no longer matches the frozen baseline variant "
            f"{grid} — expected after a non-baseline winner, not drift. "
            f"Rebuilding the variants would rewrite the provenance of the "
            f"already-scored grid (Rule 22); leave them as they are."
        )
    assert paths.GRAPHS.exists(), "cache/graphs.parquet missing"
    cached = pd.read_parquet(paths.GRAPHS)
    base = variants["baseline"]
    assert len(base) == len(cached), (
        f"baseline variant has {len(base):,} rows, cached dynamic graph "
        f"has {len(cached):,} — reproduction failed"
    )
    assert base.equals(cached), (
        "baseline variant != cache/graphs.parquet — the Phase 2 dynamic "
        "graph is not reproduced byte-for-byte. Any grid result would "
        "be rebuild drift, not k sensitivity."
    )


def test_variant_counts_match_their_k(variants, cfg):
    for name, df in variants.items():
        vcfg = sensitivity.variant_configs(cfg)[name]
        expected = _expected_per_date(vcfg)
        per_date = df.groupby("date").size()
        assert (per_date == expected).all(), (
            f"variant {name}: edge counts != {expected} on some date"
        )
        # no self-loops, indices in range
        n = len(cfg["universe"]["tickers"])
        assert (df["src"] != df["dst"]).all()
        assert df["src"].between(0, n - 1).all()
        assert df["dst"].between(0, n - 1).all()


def test_sweeps_actually_change_the_graph(variants):
    """Each non-baseline variant must differ from the baseline on at
    least one date's edge list — otherwise the sweep is a no-op."""
    base = variants["baseline"]
    base_dates = pd.DatetimeIndex(sorted(base["date"].unique()))
    probe = base_dates[len(base_dates) // 2]
    base_day = set(zip(base[base["date"] == probe]["src"],
                       base[base["date"] == probe]["dst"],
                       base[base["date"] == probe]["edge_type"]))
    for name in ["kpos6", "kpos10", "kneg3", "kneg5", "win45", "win90"]:
        v = variants[name]
        v_day = set(zip(v[v["date"] == probe]["src"],
                        v[v["date"] == probe]["dst"],
                        v[v["date"] == probe]["edge_type"]))
        assert v_day != base_day, (
            f"variant {name} has identical edges to the baseline on "
            f"{probe.date()} — its parameter change did nothing"
        )


def test_window_variants_cover_same_dates_aside_from_warmup(variants):
    """corr_window 45 needs 45 returns, 90 needs 90 — the covered date
    ranges shrink/grow at the START only; the end date must match the
    baseline's, and win45 must cover MORE dates than baseline while
    win90 covers FEWER."""
    base_dates = set(variants["baseline"]["date"].unique())
    win45_dates = set(variants["win45"]["date"].unique())
    win90_dates = set(variants["win90"]["date"].unique())
    assert max(base_dates) == max(win45_dates) == max(win90_dates)
    assert len(win45_dates) > len(base_dates), "win45 should start earlier"
    assert len(win90_dates) < len(base_dates), "win90 should start later"
    assert win90_dates <= base_dates
    assert base_dates <= win45_dates


@pytest.mark.parametrize("name", ["baseline", "kpos6", "kpos10", "kneg3",
                                  "kneg5", "win45", "win90"])
def test_nan_corruption_rejected(cfg, name):
    """Rule 4, once per variant parameter set: a NaN inside the
    correlation window must abort the build, never silently emit a
    partial graph."""
    vcfg = sensitivity.variant_configs(cfg)[name]
    universe = list(vcfg["universe"]["tickers"])
    n = len(universe)
    corr_window = int(vcfg["graph"]["corr_window"])
    dates = pd.bdate_range("2020-01-01", periods=corr_window + 5)
    rng = np.random.default_rng(3)
    arr = 100 + np.cumsum(rng.standard_normal((len(dates), n)), axis=0)
    arr[corr_window // 2, 0] = np.nan  # hole inside the LAST window
    wide = pd.DataFrame(arr, index=dates, columns=universe)
    prices = wide.stack().rename("close").reset_index()
    prices.columns = ["date", "ticker", "close"]
    with pytest.raises(AssertionError, match="NaN"):
        graph.compute_daily_edges(prices, dates[-1], vcfg)


def test_variant_hashes_recorded(variants):
    """Every variant parquet must be in hashes.json — Kaggle verifies
    these before session A trains on them."""
    if not paths.HASHES_JSON.exists():
        pytest.fail("hashes.json missing — run sensitivity.py --build")
    hashes = paths.read_hashes()
    for name in variants:
        key = f"graphs_{name}.parquet"
        assert key in hashes, f"{key} not in hashes.json"
        actual = paths.sha256_of_file(paths.graph_variant_path(name))
        assert hashes[key] == actual, (
            f"{key} hash mismatch — file changed after hashing; "
            f"re-run sensitivity.py --build"
        )
    # The five main cache entries must have survived the merge.
    for name in paths.CACHE_PARQUETS:
        assert name in hashes, f"{name} dropped from hashes.json"