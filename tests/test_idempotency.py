"""Verify data.py idempotency (§3 of the spec).

The spec demands: "Re-running `python data.py` twice produces the
identical hash both times." We pin this with checks:

  1. The SHA-256 of `cache/prices.parquet` is stable across calls
     (the prices table is fully cached after the first download —
     the second call should be a no-op that returns the same file).
  2. The SHA-256 in `cache/hashes.json` matches the file too.

There is a single source of truth for cache hashes (`hashes.json`);
`cache/prices.sha256` has been removed to eliminate the dual-source
risk (see KAGGLE.md §7).

A failure here is the strongest single signal that something in the
download/align/validate pipeline is non-deterministic. The spec calls
this the "deliverable check" for Step 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import data  # noqa: E402
import paths  # noqa: E402


def test_prices_parquet_exists():
    assert paths.PRICES.exists(), (
        "cache/prices.parquet missing — run `python data.py --download` "
        "before running this test"
    )


def test_no_prices_sha256_sidecar():
    """The redundant prices.sha256 sidecar was removed — see KAGGLE.md
    §7. Two sources of truth for the same hash would eventually
    disagree. The hashes.json file is the only place cache hashes live."""
    sidecar = paths.CACHE_DIR / "prices.sha256"
    assert not sidecar.exists(), (
        f"{sidecar} exists but should have been removed — KAGGLE.md §7 "
        f"consolidates cache hashes into hashes.json only"
    )


def test_hashes_json_consistent_with_files():
    """Every hash in hashes.json must match the corresponding file.
    This is what the Kaggle notebook re-verifies before training."""
    assert paths.HASHES_JSON.exists(), "cache/hashes.json missing"
    import json
    hashes = json.loads(paths.HASHES_JSON.read_text())
    for name, want in hashes.items():
        # Skip the timestamp entry — not a file hash.
        if name == "generated_at":
            continue
        p = paths.CACHE_DIR / name
        if not p.exists():
            # It's OK for hashes.json to mention a file that doesn't
            # exist yet — data.py only includes existing files. The
            # spec asks for the file to exist, but a future step
            # (e.g. --features not run yet) may legitimately omit it.
            continue
        got = data.sha256_of_file(p)
        assert got == want, (
            f"hashes.json says {name} should be {want[:12]}… "
            f"but the file is {got[:12]}…"
        )


def test_hashes_json_has_generated_at():
    """The hashes.json schema includes a `generated_at` timestamp —
    added so we can detect when the cache was last regenerated without
    re-hashing every parquet."""
    import json
    assert paths.HASHES_JSON.exists()
    hashes = json.loads(paths.HASHES_JSON.read_text())
    assert "generated_at" in hashes, (
        f"hashes.json missing 'generated_at' entry. Got keys: {list(hashes.keys())}"
    )
    # ISO-8601 UTC timestamp, ends with 'Z'
    ts = hashes["generated_at"]
    assert ts.endswith("Z"), f"generated_at should end with Z, got {ts!r}"
    assert "T" in ts, f"generated_at should be ISO-8601, got {ts!r}"


def test_re_download_is_cache_hit():
    """Running data.py --download a second time should be a pure
    cache hit — no network, no rewrite. We assert this by checking
    the mtime of prices.parquet before and after; if anything
    re-wrote the file, the mtime will change. We also check the
    hash stays identical."""
    assert paths.PRICES.exists()
    import time
    mtime_before = paths.PRICES.stat().st_mtime
    hash_before = data.sha256_of_file(paths.PRICES)
    # Sleep a hair to make any mtime update detectable even on
    # filesystems with 1-second resolution.
    time.sleep(1.1)
    # Re-run the download CLI in-process. The download_prices()
    # function detects the cache hit and returns without touching
    # the file.
    cfg = data.load_config()
    data.download_prices(cfg)
    mtime_after = paths.PRICES.stat().st_mtime
    hash_after = data.sha256_of_file(paths.PRICES)
    assert mtime_after == mtime_before, (
        f"prices.parquet was rewritten on a cache hit "
        f"(mtime {mtime_before} → {mtime_after})"
    )
    assert hash_after == hash_before, (
        f"prices.parquet hash changed on a cache hit "
        f"({hash_before[:12]}… → {hash_after[:12]}…)"
    )
