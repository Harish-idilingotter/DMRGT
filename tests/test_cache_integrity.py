"""Verify the integrity of the local cache/ directory.

This test suite is what protects the `dmrgt-cache` dataset upload from
silent corruption. Three classes of failure it catches:

  1. `hashes.json` is missing an entry for a parquet that exists in
     cache/. The Kaggle notebook re-verifies hashes before training
     and would halt if a hash is missing — better to fail here.

  2. A recorded hash does not match the actual file content. This
     catches both `data.py` writing the wrong hash and a file being
     modified outside the data pipeline.

  3. The cache/ directory contains junk that would be uploaded to
     Kaggle as part of the dataset (`.gitkeep`, `.DS_Store`,
     `.ipynb_checkpoints/`, the redundant `prices.sha256` sidecar).
     These pollute the dataset and slow uploads.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import data  # noqa: E402
import paths  # noqa: E402


EXPECTED_PARQUETS = [
    "prices.parquet",
    "features.parquet",
    "targets.parquet",
]


def test_hashes_json_exists():
    """A missing hashes.json means data.py was never run, or the
    file was deleted by hand."""
    assert paths.HASHES_JSON.exists(), (
        f"cache/hashes.json missing — run `python data.py --features` "
        f"to regenerate it"
    )


def test_hashes_json_covers_every_parquet():
    """Every parquet in cache/ must have a hash entry. A parquet
    without an entry is silently unverifiable on Kaggle."""
    assert paths.HASHES_JSON.exists()
    hashes = json.loads(paths.HASHES_JSON.read_text())
    missing = []
    for p in [paths.PRICES, paths.FEATURES, paths.TARGETS]:
        if p.exists() and p.name not in hashes:
            missing.append(p.name)
    assert not missing, (
        f"hashes.json is missing entries for: {missing}. "
        f"Re-run data.py so it can call update_hashes() and "
        f"_verify_hashes_complete() at the end of each stage."
    )


@pytest.mark.parametrize("parquet", EXPECTED_PARQUETS)
def test_each_parquet_hash_matches(parquet: str):
    """Each parquet's recorded hash must equal its actual SHA-256.
    Parametrised so a failure pinpoints which file is corrupt."""
    if not paths.HASHES_JSON.exists():
        pytest.skip("hashes.json missing — run data.py first")
    p = paths.CACHE_DIR / parquet
    if not p.exists():
        pytest.skip(f"{p} missing — run `data.py --{parquet.split('.')[0]}` first")
    hashes = json.loads(paths.HASHES_JSON.read_text())
    assert parquet in hashes, (
        f"hashes.json missing entry for {parquet} "
        f"(present: {list(hashes.keys())})"
    )
    got = data.sha256_of_file(p)
    want = hashes[parquet]
    assert got == want, (
        f"{parquet} hash mismatch:\n"
        f"  file:     {got}\n"
        f"  recorded: {want}\n"
        f"The file was likely modified after data.py ran. "
        f"Re-run `data.py --{parquet.split('.')[0]}` to refresh hashes."
    )


def test_cache_has_no_unexpected_files():
    """cache/ must contain exactly the 3 parquets + hashes.json
    + dataset-metadata.json. Anything else (e.g. .DS_Store,
    .gitkeep, .ipynb_checkpoints, prices.sha256) would be uploaded
    to Kaggle as part of the dataset and pollute it.
    """
    if not paths.CACHE_DIR.exists():
        pytest.skip("cache/ doesn't exist yet")
    allowed = set(EXPECTED_PARQUETS) | {"hashes.json", "dataset-metadata.json"}
    # Also allow hidden dotfiles that aren't the listed junk
    actual = {p.name for p in paths.CACHE_DIR.iterdir() if not p.name.startswith(".")}
    junk = actual - allowed
    assert not junk, (
        f"Unexpected files in cache/ that would be uploaded to Kaggle: {junk}. "
        f"Remove them before uploading the dmrgt-cache dataset."
    )


def test_cache_has_no_junk_dotfiles():
    """Catch the specific files known to leak into cache/:
    .gitkeep (no purpose), .DS_Store (macOS), prices.sha256 (redundant),
    .ipynb_checkpoints/ (notebook state)."""
    if not paths.CACHE_DIR.exists():
        pytest.skip("cache/ doesn't exist yet")
    junk_dotfiles = []
    for p in paths.CACHE_DIR.iterdir():
        if p.name in (".DS_Store", ".gitkeep"):
            junk_dotfiles.append(p.name)
        if p.is_dir() and p.name == ".ipynb_checkpoints":
            junk_dotfiles.append(p.name)
    assert not junk_dotfiles, (
        f"Found junk files in cache/ that should be removed: {junk_dotfiles}. "
        f"These would be uploaded to Kaggle as part of the dataset."
    )


def test_prices_sha256_sidecar_absent():
    """The redundant prices.sha256 sidecar was removed (KAGGLE.md §7).
    hashes.json is the single source of truth."""
    if not paths.CACHE_DIR.exists():
        pytest.skip("cache/ doesn't exist yet")
    sidecar = paths.CACHE_DIR / "prices.sha256"
    assert not sidecar.exists(), (
        f"{sidecar} exists — it was removed because hashes.json is the "
        f"single source of truth for cache hashes. Two sources of truth "
        f"will eventually disagree."
    )


# ─── verify_cache_hashes() (defect 2 regression) ─────────────────────────

def test_verify_cache_hashes_skips_metadata_keys(tmp_path, monkeypatch):
    """Non-.parquet keys (e.g. generated_at) must not be treated as
    filenames. We construct a hashes.json with a metadata key and
    check the verifier returns cleanly."""
    import json
    import train
    # Point paths at a temp dir with valid parquets + a hashes.json
    # that has an extra non-.parquet key.
    monkeypatch.setattr(train.paths, "CACHE_DIR", tmp_path, raising=False)
    monkeypatch.setattr(train.paths, "HASHES_JSON",
                        tmp_path / "hashes.json", raising=False)
    monkeypatch.setattr(train.paths, "CACHE_PARQUETS",
                        ("a.parquet", "b.parquet"), raising=False)

    (tmp_path / "a.parquet").write_bytes(b"AAA")
    (tmp_path / "b.parquet").write_bytes(b"BBB")
    (tmp_path / "hashes.json").write_text(json.dumps({
        "a.parquet": train.sha256_of_file(tmp_path / "a.parquet"),
        "b.parquet": train.sha256_of_file(tmp_path / "b.parquet"),
        "generated_at": "2026-08-05T20:00:00Z",   # metadata — must be skipped
        "future_field": "ignore me",
    }))

    found = train.verify_cache_hashes(stop_on_mismatch=True)
    assert set(found.keys()) == {"a.parquet", "b.parquet"}


def test_verify_cache_hashes_halts_when_hashes_json_missing(tmp_path, monkeypatch):
    """Missing hashes.json is a halt, not a warning."""
    import train
    monkeypatch.setattr(train.paths, "CACHE_DIR", tmp_path, raising=False)
    monkeypatch.setattr(train.paths, "HASHES_JSON",
                        tmp_path / "hashes.json", raising=False)
    with pytest.raises(RuntimeError, match="not found"):
        train.verify_cache_hashes(stop_on_mismatch=True)


def test_verify_cache_hashes_halts_on_mismatch(tmp_path, monkeypatch):
    """A wrong hash in hashes.json must halt — this is the integrity gate."""
    import json
    import train
    monkeypatch.setattr(train.paths, "CACHE_DIR", tmp_path, raising=False)
    monkeypatch.setattr(train.paths, "HASHES_JSON",
                        tmp_path / "hashes.json", raising=False)
    monkeypatch.setattr(train.paths, "CACHE_PARQUETS",
                        ("x.parquet",), raising=False)

    (tmp_path / "x.parquet").write_bytes(b"some data")
    # Wrong hash
    (tmp_path / "hashes.json").write_text(json.dumps({
        "x.parquet": "0" * 64,
    }))
    with pytest.raises(RuntimeError, match="hash mismatch"):
        train.verify_cache_hashes(stop_on_mismatch=True)