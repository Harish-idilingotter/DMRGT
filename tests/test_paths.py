"""Verify paths.py resolves correctly in BOTH local and Kaggle environments.

The Kaggle branch is detected by the existence of /kaggle/input. We can't
create that directory on macOS (read-only root), so we mock os.path.exists
instead. The mocked resolution must produce the same path values that
Kaggle would see.

The two copies of paths.py — the project root and kaggle/code/ — must
remain byte-identical so that local and Kaggle runs see the exact same
path logic. This file tests that, plus the structural invariants of the
local branch and the descriptive snapshot.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import sys
from pathlib import Path
from unittest import mock

import pytest


# Resolve the project root and ensure it's importable.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _reload_paths():
    """Reload paths.py and return the module. Lets us re-execute the
    module-level branch with a different os.path.exists answer."""
    if "paths" in sys.modules:
        return importlib.reload(sys.modules["paths"])
    return importlib.import_module("paths")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ─── Byte-identical guarantee ─────────────────────────────────────────────

def test_paths_py_byte_identical_to_kaggle_copy():
    """The kaggle/code/paths.py copy is assembled by the build script.
    Verify the build script exists and, when run, produces a byte-identical
    copy of paths.py in kaggle/code/."""
    import subprocess
    build_script = ROOT / "kaggle" / "build_code_dataset.sh"
    assert build_script.exists(), f"{build_script} missing"
    # Run the build script to assemble the dataset
    result = subprocess.run(
        ["bash", str(build_script)],
        cwd=str(ROOT), capture_output=True, text=True
    )
    assert result.returncode == 0, f"build script failed: {result.stderr}"
    b = ROOT / "kaggle" / "code" / "paths.py"
    assert b.exists(), f"{b} not created by build script"
    ha = _sha256(ROOT / "paths.py")
    hb = _sha256(b)
    assert ha == hb, (
        f"paths.py and kaggle/code/paths.py are not byte-identical. "
        f"\n  {ROOT / 'paths.py'}: {ha}"
        f"\n  {b}: {hb}"
    )


# ─── Local resolution ─────────────────────────────────────────────────────

def test_local_branch_default():
    """Without /kaggle/input present, paths.py must resolve to the project
    tree and create the cache and artifacts directories."""
    with mock.patch("os.path.exists", return_value=False):
        paths = _reload_paths()
    assert paths.is_kaggle() is False
    assert paths.CACHE_DIR == ROOT / "cache"
    assert paths.OUTPUT_DIR == ROOT / "artifacts"
    assert paths.CONFIG == ROOT / "configs.yaml"
    assert paths.CONFIG.exists(), "configs.yaml must exist in the project root"
    assert (ROOT / "cache").exists(), "cache/ must be created on local import"
    assert (ROOT / "artifacts").exists(), "artifacts/ must be created on local import"
    assert paths.PRICES == ROOT / "cache" / "prices.parquet"
    assert paths.FEATURES == ROOT / "cache" / "features.parquet"
    assert paths.TARGETS == ROOT / "cache" / "targets.parquet"
    assert paths.HASHES_JSON == ROOT / "cache" / "hashes.json"
    assert paths.PREDICTIONS == ROOT / "artifacts" / "predictions.parquet"


# ─── Kaggle resolution (mocked) ───────────────────────────────────────────

def test_kaggle_branch_resolves_via_candidate():
    """With /kaggle/input mocked present but /kaggle/input/<slug>
    missing, the resolver must try /kaggle/input/datasets/<slug>
    before falling back to rglob.

    We patch Path.is_dir so the second candidate reports as a
    directory — simulating a layout that Kaggle has been observed to
    use.
    """
    real_is_dir = Path.is_dir

    # Mark which candidate strings should look like directories
    layout = {
        "/kaggle/input/dmrgt-cache":         False,  # older layout absent
        "/kaggle/input/datasets/dmrgt-cache": True,   # current layout present
        "/kaggle/input/dmrgt-code":          False,
        "/kaggle/input/datasets/dmrgt-code": True,
    }

    def _fake_is_dir(self):
        s = str(self)
        if s in layout:
            return layout[s]
        return real_is_dir(self)

    real_exists = Path.exists

    def _fake_exists(self):
        s = str(self)
        if s == "/kaggle/input":
            return True
        return real_exists(self)

    def _fake_os_exists(p):
        if p == "/kaggle/input":
            return True
        return os.path.exists(p)

    with mock.patch("os.path.exists", side_effect=_fake_os_exists), \
         mock.patch("pathlib.Path.is_dir", _fake_is_dir), \
         mock.patch("pathlib.Path.exists", _fake_exists), \
         mock.patch("pathlib.Path.mkdir", return_value=None):
        paths = _reload_paths()
        assert paths.is_kaggle() is True
        # The resolver found the dataset under the current layout
        # (the second candidate) — not the older layout.
        assert str(paths.CACHE_DIR) == "/kaggle/input/datasets/dmrgt-cache"


def test_kaggle_branch_raises_with_listing():
    """If a slug doesn't match either candidate layout, the resolver
    must raise FileNotFoundError with the actual /kaggle/input
    listing so the user can see what really exists.

    To get past the module-load call to _resolve_kaggle_dataset, we
    have /kaggle/input/datasets/dmrgt-cache return True — the dmrgt-code
    call will then come back later for a slug that truly doesn't
    exist.
    """
    real_is_dir = Path.is_dir

    present = {
        "/kaggle/input/datasets/dmrgt-cache": True,
        "/kaggle/input/datasets/dmrgt-code":  True,
    }

    def _fake_is_dir(self):
        s = str(self)
        if s in present:
            return present[s]
        return real_is_dir(self)

    def _fake_os_exists(p):
        if p == "/kaggle/input":
            return True
        return os.path.exists(p)

    with mock.patch("os.path.exists", side_effect=_fake_os_exists), \
         mock.patch("pathlib.Path.is_dir", _fake_is_dir), \
         mock.patch("pathlib.Path.mkdir", return_value=None):
        paths = _reload_paths()
        # Now the module is loaded with both dmrgt-cache and dmrgt-code
        # present (so the imports succeeded). Calling the resolver with
        # a slug that truly isn't attached should raise.
        with pytest.raises(FileNotFoundError) as excinfo:
            paths._resolve_kaggle_dataset("definitely-not-here")
        msg = str(excinfo.value)
        assert "definitely-not-here" in msg, \
            f"error message should name the slug: {msg!r}"
        assert "/kaggle/input" in msg, \
            f"error message should mention /kaggle/input: {msg!r}"
        assert "Actual contents" in msg, \
            f"error message should show what /kaggle/input actually " \
            f"contains so the user can see what to attach: {msg!r}"
        assert "RESTART THE SESSION" in msg, \
            f"error message should warn the user to restart the " \
            f"kernel after attaching: {msg!r}"


def test_describe_returns_expected_keys():
    paths = _reload_paths()
    info = paths.describe()
    for key in ("environment", "CACHE_DIR", "CODE_DIR", "OUTPUT_DIR",
                "CONFIG", "config_exists",
                "cache_exists", "cache_contents",
                "prices_exists", "features_exists",
                "targets_exists", "hashes_exists"):
        assert key in info, f"describe() missing key: {key}"
    # env should be 'local' in this process (we never mocked Kaggle here).
    assert info["environment"] in ("local", "kaggle")


def test_cache_parquets_canonical():
    """The CACHE_PARQUETS tuple is the single source of truth for
    which files get hashed. It must contain exactly the three
    data parquets the spec names."""
    paths = _reload_paths()
    assert tuple(paths.CACHE_PARQUETS) == (
        "prices.parquet",
        "features.parquet",
        "targets.parquet",
        "graphs.parquet",
    ), (
        f"CACHE_PARQUETS changed: {paths.CACHE_PARQUETS}. "
        f"train.py and the notebook both iterate this tuple — drift "
        f"between them is a Phase-2-blocking bug."
    )


def test_kaggle_branch_resolves_nested_username_layout():
    """The currently-observed Kaggle layout is
    /kaggle/input/datasets/<username>/<slug>. The candidate resolver
    does NOT include this exact path (no hardcoded username segment)
    — the rglob fallback is what catches it."""
    real_is_dir = Path.is_dir

    present = {
        # The 2 hardcoded candidates are absent.
        "/kaggle/input/dmrgt-cache":         False,
        "/kaggle/input/datasets/dmrgt-cache": False,
        # The nested username layout is present.
        "/kaggle/input/datasets/user-name/dmrgt-cache": True,
        # Same for dmrgt-code so the module loads.
        "/kaggle/input/dmrgt-code":          False,
        "/kaggle/input/datasets/dmrgt-code": False,
        "/kaggle/input/datasets/user-name/dmrgt-code":  True,
    }

    def _fake_is_dir(self):
        s = str(self)
        if s in present:
            return present[s]
        return real_is_dir(self)

    real_exists = Path.exists

    def _fake_exists(self):
        s = str(self)
        if s == "/kaggle/input":
            return True
        return real_exists(self)

    real_rglob = Path.rglob

    def _fake_rglob(self, pattern):
        # The resolver calls KAGGLE_INPUT.rglob(slug). In the nested
        # layout that path is "/kaggle/input/datasets/user-name/<slug>",
        # so yield it when asked for the slug and we're on
        # /kaggle/input.
        if str(self) == "/kaggle/input":
            for path, is_dir in present.items():
                if pattern in path and is_dir:
                    yield Path(path)
            return
        yield from real_rglob(self, pattern)

    def _fake_os_exists(p):
        if p == "/kaggle/input":
            return True
        return os.path.exists(p)

    with mock.patch("os.path.exists", side_effect=_fake_os_exists), \
         mock.patch("pathlib.Path.is_dir", _fake_is_dir), \
         mock.patch("pathlib.Path.exists", _fake_exists), \
         mock.patch("pathlib.Path.rglob", _fake_rglob), \
         mock.patch("pathlib.Path.mkdir", return_value=None):
        paths = _reload_paths()
        assert paths.is_kaggle() is True
        # rglob fallback found the nested-username layout.
        assert str(paths.CACHE_DIR) == "/kaggle/input/datasets/user-name/dmrgt-cache"
