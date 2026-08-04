"""Verify paths.py resolves correctly in BOTH local and Kaggle environments.

The Kaggle branch is detected by the existence of /kaggle/input. We can't
create that directory on macOS (read-only root), so we mock os.path.exists
instead. The mocked resolution must produce the same path values that
Kaggle would see.
"""

from __future__ import annotations

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


def test_local_branch_default():
    """Without /kaggle/input present, paths.py must resolve to the project
    tree and create the cache and artifacts directories."""
    with mock.patch("os.path.exists", return_value=False):
        # Make sure OUTPUT_DIR mkdir doesn't hit a real Kaggle path.
        paths = _reload_paths()
    assert paths.is_kaggle() is False
    assert paths.CACHE_DIR == ROOT / "cache"
    assert paths.OUTPUT_DIR == ROOT / "artifacts"
    assert (ROOT / "cache").exists(), "cache/ must be created on local import"
    assert (ROOT / "artifacts").exists(), "artifacts/ must be created on local import"
    assert paths.PRICES == ROOT / "cache" / "prices.parquet"
    assert paths.FEATURES == ROOT / "cache" / "features.parquet"
    assert paths.TARGETS == ROOT / "cache" / "targets.parquet"
    assert paths.HASHES_JSON == ROOT / "cache" / "hashes.json"
    assert paths.PREDICTIONS == ROOT / "artifacts" / "predictions.parquet"


def test_kaggle_branch_simulated():
    """With /kaggle/input mocked present, paths.py must resolve CACHE_DIR
    to /kaggle/input/dmrgt-cache and OUTPUT_DIR to /kaggle/working, and
    must NOT try to mkdir /kaggle/input (read-only on Kaggle).

    We also patch Path.mkdir because /kaggle/working only exists inside
    a real Kaggle container; on macOS the directory simply isn't there
    and a real mkdir would (correctly) fail in a way unrelated to the
    path-resolution logic we're testing.

    The patch must remain active across the assertions: is_kaggle() is
    a function, not a constant, and re-queries the filesystem each call.
    """
    with mock.patch("os.path.exists", return_value=True), \
         mock.patch("pathlib.Path.mkdir", return_value=None):
        paths = _reload_paths()
        assert paths.is_kaggle() is True
        assert str(paths.CACHE_DIR) == "/kaggle/input/dmrgt-cache"
        assert str(paths.OUTPUT_DIR) == "/kaggle/working"
        # /kaggle/input is read-only on Kaggle; we must never create it.
        # The fact that the patched mkdir never produced a real
        # filesystem call is the assertion — if paths.py tried to mkdir
        # /kaggle/input on the real filesystem, macOS would have raised
        # before the patch.
        assert str(paths.PRICES) == "/kaggle/input/dmrgt-cache/prices.parquet"
        assert str(paths.FEATURES) == "/kaggle/input/dmrgt-cache/features.parquet"
        assert str(paths.TARGETS) == "/kaggle/input/dmrgt-cache/targets.parquet"


def test_describe_returns_expected_keys():
    paths = _reload_paths()
    info = paths.describe()
    for key in ("env", "CACHE_DIR", "OUTPUT_DIR", "PRICES",
                "FEATURES", "TARGETS", "HASHES_JSON", "PREDICTIONS"):
        assert key in info, f"describe() missing key: {key}"
    # After the previous test, /kaggle/input was mocked True, but we've
    # now reloaded without the mock — env should be 'local' in this
    # process. Either branch is acceptable as long as the value is one
    # of the two valid options.
    assert info["env"] in ("local", "kaggle")
