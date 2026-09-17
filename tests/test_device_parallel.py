"""Two-device fold scheduling in train.run_all (single-session rework, §5).

The §5 re-measurement after the assemble_fold fix came back at ~170 s/run
on the T4 — materially above the spec's ~50 s/run assumption — and the
re-derived projection (~11 h for the 160 runs) did not clear the session
cap with margin. The lever that fits is the SECOND T4 the accelerator
already provides: fold-level parallelism, pure scheduling.

These tests pin the two properties that make that safe:

  1. Scheduling cannot change any number — every run reseeds itself
     (`train_one_seed` → `set_seed(seed)`), so the parallel schedule must
     produce BITWISE-identical predictions and fold_results to the serial
     one, including row order (fold-major, chronological).
  2. `_assign_folds` balances deterministically from a static cost proxy
     (train-window length) — for the 4 walk-forward folds on 2 devices it
     must produce the perfectly balanced longest+shortest pairing.

The integration test stubs `train_one_seed` (real training is ~400 s per
run locally and the numerics-equivalence argument above makes a real
parallel-vs-serial GPU comparison untestable off-GPU anyway); everything
AROUND the training — fold assignment, thread fan-out, merge ordering,
row-count audit, manifest — runs for real against the real cache.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data  # noqa: E402
import paths  # noqa: E402
import train  # noqa: E402


# ─── _assign_folds: static, deterministic balance ──────────────────────────

def _fold_like(name: str, n_train: int):
    class _F:
        pass

    f = _F()
    f.name = name
    f.train_dates = list(range(n_train))
    return f


def test_assign_folds_single_device_is_the_serial_schedule():
    folds = [_fold_like("a", 10), _fold_like("b", 20)]
    assert train._assign_folds(folds, 1) == [folds]


def test_assign_folds_two_devices_pair_longest_with_shortest():
    # The real configuration: 4 walk-forward folds whose train windows
    # grow monotonically (fold 1 smallest, fold 4 largest).
    f1, f2, f3, f4 = (_fold_like(n, d)
                      for n, d in [("fold_1", 776), ("fold_2", 1028),
                                   ("fold_3", 1281), ("fold_4", 1533)])
    buckets = train._assign_folds([f1, f2, f3, f4], 2)
    assert len(buckets) == 2
    names = [{f.name for f in b} for b in buckets]
    assert names == [{"fold_4", "fold_1"}, {"fold_3", "fold_2"}]
    # Balanced: the two devices carry exactly equal window totals.
    loads = [sum(len(f.train_dates) for f in b) for b in buckets]
    assert loads[0] == loads[1] == 776 + 1028 + 1281 + 1533 - loads[1]
    # Every fold assigned exactly once.
    assert sorted(sum((list(n) for n in names), [])) == sorted(
        [f.name for f in (f1, f2, f3, f4)])


def test_assign_folds_round_robins_other_device_counts():
    folds = [_fold_like(n, d) for n, d in
             [("a", 100), ("b", 80), ("c", 60), ("d", 40), ("e", 20)]]
    buckets = train._assign_folds(folds, 3)
    assert len(buckets) == 3
    assigned = [f.name for b in buckets for f in b]
    assert sorted(assigned) == ["a", "b", "c", "d", "e"]
    # Descending round-robin: the two largest folds land on devices 0/1.
    assert buckets[0][0].name == "a" and buckets[1][0].name == "b"


def test_assign_folds_is_deterministic():
    folds = [_fold_like(n, d) for n, d in
             [("a", 30), ("b", 10), ("c", 20)]]
    assert train._assign_folds(folds, 2) == train._assign_folds(folds, 2)


# ─── serial ≡ parallel: scheduling cannot change a number ──────────────────

SEEDS = [0, 1]


@pytest.fixture(scope="module")
def cfg():
    cfg = data.load_config(str(paths.CONFIG))
    cfg = dict(cfg)
    cfg["folds"] = cfg["folds"][:2]          # 2 folds → both devices busy
    cfg["training"]["seeds"] = SEEDS
    return cfg


def _stub_train_one_seed(tensors, cfg, seed, device):
    """Deterministic stand-in: values depend ONLY on (fold, seed) via the
    train-window size — never on device, thread, or completion order."""
    rng = np.random.default_rng(seed * 1000 + tensors.X_train.shape[0])
    n_stocks = len(tensors.test_tickers)
    val_pred = rng.standard_normal((len(tensors.val_dates), n_stocks))
    test_pred = rng.standard_normal((len(tensors.test_dates), n_stocks))
    metrics = pd.DataFrame({"epoch": [0, 1], "val_loss": [0.10, 0.05],
                            "train_loss": [0.20, 0.15]})
    return metrics, val_pred, test_pred, {"stub_seed": int(seed)}


@pytest.fixture
def out_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(paths, "RUN_MANIFEST", tmp_path / "run_manifest.json")
    monkeypatch.setattr(paths, "CHECKPOINTS_DIR", tmp_path / "checkpoints")
    return tmp_path


@pytest.fixture
def stubbed(monkeypatch):
    monkeypatch.setattr(train, "train_one_seed", _stub_train_one_seed)


def _run(cfg, devices):
    return train.run_all(cfg, model_id="dmrgt_all3", devices=devices,
                         verify_hashes=True)


def test_parallel_matches_serial_bitwise(cfg, out_dir, stubbed):
    if not paths.FEATURES.exists():
        pytest.skip("cache/features.parquet not present — needs the cache")

    serial = _run(cfg, ["cpu"])
    serial_rows = serial.copy()
    serial_manifest = (out_dir / "run_manifest.json").read_text()

    # wipe outputs so the parallel run rebuilds everything from scratch
    (out_dir / "predictions_dmrgt_all3.parquet").unlink()
    (out_dir / "run_manifest.json").unlink()
    (out_dir / "train_metrics_dmrgt_all3.json").unlink()

    parallel = _run(cfg, ["cpu", "cpu"])

    # bitwise-identical predictions, INCLUDING row order (fold-major,
    # chronological — not device-completion order)
    pd.testing.assert_frame_equal(serial_rows, parallel)

    # identical manifest record content (the parallel record carries the
    # devices list — everything else must match)
    import json
    rec_s = json.loads(serial_manifest)[-1]
    rec_p = json.loads((out_dir / "run_manifest.json").read_text())[-1]
    assert rec_p["devices"] == ["cpu", "cpu"]
    assert rec_s["devices"] == ["cpu"]
    rec_s.pop("devices"), rec_p.pop("devices")
    rec_s.pop("timestamp_utc"), rec_p.pop("timestamp_utc")
    assert rec_s == rec_p


def test_parallel_raises_when_a_fold_fails(cfg, out_dir, monkeypatch):
    if not paths.FEATURES.exists():
        pytest.skip("cache/features.parquet not present — needs the cache")

    def _boom(tensors, cfg, seed, device):
        raise ValueError(f"boom on {device}")

    monkeypatch.setattr(train, "train_one_seed", _boom)
    with pytest.raises(RuntimeError, match="fold training failed"):
        _run(cfg, ["cpu", "cpu"])