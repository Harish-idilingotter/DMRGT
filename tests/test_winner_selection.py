"""§1 of the single-session rework — the winner-selection safeguard.

Winner selection must provably never touch ``split == "test"`` rows
before the winner is locked (Rule 18, made stricter: the grid's test
rows ARE the final dynamic-arm result, read for the first time only
after selection). These tests prove the property mechanically:

  * equivalence — selection over a grid whose test rows are garbage
    must equal selection over the same grid with test rows deleted;
  * poisoning — replacing every variant's test rows with values that
    would reorder the ranking must not move the winner or any metric;
  * mechanics — the implementation slices to val and ``del``s the
    full-frame reference before scoring, per the spec's required shape.

The safeguard lives in one place — ``sensitivity.select_winner`` — and
the Kaggle notebook's winner-selection cell calls exactly that function
before any test-split file is opened, so passing here covers the run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data
import paths
import sensitivity


@pytest.fixture(scope="module")
def cfg() -> dict:
    return data.load_config(str(paths.CONFIG))


VARIANT_ORDER = ["baseline", "kpos6", "kpos10", "kneg3", "kneg5",
                 "win45", "win90"]
DATES = [pd.Timestamp("2019-01-02"), pd.Timestamp("2019-01-03")]
SEEDS = [0]
FOLDS = ["f1", "f2"]


def _build_grid_files(output_dir: Path, cfg: dict) -> None:
    """Write one valid predictions parquet per grid variant.

    Val predictions rotate the (monotone) target pattern by the variant's
    index, so every variant has a distinct, deterministic val ranking;
    test rows are independent noise — nothing downstream may read them.
    """
    universe = list(cfg["universe"]["tickers"])
    n = len(universe)
    y_true = np.linspace(-1.0, 1.0, n)
    rng = np.random.default_rng(0)
    for i, variant in enumerate(VARIANT_ORDER):
        rows = []
        y_pred_val = np.roll(y_true, i) + 0.01 * rng.standard_normal(n)
        y_pred_test = 0.01 * rng.standard_normal(n)
        for fold in FOLDS:
            for seed in SEEDS:
                for d in DATES:
                    for split, y_pred in [("val", y_pred_val),
                                          ("test", y_pred_test)]:
                        for j, t in enumerate(universe):
                            rows.append({
                                "model": f"dmrgt_all3_grid_{variant}",
                                "fold": fold, "seed": seed,
                                "split": split, "date": d, "ticker": t,
                                "y_pred": float(y_pred[j]),
                                "y_true": float(y_true[j]),
                            })
        df = pd.DataFrame(rows)
        df.to_parquet(
            output_dir / f"predictions_dmrgt_all3_grid_{variant}.parquet",
            index=False)


def test_winner_selection_equals_selection_with_test_rows_deleted(
        cfg, tmp_path, monkeypatch):
    """The gold property: deleting every test row must change nothing."""
    monkeypatch.setattr(paths, "OUTPUT_DIR", tmp_path)
    _build_grid_files(tmp_path, cfg)

    record_with_test = sensitivity.select_winner(cfg)

    # Remove every test row from every file, re-run selection.
    for variant in VARIANT_ORDER:
        p = tmp_path / f"predictions_dmrgt_all3_grid_{variant}.parquet"
        df = pd.read_parquet(p)
        df = df[df["split"] == "val"]
        df.to_parquet(p, index=False)

    record_val_only = sensitivity.select_winner(cfg)

    assert record_with_test["winner"] == record_val_only["winner"]
    assert (record_with_test["selection_value"]
            == record_val_only["selection_value"])
    assert (record_with_test["winning_graph_params"]
            == record_val_only["winning_graph_params"])
    got = {(r["variant"], r["val_sharpe"], r["val_mean_rank_ic"])
           for r in record_with_test["grid"]}
    want = {(r["variant"], r["val_sharpe"], r["val_mean_rank_ic"])
            for r in record_val_only["grid"]}
    assert got == want, "per-variant val metrics changed when test rows vanished"


def test_winner_selection_ignores_poisoned_test_rows(
        cfg, tmp_path, monkeypatch):
    """Overwrite every variant's test rows with values that would reorder
    the ranking if anything read them — nothing may move."""
    monkeypatch.setattr(paths, "OUTPUT_DIR", tmp_path)
    _build_grid_files(tmp_path, cfg)
    record_before = sensitivity.select_winner(cfg)

    for i, variant in enumerate(VARIANT_ORDER):
        p = tmp_path / f"predictions_dmrgt_all3_grid_{variant}.parquet"
        df = pd.read_parquet(p)
        test_mask = df["split"] == "test"
        y_true = df.loc[test_mask, "y_true"].to_numpy()
        # Variant-dependent poison: kpos6 gets perfect predictions, the
        # rest anti-perfect — a selection that reads test rows would see
        # kpos6 jump to the top.
        factor = 10.0 if variant == "kpos6" else -10.0
        df.loc[test_mask, "y_pred"] = y_true * factor
        df.to_parquet(p, index=False)

    record_after = sensitivity.select_winner(cfg)

    assert record_after["winner"] == record_before["winner"], (
        "winner moved after poisoning test rows — selection read "
        "split == 'test'"
    )
    assert (record_after["selection_value"]
            == record_before["selection_value"])
    assert (record_after["winning_graph_params"]
            == record_before["winning_graph_params"])
    assert (record_after["grid"] == record_before["grid"])


def test_select_winner_mechanics_follow_the_spec():
    """§1's required shape, checked at the source: slice to val, ``del``
    the full-frame reference, never filter on test."""
    import inspect

    src = inspect.getsource(sensitivity.select_winner)
    assert 'df["split"] == "val"' in src, (
        "select_winner must slice to val rows with the spec's expression"
    )
    assert "del df" in src, (
        "select_winner must drop the full-frame reference (§1: `del df` "
        "after slicing so no lingering variable can read test rows)"
    )
    assert '"test"' not in src.replace(
        'set(df["split"].unique()) <= {"val", "test"}', ""), (
        "select_winner must not read or filter split == 'test' rows"
    )


def test_select_winner_writes_expected_artifacts(cfg, tmp_path, monkeypatch):
    """The in-session selection writes exactly the two files Cell 7 of
    the notebook requires, with the schema the frozen-graph cell reads."""
    monkeypatch.setattr(paths, "OUTPUT_DIR", tmp_path)
    _build_grid_files(tmp_path, cfg)

    record = sensitivity.select_winner(cfg)

    csv_path = tmp_path / "phase3_sensitivity_grid.csv"
    json_path = tmp_path / "phase3_frozen_graph_config.json"
    assert csv_path.exists() and json_path.exists()
    table = pd.read_csv(csv_path)
    assert set(table["variant"]) == set(VARIANT_ORDER)
    # Winner consistency across the two artifacts, whichever variant led.
    top = table.sort_values(
        f"val_{record['selection_metric']}", ascending=False).iloc[0]
    assert record["winner"] == top["variant"]
    loaded = json.loads(json_path.read_text())
    assert loaded["winner"] == record["winner"]
    assert loaded["winning_graph_params"] == record["winning_graph_params"]
    assert loaded["selection_metric"] == cfg["graph_grid"]["selection_metric"]
    assert loaded["changed_from_baseline"] == (
        loaded["winning_graph_params"] != loaded["baseline_graph_params"])


def test_select_winner_halts_on_incomplete_grid(cfg, tmp_path, monkeypatch):
    """One missing variant file must abort selection — a partial grid
    must never pick a winner from the variants that happened to finish."""
    monkeypatch.setattr(paths, "OUTPUT_DIR", tmp_path)
    _build_grid_files(tmp_path, cfg)
    (tmp_path / "predictions_dmrgt_all3_grid_kpos6.parquet").unlink()
    with pytest.raises(FileNotFoundError):
        sensitivity.select_winner(cfg)