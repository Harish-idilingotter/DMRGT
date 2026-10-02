"""DMRGT Phase 3 — walk-forward fold construction (§4.3).

Chronological splits with expanding training windows. The latest ~15% of
each training window is carved as validation for early stopping — also
chronological, never random. Rewritten fresh for Phase 3 but identical
in value to Phases 1/2: the cache and the significance test both depend
on the fold boundaries not moving.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass
class Fold:
    """One walk-forward fold with train_dates, val_dates, test_dates."""
    name: str
    fold_idx: int
    train_dates: pd.DatetimeIndex
    val_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex

    def __repr__(self) -> str:
        def span(idx: pd.DatetimeIndex) -> str:
            if len(idx) == 0:
                return "empty"
            return f"{idx[0].date()}→{idx[-1].date()} (n={len(idx)})"
        return (f"Fold({self.name!r}, train={span(self.train_dates)}, "
                f"val={span(self.val_dates)}, test={span(self.test_dates)})")


def make_folds(
    cfg: dict,
    available_dates: Iterable[pd.Timestamp] | None = None,
    val_fraction: float = 0.15,
) -> list[Fold]:
    """Build walk-forward folds from configs.yaml.

    `available_dates` restricts each configured window to dates that
    actually exist in the cache (the features table's calendar). The
    latest val_fraction of each training window becomes validation.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    avail = (pd.DatetimeIndex(sorted(set(pd.Timestamp(d) for d in available_dates)))
             if available_dates is not None else None)
    folds_cfg = cfg.get("folds", [])
    if not folds_cfg:
        raise ValueError("configs.yaml has no 'folds' section")

    out: list[Fold] = []
    for i, fc in enumerate(folds_cfg):
        name = str(fc.get("name", f"fold_{i}"))
        train_start = pd.Timestamp(fc["train_start"])
        train_end = pd.Timestamp(fc["train_end"])
        test_start = pd.Timestamp(fc["test_start"])
        test_end = pd.Timestamp(fc["test_end"])

        # ── hard chronological invariant (Rule 2) ─────────────────────
        if train_end >= test_start:
            raise AssertionError(
                f"Fold {name!r}: train_end ({train_end.date()}) must be "
                f"strictly before test_start ({test_start.date()}) — "
                f"otherwise the model sees the test window during training."
            )

        # ── build the full training and test index ─────────────────────
        if avail is not None:
            # Configured window, intersected with dates the cache has.
            # The boundary itself is owned by the config; the data
            # restricts what we can actually use.
            train_mask = (avail >= train_start) & (avail <= train_end)
            test_mask = (avail >= test_start) & (avail <= test_end)
            train_full = avail[train_mask]
            test_idx = avail[test_mask]
        else:
            # No availability constraint (tests / running splits without
            # the cache): synthesise business-day dates. The next consumer
            # that loads the features table intersects again.
            train_full = pd.bdate_range(train_start, train_end)
            test_idx = pd.bdate_range(test_start, test_end)

        if len(train_full) == 0:
            raise ValueError(f"Fold {name!r}: no train dates after intersecting with available dates")
        if len(test_idx) == 0:
            raise ValueError(f"Fold {name!r}: no test dates after intersecting with available dates")

        # ── split off the validation portion ───────────────────────────
        # The latest val_fraction of the training window becomes the
        # validation set. Chronological, not random.
        n_train_full = len(train_full)
        n_val = int(round(n_train_full * val_fraction))
        if n_val <= 0 or n_val >= n_train_full:
            raise ValueError(
                f"Fold {name!r}: val_fraction={val_fraction} produced "
                f"n_val={n_val} from n_train_full={n_train_full}; "
                f"need at least 1 train and 1 val date"
            )
        train_idx = train_full[:-n_val]
        val_idx = train_full[-n_val:]

        # ── zero-overlap asserts — fail loudly, never silently ─────────
        assert len(set(train_idx).intersection(val_idx)) == 0
        assert len(set(train_idx).intersection(test_idx)) == 0
        assert len(set(val_idx).intersection(test_idx)) == 0
        # Chronological: last train date precedes first val date, which
        # precedes the first test date.
        assert train_idx.max() < val_idx.min()
        assert val_idx.max() < test_idx.min()
        assert train_idx.max() < test_idx.min()

        out.append(Fold(
            name=name,
            fold_idx=i,
            train_dates=train_idx,
            val_dates=val_idx,
            test_dates=test_idx,
        ))

    return out


if __name__ == "__main__":
    import data
    import paths

    cfg = data.load_config(str(paths.CONFIG))
    if paths.FEATURES.exists():
        feats = pd.read_parquet(paths.FEATURES, columns=["date"])
        avail = pd.DatetimeIndex(feats["date"].unique())
    else:
        avail = None
    for f in make_folds(cfg, available_dates=avail):
        print(f)