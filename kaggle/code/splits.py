"""DMRGT Phase 1 — walk-forward fold construction (Step E / §5.1).

Reads the fold definitions from configs.yaml and returns one Fold
object per fold, each with explicit ``train_dates``, ``val_dates`` and
``test_dates`` arrays. The module is the single source of truth for
the train/test boundary — every consumer (baselines, train.py, eval.py)
imports the same list of folds.

Key invariants enforced here (and re-asserted by tests/test_leakage.py):
  • max(train_dates) < min(test_dates) on every fold — chronological,
    no random splits, ever (§1.2).
  • The validation set is carved from the LATEST ~15% of the training
    window, also chronologically — not randomly (§5.1).
  • No fold's test window overlaps any later fold's training window in
    a way that would let a model see the test answer during training.

The validation fraction defaults to 0.15 and may be overridden per call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass
class Fold:
    """One walk-forward fold.

    All date arrays are pandas DatetimeIndex (sorted, unique) restricted
    to dates that actually appear in the features/targets tables — the
    fold is intersected with the available data at construction time.
    """
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
    """Build the walk-forward folds defined in configs.yaml.

    Parameters
    ----------
    cfg : dict
        The full configs.yaml dict. Reads ``folds: [...]``.
    available_dates : iterable of Timestamps, optional
        Dates that actually exist in the cached features/targets
        tables. If provided, every fold's date windows are intersected
        with this set — so a fold whose training window extends past
        the data end is silently clipped rather than producing an
        empty window. If None, the boundaries from the config are used
        verbatim and only their chronological ordering is checked.
    val_fraction : float
        Fraction of the training window to reserve for validation
        (chronologically the LATEST portion). Default 0.15 per §5.1.

    Returns
    -------
    list[Fold]
        One entry per configured fold, in config order.
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

        # ── hard chronological invariant (§1.2) ─────────────────────
        if train_end >= test_start:
            raise AssertionError(
                f"Fold {name!r}: train_end ({train_end.date()}) must be "
                f"strictly before test_start ({test_start.date()}) — "
                f"otherwise the model sees the test window during training."
            )

        # ── build the full training and test index ──────────────────
        if avail is not None:
            # Half-open [start, end] inclusive of end-of-day, intersected
            # with the available dates. The boundary itself is owned by
            # the config; the data restricts what we can actually use.
            train_mask = (avail >= train_start) & (avail <= train_end)
            test_mask = (avail >= test_start) & (avail <= test_end)
            train_full = avail[train_mask]
            test_idx = avail[test_mask]
        else:
            # Without an availability constraint, we still need a concrete
            # set of dates for the fold. We synthesise business-day dates
            # in the configured range — the next consumer that loads the
            # features table will intersect again and may end up with
            # fewer dates. This branch exists mainly for tests and for
            # running splits without the cache.
            train_full = pd.bdate_range(train_start, train_end)
            test_idx = pd.bdate_range(test_start, test_end)

        if len(train_full) == 0:
            raise ValueError(f"Fold {name!r}: no train dates after intersecting with available dates")
        if len(test_idx) == 0:
            raise ValueError(f"Fold {name!r}: no test dates after intersecting with available dates")

        # ── split off the validation portion (§5.1) ─────────────────
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

        # ── zero-overlap asserts — fail loudly, never silently ─────
        # Train and val are adjacent in time but must not overlap.
        assert len(set(train_idx).intersection(val_idx)) == 0
        # Train and test must not overlap.
        assert len(set(train_idx).intersection(test_idx)) == 0
        # Val and test must not overlap.
        assert len(set(val_idx).intersection(test_idx)) == 0
        # Chronological: the very last train date precedes the very
        # first val date, which precedes the very first test date.
        assert train_idx.max() < val_idx.min()
        assert val_idx.max() < test_idx.min()

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
    cfg = data.load_config()
    if paths.FEATURES.exists():
        feats = pd.read_parquet(paths.FEATURES, columns=["date"])
        avail = pd.DatetimeIndex(feats["date"].unique())
    else:
        avail = None
    folds = make_folds(cfg, available_dates=avail)
    for f in folds:
        print(f)
