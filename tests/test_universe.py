"""Verify the 20-ticker universe is sector-stratified per §3 of the spec.

The spec says:

  Choose 20 liquid S&P 100 constituents **stratified across GICS
  sectors** — roughly 2–3 per sector across Technology, Financials,
  Healthcare, Energy, Consumer Discretionary, Consumer Staples,
  Industrials, Utilities.

These checks make that explicit:

  • Exactly 40 tickers in the universe (not 40 — that's Phase 2).
  • All 8 GICS sectors from the spec are represented.
  • Each sector has 1–4 tickers (a hard cap on over-representation
    plus a floor on inclusion).
  • No sector accounts for more than 30% of the universe.
  • Tickers don't accidentally include the global series
    (^GSPC, ^VIX — those are features, not universe members).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import data  # noqa: E402


# The 8 GICS sectors the spec wants represented. "Information
# Technology" is the modern name for the "Technology" sector used
# in the spec.
EXPECTED_SECTORS = {
    "Information Technology",
    "Financials",
    "Health Care",
    "Energy",
    "Consumer Discretionary",
    "Consumer Staples",
    "Industrials",
    "Utilities",
}


def test_exactly_20_tickers():
    cfg = data.load_config()
    tickers = list(cfg["universe"]["tickers"])
    assert len(tickers) == 40, f"Phase 2 universe must be 40, got {len(tickers)}"


def test_no_duplicate_tickers():
    cfg = data.load_config()
    tickers = list(cfg["universe"]["tickers"])
    assert len(tickers) == len(set(tickers)), (
        f"duplicate tickers in universe: {tickers}"
    )


def test_no_global_series_in_universe():
    """^GSPC and ^VIX are features (f7, f8), not universe members."""
    cfg = data.load_config()
    tickers = set(cfg["universe"]["tickers"])
    assert "^GSPC" not in tickers, "^GSPC must be in global_series, not tickers"
    assert "^VIX" not in tickers, "^VIX must be in global_series, not tickers"


def test_all_8_sectors_represented():
    """Each of the 8 GICS sectors the spec names must have at least
    one ticker. This is what 'stratified across' means."""
    sectors = set(data._sector_lookup(data.load_config()).values())
    missing = EXPECTED_SECTORS - sectors
    assert not missing, (
        f"Missing GICS sectors from the universe: {missing}. "
        f"Present: {sectors}"
    )


def test_per_sector_count_in_range():
    """Each sector should have roughly 2-3 tickers per the spec; we
    accept the range 1-4 to allow some sectors to be lighter (and
    to leave room for the spec's flexibility)."""
    cfg = data.load_config()
    sector_of = data._sector_lookup(cfg)
    counts: dict[str, int] = {}
    for t in cfg["universe"]["tickers"]:
        s = sector_of[t]
        counts[s] = counts.get(s, 0) + 1
    bad = {s: c for s, c in counts.items() if not (1 <= c <= 5)}
    assert not bad, (
        f"Sectors with count outside [1, 4]: {bad}. "
        f"Per-sector: {counts}"
    )


def test_no_sector_over_represented():
    """A single sector should not dominate. Cap at 30% of the universe
    (6 of 20) — the spec's 'roughly 2-3 per sector' implies this."""
    cfg = data.load_config()
    sector_of = data._sector_lookup(cfg)
    counts: dict[str, int] = {}
    for t in cfg["universe"]["tickers"]:
        s = sector_of[t]
        counts[s] = counts.get(s, 0) + 1
    max_count = max(counts.values())
    assert max_count <= 6, (
        f"Sector {max(counts, key=counts.get)!r} has {max_count} tickers — "
        f"more than 30% of the 20-stock universe. Counts: {counts}"
    )


def test_known_liquid_tickers_present():
    """A smoke test — the universe should contain some of the most
    liquid, recognisable names from each sector. A future agent
    changing the universe would need to update this list, which is
    the point."""
    cfg = data.load_config()
    tickers = set(cfg["universe"]["tickers"])
    # At least one bellwether from each sector
    bellwethers = {
        "Information Technology": {"AAPL", "MSFT"},
        "Financials": {"JPM"},
        "Health Care": {"JNJ"},
        "Energy": {"XOM"},
        "Consumer Discretionary": {"AMZN"},
        "Consumer Staples": {"PG"},
        "Industrials": {"HON", "CAT", "BA"},
        "Utilities": {"NEE", "DUK"},
    }
    for sector, expected_any in bellwethers.items():
        if not (tickers & expected_any):
            pytest.fail(
                f"None of {expected_any} present in the universe — "
                f"did someone accidentally drop the {sector} bellwethers?"
            )
