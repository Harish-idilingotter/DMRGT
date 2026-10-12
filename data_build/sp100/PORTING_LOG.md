# S&P 100 cache build — porting log

All pipeline code in `src/` is copied from git, not rewritten. `diff` against the source commit was run and
printed no differences for the first three files.

| Piece | Source (commit : file) | Copied to | Diff |
|---|---|---|---|
| Price download, calendar alignment, validation, `download_prices`, `build_targets`, hashes helpers | `fd8125d` : `data.py` | `src/data.py` | **identical** |
| 8 features (`build_features`, `FEATURE_COLUMNS`) | `fd8125d` : `features.py` | `src/features.py` | **identical** |
| Graph builder (`build_daily_graph_all_dates`, `_edges_for_returns_window`, `_sector_edges`, `_validate_edge_frame`) | `175768d` (tag `phase3-complete`) : `graph.py` | `src/graph.py` | **identical** |
| Config (universe, sectors, features periods, graph, folds) | `fd8125d` : `configs.yaml` (universe/sectors/graph/folds equal to the Phase 3 file) | `configs_40.yaml` | identical |
| Paths | Phase 2/3 `paths.py` | `src/paths.py` | **replaced** (output dir and config from env vars; same constant names the ported modules read) |

Why `fd8125d`: it is the last commit that contains `data.py` and `features.py` in full (Phase 3 reduced
`data.py` to a config loader and deleted `features.py`). The Phase 3 cache's prices, features and targets were
built with that code; its graphs with the Phase 3 `graph.py`.

## Non-verbatim pieces (all outside `src/`)

* `build.py` — driver; calls the verbatim functions in order. Two adaptations: a yfinance shim
  (`--yf-cache`) so the verbatim `download_prices` runs on the pre-fetched, screened frames, and
  `--relax-sector-cap` (S&P 100 only) which replaces `data._sector_lookup` for the printed summary because the
  40-stock guard "no sector above 20%" cannot hold for real GICS weights (IT is 21/95).
* `universe.py` — new (spec §2 rules). `regression.py`, `validate.py` — new (spec §3, §5).

## Results

* §3 regression: `regression_test_40.json`. The port is **bit-exact** on the original prices (features, targets
  and graphs max difference 0.0). A fresh download differs only through Yahoo revising adjusted prices
  (max 1.6%, median 0.27% relative), which is the case the spec says to document and proceed on.
* §5 validation: `out_sp100/validation_report.json`, all eight checks pass.
* Universe: 95 of 101 snapshot rows (excluded: GOOG second class; DELL, GEV, PLTR, UBER listed after start; SNDK no data).
