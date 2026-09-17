#!/usr/bin/env bash
# Stage the Phase 3 code dataset for Kaggle upload.
#
# Copies every module the SINGLE-SESSION notebook imports into
# kaggle/code_phase3/ (the directory the notebook treats as the
# dmrgt-phase-3-code dataset root). Per §2 of the single-session rework,
# one committed run does the grid (Cells 2–3: train + sensitivity),
# the frozen-graph build (Cell 4: graph + splits), and the significance
# tests (Cell 6: stats + eval) in-session — so eval.py, graph.py,
# sensitivity.py, and stats.py ARE shipped. What stays local-only:
# baselines.py and compare.py — the floor models and the one-shot final
# report belong to the local epilogue on the downloaded outputs (§8),
# and data.py stays the slim config loader so no notebook path can ever
# fetch fresh data (Rule 9).
#
# Usage:
#   ./kaggle/build_code_dataset_phase3.sh            # stage files
#   ./kaggle/build_code_dataset_phase3.sh --upload  # stage + push via kaggle CLI
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$ROOT/kaggle/code_phase3"

CODE_FILES=(
  configs.yaml
  paths.py
  data.py
  splits.py
  model.py
  train.py
  eval.py
  graph.py
  sensitivity.py
  stats.py
  verify_env.py
)

echo "[stage] Phase 3 code dataset → $STAGE"
mkdir -p "$STAGE"
for f in "${CODE_FILES[@]}"; do
  if [[ ! -f "$ROOT/$f" ]]; then
    echo "[stage] MISSING: $f — write it before staging." >&2
    exit 1
  fi
  cp "$ROOT/$f" "$STAGE/$f"
  echo "  + $f"
done

if [[ "${1:-}" == "--upload" ]]; then
  # No --dir-mode: the staged directory is flat, and the CLI's default
  # (--dir-mode skip) uploads a flat directory's files individually.
  # --dir-mode zip would archive the directory into a single .zip instead,
  # leaving the modules IN the archive where `import paths` cannot find
  # them.
  echo "[stage] pushing new version of harishsivakumar1912/dmrgt-phase-3-code"
  kaggle datasets version -p "$STAGE" -m "Phase 3 code update" || {
      echo "[stage] kaggle CLI push failed — run manually:" >&2
      echo "  kaggle datasets version -p $STAGE -m 'Phase 3 code update'" >&2
      exit 1
    }
fi

echo "[stage] done. Files:"
ls -1 "$STAGE"

# §1.3 — print the fingerprint the Kaggle session will print in Cell 2.
# A dataset already attached to a notebook stays pinned to the version it
# was added at: pushing a new version does NOT update the attachment, and
# the session keeps running the old code with no error. After uploading,
# remove + re-add the dataset in Add Input, restart the session, and
# confirm verify_env's "[env] code fingerprint:" line matches this one.
# Byte-for-byte what verify_env.code_fingerprint() hashes: the file name,
# then the file's raw contents, per module, in CODE_FILES order. No
# newline stripping — `cat` emits exact bytes and `printf` adds none.
FP="$(for f in "${CODE_FILES[@]}"; do printf '%s' "$f"; cat "$ROOT/$f"; done \
      | shasum -a 256 | cut -d' ' -f1)"
echo
echo "[stage] code fingerprint: $FP"
echo "[stage]   ^ Cell 2 must print this exact value. If it differs, the"
echo "[stage]     session is running a STALE dataset version — remove and"
echo "[stage]     re-add dmrgt-phase-3-code from Add Input, then RESTART."