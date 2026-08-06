#!/bin/bash
# Assemble the dmrgt-code Kaggle dataset from root source files.
# Run from the project root before `kaggle datasets version -p kaggle/code/`.
#
# The kaggle/code/ directory tracks only the files that DIFFER from root:
#   - data.py (slimmed down — only load_config + all_tickers)
#   - dataset-metadata.json
#
# Everything else is copied from root at upload time so there is exactly
# one copy of each source file in the repo.

set -euo pipefail
cd "$(dirname "$0")/.."

FILES=("configs.yaml" "model.py" "paths.py" "splits.py" "train.py")

for f in "${FILES[@]}"; do
    cp "$f" "kaggle/code/$f"
    echo "  copied $f -> kaggle/code/$f"
done

echo "Done. Now run: kaggle datasets version -p kaggle/code/ -m '<message>'"
