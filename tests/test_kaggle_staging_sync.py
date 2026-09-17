"""§7 — root vs. Kaggle staging directory drift.

Kaggle only ever executes the copy in `kaggle/code_phase3/`. A fix applied
at the project root and not re-staged changes nothing about what actually
runs, and reports no error while doing it — the run just silently keeps the
old behaviour. That cost a debugging cycle once.

This test makes forgetting impossible rather than merely unlikely: it
re-derives the staged-file list from the build script itself, so adding a
module to `build_code_dataset_phase3.sh` automatically extends the check.
Run `./kaggle/build_code_dataset_phase3.sh` to fix any reported drift.
"""

import hashlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "kaggle" / "code_phase3"
BUILD_SCRIPT = ROOT / "kaggle" / "build_code_dataset_phase3.sh"

# Modules the single-session notebook never imports are deliberately NOT
# staged: baselines.py (the floor models) and compare.py (the one-shot
# final report) belong to the LOCAL epilogue on the downloaded outputs
# (§8 of the single-session rework) — keeping them out of the session
# means the run's outputs come only from the notebook cells, and the
# once-only test scoring (Rule 18) cannot be re-run server-side by
# accident. `data.py` is the other deliberate asymmetry — it must stay
# the slim config loader so no notebook path can ever fetch fresh data
# (Rule 9). eval.py / graph.py / sensitivity.py / stats.py ARE staged
# now: Cells 3–6 run winner selection, the frozen-graph build, and the
# significance tests in-session (§2 of the rework). tune.py is gone —
# the post-processing tuning stage was dropped by the rework.
NEVER_STAGED = (
    "baselines.py",
    "compare.py",
)


def _staged_files_from_build_script() -> list[str]:
    """Parse CODE_FILES=( ... ) out of the staging script.

    Parsed, not hardcoded, so the test cannot fall out of step with the
    script it is policing.
    """
    text = BUILD_SCRIPT.read_text()
    match = re.search(r"CODE_FILES=\((.*?)\)", text, re.DOTALL)
    assert match, (
        f"could not find a CODE_FILES=( ... ) array in {BUILD_SCRIPT.name} — "
        f"if the staging mechanism changed, update this test with it"
    )
    names = [
        line.strip().strip('"').strip("'")
        for line in match.group(1).splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert names, f"CODE_FILES parsed empty from {BUILD_SCRIPT.name}"
    return names


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("name", _staged_files_from_build_script())
def test_root_and_staged_copy_are_identical(name: str) -> None:
    root_file = ROOT / name
    staged_file = STAGE / name

    assert root_file.exists(), f"{name} missing at the project root"
    assert staged_file.exists(), (
        f"{name} is not staged. Kaggle runs the staged copy only — "
        f"run ./kaggle/build_code_dataset_phase3.sh"
    )
    assert _sha256(root_file) == _sha256(staged_file), (
        f"{name} differs between the project root and kaggle/code_phase3/.\n"
        f"Kaggle executes the STAGED copy, so an edit at the root has no "
        f"effect on a run and raises no error.\n"
        f"Fix: ./kaggle/build_code_dataset_phase3.sh"
    )


def test_no_extra_files_staged() -> None:
    """The staged dir holds exactly the build script's list (+ metadata).

    Catches the other direction of drift: a module deleted from CODE_FILES
    but left behind on disk, which would still be shipped and still be
    importable inside the session.
    """
    allowed = set(_staged_files_from_build_script()) | {"dataset-metadata.json"}
    present = {
        p.name
        for p in STAGE.iterdir()
        if p.is_file() and not p.name.startswith(".")
    }
    unexpected = sorted(present - allowed)
    assert not unexpected, (
        f"unexpected files in {STAGE.relative_to(ROOT)}: {unexpected}. "
        f"Either add them to CODE_FILES in build_code_dataset_phase3.sh or "
        f"delete them — `kaggle datasets create` uploads the folder verbatim "
        f"(§9.3)."
    )


def test_local_only_modules_are_not_staged() -> None:
    """The local-epilogue boundary: the floor models and the one-shot
    final scorer stay out of the code dataset, so the committed session's
    outputs can only come from the notebook cells (§8 of the rework) and
    Rule 18's once-only scoring cannot be re-run server-side."""
    shipped = [m for m in NEVER_STAGED if (STAGE / m).exists()]
    assert not shipped, (
        f"{shipped} must NOT be in the Kaggle code dataset — baselines.py "
        f"and compare.py belong to the local epilogue on the downloaded "
        f"outputs (§8 of the single-session rework)."
    )


def test_staged_data_py_stays_a_slim_config_loader() -> None:
    """Rule 9: data.py must never contain fetch/download logic."""
    src = (STAGE / "data.py").read_text()
    forbidden = [
        "yfinance", "urllib", "requests", "subprocess",
        "read_csv", "to_parquet", "Ticker(",
    ]
    hits = [tok for tok in forbidden if tok in src]
    assert not hits, (
        f"staged data.py contains {hits} — it must stay the slim "
        f"load_config() + all_tickers() stub so no Kaggle code path can "
        f"fetch fresh data (Rule 9)."
    )
    assert "def load_config" in src and "def all_tickers" in src, (
        "staged data.py no longer exposes load_config/all_tickers"
    )


def test_staged_file_order_matches_verify_env() -> None:
    """CODE_FILES order must equal verify_env.STAGED_MODULES order.

    Both compute the §1.3 code fingerprint by concatenating name+bytes in
    their own order. Same files, different order, different hash — the
    operator would see a mismatch and go re-attach a dataset that was
    never stale. Pin the order here so it cannot drift apart.
    """
    import verify_env  # noqa: PLC0415  (imported for its constant only)

    assert _staged_files_from_build_script() == list(verify_env.STAGED_MODULES), (
        f"order mismatch:\n"
        f"  {BUILD_SCRIPT.name} CODE_FILES : {_staged_files_from_build_script()}\n"
        f"  verify_env.STAGED_MODULES      : {list(verify_env.STAGED_MODULES)}\n"
        f"The §1.3 code fingerprint concatenates name+bytes in this order, so "
        f"the two ends must agree exactly."
    )


def test_build_script_lists_every_module_the_notebook_imports() -> None:
    """The staged set must cover what the notebook actually imports.

    The notebook imports paths, verify_env, data, train (and train imports
    splits + model), plus — per §2 of the single-session rework —
    sensitivity (Cell 3), graph + splits (Cell 4), and stats + eval
    (Cell 6). A module dropped from CODE_FILES would surface as an
    ImportError mid-session rather than here.
    """
    needed = {"paths.py", "data.py", "train.py", "splits.py",
              "model.py", "verify_env.py", "configs.yaml",
              "eval.py", "graph.py", "sensitivity.py", "stats.py"}
    staged = set(_staged_files_from_build_script())
    missing = sorted(needed - staged)
    assert not missing, (
        f"{missing} are imported by the Kaggle notebook (directly or via "
        f"train.py) but are not in CODE_FILES — the session would fail with "
        f"an ImportError."
    )


def test_staged_tree_runs_with_nothing_but_the_dataset_on_the_path(
        tmp_path: Path) -> None:
    """Reproduce Cell 2 against a FAITHFUL Kaggle import environment.

    This exists because every check in this file passed while the Kaggle
    session died at Cell 2 with `ModuleNotFoundError: No module named
    'graph'`: verify_env.py imported graph.py, which is deliberately not
    shipped (Rule 9). Locally the repo root satisfied that import, so the
    gap was invisible from here — the staged tree has to be exercised in
    a copy with nothing else on sys.path for the difference to show.

    Any staged module importing a local module that is not staged fails
    here, whatever it is, rather than at the top of a paid GPU session.
    """
    import os
    import shutil
    import subprocess
    import sys

    for p in STAGE.iterdir():
        if p.is_file() and p.suffix in {".py", ".yaml"}:
            shutil.copy(p, tmp_path / p.name)
    # paths.py resolves CACHE_DIR next to itself in local mode; point that
    # at the real cache so importing it neither needs nor creates one.
    (tmp_path / "cache").symlink_to(ROOT / "cache")

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = subprocess.run(
        [sys.executable, "-c",
         "import paths, data, splits, model, train, verify_env, eval, "
         "graph, sensitivity, stats\n"
         "verify_env.check_versions()\n"
         "verify_env.check_bootstrap()\n"
         "print('[sim] staged tree imported and ran cleanly')"],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    assert proc.returncode == 0, (
        f"the staged tree does not run on its own.\n"
        f"A Kaggle session has ONLY the code dataset on sys.path, so every "
        f"module it imports must be in CODE_FILES — either stage the "
        f"missing module or make the import optional.\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
