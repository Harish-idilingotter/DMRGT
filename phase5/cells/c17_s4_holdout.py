# ─── Cell 17 (S4): holdout training — winner, A, P on folds 5-6 (after all four locks exist; test still unread) ───
_t0 = time.time()
assert d4_data.test_unlocked(), "S4 runs only after all four lock files exist"
_jobs = arm_jobs([WINNER, "A", "P"], HOLDOUT_NAMES, HOLDOUT_SEEDS, GRAPH_WINNER)
launch_stage("S4", _jobs)
for _a in (WINNER, "A", "P"):
    for _f in HOLDOUT_NAMES:
        for _s in HOLDOUT_SEEDS:
            assert d4_train.ckpt_paths(WORK, _a, _f, _s)["done"].exists(), (_a, _f, _s)
S4_SECONDS = time.time() - _t0
print(f"S4 done in {S4_SECONDS / 3600:.2f} h")
