# ─── Cell 14 (S2): train P, B, C on the winning graph (arm A is reused from S1) ───
_t0 = time.time()
_jobs = arm_jobs(S2_ARMS, FOLD_NAMES, CONFIG["seeds"], GRAPH_WINNER)
# top up arm A to all 5 seeds if the winning variant ran with 3 (budget rule); same job shape as the grid runs
_jobs += [(PER_RUN[("A", f)], {"arm": "A", "fold": f, "seed": s, "graph": GRAPH_WINNER, "tag": A_TAG})
          for f in FOLD_NAMES for s in CONFIG["seeds"]
          if not d4_train.ckpt_paths(WORK, A_TAG, f, s)["done"].exists()]
launch_stage("S2", _jobs)

def tag_for(arm, fold):
    """Checkpoint tag of an arm on a fold: arm A on folds 1-4 is the winning grid variant's runs."""
    return A_TAG if (arm == "A" and fold in FOLD_NAMES) else arm

def seeds_for(fold):
    return list(CONFIG["seeds"]) if fold in FOLD_NAMES else list(HOLDOUT_SEEDS)

for _a in ["P", "A"] + S2_ARMS[1:]:
    for _f in FOLD_NAMES:
        for _s in CONFIG["seeds"]:
            assert d4_train.ckpt_paths(WORK, tag_for(_a, _f), _f, _s)["done"].exists(), (_a, _f, _s)
S2_SECONDS = time.time() - _t0
print(f"S2 done in {S2_SECONDS / 3600:.2f} h; arms P, A({A_TAG}), {S2_ARMS[1:]} complete (20 runs each)")
