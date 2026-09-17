"""DMRGT Phase 3 — random-init determinism tests (Rule 15).

Every training run must start from weights drawn from a SEEDED random
initialisation — never a loaded checkpoint, never an unseeded global
RNG. Byte-level fingerprints prove it:

  * same seed → identical fingerprints across constructions (the seed
    fully determines the init);
  * different seeds → different fingerprints (weights are really drawn,
    not constant);
  * the fingerprint is over raw parameter BYTES, not float sums — sums
    differ across devices because float addition is not associative,
    while .cpu() copies preserve the bit pattern exactly;
  * perturbing any single weight changes the fingerprint (nothing is
    silently excluded from the hash).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_same_seed_same_fingerprint():
    train.set_seed(0)
    net_a = train.make_model(n_features=8)
    fp_a = train._param_fingerprint(net_a)

    train.set_seed(0)
    net_b = train.make_model(n_features=8)
    fp_b = train._param_fingerprint(net_b)

    assert fp_a == fp_b, (
        "Same seed produced different parameter bytes — construction is "
        "not deterministic. Training runs would not be reproducible."
    )


def test_different_seed_different_fingerprint():
    train.set_seed(0)
    net_a = train.make_model(n_features=8)
    fp_a = train._param_fingerprint(net_a)

    train.set_seed(1)
    net_b = train.make_model(n_features=8)
    fp_b = train._param_fingerprint(net_b)

    assert fp_a != fp_b, (
        "Different seeds produced identical parameters — the model is "
        "ignoring the seed (or weights are constant)."
    )


def test_fingerprint_device_independent(device):
    """The fingerprint of a model built on CPU and moved to a device
    must equal the fingerprint of the same model on CPU — .cpu() in
    _param_fingerprint preserves bit patterns, so the hash is stable
    wherever the model lives."""
    if device.type == "cpu":
        pytest.skip("no CUDA device — CPU-vs-CUDA identity is vacuous here")
    train.set_seed(0)
    net = train.make_model(n_features=8)
    fp_cpu = train._param_fingerprint(net)
    net = net.to(device)
    fp_moved = train._param_fingerprint(net)
    assert fp_cpu == fp_moved, (
        "Fingerprint changed after moving the model across devices — "
        "it would spuriously fail on Kaggle's T4."
    )


def test_fingerprint_covers_all_parameters():
    """Perturbing ANY single parameter must change the fingerprint —
    no weight may be silently excluded from the hash."""
    train.set_seed(0)
    net = train.make_model(n_features=8)
    fp_before = train._param_fingerprint(net)

    with torch.no_grad():
        params = list(net.parameters())
        for i, p in enumerate(params):
            p.reshape(-1)[0] += 0.001  # one-weight perturbation
            fp_perturbed = train._param_fingerprint(net)
            assert fp_perturbed != fp_before, (
                f"Perturbing parameter #{i} did not change the "
                f"fingerprint — it is excluded from the hash."
            )
            p.reshape(-1)[0] -= 0.001  # restore for the next iteration


def test_all_five_seeds_reproducible():
    """The actual training seeds (configs.yaml training.seeds) each
    give a distinct, reproducible init."""
    import data
    import paths
    cfg = data.load_config(str(paths.CONFIG))
    fingerprints = {}
    for seed in cfg["training"]["seeds"]:
        train.set_seed(seed)
        fingerprints[seed] = train._param_fingerprint(
            train.make_model(n_features=8))
    # reproducible…
    for seed in cfg["training"]["seeds"]:
        train.set_seed(seed)
        again = train._param_fingerprint(train.make_model(n_features=8))
        assert again == fingerprints[seed], f"seed {seed} not reproducible"
    # …and mutually distinct
    values = list(fingerprints.values())
    assert len(set(values)) == len(values), "two seeds share an init"


def test_model_shapes():
    """The DMRGT forward contract used by train_one_seed."""
    train.set_seed(0)
    net = train.make_model(n_features=8)
    X = torch.randn(40, 30, 8)
    edge_index = torch.randint(0, 40, (2, 640))
    edge_type = torch.cat([
        torch.zeros(320, dtype=torch.long),
        torch.ones(160, dtype=torch.long),
        torch.full((160,), 2, dtype=torch.long),
    ])
    y = net(X, edge_index, edge_type)
    assert y.shape == (40,)
    # batched path (val/test prediction batching)
    Xb = torch.randn(8, 40, 30, 8)
    yb = net(Xb, edge_index, edge_type)
    assert yb.shape == (8, 40)