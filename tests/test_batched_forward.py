"""DMRGT Phase 3 — batched message passing ≡ per-sample message passing.

`DMRGT.forward_with_edge_lists` exists so one optimizer step costs ONE
R-GCN call instead of one per date. The original loop over `range(B)` was
launch-bound — ~640 edges behind each call — and made the 140-run grid
take ~16 h against a 12 h Kaggle session cap.

That rewrite is only safe if it is arithmetically the same model, so the
per-sample loop is kept here as the reference implementation and the two
are compared directly. The interesting failure modes are the ones an
offset bug would produce, and they are all silent:

  * a wrong offset makes sample b's nodes read sample b±1's neighbours,
    so predictions are near-right and the training curve looks normal;
  * a lost or duplicated edge changes degree for a subset of nodes;
  * ragged batches (edge counts differing per date) are where a
    naive reshape breaks.

Tolerances rather than exact equality: the batched call is one larger
GEMM, so a given dot product may accumulate in a different order. The
arithmetic is the same; the last bits need not be.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import model as model_mod  # noqa: E402
import train  # noqa: E402

N_STOCKS = 40
LOOKBACK = 30
N_FEATURES = 8


def _ragged_edges(
    n_samples: int, n_nodes: int = N_STOCKS, seed: int = 0
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Per-sample edge lists with a DIFFERENT edge count each — the real
    situation (k_pos=6 vs k_pos=10, and dates whose history is short)."""
    g = torch.Generator().manual_seed(seed)
    eis, ets = [], []
    for b in range(n_samples):
        k_pos, k_neg = 8 + (b % 3), 4 + (b % 2)
        ei = torch.randint(0, n_nodes, (2, (k_pos + k_neg) * n_nodes + 160),
                           generator=g)
        et = torch.cat([
            torch.zeros(k_pos * n_nodes, dtype=torch.long),
            torch.ones(k_neg * n_nodes, dtype=torch.long),
            torch.full((160,), 2, dtype=torch.long),
        ])
        eis.append(ei)
        ets.append(et)
    return eis, ets


def test_batched_matches_per_sample_loop_with_ragged_edges() -> None:
    """The regression that matters: batched ≡ loop, on ragged edge counts."""
    train.set_seed(0)
    net = train.make_model(n_features=N_FEATURES).eval()  # eval: no dropout
    B = 8
    torch.manual_seed(1)
    X = torch.randn(B, N_STOCKS, LOOKBACK, N_FEATURES)
    eis, ets = _ragged_edges(B)

    with torch.no_grad():
        batched = net.forward_with_edge_lists(X, eis, ets)
        looped = torch.cat(
            [net(X[b: b + 1], eis[b], ets[b]) for b in range(B)], dim=0
        )

    assert batched.shape == (B, N_STOCKS)
    assert torch.allclose(batched, looped, atol=1e-5, rtol=1e-4), (
        f"batched and per-sample message passing disagree "
        f"(max abs diff {(batched - looped).abs().max().item():.3e}). "
        f"A node-offset bug reads the wrong sample's neighbours and is "
        f"otherwise invisible — check stack_per_sample_edges."
    )


def test_batched_does_not_leak_across_samples() -> None:
    """Changing sample 0's edges must move ONLY sample 0's output.

    Sharper than an allclose: a correct offset keeps samples disjoint,
    while an offset that is off by one shifts every sample's neighbours
    and can still land within a loose tolerance.
    """
    train.set_seed(0)
    net = train.make_model(n_features=N_FEATURES).eval()
    B = 4
    torch.manual_seed(2)
    X = torch.randn(B, N_STOCKS, LOOKBACK, N_FEATURES)
    eis, ets = _ragged_edges(B, seed=3)

    with torch.no_grad():
        before = net.forward_with_edge_lists(X, eis, ets)
        # Rewire sample 0 only, keeping its edge COUNT (so the ragged
        # layout is unchanged and only the wiring differs).
        tampered = [eis[0].flip(1)] + eis[1:]
        after = net.forward_with_edge_lists(X, tampered, ets)

    moved = (before - after).abs().amax(dim=1) > 1e-6
    assert moved[0], "sample 0's own output did not respond to its edges"
    assert not moved[1:].any(), (
        f"rewiring sample 0 moved samples "
        f"{moved.nonzero().flatten().tolist()} — samples are not disjoint, "
        f"so the batch offset is wrong"
    )


def test_forward_applies_one_edge_list_to_every_sample() -> None:
    """The B>1 path of forward(): the SAME edges for all samples must
    equal B independent single-sample calls with those edges."""
    train.set_seed(0)
    net = train.make_model(n_features=N_FEATURES).eval()
    B = 5
    torch.manual_seed(4)
    X = torch.randn(B, N_STOCKS, LOOKBACK, N_FEATURES)
    g = torch.Generator().manual_seed(5)
    ei = torch.randint(0, N_STOCKS, (2, 640), generator=g)
    et = torch.cat([torch.zeros(320, dtype=torch.long),
                    torch.ones(160, dtype=torch.long),
                    torch.full((160,), 2, dtype=torch.long)])

    with torch.no_grad():
        shared = net(X, ei, et)
        singles = torch.cat(
            [net(X[b], ei, et).unsqueeze(0) for b in range(B)], dim=0
        )

    assert shared.shape == (B, N_STOCKS)
    assert torch.allclose(shared, singles, atol=1e-5, rtol=1e-4)


def test_edge_list_count_must_match_the_batch() -> None:
    """A silently-short list would apply the wrong graph to the tail of
    the batch, so it has to raise."""
    net = train.make_model(n_features=N_FEATURES).eval()
    torch.manual_seed(6)
    X = torch.randn(3, N_STOCKS, LOOKBACK, N_FEATURES)
    eis, ets = _ragged_edges(2)
    with pytest.raises(ValueError, match="one per sample"):
        net.forward_with_edge_lists(X, eis, ets)


def test_batched_path_is_reproducible_under_a_seed() -> None:
    """dropout must consume the RNG stream identically run to run.

    train_one_seed seeds once per (fold, seed) and then trains; if the
    batched path drew RNG in a batch-size-dependent order, a re-run at a
    different batch size would silently stop reproducing.
    """
    outs = []
    for _ in range(2):
        train.set_seed(7)
        net = train.make_model(n_features=N_FEATURES)  # train() mode: dropout on
        X = torch.randn(4, N_STOCKS, LOOKBACK, N_FEATURES)
        eis, ets = _ragged_edges(4)
        with torch.no_grad():
            outs.append(net.forward_with_edge_lists(X, eis, ets).clone())
    assert torch.equal(outs[0], outs[1]), (
        "two identically-seeded batched forwards differ — the RNG stream "
        "is not reproducible"
    )


@pytest.mark.parametrize("bad_shape", [(N_STOCKS, LOOKBACK)])
def test_bad_input_rank_still_raises(bad_shape: tuple[int, ...]) -> None:
    net = train.make_model(n_features=N_FEATURES).eval()
    X = torch.randn(*bad_shape)
    ei = torch.randint(0, N_STOCKS, (2, 640))
    et = torch.zeros(640, dtype=torch.long)
    with pytest.raises(ValueError, match="Expected X"):
        net.forward_with_edge_lists(X, [ei], [et])
