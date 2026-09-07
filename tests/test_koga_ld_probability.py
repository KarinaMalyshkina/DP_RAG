"""The closed-form LD-Gumbel selection probability must match the mechanism.

``membership_score`` used to estimate P('Yes') by drawing the mechanism 200 times.
That resolves nothing finer than 1/200 while the real values are ~1e-4 and below, so
every target scored exactly 0.0 and AUC came out exactly 0.500 — indistinguishable
from a perfect defence. It was also the whole cost of a Koga cell (185 s per target,
because each draw adds Gumbel noise across a 248k-token vocabulary).

The replacement reads the distribution analytically. These tests check it against a
direct simulation of the upstream rule and against the properties the attack metrics
depend on. No model is loaded: only the histogram maths is exercised.
"""

from __future__ import annotations

import math

import pytest
import torch

from dp_rag_eval.pipelines.koga_pipeline import KogaPipeline


class _Bare:
    """Just enough of the pipeline to call the probability helper."""

    def __init__(self, vocab_size: int, n_split: int, eps: float, delta: float, yes_ids: list[int]) -> None:
        self.vocab_size = vocab_size
        self.n_split = n_split
        self.dp_eps = eps
        self.dp_delta = delta
        self._yes_ids = yes_ids

    log_yes_probability = KogaPipeline._ld_yes_log_probability


def simulate(votes: torch.Tensor, vocab: int, n_split: int, eps: float, delta: float, yes: list[int], draws: int):
    """Draw the upstream rule directly: Gumbel(0, 1/eps) on the top-k plus reject."""
    counts = torch.bincount(votes, minlength=vocab).to(torch.float64)
    sorted_counts, sorted_idx = torch.sort(counts, descending=True)
    h_perp = float(sorted_counts[n_split])
    h_perp += 1.0 + math.log(min(2.0, n_split, vocab - n_split) / delta) / eps
    values = torch.cat([sorted_counts[:n_split], torch.tensor([h_perp], dtype=torch.float64)])

    gumbel = torch.distributions.Gumbel(0.0, 1.0 / eps)
    noisy = values.unsqueeze(0) + gumbel.sample((draws, values.numel())).to(torch.float64)
    winners = noisy.argmax(dim=1)
    yes_positions = {i for i in range(n_split) if int(sorted_idx[i]) in set(yes)}
    return sum(1 for w in winners.tolist() if w in yes_positions) / draws


def score(votes, *, vocab=1000, n_split=16, eps=1.0, delta=1e-6, yes=(7,)) -> float:
    """P(Yes) on the probability scale, for tests written against probabilities.

    The helper now returns a logarithm — the probability spans hundreds of orders of
    magnitude and float64 cannot hold the small end — so these tests exponentiate it
    back. That is safe here: the cases are constructed with a few voters and a wide
    epsilon, where the value is nowhere near underflow.
    """
    log_p = _Bare(vocab, n_split, eps, delta, list(yes)).log_yes_probability(torch.tensor(votes))
    return math.exp(log_p)


def log_score(votes, *, vocab=1000, n_split=16, eps=1.0, delta=1e-6, yes=(7,)) -> float:
    return _Bare(vocab, n_split, eps, delta, list(yes)).log_yes_probability(torch.tensor(votes))


# ---------------------------------------------------------------------------
# Agreement with the mechanism
# ---------------------------------------------------------------------------


def test_closed_form_matches_a_direct_simulation():
    # eps large enough that the reject candidate does not dominate, so the sampled
    # estimate has something to resolve.
    torch.manual_seed(0)
    vocab, n_split, eps, delta, yes = 200, 8, 4.0, 1e-3, [7]
    votes = torch.tensor([7] * 6 + [3, 5])

    exact = math.exp(_Bare(vocab, n_split, eps, delta, yes).log_yes_probability(votes))
    sampled = simulate(votes, vocab, n_split, eps, delta, yes, draws=40000)

    assert exact == pytest.approx(sampled, abs=0.01)


def test_it_is_a_probability_and_monotone_in_the_vote_lead():
    # More voters agreeing on 'Yes' must never lower P('Yes').
    previous = -1.0
    for n_yes in range(0, 17):
        value = score([7] * n_yes + [3] * (16 - n_yes))
        assert 0.0 <= value <= 1.0
        assert value >= previous - 1e-15
        previous = value


def test_more_yes_votes_beat_fewer():
    assert score([7] * 12 + [3] * 4) > score([7] * 4 + [3] * 12)


# ---------------------------------------------------------------------------
# The properties the attack metrics need
# ---------------------------------------------------------------------------


def test_resolution_survives_where_monte_carlo_collapsed():
    """Distinct vote leads give distinct scores even when all are far below 1/200."""
    values = [score([7] * n + [3] * (16 - n), eps=0.35, delta=3.1e-8) for n in (1, 2, 3, 4)]

    assert all(v < 5e-3 for v in values), "expected the regime where 200 draws resolve nothing"
    assert len(set(values)) == len(values), "scores must still be distinguishable"
    assert values == sorted(values)


def test_a_yes_token_with_no_votes_keeps_a_tiny_but_real_chance():
    """Faithful to the mechanism, and easy to mistake for a bug.

    The limited domain is the top ``k_bar`` *slots*, not the tokens that were voted
    for. With 16 voters all choosing one token, the remaining slots are filled by
    zero-count tokens — and the mechanism adds Gumbel noise to those too, so any of
    them can win. A Yes token sitting in one of those slots therefore has a nonzero
    selection probability even with no votes at all. Bounding how many such tokens
    compete is precisely what the limited domain is for.

    Asserted on the log scale, which is where the claim can actually be checked.
    On the probability scale this read ``0.0 < no_votes`` and passed or failed with
    the platform: the value is small enough that one machine returned a denormal and
    another returned exactly zero. A finite logarithm says "possible but unlikely"
    on every machine, which is what the mechanism guarantees.
    """
    no_votes = log_score([3] * 16)
    with_votes = log_score([7] * 8 + [3] * 8)

    assert math.isfinite(no_votes)
    assert no_votes < math.log(1e-6)
    assert with_votes > no_votes


def test_zero_when_no_yes_token_is_inside_the_domain():
    # k_bar=2 leaves only the two most-voted tokens in the domain, neither of them 7.
    assert log_score([3] * 8 + [5] * 8, n_split=2) == float("-inf")


def test_no_yes_token_ids_gives_zero():
    assert log_score([7] * 16, yes=()) == float("-inf")


def test_tighter_epsilon_lowers_the_score():
    """More noise must reduce leakage, not merely rescale it."""
    strong = score([7] * 8 + [3] * 8, eps=0.1)
    weak = score([7] * 8 + [3] * 8, eps=8.0)

    assert strong < weak


def test_several_yes_spellings_are_summed():
    one = score([7] * 8 + [9] * 8, yes=(7,))
    both = score([7] * 8 + [9] * 8, yes=(7, 9))

    assert both > one
    assert both == pytest.approx(2 * one, rel=1e-9)


def test_handles_a_realistic_vocabulary_without_overflow():
    votes = torch.tensor([2665] * 100 + [9454] * 28)
    value = _Bare(248320, 128, 0.35, 3.1e-8, [9454]).log_yes_probability(votes)

    # Finite and negative: a log-probability. The point of the case is a 248k-token
    # vocabulary and 128 voters, where the softmax must not overflow — and where the
    # probability itself is far below what float64 can represent.
    assert math.isfinite(value)
    assert value <= 0.0
