"""CPU tests for Koga et al. DP mechanisms (from original DPRAG repo)."""

import torch
import pytest


@pytest.fixture
def koga(koga_available):
    """Import Koga wrapper (skips if baseline not cloned)."""
    from dp_rag_eval.wrappers import koga_imports
    return koga_imports


class TestLDGumbelMechanism:
    """Tests for the Limited Domain Gumbel Mechanism."""

    def test_selects_dominant_token_at_high_eps(self, koga):
        """At very high epsilon (low noise), the mechanism should select the dominant token."""
        engine = koga.LDGumbelMechanism(eps=100.0, delta=1e-5)
        # Histogram: token 0 has 50 votes, token 1 has 1 vote
        cnts = torch.tensor([50, 1], dtype=torch.float)
        result = engine.get_top1(cnts, dim=100, k_bar=10)
        # With eps=100 the noise is negligible, so token 0 (index 0) should win
        assert result == 0

    def test_budget_tracking_grows(self, koga):
        """get_dp_expense() should return increasing epsilon after each call."""
        engine = koga.LDGumbelMechanism(eps=50.0, delta=1e-5, compose_method="rdp")
        # Large dominant count + high eps ensures get_top1 succeeds deterministically
        cnts = torch.tensor([100, 1, 1, 1, 1], dtype=torch.float)

        engine.get_top1(cnts, dim=1000, k_bar=10)
        eps_after_1, delta_after_1 = engine.get_dp_expense()

        engine.get_top1(cnts, dim=1000, k_bar=10)
        eps_after_2, delta_after_2 = engine.get_dp_expense()

        assert eps_after_2 > eps_after_1
        assert eps_after_1 > 0
        assert engine.total_queries == 2

    def test_budget_overflow_raises(self, koga):
        """When target budget is exhausted, DPExpenseOverflow should be raised."""
        engine = koga.LDGumbelMechanism(
            eps=1.0, delta=1e-5,
            target_eps=0.001,  # very tight budget
            target_delta=1e-5,
            delta1=1e-5,
        )
        cnts = torch.tensor([10, 1], dtype=torch.float)
        with pytest.raises(koga.DPExpenseOverflow):
            for _ in range(1000):
                engine.get_top1(cnts, dim=100, k_bar=10)


class TestExpMechanism:
    """Tests for the Exponential Mechanism."""

    def test_selects_highest_at_high_eps(self, koga):
        """At very high epsilon, ExpMechanism should select items by true score order."""
        mech = koga.ExpMechanism(eps=100.0, target_eps=200.0, target_delta=1e-5)
        scores = torch.tensor([1.0, 5.0, 3.0, 2.0, 4.0])
        top2 = mech.get_topk(scores, k=2)
        # With eps=100, noise is negligible. Top-2 by score: indices 1 (5.0) and 4 (4.0)
        top2_set = set(top2.tolist())
        assert 1 in top2_set  # score 5.0


class TestMajorityVote:
    """Tests for the majority_vote function."""

    def test_picks_dominant_token(self, koga):
        """majority_vote should return the token with the most votes."""
        tokens = torch.tensor([42, 42, 42, 7, 7, 99])
        winner, votes = koga.majority_vote(tokens, dim=100)
        assert winner.item() == 42
        assert votes is not None

    def test_dp_vote_at_high_eps(self, koga):
        """With DP engine at high epsilon, should still pick the dominant token."""
        engine = koga.LDGumbelMechanism(eps=100.0, delta=1e-5)
        tokens = torch.tensor([42, 42, 42, 42, 42, 7])
        winner, votes = koga.majority_vote(tokens, dim=100, dp_engine=engine)
        assert winner.item() == 42


class TestRDPComposition:
    """Tests for the RDP-based privacy accounting."""

    def test_composition_returns_finite_positive_eps(self, koga):
        """compose_subsampled_LimitedDomain_to_approxDP should return finite eps > 0."""
        eps, delta = koga.compose_subsampled_LimitedDomain_to_approxDP(
            eps=1.0,
            delta=1e-6,
            total_delta=None,
            subsampling_rate=1.0,
            niter=10,
            delta1=1e-4,
        )
        assert 0 < eps < float("inf")
        assert delta > 0
