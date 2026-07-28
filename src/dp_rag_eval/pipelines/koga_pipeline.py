"""Koga DPVoteRAG baseline behind a thin run interface (HF backend).

The upstream code drives token-by-token voting through vLLM
(``ensemble_generate_token_ids_vllm``). vLLM is not available on CPU/Windows, so
this wrapper re-implements *only the generation backend* — a token-by-token loop
over voter prompts using HuggingFace ``forward`` for greedy (temperature=0)
next-token logits. The DP-sensitive parts are the **original** baseline code:

* ``majority_vote`` / ``get_majority_token_vote`` — vote aggregation, fail modes.
* ``LDGumbelMechanism``                          — LD-Gumbel noise + RDP accounting.

Retrieval has no DP in Koga's method (privacy is enforced at generation), so we
use a plain dense top-(n_split*n_docs) retrieval to pick the voter documents.

All voters are driven as a **single padded batch** sharing one KV cache: the vote
at each step needs every voter's next-token argmax, so one forward over the batch
replaces ``n_split`` sequential forwards.

Three membership scores are exposed, because they answer different questions:

* :meth:`membership_score`   — samples the *actual* LD-Gumbel mechanism, so it
  responds to epsilon. The primary DP-faithful signal.
* :meth:`membership_score_upper_bound` — raw voter logits, before the noise.
  An adversarial upper bound; epsilon-independent by construction.
* :meth:`answer`             — the black-box setting of the paper (parse Yes/No).
"""

from __future__ import annotations

import contextlib
import io
import logging
from typing import Any

import torch

from dp_rag_eval.core.types import CalibrationResult, Document, PrivacyBudget, PrivacyTarget
from dp_rag_eval.pipelines.loading import (
    Embedder,
    chat_prompt,
    forward_logits,
    yes_no_token_ids,
    yes_probability,
)
from dp_rag_eval.wrappers.koga_imports import LDGumbelMechanism, get_majority_token_vote, majority_vote

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _quiet_votes():
    """Swallow the baseline's per-failure ``print`` calls.

    ``get_majority_token_vote`` prints "LD failed. Stop." every time the
    Limited-Domain selection finds no valid top-1. That is a normal, frequent
    outcome at small epsilon — and :meth:`KogaPipeline.membership_score` triggers it
    up to ``mc_samples`` times per query, which at sweep scale means millions of
    lines of stdout. Failures are counted by the caller (a ``None`` token) and
    reported through ``logging`` instead.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        yield


class KogaPipeline:
    """Run Koga's DP token-voting end to end on a corpus of :class:`Document`."""

    name = "koga"

    def __init__(
        self,
        generator: Any,
        tokenizer: Any,
        embedder: Embedder,
        *,
        n_split: int = 16,
        n_docs: int = 1,
        max_new_tokens: int = 12,
        # Per-token budget chosen so the Limited-Domain threshold (~log(1/delta)/eps
        # votes) sits low enough that a handful of agreeing voters can emit a token
        # on this small smoke setup. This is a demonstrative budget, not a tight one.
        dp_eps: float = 4.0,
        dp_delta: float = 1e-5,
        target_eps: float = 500.0,
        target_delta: float = 1e-2,
        dp: bool = True,
        mc_samples: int = 200,
        device: str = "cpu",
    ) -> None:
        """Args:
        dp: When ``False`` the LD-Gumbel mechanism is skipped and the winning token
            is the plain plurality of voter argmaxes. This is Koga's non-private
            VoteRAG baseline — the ``epsilon = infinity`` reference point.
        mc_samples: Number of independent draws of the DP vote used by
            :meth:`membership_score` to estimate P('Yes') under the mechanism.
        """
        self.model = generator
        self.tokenizer = tokenizer
        self.embedder = embedder
        self.n_split = n_split
        self.n_docs = n_docs
        self.max_new_tokens = max_new_tokens
        self.dp_eps = dp_eps
        self.dp_delta = dp_delta
        self.target_eps = target_eps
        self.target_delta = target_delta
        self.dp = dp
        self.mc_samples = mc_samples
        self.device = device
        # Limited-domain residual delta. Upstream defaults it to dp_delta; calibrate()
        # sets it explicitly so the derived total delta lands on the requested value.
        self.delta1: float | None = None
        self._calibration: CalibrationResult | None = None
        self.vocab_size = int(generator.config.vocab_size)
        # LD-Gumbel failure bookkeeping, surfaced per experiment cell. A high rate means
        # the Limited-Domain threshold (~log(2/delta)/eps votes) exceeds what n_split
        # voters can supply, so the mechanism returns nothing rather than a noisy answer.
        # The attack then scores every target 0 and AUC collapses to 0.5 — which looks
        # like "DP worked" but is really "the configuration was infeasible".
        self.ld_draws = 0
        self.ld_failures = 0
        self._texts: list[str] = []
        self._doc_embeddings: Any = None
        self._yes_ids: list[int] | None = None
        self._no_ids: list[int] | None = None

    # Above this share of the corpus, retrieval stops being a selection: nearly every
    # document ends up in some voter's context regardless of the query.
    SELECTIVITY_LIMIT = 0.25

    def index(self, documents: list[Document]) -> None:
        """Embed and cache the corpus once so many queries can be answered cheaply."""
        self._texts = [doc.text for doc in documents]
        self._doc_embeddings = self.embedder.encode(self._texts) if self._texts else None

        needed = self.n_split * self.n_docs
        if documents and needed > self.SELECTIVITY_LIMIT * len(documents):
            # Not a privacy problem — the per-document guarantee is unaffected — but it
            # undermines the comparison, since the pipeline is then answering from a
            # large slice of the corpus rather than from retrieved context.
            logger.warning(
                "n_split*n_docs=%d pulls %.0f%% of the %d-document corpus into context; "
                "retrieval is no longer selective. Lower the voter count or index a larger corpus.",
                needed,
                100 * needed / len(documents),
                len(documents),
            )

    def _retrieve(self, query: str, k: int) -> list[str]:
        """Plain (non-DP) dense top-k retrieval — Koga adds no noise at retrieval."""
        query_embedding = self.embedder.encode([query])
        scores = (query_embedding @ self._doc_embeddings.T)[0]
        k = min(k, len(self._texts))
        top = torch.topk(scores, k).indices.tolist()
        return [self._texts[i] for i in top]

    def _build_voter_prompts(self, docs: list[str], question: str) -> list[str]:
        """Split retrieved docs into ``n_split`` voter groups and chat-format prompts.

        Each voter is given its document group as context via the model's chat
        template, so an instruct model produces an actual answer turn instead of an
        immediate end-of-text token.
        """
        question = question[:300]
        prompts: list[str] = []
        for split_id in range(self.n_split):
            group = docs[split_id * self.n_docs : (split_id + 1) * self.n_docs]
            if not group:  # corpus smaller than n_split*n_docs — reuse first doc
                group = docs[:1]
            context = " ".join(f"{i + 1}. {text}" for i, text in enumerate(group))
            messages = [
                {"role": "system", "content": "Give a short answer to the question based only on the context."},
                {"role": "user", "content": f"Context: {context}\nQuestion: {question}"},
            ]
            prompts.append(chat_prompt(self.tokenizer, messages, tokenize=False, add_generation_prompt=True))
        return prompts

    def calibrate(self, target: PrivacyTarget) -> CalibrationResult:
        """Solve the per-selection budget for a total guarantee over ``n_tokens``.

        May return ``feasible=False``. Below some epsilon the Limited-Domain cutoff
        exceeds what ``n_split`` voters can ever supply, and no parameter choice
        fixes that — only more voters would. The caller is expected to skip such
        cells rather than run them, because a run would produce an AUC of 0.5 that
        is indistinguishable in the results from a mechanism that actually worked.
        """
        from dp_rag_eval.privacy import calibrate_koga

        result = calibrate_koga(target, n_split=self.n_split)
        self.dp = target.is_private
        self.max_new_tokens = target.n_tokens
        if target.is_private:
            self.dp_eps = result.params["dp_eps"]
            self.dp_delta = result.params["dp_delta"]
            self.delta1 = result.params["delta1"]
            self.target_eps = target.epsilon
            self.target_delta = target.delta
        self._calibration = result
        return result

    def privacy_spent(self) -> PrivacyBudget:
        """The composed guarantee for one answered query, per Koga's own accountant."""
        if not self.dp:
            return PrivacyBudget(epsilon=float("inf"), delta=0.0)
        if self._calibration is not None:
            return self._calibration.realized
        engine = self._make_dp_engine()
        epsilon, delta = engine.get_dp_expense(total_queries=self.max_new_tokens)
        return PrivacyBudget(epsilon=float(epsilon), delta=float(delta))

    def ld_threshold(self) -> float:
        """Votes the LD-Gumbel mechanism needs before it can select anything.

        From ``LDGumbelMechanism.get_top1``: the limited-domain cutoff sits at
        ``1 + log(min(2, k_bar) / delta) / eps`` votes above the runner-up. If that
        exceeds ``n_split`` no vote distribution can clear it and every query fails,
        regardless of the data. Use it to sanity-check an (epsilon, n_split) pair
        before spending GPU hours on it.
        """
        if not self.dp:
            return 0.0
        from dp_rag_eval.privacy import ld_vote_threshold

        return ld_vote_threshold(self.dp_eps, self.dp_delta)

    def reset_ld_stats(self) -> None:
        self.ld_draws = 0
        self.ld_failures = 0

    @property
    def ld_failure_rate(self) -> float:
        """Fraction of DP vote draws that produced no token. NaN before any draw."""
        return self.ld_failures / self.ld_draws if self.ld_draws else float("nan")

    def _make_dp_engine(self) -> LDGumbelMechanism:
        """A fresh LD-Gumbel mechanism with its own accountant."""
        return LDGumbelMechanism(
            self.dp_eps,
            self.dp_delta,
            k_bar=self.n_split,
            target_eps=self.target_eps,
            target_delta=self.target_delta,
            delta1=self.delta1,
            subsampling_rate=1.0,
            compose_method="rdp",
            fail_mode="stop",
        )

    def _vote(self, next_tokens: torch.Tensor, dp_engine: LDGumbelMechanism | None) -> int | None:
        """Aggregate voter tokens into one winner; ``None`` on LD failure/budget exhaustion.

        With ``dp_engine=None`` this is the plain plurality vote (non-private
        VoteRAG); otherwise the original DP path is used unchanged.
        """
        if dp_engine is None:
            token, _ = majority_vote(next_tokens, self.vocab_size, dp_engine=None)
            return int(token)
        token = get_majority_token_vote(next_tokens, self.vocab_size, dp_engine)
        return None if token is None else int(token)

    @torch.no_grad()
    def _voter_prefill(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor, Any]:
        """Encode all voter prompts as one padded batch and run a single forward.

        The tokenizer pads on the **left** (set in ``load_generator``), so the final
        position of every row is that voter's real last token and one shared slice
        gives every voter's next-token logits.

        Returns:
            ``(next_logits [n_voters, vocab], attention_mask, past_key_values)``.
        """
        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        attention_mask = encoded["attention_mask"]
        output, kept = forward_logits(
            self.model,
            keep=1,
            input_ids=encoded["input_ids"],
            attention_mask=attention_mask,
            use_cache=True,
        )
        return kept[:, -1, :], attention_mask, output.past_key_values

    @torch.no_grad()
    def answer(self, query: str, seed: int = 42) -> dict[str, Any]:
        """Vote token-by-token under the LD-Gumbel mechanism on the indexed corpus."""
        torch.manual_seed(seed)
        retrieved = self._retrieve(query, self.n_split * self.n_docs)
        prompts = self._build_voter_prompts(retrieved, query)

        dp_engine = self._make_dp_engine() if self.dp else None
        eos_id = self.tokenizer.eos_token_id
        generated: list[int] = []
        exit_status = "success"

        next_logits, attention_mask, cache = self._voter_prefill(prompts)
        n_voters = next_logits.shape[0]

        for _ in range(self.max_new_tokens):
            # temperature=0 => each voter's vote is its argmax next token. Voting is a
            # tiny histogram operation, so it runs on CPU to avoid per-draw GPU launches.
            next_tokens = torch.argmax(next_logits, dim=-1).cpu()
            with _quiet_votes():
                token = self._vote(next_tokens, dp_engine)
            if token is None:  # LD failure or budget exhausted -> baseline stops
                exit_status = "budget_exhausted_or_ld_failed"
                break
            generated.append(token)
            if eos_id is not None and token == eos_id:
                break
            # Advance every voter's KV cache by the shared winning token.
            step_input = torch.full((n_voters, 1), token, dtype=torch.long, device=self.device)
            attention_mask = torch.cat([attention_mask, torch.ones_like(step_input)], dim=1)
            output, kept = forward_logits(
                self.model,
                keep=1,
                input_ids=step_input,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
            )
            next_logits = kept[:, -1, :]
            cache = output.past_key_values

        answer = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        if dp_engine is None:
            epsilon, delta = float("inf"), 0.0
        else:
            epsilon, delta = dp_engine.get_dp_expense()
        return {
            "answer": answer,
            "epsilon": float(epsilon),
            "delta": float(delta),
            "exit_status": exit_status,
            "n_indexed": len(self._texts),
            "n_retrieved": len(retrieved),
            "n_voters": self.n_split,
            "n_dp_tokens": 0 if dp_engine is None else dp_engine.total_queries,
        }

    def _ensure_yes_no_ids(self) -> None:
        if self._yes_ids is None:
            self._yes_ids, self._no_ids = yes_no_token_ids(self.tokenizer)

    @torch.no_grad()
    def _first_token_voter_logits(self, query: str) -> torch.Tensor:
        """Next-token logits of every voter for the attack prompt, ``[n_voters, vocab]``."""
        retrieved = self._retrieve(query, self.n_split * self.n_docs)
        prompts = self._build_voter_prompts(retrieved, query)
        next_logits, _, _ = self._voter_prefill(prompts)
        return next_logits

    @torch.no_grad()
    def membership_score(self, query: str, seed: int | None = None) -> float:
        """P(the system answers 'Yes'), estimated by sampling the LD-Gumbel vote.

        The mechanism's output is a *token*, not a distribution, so its P('Yes') is
        estimated by Monte Carlo: the voter logits are computed once, then the vote
        itself is re-drawn ``mc_samples`` times, each draw with a fresh accountant
        (the repeats estimate one query's output distribution — they are not
        additional queries against the corpus, and must not be charged as such).

        The score is the *unconditional* fraction of draws won by a 'Yes' token.
        Draws that the mechanism loses to noise — a random token, or an LD failure —
        count as not-Yes, which is what makes the score fall towards chance as
        epsilon shrinks. That is the intended behaviour: strong DP should erase the
        signal, not merely rescale it.

        Returns a value in ``[0, 1]``. With ``dp=False`` this degenerates to the
        deterministic plurality vote and returns 0.0 or 1.0.
        """
        if seed is not None:
            torch.manual_seed(seed)
        self._ensure_yes_no_ids()
        next_logits = self._first_token_voter_logits(query)
        next_tokens = torch.argmax(next_logits, dim=-1).cpu()

        yes_ids = set(self._yes_ids or [])
        if not self.dp:
            token = self._vote(next_tokens, None)
            return 1.0 if token in yes_ids else 0.0

        n_yes = 0
        n_fail = 0
        with _quiet_votes():
            for _ in range(self.mc_samples):
                # A fresh engine per draw: each draw is an independent realisation of the
                # same single DP query, so budget must not accumulate across draws.
                token = self._vote(next_tokens, self._make_dp_engine())
                if token is None:
                    n_fail += 1
                elif token in yes_ids:
                    n_yes += 1
        self.ld_draws += self.mc_samples
        self.ld_failures += n_fail
        if n_fail:
            logger.debug("mc score: %d/%d draws lost to LD failure", n_fail, self.mc_samples)
        return n_yes / self.mc_samples

    @torch.no_grad()
    def membership_score_upper_bound(self, query: str) -> float:
        """Voter-averaged P('Yes') from **raw** logits, before any DP noise.

        This is deliberately *not* the framework's common estimand: Koga's noise is
        applied to token selection rather than to logits, so averaging the voters'
        pre-noise probabilities describes an attacker who can read voter internals,
        not the deployed system's output. It does not vary with epsilon and must
        never be plotted on the same line as :meth:`membership_score`.

        It is kept because it costs nothing — the same batched forward produces it —
        and because the gap between it and :meth:`membership_score` quantifies how
        much leakage the DP mechanism actually removes.
        """
        self._ensure_yes_no_ids()
        next_logits = self._first_token_voter_logits(query).float()
        probs = [yes_probability(row, self._yes_ids, self._no_ids) for row in next_logits]
        return sum(probs) / len(probs) if probs else 0.0

    @torch.no_grad()
    def completion_logprob(self, query: str, completion: str) -> float:
        """Gray-box canary score: voter-averaged log P(completion | query).

        Caveat (same as :meth:`membership_score_upper_bound`): Koga's DP noise is applied to token
        *selection* (the LD-Gumbel vote), not to a scorable per-token distribution, so
        there is no faithful probability for an arbitrary completion. This returns the
        mean over voters of the raw teacher-forced log-probability of ``completion`` —
        the leakage available to an attacker with voter-logit access (an adversarial
        upper bound), not the noised output. The black-box generation attack is the
        faithful Koga setting for canary extraction.
        """
        retrieved = self._retrieve(query, self.n_split * self.n_docs)
        prompts = self._build_voter_prompts(retrieved, query)
        completion_ids = (
            self.tokenizer(completion, add_special_tokens=False, return_tensors="pt").input_ids[0].to(self.device)
        )
        n_completion = completion_ids.numel()
        if n_completion == 0:
            return float("-inf")

        voter_logprobs: list[float] = []
        for prompt in prompts:
            prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            input_ids = torch.cat([prompt_ids, completion_ids.view(1, -1)], dim=1)
            # logits[t] predicts token t+1, so completion token i sits at row
            # prompt_len+i-1: the last (n_completion + 1) rows minus the final one.
            _, kept = forward_logits(self.model, keep=n_completion + 1, input_ids=input_ids)
            log_probs = torch.log_softmax(kept[0, :-1].float(), dim=-1)
            token_logprobs = log_probs[torch.arange(n_completion), completion_ids]
            voter_logprobs.append(float(token_logprobs.sum()))
        return sum(voter_logprobs) / len(voter_logprobs) if voter_logprobs else float("-inf")

    def run(self, documents: list[Document], question: str, seed: int = 42) -> dict[str, Any]:
        """Index the corpus then answer one query (convenience for the smoke runner)."""
        self.index(documents)
        return self.answer(question, seed=seed)
