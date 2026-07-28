"""Naive (non-DP) RAG baseline — the epsilon=infinity reference point.

Plain dense retrieval followed by standard greedy generation with no privacy
mechanism whatsoever. Used as the upper bound on membership-inference attack
success: if an attacker cannot beat naive RAG, DP adds nothing meaningful.

The interface mirrors KogaPipeline / GrislainPipeline so it can be dropped into
run_attack.py without changes to the attack harness.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from dp_rag_eval.core.types import CalibrationResult, Document, PrivacyBudget, PrivacyTarget
from dp_rag_eval.pipelines.loading import (
    Embedder,
    chat_prompt,
    forward_logits,
    last_token_logits,
    yes_no_token_ids,
    yes_probability,
)

logger = logging.getLogger(__name__)


class NaivePipeline:
    """Retrieve-then-generate with no DP noise (epsilon = infinity)."""

    name = "naive"

    def __init__(
        self,
        generator: Any,
        tokenizer: Any,
        embedder: Embedder,
        *,
        top_k: int = 5,
        max_new_tokens: int = 12,
        device: str = "cpu",
    ) -> None:
        self.model = generator
        self.tokenizer = tokenizer
        self.embedder = embedder
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.device = device
        self._texts: list[str] = []
        self._doc_embeddings: Any = None
        self._yes_ids: list[int] | None = None
        self._no_ids: list[int] | None = None

    def calibrate(self, target: PrivacyTarget) -> CalibrationResult:
        """No mechanism to configure — this pipeline *is* the epsilon=infinity point.

        A finite target is accepted and ignored on purpose: the runner asks every
        pipeline to calibrate, and refusing here would just mean special-casing the
        baseline in the caller. The realized budget reported back is infinite, which
        is what lands in the results.
        """
        from dp_rag_eval.privacy import calibrate_non_private

        self.max_new_tokens = target.n_tokens
        return calibrate_non_private(target)

    def privacy_spent(self) -> PrivacyBudget:
        return PrivacyBudget(epsilon=float("inf"), delta=0.0)

    def index(self, documents: list[Document]) -> None:
        self._texts = [doc.text for doc in documents]
        self._doc_embeddings = self.embedder.encode(self._texts) if self._texts else None

    def _retrieve(self, query: str) -> list[str]:
        query_emb = self.embedder.encode([query])
        scores = (query_emb @ self._doc_embeddings.T)[0]
        k = min(self.top_k, len(self._texts))
        top_idx = torch.topk(scores, k).indices.tolist()
        return [self._texts[i] for i in top_idx]

    def _build_prompt(self, docs: list[str], question: str) -> str:
        question = question[:300]
        context = " ".join(f"{i + 1}. {text}" for i, text in enumerate(docs))
        messages = [
            {"role": "system", "content": "Give a short answer to the question based only on the context."},
            {"role": "user", "content": f"Context: {context}\nQuestion: {question}"},
        ]
        return chat_prompt(self.tokenizer, messages, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def answer(self, query: str, seed: int = 42) -> dict[str, Any]:
        """Standard greedy generation — no DP noise applied."""
        torch.manual_seed(seed)
        docs = self._retrieve(query)
        prompt = self._build_prompt(docs, query)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        new_tokens = output_ids[0, input_ids.shape[1] :]
        answer_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return {
            "answer": answer_text,
            "epsilon": float("inf"),
            "n_indexed": len(self._texts),
            "n_retrieved": len(docs),
        }

    @torch.no_grad()
    def completion_logprob(self, query: str, completion: str) -> float:
        """Gray-box canary score: total log P(completion | RAG-conditioned prompt).

        Retrieves for ``query``, builds the standard answer prompt, then sums the
        log-probabilities the model assigns to ``completion``'s tokens as the answer.
        No DP noise — this is the maximum-leakage score for a canary secret against a
        naive RAG system, i.e. the epsilon=infinity reference for Secret-Sharer exposure.
        """
        docs = self._retrieve(query)
        prompt = self._build_prompt(docs, query)
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        completion_ids = self.tokenizer(completion, add_special_tokens=False, return_tensors="pt").input_ids.to(
            self.device
        )
        if completion_ids.shape[1] == 0:
            return float("-inf")
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        # logits[t] predicts token t+1, so the distribution for completion token i
        # (at absolute position prompt_len + i) sits at row prompt_len + i - 1. Those
        # are the last (n_completion + 1) rows minus the final one, so only that many
        # positions need to be materialised.
        n_completion = completion_ids.shape[1]
        _, kept = forward_logits(self.model, keep=n_completion + 1, input_ids=input_ids)
        log_probs = torch.log_softmax(kept[0, :-1].float(), dim=-1)
        target = completion_ids[0]
        token_logprobs = log_probs[torch.arange(target.shape[0]), target]
        return float(token_logprobs.sum())

    @torch.no_grad()
    def membership_score(self, query: str) -> float:
        """P(the system answers 'Yes'), read exactly from the first-token logits.

        The framework's common membership estimand. With no DP mechanism in the
        way, the output distribution *is* the softmax over the model's logits, so
        it is available in closed form and no sampling is needed. This is the
        maximum-leakage reference the DP pipelines are measured against.
        """
        if self._yes_ids is None:
            self._yes_ids, self._no_ids = yes_no_token_ids(self.tokenizer)
        docs = self._retrieve(query)
        prompt = self._build_prompt(docs, query)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        logits = last_token_logits(self.model, input_ids=input_ids)[0]
        return yes_probability(logits.float(), self._yes_ids, self._no_ids)

    def run(self, documents: list[Document], question: str, seed: int = 42) -> dict[str, Any]:
        self.index(documents)
        return self.answer(question, seed=seed)
