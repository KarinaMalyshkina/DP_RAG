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

from dp_rag_eval.core.types import Document
from dp_rag_eval.pipelines.loading import Embedder, yes_no_token_ids, yes_probability

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
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

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
        new_tokens = output_ids[0, input_ids.shape[1]:]
        answer_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return {
            "answer": answer_text,
            "epsilon": float("inf"),
            "n_indexed": len(self._texts),
            "n_retrieved": len(docs),
        }

    @torch.no_grad()
    def yes_no_logit_score(self, query: str) -> float:
        """Gray-box membership score: P('Yes') from first-token logits.

        No DP noise — this is the maximum-leakage score an attacker can extract
        from a naive RAG system with logit access.
        """
        if self._yes_ids is None:
            self._yes_ids, self._no_ids = yes_no_token_ids(self.tokenizer)
        docs = self._retrieve(query)
        prompt = self._build_prompt(docs, query)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        logits = self.model(input_ids=input_ids).logits[0, -1]
        return yes_probability(logits, self._yes_ids, self._no_ids)

    def run(self, documents: list[Document], question: str, seed: int = 42) -> dict[str, Any]:
        self.index(documents)
        return self.answer(question, seed=seed)
