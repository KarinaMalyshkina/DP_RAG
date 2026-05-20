"""Koga et al. DPVoteRAG pipeline implementation.

Implements the DPRagPipeline protocol for Koga et al. (2024):
1. Retrieval via DPR (non-private)
2. Document splitting into voter groups
3. Token-by-token DP voting via Limited Domain Gumbel Mechanism
4. Optional SVT (Sparse Vector Technique) for the sparse variant

References:
    Koga et al. (2024), "Privacy-Preserving RAG with Differential Privacy"
"""

from __future__ import annotations

import logging

from dp_rag_eval.core.types import (
    AnswerResult,
    Document,
    PrivacyBudget,
    RetrievalResult,
)
from dp_rag_eval.pipelines.koga.dp_mechanisms import LDGumbelMechanism
from dp_rag_eval.pipelines.koga.ensemble import ensemble_generate_token_ids
from dp_rag_eval.pipelines.koga.retriever import KogaRetriever

logger = logging.getLogger(__name__)


KOGA_PROMPT_TEMPLATE = (
    "Instruction: Give a simple short answer for the given question using the provided context.\n"
    "Context: {context}\n"
    "Question: {question}\n"
    "Answer:"
)


class KogaDPVoteRAG:
    """Koga et al. DPVoteRAG pipeline.

    Wraps retrieval, document splitting into voter groups, and DP token-by-token
    voting using the Limited Domain Gumbel Mechanism.

    Args:
        llm_backend: LLM backend (vLLM or HF) for token generation.
        retriever: Document retriever (DPR or simple).
        n_split: Number of voter groups to split retrieved documents into.
        n_docs: Number of documents per voter group.
        max_length: Maximum number of tokens to generate.
        dp_eps: Per-token privacy parameter for the LD mechanism.
        dp_delta: Per-token delta for the LD mechanism.
        target_eps: Total privacy budget (epsilon).
        target_delta: Total privacy budget (delta).
        k_bar: Vocabulary size cap for the LD mechanism.
        below_threshold: Whether to use SVT (sparse variant).
        threshold: SVT agreement threshold.
        sparse_eps: SVT privacy parameter.
        fail_mode: Behavior on LD failure: "raise" or "zero".
    """

    def __init__(
        self,
        llm_backend: object,
        retriever: KogaRetriever | None = None,
        n_split: int = 50,
        n_docs: int = 1,
        max_length: int = 20,
        dp_eps: float = 0.1,
        dp_delta: float = 1e-5,
        target_eps: float = 8.0,
        target_delta: float = 1e-5,
        k_bar: int = 50257,
        below_threshold: bool = False,
        threshold: float = 0.5,
        sparse_eps: float = 1.0,
        fail_mode: str = "zero",
    ) -> None:
        self._llm = llm_backend
        self._retriever = retriever or KogaRetriever(use_dpr=False)
        self.n_split = n_split
        self.n_docs = n_docs
        self.max_length = max_length
        self.k_bar = k_bar
        self.below_threshold = below_threshold
        self.threshold = threshold
        self.sparse_eps = sparse_eps
        self._documents: list[Document] = []
        self._total_privacy = PrivacyBudget(epsilon=0.0, delta=0.0)
        self._dp_engine = LDGumbelMechanism(
            eps=dp_eps,
            delta=dp_delta,
            k_bar=k_bar,
            target_eps=target_eps,
            target_delta=target_delta,
            fail_mode=fail_mode,
        )

    def index(self, documents: list[Document]) -> None:
        """Index documents into the retrieval store."""
        self._documents = list(documents)
        self._retriever.index(documents)
        logger.info("Koga pipeline indexed %d documents", len(documents))

    def retrieve(self, query: str, eps: float, delta: float) -> RetrievalResult:
        """Retrieve documents (non-private in Koga's design).

        Privacy is enforced at generation, not retrieval.
        eps/delta params are accepted for interface compatibility but not used.
        """
        total_docs_needed = self.n_split * self.n_docs
        docs, scores = self._retriever.retrieve(query, top_k=total_docs_needed)
        return RetrievalResult(
            doc_ids=[d.id for d in docs],
            scores=scores,
            privacy_spent=PrivacyBudget(epsilon=0.0, delta=0.0),
        )

    def answer(self, query: str, eps: float, delta: float) -> AnswerResult:
        """Full DP-RAG pipeline: retrieve, split into voters, generate with DP voting.

        Args:
            query: The question to answer.
            eps: Per-token privacy parameter (overrides constructor dp_eps).
            delta: Per-token delta.

        Returns:
            AnswerResult with generated text and privacy information.
        """
        # Retrieve documents
        retrieval = self.retrieve(query, eps, delta)
        doc_map = {d.id: d for d in self._documents}
        retrieved_docs = [doc_map[did] for did in retrieval.doc_ids if did in doc_map]

        # Split into voter groups
        voter_prompts = self._build_voter_prompts(query, retrieved_docs)

        # Build public prompt for SVT variant
        public_prompt = None
        if self.below_threshold:
            public_prompt = KOGA_PROMPT_TEMPLATE.format(
                context="No context available.",
                question=query,
            )

        # Reset DP engine for this query
        self._dp_engine.reset()
        self._dp_engine.eps = eps
        self._dp_engine.target_eps = eps * self.max_length  # Budget for all tokens

        # Generate with DP voting
        token_ids, exit_status = ensemble_generate_token_ids(
            llm_backend=self._llm,  # type: ignore[arg-type]
            prompt_list=voter_prompts,
            max_new_tokens=self.max_length,
            dp_engine=self._dp_engine,
            k_bar=self.k_bar,
            below_threshold=self.below_threshold,
            threshold=self.threshold,
            sparse_eps=self.sparse_eps,
            public_prompt=public_prompt,
        )

        # Decode generated tokens
        generated_text = self._llm.tokenizer.decode(token_ids, skip_special_tokens=True)  # type: ignore[union-attr]

        # Track privacy
        generation_budget = PrivacyBudget(epsilon=self._dp_engine.get_dp_expense(), delta=delta)
        self._total_privacy = self._total_privacy + generation_budget

        return AnswerResult(
            text=generated_text.strip(),
            retrieved_doc_ids=retrieval.doc_ids,
            privacy_spent=generation_budget,
            exit_status=exit_status,
        )

    def _build_voter_prompts(self, query: str, docs: list[Document]) -> list[str]:
        """Build prompts for each voter group.

        Splits retrieved documents into n_split groups of n_docs each.
        Each voter group gets a prompt with its assigned documents as context.
        """
        prompts = []
        for i in range(self.n_split):
            start = i * self.n_docs
            end = start + self.n_docs
            group_docs = docs[start:end] if start < len(docs) else []

            if group_docs:
                context_parts = [f"{j + 1}. {d.text}" for j, d in enumerate(group_docs)]
                context = "; ".join(context_parts)
            else:
                context = "No context available."

            prompt = KOGA_PROMPT_TEMPLATE.format(context=context, question=query)
            prompts.append(prompt)

        return prompts

    def get_privacy_spent(self) -> PrivacyBudget:
        """Return cumulative privacy budget spent."""
        return self._total_privacy

    def reset(self) -> None:
        """Reset privacy accountant for a new experiment run."""
        self._total_privacy = PrivacyBudget(epsilon=0.0, delta=0.0)
        self._dp_engine.reset()
        logger.info("Koga pipeline privacy state reset")
