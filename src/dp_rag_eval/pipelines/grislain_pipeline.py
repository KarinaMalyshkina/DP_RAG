"""Grislain DP-RAG baseline behind a thin run interface.

Wraps the upstream ``DPRAGEngine`` building blocks (``PUPVectorStore`` for DP
retrieval, ``DPModel`` + ``DPLogitsAggregator`` for DP logit-aggregation
generation). Preloaded generator/embedder weights are injected into the baseline
objects so the runner can reuse them across dataset variants.

This is a faithful use of the original mechanism: only the model loading and the
indexing entry point are ours; the DP retrieval and DP generation are upstream.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from dp_rag_eval.core.types import Document
from dp_rag_eval.pipelines.loading import Embedder, yes_no_token_ids, yes_probability
from dp_rag_eval.wrappers.grislain_imports import (
    DPGenerationConfig,
    DPLogitsAggregator,
    DPModel,
    PUPVectorStore,
    PUPVectorStoreConfig,
)

logger = logging.getLogger(__name__)


class GrislainPipeline:
    """Run Grislain's DP-RAG end to end on a corpus of :class:`Document`."""

    name = "grislain"

    def __init__(
        self,
        generator: Any,
        gen_tokenizer: Any,
        embedder: Embedder,
        *,
        epsilon: float = 3.0,
        delta: float = 1e-3,
        max_new_tokens: int = 30,
        alpha: float = 1.0,
        omega: float = 0.01,
        temperature: float = 1.0,
        retrieval_epsilon: float = 0.2,
        top_k: int = 5,
        max_retrieve: int = 8,
    ) -> None:
        self.embedder = embedder
        self.retrieval_epsilon = retrieval_epsilon
        # Bounded top-k retrieval keeps the generation batch small enough for CPU
        # (the batch is one prompt per retrieved doc + a public prior).
        self.top_k = top_k
        self.max_retrieve = max_retrieve
        self.report_delta = delta
        # Total generation budget; per-token epsilon is derived internally via bisection.
        self.gen_config = DPGenerationConfig(
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            alpha=alpha,
            omega=omega,
            epsilon=epsilon,
            delta=delta,
        )
        # Build a DPModel but bypass its lazy 'cuda' loader by injecting preloaded weights.
        self.dp_model = DPModel(model_id="__preloaded__")
        self.dp_model.__dict__["model"] = generator
        self.dp_model.__dict__["tokenizer"] = gen_tokenizer
        self._store: PUPVectorStore | None = None
        self._yes_ids: list[int] | None = None
        self._no_ids: list[int] | None = None

    def _make_store(self) -> PUPVectorStore:
        """Fresh DP vector store per run, reusing the preloaded embedder weights."""
        store = PUPVectorStore(
            PUPVectorStoreConfig(
                top_k=self.top_k,
                epsilon=self.retrieval_epsilon,
                max_retrieve=self.max_retrieve,
            )
        )
        store.__dict__["model"] = self.embedder.model
        store.__dict__["tokenizer"] = self.embedder.tokenizer
        return store

    def index(self, documents: list[Document]) -> None:
        """Build the DP vector store once and precompute document embeddings."""
        self._store = self._make_store()
        for doc in documents:
            self._store.add(doc.text)
        self._store.embeddings()  # force the embedding cache so queries are cheap

    def answer(self, question: str) -> dict[str, Any]:
        """Answer one query against the indexed corpus and report the privacy budget.

        Returns a dict with the generated answer, the (epsilon, delta) of the
        composed retrieval+generation guarantee, retrieval counts, and an
        ``exit_status``.
        """
        store = self._store
        if store is None:
            raise RuntimeError("GrislainPipeline.answer called before index()")

        retrieved = store.pup_retrieve(question)
        if not retrieved:
            # The DP threshold admitted no documents — the generator has no context
            # to aggregate, so report instead of crashing inside the aggregator.
            return {
                "answer": "",
                "epsilon": float("nan"),
                "delta": self.report_delta,
                "exit_status": "no_documents_retrieved",
                "n_indexed": len(store.store),
                "n_retrieved": 0,
            }

        answer = self.dp_model.dp_chat(retrieved, question, self.gen_config)
        composed = store.privacy_loss_distribution.compose(self.gen_config.privacy_loss_distribution)
        epsilon = float(composed.get_epsilon_for_delta(self.report_delta))
        return {
            "answer": answer.strip(),
            "epsilon": epsilon,
            "delta": self.report_delta,
            "exit_status": "success",
            "n_indexed": len(store.store),
            "n_retrieved": len(retrieved),
        }

    @torch.no_grad()
    def yes_no_logit_score(self, query: str) -> float:
        """Gray-box membership score: P('Yes') from the DP-aggregated first-token logits.

        Replicates the first generation step of ``dp_chat`` (same prior + per-document
        prompts, same ``DPLogitsAggregator``) and reads the probability mass on 'Yes'
        vs 'No'. This is the system's *actual* DP output distribution, so it is a
        faithful gray-box signal. Returns a value in ``[0, 1]``.
        """
        store = self._store
        if store is None:
            raise RuntimeError("GrislainPipeline.yes_no_logit_score called before index()")
        retrieved = store.pup_retrieve(query)
        if not retrieved:
            return 0.0  # no context retrieved -> treat as non-member

        tokenizer = self.dp_model.tokenizer
        model = self.dp_model.model
        messages = [
            [
                {"role": "system", "content": "You give a short response based on a predefined set documents."},
                {"role": "user", "content": query},
            ]
        ] + [
            [
                {
                    "role": "system",
                    "content": f"You give a short responses based on this document or a predefined set of "
                    f'similar documents.\nDocument:\n"{doc}"',
                },
                {"role": "user", "content": query},
            ]
            for doc in retrieved
        ]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, padding=True, return_tensors="pt", return_dict=True, add_generation_prompt=True
        ).to(model.device)
        last_logits = model(**inputs).logits[:, -1, :]
        aggregated = DPLogitsAggregator(self.gen_config)(inputs["input_ids"], last_logits)[0]
        if self._yes_ids is None:
            self._yes_ids, self._no_ids = yes_no_token_ids(tokenizer)
        return yes_probability(aggregated, self._yes_ids, self._no_ids)

    def run(self, documents: list[Document], question: str) -> dict[str, Any]:
        """Index the corpus then answer one query (convenience for the smoke runner)."""
        self.index(documents)
        return self.answer(question)
