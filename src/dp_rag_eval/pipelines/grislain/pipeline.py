"""Grislain DP-RAG pipeline implementation.

Implements the DPRagPipeline protocol for Grislain (2025):
1. DP retrieval via PUP Vector Store (Exponential Mechanism)
2. Batch inference: one prompt per document + public prior
3. DP logit aggregation via DPLogitsAggregator (custom LogitsProcessor)
4. Privacy accounting via PLD composition (dp-accounting)

References:
    Grislain (2025), "RAG with Differential Privacy"
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from dp_rag_eval.core.types import (
    AnswerResult,
    Document,
    PrivacyBudget,
    RetrievalResult,
)
from dp_rag_eval.pipelines.grislain.dp_model import DPGenerationConfig, DPLogitsAggregator
from dp_rag_eval.pipelines.grislain.pup_store import PUPVectorStore, PUPVectorStoreConfig

logger = logging.getLogger(__name__)


GRISLAIN_PROMPT_TEMPLATE = (
    "Answer the following question based on the provided context.\n\n"
    "Context: {context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

GRISLAIN_PUBLIC_PROMPT_TEMPLATE = "Answer the following question.\n\nQuestion: {question}\n\nAnswer:"


class GrislainDPRAG:
    """Grislain DP-RAG pipeline.

    Uses DP logit aggregation during generation: each retrieved document
    is placed in a separate prompt, all prompts are processed simultaneously,
    and per-document logits are aggregated with clipping and noise via a
    custom LogitsProcessor.

    Args:
        llm_backend: HuggingFace Transformers backend (requires .model and .tokenizer).
        pup_store_config: Configuration for the PUP Vector Store.
        dp_generation_config: DP generation configuration.
    """

    def __init__(
        self,
        llm_backend: Any,
        pup_store_config: PUPVectorStoreConfig | None = None,
        dp_generation_config: DPGenerationConfig | None = None,
    ) -> None:
        self._llm = llm_backend
        self._pup_config = pup_store_config or PUPVectorStoreConfig()
        self._dp_config = dp_generation_config or DPGenerationConfig()
        self._pup_store = PUPVectorStore(self._pup_config)
        self._documents: list[Document] = []
        self._total_privacy = PrivacyBudget(epsilon=0.0, delta=0.0)

    def index(self, documents: list[Document]) -> None:
        """Add documents to the PUP Vector Store."""
        self._documents = list(documents)
        self._pup_store.add(documents)
        logger.info("Grislain pipeline indexed %d documents", len(documents))

    def retrieve(self, query: str, eps: float, delta: float) -> RetrievalResult:
        """Retrieve documents using the DP PUP Vector Store.

        Args:
            query: Query string.
            eps: Privacy budget for retrieval.
            delta: Privacy delta.

        Returns:
            RetrievalResult with document IDs, scores, and privacy spent.
        """
        docs, scores, privacy = self._pup_store.retrieve(query, epsilon=eps)
        return RetrievalResult(
            doc_ids=[d.id for d in docs],
            scores=scores,
            privacy_spent=privacy,
        )

    def answer(self, query: str, eps: float, delta: float) -> AnswerResult:
        """Full DP-RAG pipeline: retrieve + generate with DP logit aggregation.

        Args:
            query: The question to answer.
            eps: Total privacy budget (split between retrieval and generation).
            delta: Privacy delta.

        Returns:
            AnswerResult with generated text and privacy information.
        """
        # Split budget: half for retrieval, half for generation
        retrieval_eps = eps / 2.0
        generation_eps = eps / 2.0

        # DP retrieval
        retrieval = self.retrieve(query, retrieval_eps, delta)
        doc_map = {d.id: d for d in self._documents}
        retrieved_docs = [doc_map[did] for did in retrieval.doc_ids if did in doc_map]

        if not retrieved_docs:
            # No documents retrieved — generate from public prompt only
            public_prompt = GRISLAIN_PUBLIC_PROMPT_TEMPLATE.format(question=query)
            outputs = self._llm.generate([public_prompt], max_new_tokens=self._dp_config.max_new_tokens or 50)
            return AnswerResult(
                text=outputs[0].strip(),
                retrieved_doc_ids=[],
                privacy_spent=retrieval.privacy_spent,
                exit_status="success",
            )

        # Build prompts: public prior + one per document
        prompts = self._build_prompts(query, retrieved_docs)
        n_docs = len(retrieved_docs)

        # Create DP generation config for this query
        gen_config = DPGenerationConfig(
            alpha=self._dp_config.alpha,
            omega=self._dp_config.omega,
            temperature=self._dp_config.temperature,
            epsilon=generation_eps,
            dp_delta=delta,
            max_new_tokens=self._dp_config.max_new_tokens,
            do_sample=True,
        )

        # Create the DP logits aggregator
        aggregator = DPLogitsAggregator(n_docs=n_docs, config=gen_config)

        # Tokenize all prompts
        tokenizer = self._llm.tokenizer
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self._llm.model.device) for k, v in inputs.items()}

        # Generate with DP logit aggregation
        try:
            with torch.no_grad():
                output_ids = self._llm.model.generate(
                    **inputs,
                    generation_config=gen_config,
                    logits_processor=[aggregator],
                )

            # Decode the first prompt's output (all prompts get the same aggregated result)
            input_length = inputs["input_ids"].shape[1]
            generated_ids = output_ids[0, input_length:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            exit_status = "success"
        except Exception as e:
            logger.error("Generation failed: %s", e)
            generated_text = ""
            exit_status = "ld_failed"

        # Compose retrieval + generation privacy
        generation_budget = PrivacyBudget(epsilon=generation_eps, delta=delta)
        total_query_budget = retrieval.privacy_spent + generation_budget
        self._total_privacy = self._total_privacy + total_query_budget

        return AnswerResult(
            text=generated_text.strip(),
            retrieved_doc_ids=retrieval.doc_ids,
            privacy_spent=total_query_budget,
            exit_status=exit_status,
        )

    def _build_prompts(self, query: str, docs: list[Document]) -> list[str]:
        """Build prompts: public prior (index 0) + one per document.

        Args:
            query: The question.
            docs: Retrieved documents.

        Returns:
            List of prompts where prompts[0] is public, prompts[1:] are per-document.
        """
        prompts = [GRISLAIN_PUBLIC_PROMPT_TEMPLATE.format(question=query)]
        for doc in docs:
            prompt = GRISLAIN_PROMPT_TEMPLATE.format(context=doc.text, question=query)
            prompts.append(prompt)
        return prompts

    def get_privacy_spent(self) -> PrivacyBudget:
        """Return cumulative privacy budget spent."""
        return self._total_privacy

    def reset(self) -> None:
        """Reset privacy accountant for a new experiment run."""
        self._total_privacy = PrivacyBudget(epsilon=0.0, delta=0.0)
        logger.info("Grislain pipeline privacy state reset")
