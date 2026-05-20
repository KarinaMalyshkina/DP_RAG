"""Retriever for the Koga pipeline.

Wraps DPR (Dense Passage Retrieval) for document retrieval.
No DP noise at retrieval — privacy is enforced at generation stage.
Also supports simple in-memory retrieval for testing.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from dp_rag_eval.core.types import Document

logger = logging.getLogger(__name__)


class KogaRetriever:
    """Retriever for the Koga pipeline.

    Supports two modes:
    1. DPR-based retrieval using facebook/rag-sequence-nq (production)
    2. Simple TF-IDF or embedding-based in-memory retrieval (testing)

    In both cases, retrieval is NOT differentially private.
    Privacy is enforced at the generation stage via token voting.

    Args:
        use_dpr: Whether to use DPR for retrieval. If False, uses simple keyword matching.
        dpr_model_name: DPR model name for HuggingFace.
        device: Device for DPR model.
    """

    def __init__(
        self,
        use_dpr: bool = False,
        dpr_model_name: str = "facebook/dpr-ctx_encoder-single-nq-base",
        device: str = "cuda",
    ) -> None:
        self._use_dpr = use_dpr
        self._dpr_model_name = dpr_model_name
        self._device = device
        self._documents: list[Document] = []
        self._dpr_index: Any = None
        self._embeddings: np.ndarray | None = None

    def index(self, documents: list[Document]) -> None:
        """Index documents for retrieval.

        Args:
            documents: List of documents to index.
        """
        self._documents = list(documents)
        if self._use_dpr:
            self._build_dpr_index()
        else:
            self._build_simple_index()
        logger.info("Indexed %d documents", len(self._documents))

    def _build_simple_index(self) -> None:
        """Build a simple keyword-based index for testing."""
        # For testing: just store documents, retrieve by keyword overlap
        pass

    def _build_dpr_index(self) -> None:
        """Build a DPR-based FAISS index."""
        try:
            import faiss
            from transformers import DPRContextEncoder, DPRContextEncoderTokenizer

            tokenizer = DPRContextEncoderTokenizer.from_pretrained(self._dpr_model_name)
            encoder = DPRContextEncoder.from_pretrained(self._dpr_model_name).to(self._device)
            encoder.eval()

            import torch

            embeddings = []
            batch_size = 32
            for i in range(0, len(self._documents), batch_size):
                batch_texts = [d.text for d in self._documents[i : i + batch_size]]
                inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = encoder(**inputs)
                embeddings.append(outputs.pooler_output.cpu().numpy())

            self._embeddings = np.vstack(embeddings).astype(np.float32)
            dim = self._embeddings.shape[1]
            self._dpr_index = faiss.IndexFlatIP(dim)
            faiss.normalize_L2(self._embeddings)
            self._dpr_index.add(self._embeddings)
            logger.info("Built DPR index with %d vectors of dim %d", len(self._documents), dim)
        except ImportError:
            logger.warning("FAISS or DPR not available, falling back to simple retrieval")
            self._use_dpr = False

    def retrieve(self, query: str, top_k: int = 10) -> tuple[list[Document], list[float]]:
        """Retrieve top-k documents for a query.

        Args:
            query: Query string.
            top_k: Number of documents to retrieve.

        Returns:
            Tuple of (retrieved documents, similarity scores).
        """
        if not self._documents:
            return [], []

        top_k = min(top_k, len(self._documents))

        if self._use_dpr and self._dpr_index is not None:
            return self._retrieve_dpr(query, top_k)
        return self._retrieve_simple(query, top_k)

    def _retrieve_simple(self, query: str, top_k: int) -> tuple[list[Document], list[float]]:
        """Simple keyword overlap retrieval for testing."""
        query_words = set(query.lower().split())
        scores = []
        for doc in self._documents:
            doc_words = set(doc.text.lower().split())
            overlap = len(query_words & doc_words) / max(len(query_words), 1)
            scores.append(overlap)

        indices = np.argsort(scores)[::-1][:top_k]
        retrieved_docs = [self._documents[i] for i in indices]
        retrieved_scores = [scores[i] for i in indices]
        return retrieved_docs, retrieved_scores

    def _retrieve_dpr(self, query: str, top_k: int) -> tuple[list[Document], list[float]]:
        """DPR-based retrieval using FAISS index."""
        import torch
        from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

        q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to(self._device)
        q_encoder.eval()

        inputs = q_tokenizer(query, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            q_embedding = q_encoder(**inputs).pooler_output.cpu().numpy().astype(np.float32)

        import faiss

        faiss.normalize_L2(q_embedding)
        scores, indices = self._dpr_index.search(q_embedding, top_k)
        retrieved_docs = [self._documents[int(i)] for i in indices[0] if i >= 0]
        retrieved_scores = [float(s) for s in scores[0] if s > -1e9]
        return retrieved_docs, retrieved_scores
