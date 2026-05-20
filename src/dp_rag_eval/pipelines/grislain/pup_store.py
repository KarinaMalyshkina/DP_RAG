"""PUP (Private Unconstrained Partition) Vector Store for the Grislain pipeline.

Ported from Grislain (2025) pup_vector_store.py:
- Stores document embeddings and retrieves using the Exponential Mechanism
  for private threshold/top-k selection.

References:
    Grislain (2025), "RAG with Differential Privacy"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from dp_rag_eval.core.types import Document, PrivacyBudget

logger = logging.getLogger(__name__)


@dataclass
class PUPVectorStoreConfig:
    """Configuration for the PUP Vector Store.

    Args:
        embedding_model: HuggingFace model ID for sentence embeddings.
        top_k: Number of documents to retrieve (if using top-k strategy).
        top_p: Cumulative score threshold (if using top-p strategy).
        epsilon: Privacy budget for retrieval.
        delta: Privacy delta for retrieval.
        alpha: Exponent for score transformation in the exponential mechanism.
        device: Device for embedding model.
    """

    embedding_model: str = "Snowflake/snowflake-arctic-embed-m-v1.5"
    top_k: int | None = 10
    top_p: float | None = None
    epsilon: float = 1.0
    delta: float = 1e-5
    alpha: float = 4.0
    device: str = "cuda"


class PUPVectorStore:
    """Private Unconstrained Partition Vector Store.

    Stores document embeddings and retrieves documents using the Exponential
    Mechanism for differentially private selection.

    The retrieval process:
    1. Compute dot-product similarity between query and all document embeddings
    2. Use the Exponential Mechanism to select a score threshold
    3. Return all documents with scores above the threshold

    Args:
        config: PUPVectorStoreConfig with retrieval parameters.
    """

    def __init__(self, config: PUPVectorStoreConfig | None = None) -> None:
        self._config = config or PUPVectorStoreConfig()
        self._documents: list[Document] = []
        self._embeddings: np.ndarray | None = None
        self._embed_model: Any = None

    def _load_embedding_model(self) -> None:
        """Lazy-load the embedding model."""
        if self._embed_model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._embed_model = SentenceTransformer(
                self._config.embedding_model,
                device=self._config.device,
            )
            logger.info("Loaded embedding model: %s", self._config.embedding_model)
        except ImportError:
            logger.warning("sentence-transformers not available, using random embeddings for testing")
            self._embed_model = None

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Compute embeddings for a list of texts."""
        if self._embed_model is not None:
            return self._embed_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        # Fallback: random embeddings for testing
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((len(texts), 384)).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms

    def add(self, documents: list[Document]) -> None:
        """Add documents to the store and compute their embeddings.

        Args:
            documents: Documents to add.
        """
        self._load_embedding_model()
        self._documents.extend(documents)
        texts = [d.text for d in documents]
        new_embeddings = self._embed(texts)

        if self._embeddings is None:
            self._embeddings = new_embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, new_embeddings])

        logger.info("Added %d documents (total: %d)", len(documents), len(self._documents))

    def retrieve(
        self,
        query: str,
        epsilon: float | None = None,
    ) -> tuple[list[Document], list[float], PrivacyBudget]:
        """Retrieve documents using the Exponential Mechanism.

        Args:
            query: Query string.
            epsilon: Privacy budget for this retrieval (overrides config).

        Returns:
            Tuple of (retrieved documents, similarity scores, privacy spent).
        """
        if not self._documents or self._embeddings is None:
            return [], [], PrivacyBudget(epsilon=0.0, delta=0.0)

        eps = epsilon if epsilon is not None else self._config.epsilon

        # Compute query embedding and similarities
        query_embedding = self._embed([query])[0]
        similarities = self._embeddings @ query_embedding  # dot product

        # Apply the exponential mechanism for threshold selection
        selected_indices, selected_scores = self._exponential_mechanism_select(similarities, eps)

        retrieved_docs = [self._documents[i] for i in selected_indices]
        privacy_spent = PrivacyBudget(epsilon=eps, delta=self._config.delta)

        return retrieved_docs, selected_scores, privacy_spent

    def _exponential_mechanism_select(
        self,
        similarities: np.ndarray,
        epsilon: float,
    ) -> tuple[list[int], list[float]]:
        """Select documents using the exponential mechanism.

        Strategy:
        - top_k: Select top-k documents by noisy scores
        - top_p: Select documents above a privately chosen threshold

        Args:
            similarities: Dot-product similarity scores for all documents.
            epsilon: Privacy parameter.

        Returns:
            Tuple of (selected indices, selected scores).
        """
        n_docs = len(similarities)

        if self._config.top_k is not None:
            # Top-k with exponential mechanism noise
            k = min(self._config.top_k, n_docs)
            sensitivity = 1.0  # Normalized embeddings → max similarity change is bounded

            # Add exponential mechanism noise
            noise_scale = 2.0 * sensitivity / epsilon if epsilon > 0 else 1e10
            noisy_scores = similarities + np.random.laplace(0, noise_scale, size=n_docs)

            top_indices = np.argsort(noisy_scores)[::-1][:k]
            return top_indices.tolist(), similarities[top_indices].tolist()

        elif self._config.top_p is not None:
            # Top-p: select threshold via exponential mechanism
            sorted_indices = np.argsort(similarities)[::-1]
            sorted_scores = similarities[sorted_indices]

            # Find threshold using exponential mechanism
            # Utility of threshold t: number of docs above t that are "relevant"
            candidate_thresholds = np.linspace(sorted_scores[-1], sorted_scores[0], num=50)

            utilities = np.array([np.sum(similarities >= t) for t in candidate_thresholds], dtype=np.float64)
            # Prefer thresholds that give approximately top_p fraction
            target_count = max(1, int(self._config.top_p * n_docs))
            utilities = -np.abs(utilities - target_count)  # penalize deviation from target

            # Exponential mechanism selection
            scaled = epsilon * utilities / 2.0
            scaled -= np.max(scaled)
            probs = np.exp(scaled)
            probs /= probs.sum()

            chosen_idx = np.random.choice(len(candidate_thresholds), p=probs)
            threshold = candidate_thresholds[chosen_idx]

            selected = np.where(similarities >= threshold)[0]
            if len(selected) == 0:
                selected = np.array([np.argmax(similarities)])

            return selected.tolist(), similarities[selected].tolist()

        else:
            # Fallback: return all documents
            indices = list(range(n_docs))
            return indices, similarities.tolist()

    def reset(self) -> None:
        """Clear all stored documents and embeddings."""
        self._documents = []
        self._embeddings = None
