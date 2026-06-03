"""Dataset loaders that produce a base corpus of :class:`Document`.

Currently implements the HealthCareMagic loader. Loaders return documents with
unique, stable ids; the four controlled variants in :mod:`dp_rag_eval.datasets.variants`
are applied on top of a loader's output.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

from dp_rag_eval.core.types import Document

logger = logging.getLogger(__name__)

HEALTHCARE_MAGIC_HF_NAME = "RafaelMPereira/HealthCareMagic-100k-Chat-Format-en"

_HUMAN_TAG = "<human>:"
_BOT_TAG = "<bot>:"


def _parse_chat(text: str) -> tuple[str, str] | None:
    """Split a HealthCareMagic ``text`` field into (patient question, doctor answer).

    The corpus stores each example as a single string of the form::

        <human>: <patient question>
        <bot>: <doctor answer>

    Args:
        text: Raw ``text`` field from the dataset.

    Returns:
        ``(question, answer)`` with both sides stripped, or ``None`` if the string
        does not contain exactly one human/bot turn (skipped rather than guessed).
    """
    if text.count(_BOT_TAG) != 1 or text.count(_HUMAN_TAG) != 1:
        return None
    human_part, bot_part = text.split(_BOT_TAG, 1)
    question = human_part.replace(_HUMAN_TAG, "", 1).strip()
    answer = bot_part.strip()
    if not question or not answer:
        return None
    return question, answer


class HealthCareMagicLoader:
    """Loads HealthCareMagic patient↔doctor chats as a corpus.

    Each document's ``text`` is the **doctor's answer** (the medical content into
    which canaries are naturally injected); the patient's question is preserved in
    ``metadata["query"]`` so it can drive retrieval evaluation later.

    Args:
        n_docs: If set, subsample to this many documents (after a seeded shuffle).
            ``None`` loads the full split.
        split: HuggingFace split name.
        seed: Seed for the subsampling shuffle (reproducibility).
        hf_name: HuggingFace dataset id.
        cache_dir: Optional HuggingFace ``datasets`` cache directory.
    """

    source = "healthcaremagic"

    def __init__(
        self,
        n_docs: int | None = 1000,
        split: str = "train",
        seed: int = 42,
        hf_name: str = HEALTHCARE_MAGIC_HF_NAME,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.n_docs = n_docs
        self.split = split
        self.seed = seed
        self.hf_name = hf_name
        self.cache_dir = str(cache_dir) if cache_dir is not None else None

    def load(self) -> list[Document]:
        """Download (or read from cache), parse, and return the corpus.

        Returns:
            A list of documents with ids ``hcm-000000``, ``hcm-000001``, …
        """
        # Imported lazily so the dependency is only needed when actually loading.
        from datasets import load_dataset

        logger.info("Loading %s (split=%s)", self.hf_name, self.split)
        ds = load_dataset(self.hf_name, split=self.split, cache_dir=self.cache_dir)

        indices = list(range(len(ds)))
        if self.n_docs is not None and self.n_docs < len(indices):
            rng = random.Random(self.seed)
            rng.shuffle(indices)
            indices = sorted(indices[: self.n_docs])

        documents: list[Document] = []
        skipped = 0
        for i in indices:
            parsed = _parse_chat(ds[i]["text"])
            if parsed is None:
                skipped += 1
                continue
            question, answer = parsed
            doc_id = f"hcm-{len(documents):06d}"
            documents.append(
                Document(
                    id=doc_id,
                    text=answer,
                    metadata={
                        "source": self.source,
                        "query": question,
                        "hf_index": i,
                        "variant": "original",
                    },
                )
            )

        if skipped:
            logger.warning("Skipped %d malformed examples while parsing %s", skipped, self.hf_name)
        logger.info("Loaded %d documents from %s", len(documents), self.hf_name)
        return documents
