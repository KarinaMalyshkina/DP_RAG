"""Dataset loading and the four controlled corpus variants."""

from dp_rag_eval.datasets.io import (
    load_documents_jsonl,
    save_documents_jsonl,
    to_texts,
)
from dp_rag_eval.datasets.loaders import HealthCareMagicLoader
from dp_rag_eval.datasets.variants import (
    VARIANTS,
    CanaryInjectionVariant,
    FullDuplicateVariant,
    OriginalVariant,
    PartialOverlapVariant,
    split_sentences,
)

__all__ = [
    "HealthCareMagicLoader",
    "OriginalVariant",
    "FullDuplicateVariant",
    "PartialOverlapVariant",
    "CanaryInjectionVariant",
    "VARIANTS",
    "split_sentences",
    "save_documents_jsonl",
    "load_documents_jsonl",
    "to_texts",
]
