"""Core types and protocols for the dp-rag-eval framework."""

from dp_rag_eval.core.protocols import DatasetLoader, DatasetVariant
from dp_rag_eval.core.types import Document

__all__ = ["Document", "DatasetLoader", "DatasetVariant"]
