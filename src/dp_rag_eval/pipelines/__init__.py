"""DP-RAG pipeline wrappers behind a common ``run(documents, question)`` call.

* :class:`NaivePipeline`    — plain RAG, no DP noise (epsilon = infinity baseline).
* :class:`KogaPipeline`     — DP token-voting (LD-Gumbel) over voter prompts.
* :class:`GrislainPipeline` — DP retrieval + DP logit aggregation.

All reuse preloaded weights from :mod:`dp_rag_eval.pipelines.loading`.
"""

from dp_rag_eval.pipelines.grislain_pipeline import GrislainPipeline
from dp_rag_eval.pipelines.koga_pipeline import KogaPipeline
from dp_rag_eval.pipelines.loading import Embedder, load_generator, resolve_device
from dp_rag_eval.pipelines.naive_pipeline import NaivePipeline

__all__ = [
    "NaivePipeline",
    "GrislainPipeline",
    "KogaPipeline",
    "Embedder",
    "load_generator",
    "resolve_device",
]
