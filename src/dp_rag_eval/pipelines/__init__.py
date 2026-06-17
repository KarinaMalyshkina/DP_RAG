"""DP-RAG pipeline wrappers behind a common ``run(documents, question)`` call.

* :class:`KogaPipeline`     — DP token-voting (LD-Gumbel) over voter prompts.
* :class:`GrislainPipeline` — DP retrieval + DP logit aggregation.

Both reuse preloaded weights from :mod:`dp_rag_eval.pipelines.loading`.
"""

from dp_rag_eval.pipelines.grislain_pipeline import GrislainPipeline
from dp_rag_eval.pipelines.koga_pipeline import KogaPipeline
from dp_rag_eval.pipelines.loading import Embedder, load_generator, resolve_device

__all__ = [
    "GrislainPipeline",
    "KogaPipeline",
    "Embedder",
    "load_generator",
    "resolve_device",
]
