"""DP-RAG pipeline implementations."""

from __future__ import annotations

from typing import Any

from dp_rag_eval.core.config import PipelineConfig
from dp_rag_eval.pipelines.grislain.pipeline import GrislainDPRAG
from dp_rag_eval.pipelines.koga.pipeline import KogaDPVoteRAG


def create_pipeline(config: PipelineConfig, llm_backend: Any) -> KogaDPVoteRAG | GrislainDPRAG:
    """Factory: create a DP-RAG pipeline from configuration.

    Args:
        config: Pipeline configuration specifying type and parameters.
        llm_backend: Initialized LLM backend (LocalHFBackend or LocalVLLMBackend).

    Returns:
        An initialized pipeline instance.

    Raises:
        ValueError: If the pipeline type is not recognized.
    """
    if config.type == "koga":
        from dp_rag_eval.pipelines.koga.retriever import KogaRetriever

        retriever = KogaRetriever(use_dpr=False, device=config.device)
        return KogaDPVoteRAG(
            llm_backend=llm_backend,
            retriever=retriever,
            n_split=config.koga.n_split,
            n_docs=config.koga.n_docs,
            max_length=config.koga.max_length,
            below_threshold=config.koga.below_threshold,
            threshold=config.koga.threshold,
            sparse_eps=config.koga.sparse_eps,
            fail_mode=config.koga.fail_mode,
        )
    elif config.type == "grislain":
        from dp_rag_eval.pipelines.grislain.dp_model import DPGenerationConfig
        from dp_rag_eval.pipelines.grislain.pup_store import PUPVectorStoreConfig

        pup_config = PUPVectorStoreConfig(
            embedding_model=config.grislain.embedding_model,
            top_k=config.grislain.top_k,
            top_p=config.grislain.top_p,
            alpha=config.grislain.top_p_alpha,
            device=config.device,
        )
        dp_gen_config = DPGenerationConfig(
            alpha=config.grislain.alpha,
            omega=config.grislain.omega,
            temperature=config.grislain.temperature,
            max_new_tokens=config.grislain.max_new_tokens,
        )
        return GrislainDPRAG(
            llm_backend=llm_backend,
            pup_store_config=pup_config,
            dp_generation_config=dp_gen_config,
        )
    else:
        raise ValueError(f"Unknown pipeline type: {config.type!r}. Expected 'koga' or 'grislain'.")


__all__ = ["GrislainDPRAG", "KogaDPVoteRAG", "create_pipeline"]
