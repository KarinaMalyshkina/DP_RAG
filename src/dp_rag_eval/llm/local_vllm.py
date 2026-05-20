"""vLLM-based LLM backend for Koga pipeline."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class LocalVLLMBackend:
    """LLM backend using vLLM for fast batched single-token generation.

    Used by Koga's voting approach where each voter independently generates
    one token at a time via greedy decoding.
    """

    def __init__(
        self,
        model_name: str = "facebook/opt-1.3b",
        device: str = "cuda",
        model_revision: str | None = None,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model_revision = model_revision
        self._gpu_memory_utilization = gpu_memory_utilization
        self._max_model_len = max_model_len
        self._llm: Any = None
        self._tokenizer: Any = None

    def _load(self) -> None:
        """Lazy-load the vLLM engine."""
        if self._llm is not None:
            return
        from vllm import LLM

        logger.info("Loading vLLM model: %s", self._model_name)
        kwargs: dict[str, Any] = {
            "model": self._model_name,
            "gpu_memory_utilization": self._gpu_memory_utilization,
            "trust_remote_code": True,
        }
        if self._model_revision:
            kwargs["revision"] = self._model_revision
        if self._max_model_len:
            kwargs["max_model_len"] = self._max_model_len
        self._llm = LLM(**kwargs)
        self._tokenizer = self._llm.get_tokenizer()
        logger.info("vLLM model loaded: %s", self._model_name)

    @property
    def tokenizer(self) -> Any:
        """Return the tokenizer (lazy-loaded)."""
        self._load()
        return self._tokenizer

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        self._load()
        return self._tokenizer.vocab_size

    def generate_single_token(self, prompts: list[str]) -> list[int]:
        """Generate a single next token for each prompt using vLLM."""
        self._load()
        from vllm import SamplingParams

        params = SamplingParams(temperature=0, max_tokens=1)
        outputs = self._llm.generate(prompts, params)
        token_ids = []
        for output in outputs:
            generated = output.outputs[0]
            if generated.token_ids:
                token_ids.append(generated.token_ids[0])
            else:
                token_ids.append(self._tokenizer.eos_token_id)
        return token_ids

    def generate_single_token_from_ids(self, prompt_token_ids: list[list[int]]) -> list[int]:
        """Generate a single next token from pre-tokenized prompt IDs."""
        self._load()
        from vllm import SamplingParams

        params = SamplingParams(temperature=0, max_tokens=1)
        outputs = self._llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=params)
        token_ids = []
        for output in outputs:
            generated = output.outputs[0]
            if generated.token_ids:
                token_ids.append(generated.token_ids[0])
            else:
                token_ids.append(self._tokenizer.eos_token_id)
        return token_ids

    def generate(self, prompts: list[str], max_new_tokens: int = 50) -> list[str]:
        """Generate text completions for the given prompts."""
        self._load()
        from vllm import SamplingParams

        params = SamplingParams(temperature=0, max_tokens=max_new_tokens)
        outputs = self._llm.generate(prompts, params)
        return [output.outputs[0].text for output in outputs]
