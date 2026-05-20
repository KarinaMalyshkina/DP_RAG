"""HuggingFace Transformers LLM backend for Grislain pipeline."""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class LocalHFBackend:
    """LLM backend using HuggingFace Transformers for local inference.

    Provides direct access to model and tokenizer, which is required by
    Grislain's pipeline for custom LogitsProcessor-based DP aggregation.
    """

    def __init__(
        self,
        model_name: str = "microsoft/Phi-3.5-mini-instruct",
        device: str = "cuda",
        model_revision: str | None = None,
        torch_dtype: torch.dtype | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model_revision = model_revision
        self._torch_dtype = torch_dtype or (torch.float16 if device == "cuda" else torch.float32)
        self._model: AutoModelForCausalLM | None = None
        self._tokenizer: AutoTokenizer | None = None

    def _load(self) -> None:
        """Lazy-load model and tokenizer."""
        if self._model is not None:
            return
        logger.info("Loading HF model: %s", self._model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            revision=self._model_revision,
            padding_side="left",
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            revision=self._model_revision,
            torch_dtype=self._torch_dtype,
            device_map=self._device if self._device != "cpu" else None,
        )
        if self._device == "cpu":
            self._model = self._model.to("cpu")
        self._model.eval()
        logger.info("Model loaded: %s (%d params)", self._model_name, sum(p.numel() for p in self._model.parameters()))

    @property
    def model(self) -> Any:
        """Return the underlying HF model (lazy-loaded)."""
        self._load()
        return self._model

    @property
    def tokenizer(self) -> Any:
        """Return the tokenizer (lazy-loaded)."""
        self._load()
        return self._tokenizer

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        self._load()
        return self._tokenizer.vocab_size  # type: ignore[union-attr]

    def generate_single_token(self, prompts: list[str]) -> list[int]:
        """Generate a single next token for each prompt."""
        self._load()
        assert self._tokenizer is not None and self._model is not None
        inputs = self._tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        # Get the last token logits for each sequence
        last_token_logits = outputs.logits[:, -1, :]
        token_ids = torch.argmax(last_token_logits, dim=-1).tolist()
        return token_ids

    def generate(self, prompts: list[str], max_new_tokens: int = 50) -> list[str]:
        """Generate text completions for the given prompts."""
        self._load()
        assert self._tokenizer is not None and self._model is not None
        inputs = self._tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        # Decode only the newly generated tokens
        input_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_length:]
        return self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
