"""Shared model loading and dense retrieval for the smoke pipelines.

The experiment runner loads each model **once** through these helpers and injects
the loaded weights into both baselines, so a smoke run over several dataset
variants does not reload the generator/embedder for every variant.

All defaults are small and CPU-friendly: the goal here is to prove that both
DP-RAG baselines accept our corpus variants and produce an answer plus a spent
privacy budget, not to obtain research-grade utility numbers.
"""

from __future__ import annotations

import logging
import math

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def yes_no_token_ids(tokenizer: AutoTokenizer) -> tuple[list[int], list[int]]:
    """First-token ids for common 'Yes'/'No' spellings, for gray-box scoring."""
    yes: set[int] = set()
    no: set[int] = set()
    for word in ("Yes", "yes", " Yes", " yes", "YES"):
        ids = tokenizer.encode(word, add_special_tokens=False)
        if ids:
            yes.add(ids[0])
    for word in ("No", "no", " No", " no", "NO"):
        ids = tokenizer.encode(word, add_special_tokens=False)
        if ids:
            no.add(ids[0])
    return sorted(yes), sorted(no)


def yes_probability(logits: torch.Tensor, yes_ids: list[int], no_ids: list[int]) -> float:
    """P('Yes') vs 'No' from a next-token logit vector: softmax over the two classes.

    Uses the max logit across spelling variants for each class. Returns a value in
    ``[0, 1]`` — a continuous membership score for the gray-box attack.
    """
    yes_logit = max(float(logits[i]) for i in yes_ids)
    no_logit = max(float(logits[i]) for i in no_ids)
    shift = max(yes_logit, no_logit)
    exp_yes = math.exp(yes_logit - shift)
    exp_no = math.exp(no_logit - shift)
    return exp_yes / (exp_yes + exp_no)


# Minimal chat template so tiny test models that ship without one still work with
# Grislain's ``apply_chat_template`` call. Real chat models override this with
# their own template (we only set it when none is present).
_FALLBACK_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ message['role'] + ': ' + message['content'] + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
)


def resolve_device(device: str | None = None) -> str:
    """Return an explicit device string, auto-detecting CUDA when not given."""
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_generator(model_id: str, device: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a causal-LM generator shared by both pipelines.

    The tokenizer is configured the way Grislain's ``DPModel`` expects
    (left padding, a pad token) and given a fallback chat template if it lacks one.
    """
    logger.info("Loading generator %s on %s", model_id, device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        logger.info("Generator tokenizer has no chat template; installing a minimal fallback")
        tokenizer.chat_template = _FALLBACK_CHAT_TEMPLATE
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device).eval()
    return model, tokenizer


class Embedder:
    """CLS-pooled, L2-normalised dense encoder.

    Mirrors Grislain's ``PUPVectorStore.encode`` (CLS pooling + L2 norm) so the
    same loaded weights can back both Grislain's retrieval and Koga's retrieval.
    """

    def __init__(self, model_id: str, device: str) -> None:
        logger.info("Loading embedder %s on %s", model_id, device)
        self.model_id = model_id
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        """Encode texts to L2-normalised CLS embeddings, shape ``[n, dim]``."""
        encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        output = self.model(**encoded, return_dict=True)
        embeddings = output.last_hidden_state[:, 0]
        embeddings = embeddings / torch.sqrt(torch.sum(torch.square(embeddings), dim=1, keepdim=True))
        return embeddings
