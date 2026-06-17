"""Koga DPVoteRAG baseline behind a thin run interface (CPU / HF backend).

The upstream code drives token-by-token voting through vLLM
(``ensemble_generate_token_ids_vllm``). vLLM is not available on CPU/Windows, so
this wrapper re-implements *only the generation backend* — a token-by-token loop
over voter prompts using HuggingFace ``forward`` for greedy (temperature=0)
next-token logits. The DP-sensitive parts are the **original** baseline code:

* ``get_majority_token_vote`` — vote aggregation, fail-mode handling.
* ``LDGumbelMechanism``       — Limited Domain Gumbel noise + RDP accounting.

Retrieval has no DP in Koga's method (privacy is enforced at generation), so we
use a plain dense top-(n_split*n_docs) retrieval to pick the voter documents.

On a GPU machine the faithful path is the upstream vLLM script; this wrapper is
for the CPU smoke check.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from dp_rag_eval.core.types import Document
from dp_rag_eval.pipelines.loading import Embedder, yes_no_token_ids, yes_probability
from dp_rag_eval.wrappers.koga_imports import LDGumbelMechanism, get_majority_token_vote

logger = logging.getLogger(__name__)


class KogaPipeline:
    """Run Koga's DP token-voting end to end on a corpus of :class:`Document`."""

    name = "koga"

    def __init__(
        self,
        generator: Any,
        tokenizer: Any,
        embedder: Embedder,
        *,
        n_split: int = 16,
        n_docs: int = 1,
        max_new_tokens: int = 12,
        # Per-token budget chosen so the Limited-Domain threshold (~log(1/delta)/eps
        # votes) sits low enough that a handful of agreeing voters can emit a token
        # on this small smoke setup. This is a demonstrative budget, not a tight one.
        dp_eps: float = 4.0,
        dp_delta: float = 1e-5,
        target_eps: float = 500.0,
        target_delta: float = 1e-2,
        device: str = "cpu",
    ) -> None:
        self.model = generator
        self.tokenizer = tokenizer
        self.embedder = embedder
        self.n_split = n_split
        self.n_docs = n_docs
        self.max_new_tokens = max_new_tokens
        self.dp_eps = dp_eps
        self.dp_delta = dp_delta
        self.target_eps = target_eps
        self.target_delta = target_delta
        self.device = device
        self.vocab_size = int(generator.config.vocab_size)
        self._texts: list[str] = []
        self._doc_embeddings: Any = None
        self._yes_ids: list[int] | None = None
        self._no_ids: list[int] | None = None

    def index(self, documents: list[Document]) -> None:
        """Embed and cache the corpus once so many queries can be answered cheaply."""
        self._texts = [doc.text for doc in documents]
        self._doc_embeddings = self.embedder.encode(self._texts) if self._texts else None

    def _retrieve(self, query: str, k: int) -> list[str]:
        """Plain (non-DP) dense top-k retrieval — Koga adds no noise at retrieval."""
        query_embedding = self.embedder.encode([query])
        scores = (query_embedding @ self._doc_embeddings.T)[0]
        k = min(k, len(self._texts))
        top = torch.topk(scores, k).indices.tolist()
        return [self._texts[i] for i in top]

    def _build_voter_prompts(self, docs: list[str], question: str) -> list[str]:
        """Split retrieved docs into ``n_split`` voter groups and chat-format prompts.

        Each voter is given its document group as context via the model's chat
        template, so an instruct model produces an actual answer turn instead of an
        immediate end-of-text token.
        """
        question = question[:300]
        prompts: list[str] = []
        for split_id in range(self.n_split):
            group = docs[split_id * self.n_docs : (split_id + 1) * self.n_docs]
            if not group:  # corpus smaller than n_split*n_docs — reuse first doc
                group = docs[:1]
            context = " ".join(f"{i + 1}. {text}" for i, text in enumerate(group))
            messages = [
                {"role": "system", "content": "Give a short answer to the question based only on the context."},
                {"role": "user", "content": f"Context: {context}\nQuestion: {question}"},
            ]
            prompts.append(self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        return prompts

    @torch.no_grad()
    def answer(self, query: str, seed: int = 42) -> dict[str, Any]:
        """Vote token-by-token under the LD-Gumbel mechanism on the indexed corpus."""
        torch.manual_seed(seed)
        retrieved = self._retrieve(query, self.n_split * self.n_docs)
        prompts = self._build_voter_prompts(retrieved, query)

        dp_engine = LDGumbelMechanism(
            self.dp_eps,
            self.dp_delta,
            k_bar=self.n_split,
            target_eps=self.target_eps,
            target_delta=self.target_delta,
            subsampling_rate=1.0,
            compose_method="rdp",
            fail_mode="stop",
        )

        eos_id = self.tokenizer.eos_token_id
        generated: list[int] = []
        exit_status = "success"

        # Encode each voter prompt once; keep its KV cache and current next-token logits.
        # Token-by-token voting then only forwards a single new token per voter per step
        # (the shared winning token), which is what makes this tractable on CPU.
        caches: list = []
        last_logits: list = []
        for prompt in prompts:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            output = self.model(input_ids=input_ids, use_cache=True)
            caches.append(output.past_key_values)
            last_logits.append(output.logits[0, -1])

        for _ in range(self.max_new_tokens):
            # temperature=0 => each voter's vote is its argmax next token.
            next_tokens = torch.stack([torch.argmax(logits) for logits in last_logits])
            token = get_majority_token_vote(next_tokens, self.vocab_size, dp_engine)
            if token is None:  # LD failure or budget exhausted -> baseline stops
                exit_status = "budget_exhausted_or_ld_failed"
                break
            token = int(token)
            generated.append(token)
            if eos_id is not None and token == eos_id:
                break
            # Advance every voter's KV cache by the shared winning token.
            step_input = torch.tensor([[token]], device=self.device)
            for v in range(len(prompts)):
                output = self.model(input_ids=step_input, past_key_values=caches[v], use_cache=True)
                caches[v] = output.past_key_values
                last_logits[v] = output.logits[0, -1]

        answer = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        epsilon, delta = dp_engine.get_dp_expense()
        return {
            "answer": answer,
            "epsilon": float(epsilon),
            "delta": float(delta),
            "exit_status": exit_status,
            "n_indexed": len(self._texts),
            "n_retrieved": len(retrieved),
            "n_voters": self.n_split,
            "n_dp_tokens": dp_engine.total_queries,
        }

    @torch.no_grad()
    def yes_no_logit_score(self, query: str) -> float:
        """Gray-box membership score: voter-averaged P('Yes') at the first token.

        Caveat: Koga's DP noise is applied to token *selection* (LD-Gumbel vote), not
        to logits. This score averages each voter's raw first-token P('Yes'/'No'),
        bypassing that noise — so it measures the leakage available to an attacker
        with access to voter logits (an upper bound), not the noised output. The
        black-box Yes/No attack is the faithful Koga setting.
        """
        retrieved = self._retrieve(query, self.n_split * self.n_docs)
        prompts = self._build_voter_prompts(retrieved, query)
        if self._yes_ids is None:
            self._yes_ids, self._no_ids = yes_no_token_ids(self.tokenizer)
        probs: list[float] = []
        for prompt in prompts:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            logits = self.model(input_ids=input_ids).logits[0, -1]
            probs.append(yes_probability(logits, self._yes_ids, self._no_ids))
        return sum(probs) / len(probs) if probs else 0.0

    def run(self, documents: list[Document], question: str, seed: int = 42) -> dict[str, Any]:
        """Index the corpus then answer one query (convenience for the smoke runner)."""
        self.index(documents)
        return self.answer(question, seed=seed)
