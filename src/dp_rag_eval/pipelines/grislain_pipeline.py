"""Grislain DP-RAG baseline behind a thin run interface.

Wraps the upstream ``DPRAGEngine`` building blocks (``PUPVectorStore`` for DP
retrieval, ``DPModel`` + ``DPLogitsAggregator`` for DP logit-aggregation
generation). Preloaded generator/embedder weights are injected into the baseline
objects so the runner can reuse them across dataset variants.

This is a faithful use of the original mechanism: only the model loading and the
indexing entry point are ours; the DP retrieval and DP generation are upstream.
Setting ``dp=False`` selects the upstream non-private code paths
(``_non_dp_call`` / ``_non_dp_top_k_threshold``), giving the epsilon = infinity
reference point without changing anything else.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from dp_rag_eval.core.types import CalibrationResult, Document, PrivacyBudget, PrivacyTarget
from dp_rag_eval.pipelines.loading import (
    Embedder,
    chat_prompt,
    forward_logits,
    last_token_logits,
    yes_no_token_ids,
    yes_probability,
)
from dp_rag_eval.wrappers.grislain_imports import (
    DPGenerationConfig,
    DPLogitsAggregator,
    DPModel,
    PUPVectorStore,
    PUPVectorStoreConfig,
)

logger = logging.getLogger(__name__)


class GrislainPipeline:
    """Run Grislain's DP-RAG end to end on a corpus of :class:`Document`."""

    name = "grislain"

    def __init__(
        self,
        generator: Any,
        gen_tokenizer: Any,
        embedder: Embedder,
        *,
        epsilon: float = 3.0,
        delta: float = 1e-3,
        max_new_tokens: int = 30,
        alpha: float = 1.0,
        omega: float = 0.01,
        temperature: float = 1.0,
        retrieval_epsilon: float = 0.2,
        retrieval_share: float = 0.1,
        top_k: int = 5,
        max_retrieve: int = 8,
        dp: bool = True,
    ) -> None:
        """Args:
        dp: When ``False``, both retrieval and generation use the upstream
            non-private paths (no exponential-mechanism threshold, no clipping).
            This is the ``epsilon = infinity`` reference point.
        """
        self.embedder = embedder
        self.retrieval_epsilon = retrieval_epsilon
        # Bounded top-k retrieval keeps the generation batch small enough
        # (the batch is one prompt per retrieved doc + a public prior).
        self.top_k = top_k
        self.max_retrieve = max_retrieve
        self.report_delta = delta
        self.dp = dp
        # Kept so calibrate() can rebuild the generation config without the caller
        # having to re-supply the mechanism's shape parameters.
        self.temperature = temperature
        self.alpha = alpha
        self.omega = omega
        self.retrieval_share = retrieval_share
        self._calibration: CalibrationResult | None = None
        # Total generation budget; per-token epsilon is derived internally via bisection.
        self.gen_config = DPGenerationConfig(
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            alpha=alpha,
            omega=omega,
            epsilon=epsilon,
            delta=delta,
            differential_pivacy=dp,
        )
        # The aggregator is deterministic given the config (its constructor re-runs the
        # per-token epsilon bisection), so it is built once and reused across queries.
        self._aggregator = DPLogitsAggregator(self.gen_config)
        # Build a DPModel but bypass its lazy 'cuda' loader by injecting preloaded weights.
        self.dp_model = DPModel(model_id="__preloaded__")
        self.dp_model.__dict__["model"] = generator
        self.dp_model.__dict__["tokenizer"] = gen_tokenizer
        self._store: PUPVectorStore | None = None
        self._yes_ids: list[int] | None = None
        self._no_ids: list[int] | None = None

    def _make_store(self) -> PUPVectorStore:
        """Fresh DP vector store per run, reusing the preloaded embedder weights."""
        store = PUPVectorStore(
            PUPVectorStoreConfig(
                top_k=self.top_k,
                epsilon=self.retrieval_epsilon,
                max_retrieve=self.max_retrieve,
                differential_pivacy=self.dp,
            )
        )
        store.__dict__["model"] = self.embedder.model
        store.__dict__["tokenizer"] = self.embedder.tokenizer
        return store

    def calibrate(self, target: PrivacyTarget) -> CalibrationResult:
        """Solve the retrieval/generation budget split for a total guarantee.

        Call **before** :meth:`index`: the retrieval epsilon is baked into the
        vector store when it is built, so calibrating afterwards would leave the
        store on the old budget. If the store already exists it is discarded here
        rather than left silently stale.

        Handing the target epsilon straight to ``DPGenerationConfig`` would not be
        equivalent — retrieval spends budget too, and the two stages compose. See
        :mod:`dp_rag_eval.privacy.calibration`.
        """
        from dp_rag_eval.privacy import calibrate_grislain

        result = calibrate_grislain(target, retrieval_share=self.retrieval_share)
        self.dp = target.is_private
        self.report_delta = target.delta
        if target.is_private:
            self.retrieval_epsilon = result.params["retrieval_epsilon"]
            generation_epsilon = result.params["epsilon"]
        else:
            self.retrieval_epsilon = 1.0
            generation_epsilon = 1.0
        self.gen_config = DPGenerationConfig(
            temperature=self.temperature,
            max_new_tokens=target.n_tokens,
            alpha=self.alpha,
            omega=self.omega,
            epsilon=generation_epsilon,
            delta=target.delta,
            differential_pivacy=self.dp,
        )
        self._aggregator = DPLogitsAggregator(self.gen_config)
        self._calibration = result
        if self._store is not None:
            logger.info("calibrate() after index(): discarding the store so it picks up the new retrieval budget")
            self._store = None
        return result

    def privacy_spent(self) -> PrivacyBudget:
        """The composed retrieval+generation guarantee for one answered query."""
        if not self.dp:
            return PrivacyBudget(epsilon=float("inf"), delta=0.0)
        if self._calibration is not None:
            return self._calibration.realized
        store = self._store if self._store is not None else self._make_store()
        composed = store.privacy_loss_distribution.compose(self.gen_config.privacy_loss_distribution)
        return PrivacyBudget(epsilon=float(composed.get_epsilon_for_delta(self.report_delta)), delta=self.report_delta)

    def index(self, documents: list[Document]) -> None:
        """Build the DP vector store once and precompute document embeddings."""
        self._store = self._make_store()
        for doc in documents:
            self._store.add(doc.text)
        self._store.embeddings()  # force the embedding cache so queries are cheap

    def answer(self, question: str) -> dict[str, Any]:
        """Answer one query against the indexed corpus and report the privacy budget.

        Returns a dict with the generated answer, the (epsilon, delta) of the
        composed retrieval+generation guarantee, retrieval counts, and an
        ``exit_status``.
        """
        store = self._store
        if store is None:
            raise RuntimeError("GrislainPipeline.answer called before index()")

        retrieved = store.pup_retrieve(question)
        if not retrieved:
            # The DP threshold admitted no documents — the generator has no context
            # to aggregate, so report instead of crashing inside the aggregator.
            return {
                "answer": "",
                "epsilon": float("nan"),
                "delta": self.report_delta,
                "exit_status": "no_documents_retrieved",
                "n_indexed": len(store.store),
                "n_retrieved": 0,
            }

        answer = self.dp_model.dp_chat(retrieved, question, self.gen_config)
        if self.dp:
            composed = store.privacy_loss_distribution.compose(self.gen_config.privacy_loss_distribution)
            epsilon = float(composed.get_epsilon_for_delta(self.report_delta))
        else:
            epsilon = float("inf")
        return {
            "answer": answer.strip(),
            "epsilon": epsilon,
            "delta": self.report_delta,
            "exit_status": "success",
            "n_indexed": len(store.store),
            "n_retrieved": len(retrieved),
        }

    def _batch_messages(self, query: str, retrieved: list[str]) -> list[list[dict[str, str]]]:
        """The upstream prompt batch: a public prior plus one prompt per document.

        Mirrors ``DPModel.dp_chat`` exactly so that the scores read here come from
        the same distribution the generator samples from.
        """
        return [
            [
                {"role": "system", "content": "You give a short response based on a predefined set documents."},
                {"role": "user", "content": query},
            ]
        ] + [
            [
                {
                    "role": "system",
                    "content": f"You give a short responses based on this document or a predefined set of "
                    f'similar documents.\nDocument:\n"{doc}"',
                },
                {"role": "user", "content": query},
            ]
            for doc in retrieved
        ]

    def _encode_batch(self, messages: list[list[dict[str, str]]]) -> dict[str, torch.Tensor]:
        tokenizer = self.dp_model.tokenizer
        return chat_prompt(
            tokenizer,
            messages,
            tokenize=True,
            padding=True,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        ).to(self.dp_model.model.device)

    @torch.no_grad()
    def membership_score(self, query: str) -> float:
        """P(the system answers 'Yes'), read exactly from the DP-aggregated logits.

        The framework's common membership estimand. Replicates the first generation
        step of ``dp_chat`` (same prior + per-document prompts, same
        ``DPLogitsAggregator``) and reads the probability mass on 'Yes' vs 'No'.

        This mechanism's output distribution is available in closed form, so the
        estimand is computed exactly rather than sampled — the same quantity Koga
        has to estimate by Monte Carlo. :meth:`membership_score_sampled` computes it
        the sampled way for validating that the two estimators agree.
        """
        store = self._store
        if store is None:
            raise RuntimeError("GrislainPipeline.membership_score called before index()")
        retrieved = store.pup_retrieve(query)
        if not retrieved:
            return 0.0  # no context retrieved -> treat as non-member

        inputs = self._encode_batch(self._batch_messages(query, retrieved))
        last_logits = last_token_logits(self.dp_model.model, **inputs)
        aggregated = self._aggregator(inputs["input_ids"], last_logits)[0]
        if self._yes_ids is None:
            self._yes_ids, self._no_ids = yes_no_token_ids(self.dp_model.tokenizer)
        return yes_probability(aggregated, self._yes_ids, self._no_ids)

    @torch.no_grad()
    def membership_score_sampled(self, query: str, n_samples: int = 200) -> float:
        """The same estimand as :meth:`membership_score`, estimated by sampling.

        Draws ``n_samples`` tokens from the temperature-warped aggregated
        distribution — the distribution the generator actually samples from — and
        returns the fraction that are affirmative. This is the estimator Koga is
        forced to use, applied here where the exact answer is also available.

        Its only purpose is validation: if the sampled and exact scores agree, then
        a difference in attack success between the two pipelines cannot be
        attributed to them being measured differently. It is not used for reported
        results, where the exact value is strictly better.

        Note this is the *unconditional* fraction of affirmative draws, whereas
        :meth:`membership_score` conditions on the answer being Yes or No — the two
        agree only when the model puts most of its mass on those two tokens, which
        is the intended regime for the attack prompt. Compare like with like when
        reading the validation numbers.
        """
        store = self._store
        if store is None:
            raise RuntimeError("GrislainPipeline.membership_score_sampled called before index()")
        retrieved = store.pup_retrieve(query)
        if not retrieved:
            return 0.0

        inputs = self._encode_batch(self._batch_messages(query, retrieved))
        last_logits = last_token_logits(self.dp_model.model, **inputs)
        aggregated = self._aggregator(inputs["input_ids"], last_logits)[0]
        if self._yes_ids is None:
            self._yes_ids, self._no_ids = yes_no_token_ids(self.dp_model.tokenizer)

        temperature = max(self.gen_config.temperature, 1e-6)
        probs = torch.softmax(aggregated.float() / temperature, dim=-1)
        draws = torch.multinomial(probs, num_samples=n_samples, replacement=True)
        yes = torch.tensor(self._yes_ids, device=draws.device)
        return float(torch.isin(draws, yes).float().mean())

    @torch.no_grad()
    def completion_logprobs(self, query: str, completions: list[str]) -> list[float]:
        """Gray-box canary scores: log P(completion | query) for many completions.

        Teacher-forces each completion through the same batch of prompts as
        ``dp_chat`` (a public prior + one prompt per DP-retrieved document). At each
        step the per-document and prior logits are combined by ``DPLogitsAggregator``
        and warped by ``temperature`` — i.e. the *actual* DP output distribution the
        sampler draws from — and the log-probability it assigns to the next
        completion token is accumulated.

        All completions share one query, hence one retrieval and one prompt prefix.
        The prefix is forwarded **once** and its KV cache is reused for every
        completion, which is what makes canary extraction (256 candidates per
        canary) affordable. Without the shared prefix each candidate would re-read
        the full ~800-token prompt.

        Returns:
            One log-probability per completion, ``-inf`` where a completion is empty
            or the DP retrieval threshold admitted no documents.
        """
        store = self._store
        if store is None:
            raise RuntimeError("GrislainPipeline.completion_logprobs called before index()")
        retrieved = store.pup_retrieve(query)
        if not retrieved:
            return [float("-inf")] * len(completions)

        tokenizer = self.dp_model.tokenizer
        model = self.dp_model.model
        inputs = self._encode_batch(self._batch_messages(query, retrieved))
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        batch_size, prompt_len = input_ids.shape

        # Prefill once: these logits give the distribution over each completion's
        # *first* token, shared by every candidate.
        output, kept = forward_logits(model, keep=1, input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        prefix_cache = output.past_key_values
        first_logits = kept[:, -1, :]
        first_aggregated = self._aggregator(input_ids, first_logits)[0]

        temperature = max(self.gen_config.temperature, 1e-6)
        first_log_probs = torch.log_softmax(first_aggregated.float() / temperature, dim=-1)

        # Rewinding the cache to the prompt is what makes scoring N candidates cost one
        # prefill instead of N. `hasattr(cache, "crop")` is not enough to decide that:
        # from transformers 5.x the method exists but raises unless past-state recording
        # was switched on first, so probe for the real capability and only then keep the
        # fast path. Getting this wrong is silent at import and loud only mid-sweep.
        #
        # **Do not "fix" the probe by switching to the negative `crop(-k)` form.** It is
        # the API linear-attention layers ask for, and it does not raise — but on
        # Qwen3.5 it does not restore the prefix either: scoring the same candidate
        # twice around another one drifts by ~1.3 nats (measured on an A6000 by
        # `scripts/verify_optimizations.py`, check 1). Linear attention carries a
        # recurrent state rather than per-token keys and values, so "remove k tokens"
        # cannot reconstruct it. The positive form raising here is the honest outcome:
        # it declines exactly the architectures where the rewind would be unsound, and
        # the fallback below recomputes the prompt and is exact.
        reusable = hasattr(prefix_cache, "crop")
        if reusable and hasattr(prefix_cache, "activate_past_recording"):
            prefix_cache.activate_past_recording()
        if reusable:
            try:
                # Cropping to the length it already has is a no-op, so this asks "will
                # crop work?" without changing anything. Answering it here costs one
                # call; answering it on the first multi-token candidate would abort a
                # cell hours into a sweep.
                prefix_cache.crop(prompt_len)
            except Exception as exc:  # noqa: BLE001 - any failure means the fast path is unusable
                logger.warning("Prefix-cache rewind unavailable (%s); scoring candidates the slow way", exc)
                reusable = False

        results: list[float] = []
        for completion in completions:
            completion_ids = (
                tokenizer(completion, add_special_tokens=False, return_tensors="pt").input_ids[0].to(model.device)
            )
            n_completion = completion_ids.numel()
            if n_completion == 0:
                results.append(float("-inf"))
                continue

            total = float(first_log_probs[completion_ids[0]])
            if n_completion > 1:
                # Teacher-force the whole completion in one forward. The same tokens are
                # appended to every prompt in the batch, so a single pass yields the
                # per-document logits at every completion position at once.
                step_ids = completion_ids.view(1, -1).expand(batch_size, -1)
                step_mask = torch.cat([attention_mask, torch.ones_like(step_ids, dtype=attention_mask.dtype)], dim=1)
                if reusable:
                    _, kept = forward_logits(
                        model,
                        keep=n_completion,
                        input_ids=step_ids,
                        attention_mask=step_mask,
                        past_key_values=prefix_cache,
                        use_cache=True,
                    )
                    prefix_cache.crop(prompt_len)
                else:
                    # Older cache objects cannot be rewound; fall back to re-reading the
                    # prompt. Still one forward per candidate instead of one per token.
                    full_ids = torch.cat([input_ids, step_ids], dim=1)
                    _, kept = forward_logits(model, keep=n_completion, input_ids=full_ids, attention_mask=step_mask)
                # kept[:, j] predicts completion token j+1.
                for j in range(n_completion - 1):
                    aggregated = self._aggregator(input_ids, kept[:, j, :])[0]
                    log_probs = torch.log_softmax(aggregated.float() / temperature, dim=-1)
                    total += float(log_probs[completion_ids[j + 1]])
            results.append(total)
        return results

    def completion_logprob(self, query: str, completion: str) -> float:
        """Single-completion form of :meth:`completion_logprobs`."""
        return self.completion_logprobs(query, [completion])[0]

    def run(self, documents: list[Document], question: str) -> dict[str, Any]:
        """Index the corpus then answer one query (convenience for the smoke runner)."""
        self.index(documents)
        return self.answer(question)
