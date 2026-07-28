"""Structural protocols for pluggable framework components.

:class:`DPRagPipeline` is the contract that makes systems comparable. Two DP-RAG
methods can only be placed on one axis if they agree on what is being measured,
so the protocol pins down the parts that are easy to get silently wrong:

* the privacy guarantee is **solved for**, not assumed — see :meth:`calibrate`;
* the membership signal is one defined quantity, not "whatever this pipeline
  happens to expose";
* a pipeline reports what its **own** accountant says it spent, so a comparison
  measures the published method rather than our re-implementation of it.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from dp_rag_eval.core.types import CalibrationResult, Document, PrivacyBudget, PrivacyTarget


@runtime_checkable
class DatasetLoader(Protocol):
    """Loads a base corpus. Dataset variants are applied on top of the result."""

    def load(self) -> list[Document]:
        """Return the base corpus as a list of documents with unique ids."""
        ...


@runtime_checkable
class DatasetVariant(Protocol):
    """Transforms a base corpus into one controlled evaluation setting.

    A variant takes a base ``list[Document]`` and returns a new list, without
    mutating the input. Implementations must be deterministic given their seed so
    that experiments are reproducible.
    """

    name: str

    def apply(self, documents: list[Document]) -> list[Document]:
        """Return a new corpus for this variant. Must not mutate ``documents``."""
        ...


@runtime_checkable
class DPRagPipeline(Protocol):
    """A DP-RAG system that can be evaluated alongside the others.

    Implementing this protocol is what it takes to add a method to the framework.
    See ``docs/METHODOLOGY.md`` for the definitions the methods below must honour.
    """

    name: str

    def calibrate(self, target: PrivacyTarget) -> CalibrationResult:
        """Configure the mechanism to meet ``target``, or report why it cannot.

        Must solve the pipeline's own parameters — noise scale, per-step budget,
        budget split across stages — so that the pipeline's **own** privacy
        accountant reports the target guarantee for one ``target.n_tokens``-token
        answer. Implementations must not approximate this by scaling a knob
        linearly: composition is not linear, and the resulting axis would not be
        comparable across methods.

        Returning ``feasible=False`` is legitimate and expected in part of the
        epsilon range for some mechanisms. Do not fall back to a weaker guarantee.
        """
        ...

    def index(self, documents: list[Document]) -> None:
        """Build the retrieval structure over ``documents``."""
        ...

    def answer(self, query: str) -> dict[str, Any]:
        """Answer one query under the calibrated guarantee.

        The returned mapping carries at least ``answer`` (the generated text) and
        the realised ``epsilon`` / ``delta``. This is the utility path; the same
        configuration must be the one the attacks are run against, otherwise the
        privacy-utility curve compares two different systems.
        """
        ...

    def membership_score(self, query: str) -> float:
        """P(the system answers 'Yes') for an attack prompt, in ``[0, 1]``.

        The estimand is fixed across pipelines: the probability that the
        **deployed system's output distribution** puts on an affirmative first
        answer token. How it is computed is free — exactly where the distribution
        is available in closed form, by sampling the mechanism where it is not.
        What is not free is measuring something else, such as pre-noise logits;
        expose that separately if it is interesting.
        """
        ...

    def completion_logprob(self, query: str, completion: str) -> float:
        """Log-probability the system assigns to ``completion`` as its answer.

        Used by canary extraction to rank a true secret against distractors.
        """
        ...

    def privacy_spent(self) -> PrivacyBudget:
        """What this pipeline's own accountant reports for one answered query."""
        ...
