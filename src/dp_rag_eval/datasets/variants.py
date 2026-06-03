"""Four controlled dataset variants for studying duplication vs. DP guarantees.

Given one base corpus (``list[Document]``), each variant produces a new corpus
that isolates a single phenomenon:

* :class:`OriginalVariant`      — control: every document unique, unchanged.
* :class:`FullDuplicateVariant` — a subset of documents repeated *verbatim*.
* :class:`PartialOverlapVariant`— a subset of documents *share a sentence/paragraph*.
* :class:`CanaryInjectionVariant` — unique secret strings inserted as extraction targets.

All variants are deterministic given their ``seed`` and never mutate the input
list. Provenance for every change is written into each document's ``metadata`` so
downstream attacks and metrics can recover the ground truth.
"""

from __future__ import annotations

import logging
import random
import re
import string

from dp_rag_eval.core.types import Document

logger = logging.getLogger(__name__)

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences with a lightweight, dependency-free heuristic.

    Splits on ``.``/``!``/``?`` followed by whitespace. Good enough for selecting a
    shareable fragment; not a linguistically perfect tokenizer.
    """
    parts = [s.strip() for s in _SENTENCE_RE.split(text.strip())]
    return [s for s in parts if s]


def _select_indices(n: int, fraction: float, seed: int) -> list[int]:
    """Deterministically choose ``round(fraction * n)`` distinct indices in ``[0, n)``.

    At least one index is selected when ``fraction > 0`` and ``n > 0``.
    """
    if n == 0 or fraction <= 0:
        return []
    k = max(1, round(fraction * n))
    k = min(k, n)
    rng = random.Random(seed)
    return sorted(rng.sample(range(n), k))


class OriginalVariant:
    """Identity variant: the base corpus as-is (the control setting).

    Each HealthCareMagic chat is treated as a distinct patient interaction, so the
    base corpus is taken to be unique. Documents are copied (not mutated) and tagged
    ``metadata["variant"] = "original"``.
    """

    name = "original"

    def apply(self, documents: list[Document]) -> list[Document]:
        out = [doc.copy() for doc in documents]
        for doc in out:
            doc.metadata["variant"] = self.name
            doc.metadata.setdefault("is_duplicate", False)
        logger.info("[original] %d documents (unchanged)", len(out))
        return out


class FullDuplicateVariant:
    """Repeat a subset of documents *verbatim* to study exact duplication.

    ``duplication_rate`` of the base documents are selected; each selected document
    is appended ``n_copies`` additional times with identical text but new ids. A
    document thus appears ``1 + n_copies`` times in the output.

    Args:
        duplication_rate: Fraction of base documents to duplicate, in ``[0, 1]``.
        n_copies: Number of *extra* verbatim copies per selected document.
        seed: Determinism seed for selecting which documents to duplicate.
    """

    name = "full_duplicate"

    def __init__(self, duplication_rate: float = 0.3, n_copies: int = 1, seed: int = 42) -> None:
        self.duplication_rate = duplication_rate
        self.n_copies = n_copies
        self.seed = seed

    def apply(self, documents: list[Document]) -> list[Document]:
        selected = set(_select_indices(len(documents), self.duplication_rate, self.seed))
        out: list[Document] = []
        n_added = 0
        for i, doc in enumerate(documents):
            base = doc.copy()
            base.metadata["variant"] = self.name
            base.metadata["is_duplicate"] = False
            out.append(base)
            if i in selected:
                for j in range(self.n_copies):
                    dup = doc.copy(id=f"{doc.id}-dup{j}")
                    dup.metadata["variant"] = self.name
                    dup.metadata["is_duplicate"] = True
                    dup.metadata["duplicate_of"] = doc.id
                    dup.metadata["copy_index"] = j
                    out.append(dup)
                    n_added += 1
        logger.info(
            "[full_duplicate] %d base + %d duplicates (%d sources x %d copies) = %d total",
            len(documents),
            n_added,
            len(selected),
            self.n_copies,
            len(out),
        )
        return out


class PartialOverlapVariant:
    """Make a subset of documents *share* a sentence to study partial overlap.

    A fraction of documents act as *donors*. From each donor, one representative
    sentence (the longest) is taken as a shared fragment and inserted into
    ``n_hosts`` other documents. The hosts keep all their original content, so the
    overlap is partial rather than a full duplicate. This mirrors "the same clause
    appears in many contracts" / "the same diagnosis sentence in many records".

    Args:
        overlap_rate: Fraction of base documents used as donors, in ``[0, 1]``.
        n_hosts: Number of distinct host documents that receive each donor fragment.
        seed: Determinism seed for donor/host selection.
    """

    name = "partial_overlap"

    def __init__(self, overlap_rate: float = 0.3, n_hosts: int = 3, seed: int = 42) -> None:
        self.overlap_rate = overlap_rate
        self.n_hosts = n_hosts
        self.seed = seed

    def _pick_fragment(self, text: str) -> str | None:
        sentences = split_sentences(text)
        if not sentences:
            return None
        return max(sentences, key=len)

    def apply(self, documents: list[Document]) -> list[Document]:
        out = [doc.copy() for doc in documents]
        for doc in out:
            doc.metadata["variant"] = self.name
            doc.metadata.setdefault("is_duplicate", False)

        n = len(out)
        donor_indices = _select_indices(n, self.overlap_rate, self.seed)
        donor_set = set(donor_indices)
        non_donor = [i for i in range(n) if i not in donor_set]
        rng = random.Random(self.seed)

        n_injections = 0
        for donor_idx in donor_indices:
            fragment = self._pick_fragment(documents[donor_idx].text)
            if fragment is None or not non_donor:
                continue
            donor_id = documents[donor_idx].id
            out[donor_idx].metadata["is_fragment_donor"] = True
            k = min(self.n_hosts, len(non_donor))
            host_indices = rng.sample(non_donor, k)
            for host_idx in host_indices:
                host = out[host_idx]
                host.text = f"{host.text} {fragment}"
                shared = host.metadata.setdefault("shared_fragment_from", [])
                shared.append(donor_id)
                fragments = host.metadata.setdefault("shared_fragment", [])
                fragments.append(fragment)
                n_injections += 1

        logger.info(
            "[partial_overlap] %d donors, %d fragment injections across hosts (%d docs total)",
            len(donor_indices),
            n_injections,
            n,
        )
        return out


class CanaryInjectionVariant:
    """Insert unique secret strings into documents as extraction targets.

    Each canary is a distinguishable phrase with a known template and a random
    secret payload (Secret-Sharer style). ``n_repeats`` controls how many documents
    receive the **same** canary — letting experiments vary secret duplication, the
    central question of this thesis (a duplicated secret should be easier to extract).

    After :meth:`apply`, the ground-truth registry is available on
    :attr:`injected_canaries` for use by extraction attacks.

    Args:
        n_canaries: Number of distinct secrets to inject.
        n_repeats: Number of documents each secret is inserted into.
        secret_length: Length of the random secret payload.
        secret_alphabet: Characters the secret payload is drawn from.
        seed: Determinism seed for secrets and host selection.
    """

    name = "canary"

    # Known, attacker-visible template. ``{name}`` and ``{code}`` are the secret.
    CANARY_TEMPLATE = "CONFIDENTIAL PATIENT NOTE: the secret medical record code for {name} is {code}."

    def __init__(
        self,
        n_canaries: int = 20,
        n_repeats: int = 1,
        secret_length: int = 9,
        secret_alphabet: str = string.digits,
        seed: int = 42,
    ) -> None:
        self.n_canaries = n_canaries
        self.n_repeats = n_repeats
        self.secret_length = secret_length
        self.secret_alphabet = secret_alphabet
        self.seed = seed
        self.injected_canaries: list[dict] = []

    def _make_secret(self, rng: random.Random) -> str:
        return "".join(rng.choice(self.secret_alphabet) for _ in range(self.secret_length))

    def apply(self, documents: list[Document]) -> list[Document]:
        from faker import Faker

        out = [doc.copy() for doc in documents]
        for doc in out:
            doc.metadata["variant"] = self.name
            doc.metadata.setdefault("is_canary_host", False)

        n = len(out)
        rng = random.Random(self.seed)
        faker = Faker()
        faker.seed_instance(self.seed)

        self.injected_canaries = []
        needed_hosts = self.n_canaries * self.n_repeats
        if needed_hosts > n:
            logger.warning(
                "[canary] need %d host slots (%d canaries x %d repeats) but only %d documents; "
                "hosts will be reused across canaries",
                needed_hosts,
                self.n_canaries,
                self.n_repeats,
                n,
            )

        for k in range(self.n_canaries):
            name = faker.name()
            code = self._make_secret(rng)
            secret_text = self.CANARY_TEMPLATE.format(name=name, code=code)
            # Sample distinct hosts for this canary when possible.
            k_hosts = min(self.n_repeats, n)
            host_indices = rng.sample(range(n), k_hosts)
            for host_idx in host_indices:
                host = out[host_idx]
                host.text = f"{host.text} {secret_text}"
                host.metadata["is_canary_host"] = True
                ids = host.metadata.setdefault("canary_ids", [])
                ids.append(k)
            self.injected_canaries.append(
                {
                    "canary_id": k,
                    "name": name,
                    "secret_code": code,
                    "text": secret_text,
                    "n_repeats": self.n_repeats,
                    "host_doc_ids": [out[h].id for h in host_indices],
                }
            )

        logger.info(
            "[canary] injected %d secrets x %d repeats into %d documents",
            self.n_canaries,
            self.n_repeats,
            n,
        )
        return out


# Registry so configs / scripts can refer to variants by name.
VARIANTS: dict[str, type] = {
    OriginalVariant.name: OriginalVariant,
    FullDuplicateVariant.name: FullDuplicateVariant,
    PartialOverlapVariant.name: PartialOverlapVariant,
    CanaryInjectionVariant.name: CanaryInjectionVariant,
}
