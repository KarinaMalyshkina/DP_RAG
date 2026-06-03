"""Tests for the HealthCareMagic loader.

The HuggingFace ``load_dataset`` call is mocked so the test needs no network.
"""

from __future__ import annotations

import pytest

from dp_rag_eval.datasets.loaders import HealthCareMagicLoader, _parse_chat

# --- _parse_chat -------------------------------------------------------------


def test_parse_chat_basic():
    text = "<human>: What is wrong?\n<bot>: You have a cold."
    assert _parse_chat(text) == ("What is wrong?", "You have a cold.")


@pytest.mark.parametrize(
    "text",
    [
        "<human>: only a question, no answer",  # missing <bot>
        "<bot>: answer with no question",  # missing <human>
        "<human>: q\n<bot>: \n",  # empty answer
        "<human>: \n<bot>: a",  # empty question
        "<human>: a\n<bot>: b\n<bot>: c",  # two bot turns
    ],
)
def test_parse_chat_rejects_malformed(text):
    assert _parse_chat(text) is None


# --- HealthCareMagicLoader (mocked download) ---------------------------------


class _FakeDataset:
    """Minimal stand-in for a HuggingFace Dataset: len + integer indexing."""

    def __init__(self, rows: list[dict]):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, i):
        return self._rows[i]


@pytest.fixture
def fake_hf(monkeypatch):
    rows = [{"text": f"<human>: question {i}?\n<bot>: answer {i}."} for i in range(20)]
    # One malformed row that must be skipped.
    rows.append({"text": "<human>: dangling question with no bot turn"})

    def fake_load_dataset(name, split="train", cache_dir=None):
        return _FakeDataset(rows)

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    return rows


def test_loader_parses_and_skips_malformed(fake_hf):
    docs = HealthCareMagicLoader(n_docs=None).load()
    # 20 well-formed rows; the malformed one is dropped.
    assert len(docs) == 20
    assert docs[0].text.startswith("answer")
    assert docs[0].metadata["query"].startswith("question")
    assert docs[0].metadata["source"] == "healthcaremagic"
    # Ids are stable and contiguous.
    assert [d.id for d in docs[:3]] == ["hcm-000000", "hcm-000001", "hcm-000002"]


def test_loader_subsampling_is_deterministic(fake_hf):
    a = HealthCareMagicLoader(n_docs=5, seed=42).load()
    b = HealthCareMagicLoader(n_docs=5, seed=42).load()
    assert len(a) == 5
    assert [d.metadata["hf_index"] for d in a] == [d.metadata["hf_index"] for d in b]
