"""The generator loader must accept a new checkpoint without disturbing the old one.

Gemma 4 is published as ``Gemma4UnifiedForConditionalGeneration`` and is not
registered for causal LM, so ``load_generator`` gained a fallback through the
multimodal auto classes. The main model of the matrix (Qwen3.5-9B) loads through
``AutoModelForCausalLM`` and must keep taking exactly that path — a second model
is worth nothing if acquiring it perturbs the first.

These tests pin three things:

* the causal class is tried first, and a successful load never consults a wrapper;
* the fallback is reachable only through a failure of that first attempt;
* a wrapper that returns hidden states instead of vocabulary logits is rejected at
  load time rather than silently scoring every attack query.

No network, no downloads: the auto classes are stubbed.
"""

from __future__ import annotations

import pytest
import torch

from dp_rag_eval.pipelines import loading


class FakeAuto:
    """Stand-in for an auto class, recording how it was called."""

    def __init__(self, name: str, *, fails_with: Exception | None = None) -> None:
        self.__name__ = name
        self.fails_with = fails_with
        self.calls: list[dict] = []

    def from_pretrained(self, model_id: str, **kwargs):
        self.calls.append({"model_id": model_id, **kwargs})
        if self.fails_with is not None:
            raise self.fails_with
        return f"model<{self.__name__}>"


class FakeTokenizer:
    """Minimal tokenizer: callable, sized, enough for the logit-head probe."""

    def __init__(self, vocab_size: int = 1000) -> None:
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.vocab_size

    def __call__(self, text: str, return_tensors: str = "pt") -> dict:
        return {"input_ids": torch.zeros(1, 2, dtype=torch.long), "attention_mask": torch.ones(1, 2, dtype=torch.long)}


class FakeOutput:
    def __init__(self, logits) -> None:
        self.logits = logits


class FakeModel:
    """Emits a vector of the given width for each position."""

    def __init__(self, width: int | None) -> None:
        self.width = width

    def __call__(self, **kwargs):
        if self.width is None:
            return FakeOutput(None)
        return FakeOutput(torch.zeros(1, 2, self.width))


# ---------------------------------------------------------------------------
# Class selection
# ---------------------------------------------------------------------------


def test_wrapper_classes_never_include_the_causal_class() -> None:
    """The fallback list is strictly the alternatives, so order cannot invert."""
    assert loading.AutoModelForCausalLM not in loading._wrapper_classes()


def test_causal_class_is_tried_first_and_wins(monkeypatch) -> None:
    """The path Qwen takes: one call, to the class it has always used."""
    causal = FakeAuto("AutoModelForCausalLM")
    wrapper = FakeAuto("AutoModelForMultimodalLM")
    monkeypatch.setattr(loading, "AutoModelForCausalLM", causal)
    monkeypatch.setattr(loading, "_wrapper_classes", lambda: [wrapper])

    model = loading._load_weights("Qwen/Qwen3.5-9B", "sdpa", {"dtype": torch.bfloat16})

    assert model == "model<AutoModelForCausalLM>"
    assert len(causal.calls) == 1
    assert causal.calls[0]["attn_implementation"] == "sdpa"
    assert causal.calls[0]["dtype"] is torch.bfloat16
    assert wrapper.calls == [], "a successful causal load must not consult a wrapper class"


def test_wrapper_is_reached_only_after_the_causal_class_refuses(monkeypatch) -> None:
    """The path Gemma 4 takes: not registered for causal LM, so the wrapper answers."""
    causal = FakeAuto("AutoModelForCausalLM", fails_with=ValueError("Unrecognized configuration class"))
    wrapper = FakeAuto("AutoModelForMultimodalLM")
    monkeypatch.setattr(loading, "AutoModelForCausalLM", causal)
    monkeypatch.setattr(loading, "_wrapper_classes", lambda: [wrapper])

    model = loading._load_weights("google/gemma-4-12B-it", "sdpa", {})

    assert model == "model<AutoModelForMultimodalLM>"
    assert len(wrapper.calls) == 1


def test_every_failure_is_reported_together(monkeypatch) -> None:
    """A checkpoint nothing accepts must name each refusal, not just the last."""
    causal = FakeAuto("AutoModelForCausalLM", fails_with=ValueError("no causal mapping"))
    wrapper = FakeAuto("AutoModelForMultimodalLM", fails_with=OSError("no weights"))
    monkeypatch.setattr(loading, "AutoModelForCausalLM", causal)
    monkeypatch.setattr(loading, "_wrapper_classes", lambda: [wrapper])

    with pytest.raises(RuntimeError) as excinfo:
        loading._load_weights("nonexistent/model", "sdpa", {})

    message = str(excinfo.value)
    assert "no causal mapping" in message
    assert "no weights" in message


def test_out_of_memory_is_not_retried(monkeypatch) -> None:
    """Three more loads of the same weights cannot help, and the box is rented."""
    causal = FakeAuto("AutoModelForCausalLM", fails_with=torch.cuda.OutOfMemoryError("CUDA out of memory"))
    wrapper = FakeAuto("AutoModelForMultimodalLM")
    monkeypatch.setattr(loading, "AutoModelForCausalLM", causal)
    monkeypatch.setattr(loading, "_wrapper_classes", lambda: [wrapper])

    with pytest.raises(torch.cuda.OutOfMemoryError):
        loading._load_weights("Qwen/Qwen3.5-9B", "sdpa", {})
    assert wrapper.calls == []


# ---------------------------------------------------------------------------
# Logit head
# ---------------------------------------------------------------------------


def test_logit_head_accepts_a_vocabulary_width() -> None:
    loading._assert_lm_head(FakeModel(width=1000), FakeTokenizer(1000), "cpu")


def test_logit_head_tolerates_a_padded_vocabulary() -> None:
    """Checkpoints often pad the embedding matrix past the tokenizer's length."""
    loading._assert_lm_head(FakeModel(width=1024), FakeTokenizer(1000), "cpu")


def test_logit_head_rejects_hidden_states() -> None:
    """The failure this probe exists for: a wrapper handing back hidden states.

    3840-wide vectors from a 262k-token model would score every attack query
    plausibly and mean nothing.
    """
    with pytest.raises(RuntimeError, match="hidden states"):
        loading._assert_lm_head(FakeModel(width=3840), FakeTokenizer(262144), "cpu")


def test_logit_head_rejects_a_model_without_logits() -> None:
    with pytest.raises(RuntimeError, match="no `logits`"):
        loading._assert_lm_head(FakeModel(width=None), FakeTokenizer(1000), "cpu")


# ---------------------------------------------------------------------------
# Attention backend
# ---------------------------------------------------------------------------


class FakeConfig:
    def __init__(self, softcapping=None, text_config=None) -> None:
        if softcapping is not None:
            self.attn_logit_softcapping = softcapping
        if text_config is not None:
            self.text_config = text_config


def _patch_config(monkeypatch, config) -> None:
    class Auto:
        @staticmethod
        def from_pretrained(model_id: str):
            return config

    monkeypatch.setattr(loading, "AutoConfig", Auto)


def test_softcapping_forces_eager(monkeypatch) -> None:
    _patch_config(monkeypatch, FakeConfig(softcapping=50.0))
    assert loading._preferred_attn("google/gemma-2-9b-it", "sdpa") == "eager"


def test_softcapping_is_found_in_a_text_subconfig(monkeypatch) -> None:
    """Multimodal checkpoints keep the language model's fields one level down."""
    _patch_config(monkeypatch, FakeConfig(text_config=FakeConfig(softcapping=30.0)))
    assert loading._preferred_attn("some/multimodal", "sdpa") == "eager"


def test_models_without_softcapping_keep_sdpa(monkeypatch) -> None:
    """Qwen3.5 and Gemma 4 both land here — the fast backend is kept."""
    _patch_config(monkeypatch, FakeConfig(softcapping=None))
    assert loading._preferred_attn("Qwen/Qwen3.5-9B", "sdpa") == "sdpa"


def test_an_explicit_backend_is_never_overridden(monkeypatch) -> None:
    _patch_config(monkeypatch, FakeConfig(softcapping=50.0))
    assert loading._preferred_attn("google/gemma-2-9b-it", "eager") == "eager"
    assert loading._preferred_attn("google/gemma-2-9b-it", None) is None


def test_an_unreadable_config_does_not_fail_the_load(monkeypatch) -> None:
    class Auto:
        @staticmethod
        def from_pretrained(model_id: str):
            raise OSError("offline")

    monkeypatch.setattr(loading, "AutoConfig", Auto)
    assert loading._preferred_attn("Qwen/Qwen3.5-9B", "sdpa") == "sdpa"
