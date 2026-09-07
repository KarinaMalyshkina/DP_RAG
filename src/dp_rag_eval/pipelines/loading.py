"""Shared model loading, prompting and forward helpers for the pipelines.

The experiment runner loads each model **once** through these helpers and injects
the loaded weights into every baseline, so a sweep over dataset variants and
epsilon values does not reload the generator/embedder for each cell.

Three concerns live here because all three pipelines share them:

* **Loading** — dtype selection (bf16 on GPU, fp32 on CPU), attention backend, and
  the auto class. ``AutoModelForCausalLM`` covers every text-only chat model and is
  always tried first; checkpoints published as multimodal wrappers (Gemma 4 ships
  as ``Gemma4UnifiedForConditionalGeneration``) fall through to the wrapper classes
  and are then checked to be emitting vocabulary logits rather than hidden states.
* **Prompting** — :func:`chat_prompt`, which disables reasoning models' thinking
  mode so that the first generated token is the answer itself.
* **Forwarding** — :func:`forward_logits`, which restricts the returned logits to
  the positions actually needed. With a 250k-token vocabulary the full-sequence
  logit tensor is the dominant VRAM cost of every attack query.
"""

from __future__ import annotations

import inspect
import logging
import math

import torch
import transformers
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

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


def resolve_dtype(dtype: str | torch.dtype | None, device: str) -> torch.dtype:
    """Pick the compute dtype: bf16 on CUDA, fp32 on CPU, unless overridden.

    Loading a 9B model at the ``from_pretrained`` default (fp32) needs ~36 GB of
    VRAM for the weights alone, which does not fit on a 48 GB card alongside the
    activations. bf16 halves that. CPU keeps fp32 because bf16 matmuls are slow
    (and sometimes unimplemented) on CPU, and the CPU path is only used for smoke
    runs with tiny models.
    """
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype in (None, "auto"):
        return torch.bfloat16 if device.startswith("cuda") else torch.float32
    resolved = getattr(torch, str(dtype), None)
    if not isinstance(resolved, torch.dtype):
        raise ValueError(f"Unknown dtype {dtype!r}")
    return resolved


def _transformers_version() -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in transformers.__version__.split(".")[:3]:
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def _dtype_kwarg() -> str:
    """Name of the ``from_pretrained`` dtype argument for the installed version.

    ``torch_dtype`` was renamed to ``dtype`` in transformers 4.56. Passing the
    wrong one is silently ignored (it lands in ``**kwargs`` and ends up on the
    config), so the model would load in fp32 without any error — hence the
    explicit version check plus the post-load assertion in :func:`load_generator`.
    """
    return "dtype" if _transformers_version() >= (4, 56) else "torch_dtype"


# Auto classes tried, in order, when loading a generator.
#
# ``AutoModelForCausalLM`` is first and unconditional. Every text-only chat model in
# this framework loads through it, so that call stays byte-for-byte what it always
# was and the branches below are unreachable for them. The rest exist for
# checkpoints published as multimodal wrappers: Gemma 4 ships as
# ``Gemma4UnifiedForConditionalGeneration`` and is not registered for causal LM at
# all. They are resolved by name rather than imported, because which of them exists
# depends on the installed transformers version.
#
# Deliberately *not* done here: choosing the class up front from
# ``MODEL_FOR_CAUSAL_LM_MAPPING``. That reads better, but it would make the main
# model's load depend on how the lazy auto-mapping behaves in a given transformers
# release. Trying the causal class first cannot regress a model that already loads.
_WRAPPER_CLASS_NAMES = ("AutoModelForMultimodalLM", "AutoModelForImageTextToText", "AutoModelForVision2Seq")


def _wrapper_classes() -> list[type]:
    """Multimodal auto classes present in the installed transformers, in preference order."""
    found = []
    for name in _WRAPPER_CLASS_NAMES:
        cls = getattr(transformers, name, None)
        if cls is not None:
            found.append(cls)
    return found


def _preferred_attn(model_id: str, attn_implementation: str | None) -> str | None:
    """Force eager attention for checkpoints that soft-cap their attention logits.

    Gemma-family models with ``attn_logit_softcapping`` set are numerically wrong
    under sdpa: the capping is applied inside the attention kernel, which the fused
    path skips. It produces plausible logits rather than an error, so it has to be
    decided from the config instead of noticed later.
    """
    if attn_implementation != "sdpa":
        return attn_implementation
    try:
        config = AutoConfig.from_pretrained(model_id)
    except Exception as exc:  # noqa: BLE001 - a config we cannot read is not a reason to fail the load
        logger.debug(
            "Could not read the config of %s (%s); keeping attn_implementation=%r", model_id, exc, attn_implementation
        )
        return attn_implementation
    for candidate in (config, getattr(config, "text_config", None)):
        if candidate is not None and getattr(candidate, "attn_logit_softcapping", None):
            logger.info("%s soft-caps attention logits; using eager attention instead of sdpa", model_id)
            return "eager"
    return attn_implementation


def _from_pretrained(auto_cls, model_id: str, attn_implementation: str | None, kwargs: dict):
    """One auto class's load, with the existing attention-backend fallback."""
    try:
        return auto_cls.from_pretrained(model_id, attn_implementation=attn_implementation, **kwargs)
    except (ValueError, TypeError) as exc:
        logger.info("attn_implementation=%r unavailable (%s); using the default backend", attn_implementation, exc)
        return auto_cls.from_pretrained(model_id, **kwargs)


def _load_weights(model_id: str, attn_implementation: str | None, kwargs: dict):
    """Load the checkpoint through the first auto class that accepts it."""
    failures: list[str] = []
    for auto_cls in [AutoModelForCausalLM, *_wrapper_classes()]:
        try:
            model = _from_pretrained(auto_cls, model_id, attn_implementation, kwargs)
        except torch.cuda.OutOfMemoryError:
            # Retrying three more times cannot help and only burns rented minutes.
            raise
        except Exception as exc:  # noqa: BLE001 - each class reports its own refusal differently
            failures.append(f"{auto_cls.__name__}: {type(exc).__name__}: {exc}")
            continue
        if auto_cls is not AutoModelForCausalLM:
            logger.warning(
                "%s is not registered as a causal LM; loaded through %s. Only its text path is used - "
                "any vision or audio towers in the checkpoint are dead weight.",
                model_id,
                auto_cls.__name__,
            )
        return model
    detail = "; ".join(failures)
    raise RuntimeError(f"Could not load {model_id} through any auto class -- {detail}")


def _load_tokenizer(model_id: str):
    """The text tokenizer, taken from the processor when the repo has no standalone one."""
    try:
        return AutoTokenizer.from_pretrained(model_id, padding_side="left")
    except Exception as exc:  # noqa: BLE001 - multimodal repos may only ship a processor
        logger.info("AutoTokenizer failed for %s (%s); falling back to the processor's tokenizer", model_id, exc)
        processor = transformers.AutoProcessor.from_pretrained(model_id)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise
        tokenizer.padding_side = "left"
        return tokenizer


def _assert_lm_head(model, tokenizer, device: str) -> None:
    """Confirm the loaded object actually emits vocabulary logits, with one tiny forward.

    Every attack in this framework reads a next-token logit vector: the gray-box
    scorers take P('Yes') vs P('No') from it, and Koga votes on its argmax. A
    multimodal wrapper that returns hidden states instead would give a
    well-shaped, entirely meaningless score for every query, and nothing
    downstream would notice. Two tokens of compute is a cheap place to find out.
    """
    encoded = tokenizer("Hello", return_tensors="pt")
    with torch.no_grad():
        output = model(
            input_ids=encoded["input_ids"].to(device),
            attention_mask=encoded["attention_mask"].to(device),
        )
    logits = getattr(output, "logits", None)
    if logits is None:
        raise RuntimeError(f"{type(model).__name__} returned no `logits`; this framework cannot score with it.")
    width = int(logits.shape[-1])
    if width < len(tokenizer):
        raise RuntimeError(
            f"{type(model).__name__} emits {width}-wide vectors but the tokenizer has {len(tokenizer)} "
            "tokens - this looks like hidden states, not vocabulary logits."
        )
    logger.info("Logit head verified: %d-wide over a %d-token vocabulary", width, len(tokenizer))


def load_generator(
    model_id: str,
    device: str,
    *,
    dtype: str | torch.dtype | None = None,
    attn_implementation: str | None = "sdpa",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a causal-LM generator shared by all pipelines.

    The tokenizer is configured the way Grislain's ``DPModel`` expects
    (left padding, a pad token) and given a fallback chat template if it lacks one.

    Args:
        model_id: HuggingFace model id.
        device: ``"cuda"`` / ``"cpu"``; the model is moved here after loading.
        dtype: Compute dtype, or ``None`` for the per-device default
            (see :func:`resolve_dtype`).
        attn_implementation: Attention backend. ``"sdpa"`` is faster and uses less
            memory than eager; silently falls back if the architecture lacks it.
    """
    torch_dtype = resolve_dtype(dtype, device)
    logger.info("Loading generator %s on %s (%s)", model_id, device, torch_dtype)

    tokenizer = _load_tokenizer(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        logger.info("Generator tokenizer has no chat template; installing a minimal fallback")
        tokenizer.chat_template = _FALLBACK_CHAT_TEMPLATE

    kwargs = {_dtype_kwarg(): torch_dtype, "low_cpu_mem_usage": True}
    model = _load_weights(model_id, _preferred_attn(model_id, attn_implementation), kwargs)
    model = model.to(device).eval()

    if model.dtype != torch_dtype:
        logger.warning(
            "Requested dtype %s but the model loaded as %s — check the transformers version "
            "(dtype kwarg is %r here). A 9B model in fp32 will not fit in 48 GB.",
            torch_dtype,
            model.dtype,
            _dtype_kwarg(),
        )
    if supports_thinking(tokenizer):
        logger.info("%s is a thinking model; chat_prompt() will disable thinking mode", model_id)
    _assert_lm_head(model, tokenizer, device)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------


def supports_thinking(tokenizer: AutoTokenizer) -> bool:
    """Whether the tokenizer's chat template understands ``enable_thinking``.

    Reasoning models (Qwen3 / Qwen3.5) emit a ``<think>...</think>`` block before
    the answer unless it is switched off in the template. Older chat models
    (Qwen2.5) do not accept the flag at all, so it must not be passed to them.
    """
    template = getattr(tokenizer, "chat_template", None) or ""
    return "enable_thinking" in template


def chat_prompt(tokenizer: AutoTokenizer, messages: list, **kwargs):
    """Apply the chat template with thinking mode disabled.

    Every attack in this framework reads the model's **first** generated token —
    the gray-box scorers take P('Yes') vs P('No') from the first-token logits, and
    the black-box ones generate a handful of tokens. On a thinking model the first
    token is ``<think>``, so those scores would measure the model's propensity to
    reason rather than membership. Disabling thinking keeps the first token the
    answer itself (and makes generation an order of magnitude cheaper).

    Args mirror ``tokenizer.apply_chat_template``; ``enable_thinking=False`` is
    injected only for templates that accept it, and can still be overridden by the
    caller.
    """
    if supports_thinking(tokenizer):
        kwargs.setdefault("enable_thinking", False)
    return tokenizer.apply_chat_template(messages, **kwargs)


# ---------------------------------------------------------------------------
# Forwarding
# ---------------------------------------------------------------------------

# Resolved per model class: transformers renamed ``num_logits_to_keep`` to
# ``logits_to_keep`` in 4.49, and not every architecture supports either.
_LOGITS_KWARG_CACHE: dict[type, str | None] = {}


def _logits_kwarg(model) -> str | None:
    cls = type(model)
    if cls not in _LOGITS_KWARG_CACHE:
        try:
            params = inspect.signature(model.forward).parameters
        except (TypeError, ValueError):
            params = {}
        found = None
        for name in ("logits_to_keep", "num_logits_to_keep"):
            if name in params:
                found = name
                break
        _LOGITS_KWARG_CACHE[cls] = found
    return _LOGITS_KWARG_CACHE[cls]


def forward_logits(model, *, keep: int = 1, **inputs):
    """Forward pass returning only the last ``keep`` positions' logits.

    A full-sequence logit tensor costs ``batch x seq_len x vocab_size`` floats. For
    Qwen3.5-9B (vocab 248,320) a batch of 16 voter prompts of 800 tokens is ~6.4 GB
    in bf16 — larger than the activations and enough to OOM a 48 GB card on its
    own. Restricting the computation to the positions actually scored removes that
    cost. When the architecture does not support the kwarg the tensor is sliced
    after the fact (correct, just not cheaper).

    Args:
        model: A causal LM.
        keep: Number of trailing positions to return logits for. Must be >= 1
            (transformers treats 0 as "keep everything").
        **inputs: Forward arguments (``input_ids``, ``attention_mask``,
            ``past_key_values``, ``use_cache``, ...).

    Returns:
        ``(output, logits)`` where ``logits`` has shape ``[batch, keep, vocab]``.
        ``output`` is the full model output, so ``past_key_values`` stays reachable.
    """
    if keep < 1:
        raise ValueError(f"keep must be >= 1, got {keep}")
    kwarg = _logits_kwarg(model)
    if kwarg is not None:
        inputs[kwarg] = keep
    output = model(**inputs)
    logits = output.logits
    if logits.shape[1] > keep:
        logits = logits[:, -keep:, :]
    return output, logits


def last_token_logits(model, **inputs) -> torch.Tensor:
    """Next-token logits for every sequence in the batch, shape ``[batch, vocab]``."""
    _, logits = forward_logits(model, keep=1, **inputs)
    return logits[:, -1, :]


class Embedder:
    """CLS-pooled, L2-normalised dense encoder.

    Mirrors Grislain's ``PUPVectorStore.encode`` (CLS pooling + L2 norm) so the
    same loaded weights can back both Grislain's retrieval and Koga's retrieval.
    """

    def __init__(self, model_id: str, device: str, *, batch_size: int = 64) -> None:
        logger.info("Loading embedder %s on %s", model_id, device)
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        """Encode texts to L2-normalised CLS embeddings, shape ``[n, dim]``.

        Encoding runs in batches so that indexing a corpus of a few thousand
        documents does not build one oversized padded tensor.
        """
        if not texts:
            return torch.empty(0, device=self.device)
        chunks: list[torch.Tensor] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
            output = self.model(**encoded, return_dict=True)
            embeddings = output.last_hidden_state[:, 0]
            embeddings = embeddings / torch.sqrt(torch.sum(torch.square(embeddings), dim=1, keepdim=True))
            chunks.append(embeddings)
        return torch.cat(chunks, dim=0)
