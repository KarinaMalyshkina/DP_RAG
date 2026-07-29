# Decisions needed

Open items arising from
[`docs/empirical-coverage-report.md`](empirical-coverage-report.md). **None of these
were decided or implemented** — each one is either a scope question, a
measurement-design change, or a resource-sizing call that needs a decision before it
is built.

Written 2026-07-29, against `feature/unified-evaluation-methodology`.

---

## 1. Property inference does not exist — the "three attacks" claim is currently two

**Status:** no code at all. `src/dp_rag_eval/attacks/` contains `base.py`,
`canary.py`, `membership.py` and `__init__.py`; no property-inference symbol appears
in `attacks/__init__.py`'s `__all__`, and there is no construction site anywhere
outside prose.

The name appears only in the thesis proposal (contributions, functional
requirements, roadmap), where it is planned but not implemented.

**The claim that needs adjusting either way:** the thesis proposal states three attacks
as a contribution. In code it is two — canary extraction and membership inference.

**Why it is not being fixed here.** Implementing property inference is not a gap-fill.
It needs a defined estimand (which aggregate property, over which attribute), an
attribute-labelled corpus that HealthCareMagic does not currently provide, a decision
about whether the property is a corpus statistic or a per-document one, and its own
place in the calibration and feasibility machinery. That is a multi-day task with
research content, not a wiring change.

**Decision required:** go / no-go on implementing it. If no-go, the thesis proposal and
the thesis text should say two attacks, and the framework contribution should be framed
around the two that exist plus the evaluation layer around them.

---

## 2. How to formalize the "isolated canary effect"

The coverage report marks this **PARTIALLY FOUND**: a matched with-canary vs
without-canary contrast exists only under the *membership* metric, and even there it
was never computed.

Two ways to make it a real measurement:

**(a) Original-vs-canary AUC delta under the membership metric — cheap.**
The rows already exist. `variant_configs("base", …)` emits both `("original", {})`
and `("canary", …)`, `run_attack.py` applies each to the same member corpus at the
same seed and epsilon, and after this pass `scripts/report_degradation.py` computes
the delta and its seed-pooled CI with no further work at all. Cost: zero additional
GPU time on runs already planned.

**(b) A true no-canary control under the exposure metric — bigger.**
This is the honest version of the contrast, but the exposure metric is *undefined*
without a canary: with none injected, `CanaryExtractionAttack.run` short-circuits and
returns `{"n_canaries": 0}` with no exposure value
(`src/dp_rag_eval/attacks/canary.py:150-152`). A control arm therefore requires
defining what "exposure with no secret present" even means — for instance, scoring a
never-injected code against distractors to establish the floor the metric sits at by
construction. That is a metric-design change and needs to be defended in the thesis.

**Recommendation (not a decision):** (a) as a quick win now, since it costs nothing
and produces a number; (b) only if the thesis needs to claim the exposure metric
itself is calibrated against a null. **Not picked — your call.**

See caveat 3 below: (a) has a targeting weakness that partly limits what it measures.

---

## 3. MIA targets are sampled before the variant is applied

`scripts/run_attack.py` draws its member targets from `members_base` **before** the
variant transforms the corpus:

```python
members_base, non_members = split_members(...)
targets = [(d, True) for d in seeded_sample(members_base, ...)] + ...
for cell_spec in variants:
    indexed_corpus = make_variant_cell(cell_spec, seed).apply(members_base)
```

The scored target texts are therefore the *pre-injection*, *pre-duplication*
documents. Consequences:

- The canary secret is never itself a target, so option 2(a) above measures "does
  injecting canaries anywhere in the corpus change membership leakage for ordinary
  documents", not "can the canary document be identified as a member".
- Under `full_duplicate`, a target that happens to have been duplicated is scored
  with its original text, which is correct for a membership question but means the
  attack does not preferentially probe the duplicated documents.

**Why this is not a bug.** Holding the target set fixed across variants is what makes
`AUC_dup − AUC_orig` a clean contrast: the same questions are asked of every corpus,
so the difference is attributable to the corpus and not to a changed target sample.
Sampling targets after the variant would make the comparison across variants weaker
even as it made each individual cell more pointed.

**Decision required:** whether to add a *second*, variant-aware target set (e.g.
targets drawn from injected canary hosts and duplicated documents) alongside the
fixed one, or to state the fixed-target design as a deliberate limitation. This is a
measurement-design change either way.

---

## 4. The actual grid for the server sweep

The argparse defaults are dev-scale and collapse the joint grid to a single point:
`--epsilons inf`, `--canary-repeats 1`. `docs/RUNBOOK_GPU.md` contains worked example
commands, but no committed grid for the paper.

What needs pinning down before renting the box:

| axis | RUNBOOK example | committed value |
|---|---|---|
| ε (MIA, group 2) | `inf,0.5,1,2,4,8,16,32` | — |
| ε (duplication, group 3) | `inf,4,16` | — |
| ε (canary) | `inf,4,8,16,32` | — |
| `--dup-rates` | `0.1 … 0.9` (9 points) | — |
| `--dup-copies` | `1,2,3,5,10` | — |
| `--overlap-rates` | `0.1 … 0.9` (9 points) | — |
| `--canary-repeats` | `1,2,5,10` | — |
| seeds | `42,43,44` | — |

Constraints worth carrying into that decision: Koga and Grislain only overlap at
ε ∈ {4, 8} at `n_tokens=32` (decision of 2026-07-28), so ε values outside the
overlap only support within-system statements; and the duplication grid now carries
its zero-duplication control, which is one extra cell per (pipeline, mode, ε, seed)
and is what every delta in `degradation.md` is computed against.

---

## 5. Resource sizing for Qwen3.5-9B (flagged, deliberately not resized)

The compatibility sweep found **no blockers** for `Qwen/Qwen3.5-9B`: no hardcoded
model id in `build_pipeline`, no context-length assumption anywhere (no
`max_length` / `max_position_embeddings` / prompt truncation — only `max_new_tokens`),
`supports_thinking` / `chat_prompt` already handle the `<think>` block, `resolve_dtype`
already documents the fp32-on-48 GB trap, and `forward_logits` is already sized
against the 248,320-token vocabulary.

`--generator` now defaults to `Qwen/Qwen3.5-9B` in `run_attack.py`, `run_canary.py`
and `run_utility.py`. `verify_canary.py`, `verify_optimizations.py` and
`run_smoke.py` keep small defaults on purpose — they are the CPU dry-run path.

These defaults would need adjusting for a real run and were **not** changed, because
each is a cost decision:

| flag | current default | RUNBOOK uses | note |
|---|---|---|---|
| `run_attack --members` / `--non-members` | 40 / 40 | 1000 / 1000 | |
| `run_attack --targets-per-class` | 10 | 250 | TPR@FPR=1% needs ≥100 to resolve at all |
| `run_canary --corpus-size` | 200 | 1000 | |
| `run_utility --corpus-size` / `--n-queries` | 200 / 25 | 1000 / 200 | |
| `run_utility --max-answer-chars` | 400 | 176 | must be ~5.5 × `n_tokens` or F1 is capped |

VRAM drivers on a 48 GB card, in order:

1. **Koga's voter batch.** `--auto-n-split-cap 128` means up to 128 voter prompts
   forwarded together. That batch, not the ~18 GB of bf16 weights, is what fills the
   card. `--n-split` must not be lowered to save time — it silently invalidates the
   low-ε cells; cut `--targets-per-class` instead.
2. `--top-k 5` (Grislain forwards one prompt per retrieved document plus the prior).
3. `--mc-samples 200` (Koga's sampled `unified` score) and `--n-distractors 255`
   (canary candidate set) drive wall-clock rather than memory.
4. `Embedder(batch_size=64)` — negligible next to the generator.

**The 262K context is not exercised.** `--max-doc-chars 300` × `--top-k 5` keeps
prompts near 1K tokens. If the long context is meant to be part of the story,
`--max-doc-chars` is the lever — and raising it raises the Koga voter batch cost
proportionally.

---

## 6. Koga's `unified` score has no resolution at these probabilities (measured)

**Blocking. Found on the A6000 pod, 2026-07-29, Qwen3.5-9B.** The calibration cells
in `runs/calib` give:

| pipeline | ε | AUC | distinct scores over 50 targets | score range |
|---|---|---|---|---|
| grislain | inf | 0.802 | 50 | 0.0 … 5.0e-4 |
| koga | inf | **0.500** | **1** | 0.0 … 0.0 |

Koga's score is exactly 0.0 for every target, member and non-member alike, which is
what produces an AUC of exactly 0.5 — at ε = **infinity**, the non-private reference
that should leak the most.

**This is not the Limited-Domain infeasibility the framework already warns about**
(ε=inf runs a plain plurality vote, no LD selection, and `ld_failure_rate` does not
apply). It is a resolution mismatch:

- The estimand is P(the system answers 'Yes'). On this prompt the model essentially
  never says Yes: Grislain's *exact* values top out at 5e-4.
- Koga has no closed form, so the score is estimated by sampling the mechanism
  `--mc-samples` times (default 200). 200 draws resolve probabilities no finer than
  5e-3 — **an order of magnitude coarser than the signal**. Every draw returns 'No',
  so the estimate is identically zero and carries no information.
- Resolving 5e-4 by sampling needs ~1e5–1e6 draws per target. At 21 minutes per 25
  targets with 200 draws, that is not a tuning knob.

The framework's headline comparison — one estimand, exact where closed-form, sampled
where not — therefore does not currently work for Koga on this task. `upper_bound`
(pre-noise logits) and `black_box` (parsed text) are unaffected but are, respectively,
flat in ε by construction and binary.

**Options, none of them chosen:**

1. **Change the attack prompt so P('Yes') is not vanishingly small.** If the model
   genuinely hesitates, both pipelines move into a range 200 draws can resolve. This
   is the cheapest thing to try and it fixes the estimand rather than working around
   it — but it changes the attack, so the paper's numbers are not comparable to
   anything measured before it.
2. **Give Koga a different estimator** — e.g. the vote-histogram share for 'Yes' as
   the mechanism's own notion of probability. Resolves at 1/n_split rather than
   1/mc_samples, but it is a different quantity and needs defending as such.
3. **Report Koga on `upper_bound` + `black_box` only** and state the sampled unified
   score's lack of resolution as a finding about the mechanism's auditability. Honest,
   costs nothing, but weakens the cross-pipeline claim to a qualitative one.
4. Raise `--mc-samples`. Ruled out by the arithmetic above.

**Do not start the Koga part of the matrix before deciding.** Every `unified` cell it
produces at present is an identical 0.5.

---

## 7. Koga's finite-ε cells cost ~3.5 h each (measured)

`koga` at ε=4 with the default `--auto-n-split-cap 128` ran **21 minutes for 25
targets** and had not finished; VRAM sat at 40.6 / 47.4 GiB, so the voter batch is
also near the card's limit. Scaling to the production `--targets-per-class 250` gives
roughly **3.5 hours per cell**, against ~4 minutes for a Grislain cell (22-24 s at 25
targets).

Group 2 alone schedules Koga over 4 variants × 3 seeds × 4 reachable ε × 2 modes ≈ 96
cells. At 3.5 h that is ~330 GPU-hours for one group — far beyond the budget the plan
assumed, and this is before groups 3, canary and utility.

Also worth knowing: ε=1 was correctly recorded `INFEASIBLE` in 0 s (it needs 462
voters at `n_tokens=32`, above the 128 cap), so the reachable-range machinery works as
designed on real hardware and costs nothing.

Levers, in the order the runbook recommends them, all of which are yours to pick:
cut `--targets-per-class` to 100; run Koga at one ε rather than four; drop to one or
two seeds for Koga; lower `--auto-n-split-cap` (invalidates the low-ε cells, so it
trades range for time); or shorten `--n-tokens`, which must then also change the
utility runs.

Note this interacts with item 6: if Koga's `unified` score is not usable, the cheapest
resolution may be to not run those cells at all.
