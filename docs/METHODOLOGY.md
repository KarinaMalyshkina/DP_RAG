# Evaluation methodology for DP-RAG systems

This document defines what is measured and how, so that results from different
DP-RAG systems can be placed on the same axes. It is the written form of RQ2 —
*what components are necessary for a unified evaluation framework* — and it is the
reference for Chapter 3 of the thesis.

The short version: **nothing is compared until it has been calibrated to the same
guarantee, and every quantity plotted has a single definition that does not depend
on which mechanism produced it.**

---

## 1. The privacy unit

The guarantee is **document-level**: adding or removing one document from the
retrieval corpus. Formally, for corpora `D` and `D'` differing in one document and
any measurable set `S` of outputs,

> Pr[M(D) ∈ S] ≤ e^ε · Pr[M(D') ∈ S] + δ

Both integrated baselines claim this unit, by different routes:

- **Koga** splits retrieved documents into disjoint voter groups, so one document
  changes at most one voter's context and therefore at most one vote.
- **Grislain** places each retrieved document in its own prompt and clips each
  document's contribution before aggregation, so one document's influence on the
  output logits is bounded.

A new system must state which unit it provides. A system with a *user*-level or
*query*-level unit is not comparable to these without conversion and must not be
plotted on the same axis.

**This is also where the thesis question bites.** The unit is one document, but the
guarantee is stated over the corpus as a *set of independent records*. When the same
content appears in `k` documents, removing one leaves `k-1` copies, so the
protection an individual actually receives is weaker than the per-document ε
suggests. Measuring that gap is the point of the duplication variants.

---

## 2. The common axis: total (ε, δ) per answer

A point on the x-axis is:

> the total (ε, δ) guarantee for producing **one complete answer of `n_tokens`
> tokens**, including every stage that touches the private corpus.

Three things are deliberately fixed by that definition.

**It is a total, not a per-step budget.** Mechanisms differ in how many internal
private operations they perform — Koga runs one noisy selection per generated
token, Grislain runs one exponential mechanism for retrieval plus one per token.
A per-step ε is therefore not a shared unit. Reporting one would make a mechanism
with more steps look more private than it is.

**It includes retrieval.** Grislain spends part of its budget choosing which
documents to admit. Omitting that stage would understate its cost relative to Koga,
which retrieves without noise.

**It is stated for a specific answer length.** Composition cost grows with the
number of generated tokens, so an ε quoted without `n_tokens` is meaningless. All
experiments — privacy and utility alike — must use the same `n_tokens`, otherwise
the two halves of the privacy-utility curve describe different systems.

### Choice of δ

δ is fixed at **1e-6** for every cell.

The requirement is δ ≪ 1/n, where n is the corpus size. At or above 1/n the
definition permits a mechanism that publishes one randomly chosen record verbatim
while still technically satisfying (ε, δ)-DP. The largest corpus in the experiment
matrix is roughly 10,000 documents (1,000 base documents with nine duplicate copies
each), giving 1/n ≈ 1e-4; δ = 1e-6 is two orders of magnitude below that.

δ is **fixed rather than tied to n**. The duplication sweep deliberately varies the
corpus size, so a δ defined as a function of n would drift from cell to cell and the
cells would stop being comparable. Fixing it at a value valid for the largest corpus
keeps the guarantee at least as strong everywhere else.

The cost of the strict δ is modest and, importantly, not order-of-magnitude.
Tightening δ from 1e-5 to 1e-7 raises Koga's voter requirement by 40–54% (at
`n_tokens=32`: 704 → 1083 voters at ε=0.5, 58 → 81 at ε=8), while ε enters the
same threshold logarithmically-over-linearly and dominates it. In other words, δ
can be chosen on principled grounds rather than traded for feasibility.

---

## 3. Calibration

Given a target, each pipeline **solves its own parameters** so that **its own
privacy accountant** reports that target. Implemented in
`src/dp_rag_eval/privacy/calibration.py`.

Using the method's own accountant is a deliberate constraint. Re-accounting a
published mechanism with a tighter analysis would mean comparing our analysis
against someone else's rather than comparing the two methods.

**Koga.** Its accountant does not accept a δ to evaluate at — it *derives* the total
as `δ₁ + δ_step · n_tokens`. So δ₁ and δ_step are chosen to sum exactly to the
target δ, and the per-selection ε is then found by bisection on
`compose_subsampled_LimitedDomain_to_approxDP` until the composed total equals the
target ε.

**Grislain.** A single scale is applied to the retrieval and generation budgets in a
fixed proportion (`--retrieval-share`, default 0.1) and solved outward until the
PLD composition of the two stages, evaluated at the target δ, equals the target ε.
Handing the target straight to the stages does not give the target back: the error
is a few percent and its sign depends on the answer length.

**Non-private baseline.** ε = ∞ selects each method's own published non-private
algorithm — plain plurality voting for Koga, unclipped aggregation and a non-DP
retrieval threshold for Grislain — not DP with the noise turned down. It is a
reference point, not a limit of the sweep.

Results record `realized_epsilon` and `realized_delta`. **Plot against those**, not
against the requested value.

---

## 4. Feasibility is a result, not an error

Some targets cannot be met by some mechanisms at any parameter setting.

Koga's Limited-Domain selection returns a token only when one candidate leads by
about `1 + log(2/δ_step)/ε_step` votes. A histogram over `n_split` voters cannot
exceed `n_split`, so below some ε the mechanism returns nothing regardless of the
data. Required votes at δ = 1e-6, by target ε and answer length:

| ε_total | n_tokens=1 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| 0.5 | 35 | 143 | 287 | 555 | 890 |
| 1 | 18 | 71 | 143 | 285 | 462 |
| 2 | 10 | 36 | 72 | 145 | 242 |
| 4 | 6 | 19 | 37 | 74 | 128 |
| 8 | 4 | 11 | 20 | 38 | 69 |

(Regenerate with `scripts/report_feasibility.py`; the table is computed, not typed.)

An attack run against an infeasible configuration scores every target identically,
so AUC lands on exactly 0.5 — **visually indistinguishable from a mechanism that
genuinely defeated the attack**. Reporting it as privacy would be wrong.

The framework therefore checks feasibility during calibration, records such cells
with `status=INFEASIBLE` and the reason, and does not run them. The reachable ε
range for each system, at a given voter count and answer length, is itself a
reported finding.

A consequence to state plainly in the thesis: at `n_tokens=32` the two systems have
**barely overlapping reachable ranges** unless Koga is given hundreds of voters.
Comparisons are valid only on the overlap; outside it, the honest statement is that
one system cannot operate at that budget, not that it is more or less private.

### Voter sizing, and the confound it introduces

Because the requirement varies with ε, the framework sizes Koga's voter count per
cell to the minimum that makes the target feasible, capped at 128 (the cap that
buys an overlap with Grislain — ε ∈ {4, 8} — while keeping the voter set below a
quarter of a 1000-document corpus, past which retrieval stops being a selection).

**This means `n_split` co-varies with ε for Koga, and that is a confound.** Part of
any change in Koga's utility across the ε axis comes from the voter count changing
(128 voters at ε=4, 21 at ε=32), not from the noise alone. It is unavoidable: a
fixed count either makes the low end infeasible or wastes the high end, and Koga's
mechanism genuinely requires more voters at tighter budgets. The mitigation is
disclosure — `n_split` is recorded per cell in the results, and any statement about
Koga's privacy-utility slope must say that the configuration moved with the budget.
Grislain has no equivalent confound; its parameters are solved without changing its
retrieval breadth.

---

## 5. The membership signal

One estimand, defined independently of any mechanism:

> **score = P(the deployed system's answer begins with an affirmative token)**,
> for the attack prompt of Anderson et al. (2024).

How it is computed is free; *what* is computed is not.

| pipeline | estimator | why |
|---|---|---|
| naive | exact | output distribution is the softmax over model logits |
| Grislain | exact | output distribution is the softmax over DP-aggregated logits |
| Koga | Monte Carlo, R draws | the mechanism selects a token; there is no closed-form distribution to read |

**Estimator validation.** `GrislainPipeline.membership_score_sampled` computes the
same estimand by sampling, where the exact value is also available. Reporting the
agreement between the two rules out the objection that any difference in attack
success between pipelines is an artefact of measuring them differently.

**What is not the estimand.** `KogaPipeline.membership_score_upper_bound` averages
voter logits *before* the noise. That describes an attacker who can read mechanism
internals, is flat in ε by construction, and must never be plotted on the same line
as the unified score. It is retained because it is free to compute and because the
gap between it and the unified score quantifies how much leakage the noise removes.

For canary extraction the estimand is the log-probability the system assigns to a
candidate secret, and Secret-Sharer exposure `log2(N) − log2(rank)` is computed from
the ranking. Koga has no scorable distribution over arbitrary strings, so its
exposure is an upper bound and is labelled as such.

---

## 6. Utility

Task: answer the patient's question from the corpus. `HealthCareMagicLoader` stores
the patient question in `metadata["query"]` and the doctor's reply as the document
text, so the reply is both a corpus document and the reference answer — the RAG
setting, where the answer lives in the retrieval store.

Metrics (`src/dp_rag_eval/evaluation/metrics.py`), SQuAD-style normalisation:

- **token F1 — primary.** Bag-of-tokens overlap, multiplicity-aware.
- **ROUGE-L** — order-sensitive cross-check. A mechanism that emits the right
  vocabulary in scrambled order scores well on F1 and poorly here.
- **exact match — reported, not interpreted.** References are free-form replies of
  tens of words; EM is ~0 for every system including the non-private baseline. That
  is a property of the corpus, not an effect of DP. It is reported because a
  uniformly zero column is itself evidence that EM is the wrong instrument here.

**Utility must be measured on the same calibrated configuration as the attacks** —
same `n_tokens`, same δ, same ε targets, same corpus variant. A privacy-utility
curve assembled from differently configured cells compares two different systems.

**Reference length must match the answer budget.** A system permitted `n_tokens`
tokens cannot recall a reference many times longer, so F1 is capped by the length
ratio regardless of answer quality — at `n_tokens=32` against a 70-word reference
the ceiling is ≈0.6 even for a perfect system. That is a scaling artefact and would
read as a privacy cost if left uncorrected. Keep `--max-answer-chars` at roughly
`5.5 × n_tokens` (the runner warns when it is not), and state both numbers wherever
utility figures appear.

---

## 7. Controlled corpus conditions

Four variants, all deterministic given a seed
(`src/dp_rag_eval/datasets/variants.py`):

| variant | what it isolates | strength parameter |
|---|---|---|
| `original` | control | — |
| `full_duplicate` | verbatim repetition | `duplication_rate`, `n_copies` |
| `partial_overlap` | shared sentences across documents | `overlap_rate`, `n_hosts` |
| `canary` | a known secret repeated `n_repeats` times | `n_canaries`, `n_repeats` |

Seeds redraw the member/non-member split as well as the variant randomness, so a
seed is a replicate of the whole experiment rather than of the noise alone.

---

## 8. Adding a DP-RAG system

A system enters the framework by implementing `DPRagPipeline`
(`src/dp_rag_eval/core/protocols.py`):

1. `calibrate(target)` — solve your parameters for the target using **your own**
   accountant; return `feasible=False` rather than silently delivering a weaker
   guarantee.
2. `index(documents)`.
3. `answer(query)` — used for utility; must be the same configuration the attacks
   see.
4. `membership_score(query)` — the estimand in §5, exactly or by sampling.
5. `completion_logprob(query, completion)` — for canary ranking.
6. `privacy_spent()` — what your accountant reports for one answered query.

Then declare the privacy unit (§1), and add the pipeline to `runner/sweep.py`.
Calibration tests in `tests/test_calibration.py` are the acceptance criterion:
realized ε must track the target across the grid, and δ must land exactly.

---

## 9. Known limitations

- **The reachable ε ranges of the two baselines barely overlap** at realistic answer
  lengths (§4). Conclusions about their relative privacy hold only on the overlap.
- **Koga's canary exposure is an upper bound**, not the deployed system's leakage,
  because the mechanism has no scorable distribution over arbitrary strings.
- **Black-box scores are binary**, so TPR@FPR=1% degenerates to 0 there regardless
  of leakage. The unified mode exists to avoid making that a finding.
- **Koga's ε=∞ reference point has coarse resolution.** With the noise removed the
  vote is a deterministic plurality, so the output distribution is a point mass and
  the unified score is exactly 0 or 1. That is the correct value of the estimand,
  not an estimator artefact — but it means Koga's non-private AUC is computed from
  binary scores while the other pipelines' remains continuous. Read Koga's ε=∞
  column as a coarse upper reference rather than a precisely located point.
- **Composition across the many queries an attack issues is not accounted.** Each
  cell reports the per-answer guarantee; a real attacker issuing 500 queries faces a
  much larger composed budget. The per-answer figure is the standard reporting unit
  in both source papers, and comparing against it keeps the framework aligned with
  them, but it is an optimistic view of a persistent adversary.
