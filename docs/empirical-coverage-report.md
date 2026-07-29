# Empirical Coverage Report

Audit of `dp-rag-eval`, 2026-07-28, branch `feature/unified-evaluation-methodology` at `4d9dce9`.

Scope: does the code actually *measure* the things the thesis claims to measure — as opposed to defining machinery that is never invoked, or describing an analysis in prose that no code performs. Vendored upstream code in `baselines/koga` and `baselines/grislain` was excluded from all verdicts.

The open decisions that follow from it are listed in `docs/decisions-needed.md`.

## Summary

| # | Question | Verdict |
|---|---|---|
| 1 | Attack success rate as a joint function of duplication rate × ε, per attack | **FOUND** for canary and membership; **MISSING** for property inference (no code exists) |
| 2 | Explicit computed comparison (delta/ratio) between original and duplicated settings | **MISSING** |
| 3 | Measurement isolating canary injection's effect on leakage (with vs without, ε and duplication fixed) | **PARTIALLY FOUND** |

Cross-cutting observation, visible in all three sections: the framework is a *cell producer*, not an analysis pipeline. Every metric in the repo is computed within a single configuration and written as an independent row. Nothing anywhere reads `results.json` back to compare cells — the only re-read is `CheckpointedRun.__init__` rebuilding the resume set from cell keys (`src/dp_rag_eval/runner/sweep.py:285-292`). The architectural slot for this, `evaluation/reporter.py`, is planned in the thesis proposal but was never built; `src/dp_rag_eval/evaluation/` contains only `__init__.py` and `metrics.py`.

---

## 1. Duplication × Epsilon Joint Coverage

### Verdict per attack

| Attack | Implemented | Invoked in a sweep | Joint (duplication × ε) grid | Verdict |
|---|---|---|---|---|
| Canary extraction (Secret Sharer) | Yes | Yes | Yes — `n_repeats` × ε | **FOUND** |
| Membership inference (Anderson et al. 2024) | Yes | Yes | Yes — variant/duplication-strength × ε | **FOUND** |
| Property inference | No code at all | — | — | **MISSING** |

### 1.1 Which attacks exist as real code

- `src/dp_rag_eval/attacks/membership.py` — 226 lines, `RAGMembershipInferenceAttack`, three scoring modes declared at `src/dp_rag_eval/attacks/membership.py:36` (`MODE_METHODS`) and `:41` (`ALL_MODES`). Real implementation.
- `src/dp_rag_eval/attacks/canary.py` — 255 lines, `CanaryExtractionAttack`, exposure + generation modes. Real implementation.
- `src/dp_rag_eval/attacks/base.py:1` — shared `AttackResult` only.
- **Property inference: no source file exists.** `src/dp_rag_eval/attacks/` contains only `base.py`, `canary.py`, `membership.py`, `__init__.py` — the package `__all__` at `src/dp_rag_eval/attacks/__init__.py:13` lists no property-inference symbol. The name appears only in the thesis proposal (contributions, functional requirements, roadmap). The proposal's "three attacks" claim is two attacks in code.

Construction sites (the invocation test):

- `scripts/run_canary.py:333` — `CanaryExtractionAttack(...)` inside the sweep.
- `scripts/run_attack.py:408` — `RAGMembershipInferenceAttack(...)` inside the sweep.
- `scripts/verify_canary.py:82`, `:86` — single-cell verification harness, not a sweep.
- No other construction site exists outside docs.

### 1.2 Dataset variants and reachability

All four exist in `src/dp_rag_eval/datasets/variants.py`: `OriginalVariant:53`, `FullDuplicateVariant:72`, `PartialOverlapVariant:121`, `CanaryInjectionVariant:188`.

Reachability by name string:

- `src/dp_rag_eval/runner/sweep.py:140-149` — `make_variant(name, params, seed)` resolves all four by name via an if/elif chain. Called from `scripts/run_attack.py:334` (MIA) and `scripts/run_utility.py:250` (utility, not an attack).
- `src/dp_rag_eval/runner/sweep.py:93-137` — `variant_configs()` expands `--variant-grid` into `(variant_name, params)` pairs; `base` → four variants at fixed strength (`:109-115`), `duplication` → strength sweeps (`:116-124`). Reached from `scripts/run_attack.py:263` and `scripts/run_utility.py:181`.
- **The `VARIANTS` registry at `src/dp_rag_eval/datasets/variants.py:289-294` is dead.** It is re-exported at `src/dp_rag_eval/datasets/__init__.py:10` and never read anywhere — no lookup site exists; `make_variant` duplicates the mapping as an if/elif chain instead.
- `scripts/run_canary.py:248` constructs `CanaryInjectionVariant` **directly**, bypassing `make_variant`/`variant_configs` (its import block at `scripts/run_canary.py:53-64` does not import either). Consequence: `full_duplicate` and `partial_overlap` corpora are structurally unreachable from the canary-extraction sweep.
- `scripts/run_smoke.py:41` names all four variants but loads pre-materialised JSONL written by `scripts/prepare_dataset.py:63-67`; it takes a single scalar `--epsilon` (`scripts/run_smoke.py:67`, default 3.0) and runs no attack class.

### 1.3 Sweep structure — which axes are actually swept together

**Membership inference — `scripts/run_attack.py`.** Nested loops, all in one process:

- `:319` `for seed in seeds:` → `:333` `for variant_name, variant_params in variants:` → `:343` `for epsilon in epsilons:` → `:345` `for name in pipeline_names:` → `:394` `for mode in pending:`
- Corpus is rebuilt per variant at `:334`; the privacy target is built per epsilon at `:344`.
- This is a **genuine joint grid**: duplication variant (and, under `--variant-grid duplication`, duplication *strength* — rates, copy counts, overlap rates, canary repeats from `src/dp_rag_eval/runner/sweep.py:116-124`) crossed with epsilon in the same run.
- Caveat on defaults: `--epsilons` defaults to `"inf"` (`scripts/run_attack.py:140`) and `--variant-grid` to `base` (`:145`). A bare `python scripts/run_attack.py` therefore executes **4 variants × 1 epsilon** — the epsilon axis collapses unless multiple values are passed. Multi-axis invocations exist only in documentation: `scripts/run_attack.py:34-41` (docstring usage) and `docs/RUNBOOK_GPU.md:92-99` (group 2: 4 variants × 8 epsilons) and `docs/RUNBOOK_GPU.md:136-143` (group 3: ~26 duplication configs × `inf,4,16`).
- Note the RUNBOOK deliberately splits the two axes into separate commands: group 2 fixes duplication at base strength and sweeps ε; group 3 sweeps duplication strength at 3 ε values (`docs/RUNBOOK_GPU.md:130-160`, including the explicit instruction "Run Koga's part of this group at ε=8 only"). Group 3 is still joint (26 × 3), but the plan does not run one full strength × full ε grid.

**Canary extraction — `scripts/run_canary.py`.** Nested loops:

- `:239` `for seed in seeds:` → `:247` `for n_repeats in repeats_grid:` → `:259` `for epsilon in epsilons:` → `:261` `for name in pipeline_names:` → `:310` `for mode in pending:`
- Corpus re-injected per repeat level at `:248-249`; target built per epsilon at `:260`.
- **Genuine joint grid**, but the duplication axis is *only* secret-repetition (`n_repeats`). The corpus-level duplication variants (`full_duplicate`, `partial_overlap`) are never applied here (see 1.2).
- Defaults collapse both axes: `--canary-repeats` defaults to `"1"` (`:109`) and `--epsilons` to `"inf"` (`:112`) → a bare invocation runs one repeat level at one budget. The joint form appears in the docstring `:31-32` and `docs/RUNBOOK_GPU.md:169-175` (`--epsilons inf,4,8,16,32 --canary-repeats 1,2,5,10`).

**Property inference.** No loop, no script, no class. Nothing to sweep.

### 1.4 Per-cell keying of results

Both attack sweeps key every written row by the full configuration:

- MIA cell key: `scripts/run_attack.py:348` — `f"{name}|%s|{variant_name}|{param_str(variant_params)}|{target.label()}|seed={seed}"` → (pipeline, mode, variant, variant-params, ε-target, seed). CSV columns at `scripts/run_attack.py:90-115` include `pipeline, mode, variant, params, epsilon, realized_epsilon, realized_delta, seed, status`.
- Canary cell key: `scripts/run_canary.py:265` — `f"{name}|%s|repeats={n_repeats}|{target.label()}|seed={seed}"`. CSV columns at `scripts/run_canary.py:71-92` include `pipeline, mode, n_repeats, epsilon, realized_epsilon, seed, status`.
- Persistence and resume are keyed on that same `cell` string: `src/dp_rag_eval/runner/sweep.py:289` (`self.done = {r["cell"] for r in self.rows}`), `:294` `is_done`, `:297-308` `record`.
- Duplication *strength* is not its own column in the MIA output — it is serialised into a single `params` string by `src/dp_rag_eval/runner/sweep.py:63-65` (`param_str`), e.g. `duplication_rate=0.5,n_copies=1`. The information is present but requires string parsing to use as a numeric axis.
- Infeasible cells are recorded rather than dropped (`scripts/run_attack.py:375-389`, `scripts/run_canary.py:284-306`), so grid holes are explicit.
- There is no aggregation/plotting/analysis script in `scripts/` (contents: `bootstrap_gpu.sh`, `prepare_dataset.py`, `report_feasibility.py`, `run_attack.py`, `run_canary.py`, `run_smoke.py`, `run_utility.py`, `verify_canary.py`, `verify_optimizations.py`). Nothing reads `results.csv` back to produce a variant × epsilon table or surface; the per-cell CSV is the terminal artefact, plus the flat Markdown row-dump at `scripts/run_attack.py:232-249` and `scripts/run_canary.py:168-183`.

### 1.5 What has actually been executed (evidence from `runs/`)

- `runs/m_canary/results.csv` — the only executed joint cell set: `n_repeats ∈ {1,2}` × `epsilon ∈ {inf, 4}` × 3 pipelines, seed 42 (Koga's ε=4 cells `INFEASIBLE`). A 2×2 joint grid.
- `runs/m_attack/results.csv` — `variant=original` only × `epsilon ∈ {inf, 4}`. Epsilon axis only; duplication fixed.
- `runs/m_auto/results.csv` — `variant=original` × `epsilon ∈ {16, 32}`. Epsilon axis only.
- `runs/plan_smoke/results.csv` — `variant=original` × `epsilon ∈ {inf, 2}`. Epsilon axis only.
- `runs/attack/`, `runs/attack_fair/`, `runs/attack_fpr05_t20/`, `runs/attack_naive/`, `runs/attack_naive_fpr05/` — CSV headers contain **no `epsilon` column at all** (e.g. `pipeline,variant,status,auc,...`), i.e. pre-calibration-era runs with no privacy-budget axis.
- No executed MIA run in the repo contains more than one variant, and none contains a duplication-strength sweep.

### 1.6 Explicit axis statement

- **Canary extraction:** axes swept together in one run = `seed × n_repeats (secret duplication) × epsilon × pipeline × mode` (`scripts/run_canary.py:239/247/259/261/310`). Corpus-level duplication variants are not on any axis of this sweep.
- **Membership inference:** axes swept together in one run = `seed × variant(+strength params) × epsilon × pipeline × mode` (`scripts/run_attack.py:319/333/343/345/394`), where the duplication axis is categorical under `--variant-grid base` and numeric (rate / copies / overlap-rate / canary-repeats) under `--variant-grid duplication` (`src/dp_rag_eval/runner/sweep.py:109-124`).
- **Property inference:** no axes; no code.
- Both joint grids collapse to a single epsilon and (for canary) a single repeat level under argparse defaults (`scripts/run_attack.py:140,145`; `scripts/run_canary.py:109,112`); the joint form is only reached by explicit CLI arguments documented in the script docstrings and `docs/RUNBOOK_GPU.md`.

### Third-party note

`baselines/koga` and `baselines/grislain` are vendored upstream implementations; they contribute the pipelines constructed at `src/dp_rag_eval/runner/sweep.py:229-252` but contain no attack, variant, or sweep code of ours and were not counted in any verdict above.

---

## 2. Computed Original-vs-Duplicated Degradation

**VERDICT: MISSING.** No code anywhere in `src/dp_rag_eval/` or `scripts/` computes a delta, ratio, or degradation metric between the `original` variant and any duplicated variant. Every metric is computed strictly within one cell, written as an independent row, and rendered into a flat table. The subtraction `AUC_dup − AUC_orig` exists only in the reader's head.

### 2.1 `evaluation/metrics.py` — complete enumeration (all within-cell)

`src/dp_rag_eval/evaluation/metrics.py` is 128 lines and contains exactly seven public names, all of which operate on a single `(prediction, reference)` pair or a single cell's list of them:

| Name | Line | Scope |
|---|---|---|
| `normalize_answer` | `src/dp_rag_eval/evaluation/metrics.py:26` | one string |
| `exact_match` | `src/dp_rag_eval/evaluation/metrics.py:34` | one pair |
| `token_f1` | `src/dp_rag_eval/evaluation/metrics.py:39` | one pair |
| `_lcs_length` | `src/dp_rag_eval/evaluation/metrics.py:61` | one pair |
| `rouge_l` | `src/dp_rag_eval/evaluation/metrics.py:81` | one pair |
| `METRICS` dict | `src/dp_rag_eval/evaluation/metrics.py:102` | registry of the three above |
| `score_answer` | `src/dp_rag_eval/evaluation/metrics.py:109` | one pair → dict |
| `aggregate_scores` | `src/dp_rag_eval/evaluation/metrics.py:114` | mean over answers **within one cell** |

`aggregate_scores` is the only function with "aggregate" in the name, and its aggregation axis is *answers inside a single configuration* (`src/dp_rag_eval/evaluation/metrics.py:123-126`) — it returns `{token_f1_mean, rouge_l_mean, exact_match_mean, n_answers}`. It has no parameter for a second cell, no baseline argument, and no notion of a variant. There is no `reporter.py`, no `compare.py`, no second module: `src/dp_rag_eval/evaluation/` contains only `__init__.py` and `metrics.py`, and `__init__.py` re-exports exactly the seven names above.

The attack-side metrics are the same story. `roc_auc` (`src/dp_rag_eval/attacks/membership.py:66`), `tpr_at_fpr` (`src/dp_rag_eval/attacks/membership.py:85`) and `_binary_metrics` (`src/dp_rag_eval/attacks/membership.py:110`) all take one `(labels, scores)` pair from one indexed corpus. The emitted metric dict at `src/dp_rag_eval/attacks/membership.py:210-217` is `{auc, tpr_at_fpr_01, tpr_at_fpr_05, n_members, n_non_members, accuracy, precision, tpr, fpr, tp, fp, tn, fn}` — no field references any other cell. Canary is identical: `src/dp_rag_eval/attacks/canary.py:202-210` emits `{n_canaries, n_candidates, mean_exposure, median_exposure, max_exposure, max_possible_exposure, extraction_rate, mean_rank}`, and the generation mode at `src/dp_rag_eval/attacks/canary.py:244-247` emits `{n_canaries, extraction_rate}`.

### 2.2 Aggregation layer — does not exist

There is **no reader of `results.json` anywhere in the repo.** Every occurrence of the string is a writer or a doc comment:

- `src/dp_rag_eval/runner/sweep.py:281`, `:304-308` — writes it
- `scripts/run_smoke.py:279`, `scripts/verify_canary.py:103` — write it
- `docs/RUNBOOK_GPU.md:6`, `:301`, `:303` — prose about resuming/deleting rows

The one exception is `CheckpointedRun.__init__` at `src/dp_rag_eval/runner/sweep.py:285-292`, which re-reads `results.json` **solely to rebuild the `done` set for resumability** (`self.done = {r["cell"] for r in self.rows}`, `src/dp_rag_eval/runner/sweep.py:289`). It reads cell *keys*, never metric values.

`CheckpointedRun.finalise` (`src/dp_rag_eval/runner/sweep.py:310-315`) writes `results.csv` via a plain `csv.DictWriter` over `self.rows` and dumps a pre-rendered `report.md` string. No arithmetic.

The three `render_markdown` functions are pure per-row formatters — each loops over rows and formats each independently with no reference to any other row:

- `scripts/run_attack.py:195-250` (row loop at `:237`, emits `AUC | TPR@1% | TPR@5%` per row)
- `scripts/run_canary.py:138-184` (row loop at `:173`)
- `scripts/run_utility.py:130-169` (row loop at `:156`)

`scripts/report_feasibility.py` — despite the name — never touches results. It tabulates `min_voters_for` / `calibrate_koga` / `calibrate_grislain` a priori (`scripts/report_feasibility.py:57-84`); it is a mechanism-capability table, not an experiment analysis.

**There are no notebooks.** A repo-wide search for `*.ipynb` (excluding `.venv/`, `baselines/`) returns nothing. There is no pandas, matplotlib, or plotting dependency used by any framework script. `tests/` contains no comparison logic (`tests/test_metrics.py` tests the within-pair functions).

### 2.3 Results records are fully independent — no baseline reference field

Rows are keyed by `cell` (`src/dp_rag_eval/runner/sweep.py:289`, `:300`), a string built at `scripts/run_attack.py:348`:

```
f"{name}|%s|{variant_name}|{param_str(variant_params)}|{target.label()}|seed={seed}"
```

The variant is *part of the key*, not a field paired against a reference. The full CSV schema at `scripts/run_attack.py:90-115` is `pipeline, mode, variant, params, epsilon, realized_epsilon, realized_delta, seed, status, auc, tpr_at_fpr_01, tpr_at_fpr_05, accuracy, precision, tpr, fpr, n_members, n_non_members, n_indexed, n_split, ld_threshold, ld_failure_rate, seconds, error`. There is **no** `auc_baseline`, `auc_original`, `auc_delta`, `degradation`, or equivalent. Same for canary (`scripts/run_canary.py:71-92`) and utility (`scripts/run_utility.py:65+`). The row assembled at `scripts/run_attack.py:362-373` and `:403` carries only the cell's own config and its own metrics.

Worse for the paired comparison: **the `duplication` grid does not contain the `original` variant at all.** `variant_configs` puts `("original", {})` only in the `base`/`all` branch (`src/dp_rag_eval/runner/sweep.py:110-115`); the `duplication` branch (`src/dp_rag_eval/runner/sweep.py:116-124`) emits only `full_duplicate`, `partial_overlap`, and `canary` configs, with `dup_rates` starting at 0.1 and `overlap_rates` at 0.1 (`src/dp_rag_eval/runner/sweep.py:81-84`) — no zero-duplication anchor point in the sweep. And the RUNBOOK sends that grid to a *different output directory* (`--output-dir runs/mia_duplication`, `docs/RUNBOOK_GPU.md:137-138`) from the base grid (`--output-dir runs/mia_main`, `docs/RUNBOOK_GPU.md:117`), so the original-variant row is not even in the same `results.json` as the duplicated rows it would be compared against.

### 2.4 Doc-vs-code gap — the comparison is described but never computed

The docs repeatedly frame the original-vs-duplicated comparison as the point of the thesis, and in every case the described operation is left to a human:

- `docs/METHODOLOGY.md:36-38` — "the protection an individual actually receives is weaker than the per-document ε suggests. **Measuring that gap is the point of the duplication variants.**" No code measures a gap; it produces two independent numbers.
- `docs/METHODOLOGY.md:244` — variant table labels `original` as "control". It is a control only in the experimental-design sense; nothing consumes it as a reference value.
- `docs/RUNBOOK_GPU.md:132-133` — "This is the RQ1 axis and the largest group (~26 variant configs)."
- `docs/RUNBOOK_GPU.md:153-157` — "RQ1 is a trend *across duplication levels at fixed ε*… **The duplication effect is read from the trend across ~26 configs**." Explicitly a human reading a table. This is the closest the repo comes to naming the analysis, and it names it as a manual act.
- `scripts/run_canary.py:156-159` (rendered into every canary `report.md`) — "**The RQ1 reading:** hold epsilon fixed and vary `n_repeats`. …exposure rising with `n_repeats` is the gap between the formal guarantee and actual protection." Instruction to a reader, printed as prose above a flat table.
- `scripts/run_attack.py:227-229` — "the gap between the two is how much leakage the mechanism removes" (about `unified` vs `upper_bound` modes). Also uncomputed prose; those are separate rows too.
- The thesis proposal, contribution 4, claims "Quantitative privacy-utility trade-off analysis as a joint function of ε and duplication rate." No code performs a joint analysis.
- The thesis proposal's planned tree lists `evaluation/reporter.py # JSON logging, summary generation`. **This file does not exist**; `src/dp_rag_eval/evaluation/` contains only `__init__.py` and `metrics.py`.

The gap is therefore: the docs describe a degradation *reading*, never a degradation *computation*, and the one architectural slot that would have held it (`reporter.py`) was never built.

### 2.5 What IS computed vs what must be done by hand

**Computed by code, as named output fields (all within a single cell):**

- `auc`, `tpr_at_fpr_01`, `tpr_at_fpr_05`, `accuracy`, `precision`, `tpr`, `fpr`, `tp/fp/tn/fn`, `n_members`, `n_non_members` — `src/dp_rag_eval/attacks/membership.py:210-217`
- `mean_exposure`, `median_exposure`, `max_exposure`, `max_possible_exposure`, `extraction_rate`, `mean_rank`, `n_canaries`, `n_candidates` — `src/dp_rag_eval/attacks/canary.py:202-210`
- `token_f1_mean`, `rouge_l_mean`, `exact_match_mean`, `n_answers` — `src/dp_rag_eval/evaluation/metrics.py:114-127`
- `realized_epsilon` / `realized_delta` per cell, and calibration `feasible` / `INFEASIBLE` status — `scripts/run_attack.py:367-368`, `:375-389`
- Mechanism feasibility tables (a priori, not from results) — `scripts/report_feasibility.py:57-84`
- Mean over *answers within one cell* — the only averaging the codebase performs (`src/dp_rag_eval/evaluation/metrics.py:126`)

**Not computed anywhere — must be done by hand from `results.csv`:**

- `AUC_dup − AUC_orig`, `AUC_dup / AUC_orig`, or any named degradation/amplification metric
- Any pairing of a duplicated cell to its matching `original` cell (same pipeline, mode, ε, seed)
- Any trend/slope/regression of AUC or exposure against `duplication_rate`, `n_copies`, `overlap_rate`, or `n_repeats` — the RQ1 quantity itself
- Averaging or variance across the three seeds (`--seeds 42,43,44`); seeds are emitted as separate rows and never pooled — no mean, no std, no confidence interval anywhere in the codebase
- The `unified` vs `upper_bound` "gap" described at `scripts/run_attack.py:227-229`
- Any joining of the utility rows to the attack rows to form the privacy-utility trade-off described at `docs/METHODOLOGY.md:224`
- Any cross-file join between `runs/mia_main` and `runs/mia_duplication`

---

## 3. Isolated Canary-Injection Effect

**Verdict: PARTIALLY FOUND** — a matched with-canary vs. without-canary corpus contrast exists only in the *membership* sweep (`original` vs `canary` variant rows), and even there it is only tabulated, never computed as a contrast. Under the canary/exposure metric itself there is no no-canary control arm at all; `scripts/run_canary.py` is a canary-duplication sweep, not a canary-presence contrast.

### 3.1 What the canary attack measures

- The metric is Secret-Sharer **exposure = log2(n_candidates) − log2(rank)** (`src/dp_rag_eval/attacks/canary.py:73-82`), computed per canary in `_run_exposure` (`src/dp_rag_eval/attacks/canary.py:157-221`); rank is the position of the true code by log-prob among candidates (`src/dp_rag_eval/attacks/canary.py:182`).
- Aggregate metrics reported per cell: `mean/median/max_exposure`, `max_possible_exposure`, `extraction_rate` (fraction ranked 1), `mean_rank` (`src/dp_rag_eval/attacks/canary.py:202-212`). A per-canary score is also emitted in `predictions` (`src/dp_rag_eval/attacks/canary.py:187-199`).
- A second, black-box `generation` mode measures verbatim regurgitation rate only (`src/dp_rag_eval/attacks/canary.py:223-255`).
- **The reference population is internal, not experimental.** `_distractor_codes` (`src/dp_rag_eval/attacks/canary.py:85-101`) draws `n_distractors` random codes "never inserted", and the true code is ranked against them within the same run (`src/dp_rag_eval/attacks/canary.py:168-183`). This is a *within-cell* control over the secret's identity — it answers "is the true secret preferred over random strings of the same shape". It is **not** a no-canary control arm, and it does not tell you how much injecting the canary changed the system's leakage relative to a corpus where the canary was absent.
- The metric is therefore relative-to-distractors but **absolute with respect to corpus condition**: it is undefined without a canary. With no canaries the attack short-circuits and returns only `{"n_canaries": 0.0}` with no exposure value at all (`src/dp_rag_eval/attacks/canary.py:150-152`).

### 3.2 Axes of the canary sweep (`scripts/run_canary.py`)

Cells are (seed × canary repeats × epsilon × pipeline × mode) — stated at `scripts/run_canary.py:3` and realized in the loop nest at `scripts/run_canary.py:239-351`:

- seeds — `scripts/run_canary.py:115`, loop `:239`
- `--canary-repeats` (duplication of the *same* secret) — `scripts/run_canary.py:109`, loop `:247`
- `--epsilons` — `scripts/run_canary.py:112`, loop `:259`
- pipelines / modes — `scripts/run_canary.py:98-99`, loops `:261`, `:310`
- `--n-canaries` is a **fixed scalar, not a swept axis** — `scripts/run_canary.py:108` (default 10), consumed once at `scripts/run_canary.py:248`

**There is no control cell.** Every cell constructs `CanaryInjectionVariant(n_canaries=args.n_canaries, n_repeats=n_repeats, seed=seed)` and indexes the injected corpus (`scripts/run_canary.py:248-249`, `:308`). `OriginalVariant` is never imported or used by this script (import block `scripts/run_canary.py:48-64` pulls no `make_variant`/`variant_configs`). The output schema has no `variant` column — only `n_repeats` (`scripts/run_canary.py:71-92`), i.e. corpus condition is not even a recorded dimension.

### 3.3 Is `n_repeats=1` or `n_canaries=0` treated as a control?

- `n_repeats=1` is only the *default value* of the repeats grid (`scripts/run_canary.py:109`) and the "base" strength in the shared grid (`src/dp_rag_eval/runner/sweep.py:89`). Nothing is computed against it: `render_markdown` emits one flat row per cell with no differencing (`scripts/run_canary.py:138-184`), and `CheckpointedRun.finalise` only writes rows to JSON/CSV (`src/dp_rag_eval/runner/sweep.py:297-314`). No delta, ratio, or baseline-subtracted column exists anywhere in the repo; there is no aggregation/analysis script (no `analy*`/`aggregate*`/`plot*` files exist).
- `n_canaries=0` is never used by any runner: `scripts/run_canary.py:108` default 10, `src/dp_rag_eval/runner/sweep.py:90` `n_canaries: int = 10`, `scripts/verify_canary.py:50` default 2. It is not treated as a control anywhere.

### 3.4 `CanaryInjectionVariant` with zero canaries

- Structurally it would work: the injection loop is `for k in range(self.n_canaries)` (`src/dp_rag_eval/datasets/variants.py:255`), so `n_canaries=0` returns a copied, untagged-as-host corpus with `injected_canaries == []` (`src/dp_rag_eval/datasets/variants.py:230-243`). Default is 20 (`src/dp_rag_eval/datasets/variants.py:213`).
- **That path is never exercised.** No runner passes 0 (see refs in 3.3), and no test covers it — the variant tests all use positive counts (`tests/test_dataset_variants.py:112`, `:128`).
- Even if exercised, it yields no measurement to contrast against: the attack returns only `n_canaries=0.0` and empty predictions (`src/dp_rag_eval/attacks/canary.py:150-152`), so the "leakage without canary" side of the contrast has no numeric value in the exposure metric.

### 3.5 Can canary-present vs canary-absent be measured on the same corpus config?

- **Yes, but only under the membership metric.** `variant_configs("base", ...)` emits both `("original", {})` and `("canary", {"n_canaries": ..., "n_repeats": ...})` (`src/dp_rag_eval/runner/sweep.py:109-115`), and the duplication grid adds a canary-repeats ladder (`src/dp_rag_eval/runner/sweep.py:124`). `scripts/run_attack.py` loops over these variants applying each to the *same* `members_base` at the same seed and same epsilon (`scripts/run_attack.py:333-334`, epsilon loop `:343`), recording the variant name per row (`scripts/run_attack.py:364-365`). So `AUC(original)` vs `AUC(canary, n_repeats=1)` at fixed epsilon/seed is a genuine matched with/without-canary pair — it is just never *computed*; the rows are dumped and rendered as a table (`scripts/run_attack.py:233-246`, `src/dp_rag_eval/runner/sweep.py:310-314`).
- That contrast is also weakly targeted at the canary: MIA targets are sampled from `members_base` **before** the variant is applied (`scripts/run_attack.py:326-329`), so the scored targets are the un-injected document texts; the canary secret is never a target.
- **No, for the exposure metric.** `scripts/run_canary.py` and `scripts/run_attack.py` build different corpora: the canary runner indexes the whole loaded corpus with no member/non-member split (`scripts/run_canary.py:242-243`, `:308`, default `--corpus-size 200` at `:106`), while the MIA runner indexes members only with a held-out non-member pool (`scripts/run_attack.py:325-326`, defaults 1000/1000). No runner executes both attacks against one indexed corpus, and `run_canary.py` never instantiates `OriginalVariant`.
- `scripts/run_smoke.py` does iterate `original` and `canary` (`scripts/run_smoke.py:41`), but it asks a *different question per variant* (`scripts/run_smoke.py:70-79`) and reports only a boolean "LEAKED / not in answer" for the canary arm (`scripts/run_smoke.py:208`) — not a matched, comparable leakage measurement.

### 3.6 `scripts/verify_canary.py`

A **sanity/smoke check, not a measurement**. It builds 6 synthetic look-alike documents (`scripts/verify_canary.py:58-63`), injects 2 canaries at `n_repeats=1` (`scripts/verify_canary.py:50-51`, `:73-74`), indexes a single `NaivePipeline` (`scripts/verify_canary.py:79-80`), runs both exposure and forced-generation modes (`scripts/verify_canary.py:82-88`), and writes `results.json` (`scripts/verify_canary.py:102-114`). It has no epsilon axis, no DP pipeline, no seed sweep, and no control arm; its own docstring states it "proves the attack machinery and the `completion_logprob` scorer are wired correctly, not research-grade numbers" (`scripts/verify_canary.py:9-10`).

### 3.7 Explicit statement of what is and is not measurable

**Measurable as built:**

- Exposure / extraction rate as a function of **canary duplication** `n_repeats` ∈ {1,2,5,10} at fixed epsilon, seed, and pipeline (`scripts/run_canary.py:247`, `:259`) — a *duplication-of-canary* sweep.
- Exposure as a function of epsilon and pipeline at fixed `n_repeats`.
- Whether the true secret outranks random never-inserted distractors within a cell (`src/dp_rag_eval/attacks/canary.py:85-101`, `:182`).
- Membership AUC on `original` vs `canary` corpora at matched epsilon/seed — the raw rows exist (`src/dp_rag_eval/runner/sweep.py:109-115`, `scripts/run_attack.py:333-334`), though the difference is never computed and the MIA targets are the pre-injection texts.

**Not measurable / missing:**

- A **with-canary vs without-canary contrast under the canary/exposure metric**. There is no no-canary arm in `scripts/run_canary.py`, the exposure metric is undefined when no canary exists (`src/dp_rag_eval/attacks/canary.py:150-152`), and `n_canaries=0` is never run.
- Any computed **delta** (canary-present minus canary-absent, or repeats=k minus repeats=1) at held-fixed epsilon and duplication rate — no code in `src/dp_rag_eval/` or `scripts/` differences cells; outputs are per-cell rows only (`scripts/run_canary.py:138-184`, `src/dp_rag_eval/runner/sweep.py:297-314`).
- The distinction to state plainly: **varying `n_repeats` ∈ {1,2,5,10} is a duplication-of-canary sweep, not a with-canary/without-canary contrast.** Only the former is implemented in the canary experiment; the latter is missing.
