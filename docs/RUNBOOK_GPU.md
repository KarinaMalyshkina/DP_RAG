# GPU runbook — Qwen3.5-9B on a rented box

Operating instructions for the experiment matrix. Assumes a single 48 GB card
(RTX A6000 / L40S) on RunPod with a persistent network volume at `/workspace`.

Everything here is resumable: each finished cell is written to `results.json`
immediately and keyed by its full configuration, so a killed run restarts where it
stopped. **Always run under `tmux`** — an SSH drop otherwise kills the sweep.

---

## 0. First 15 minutes

Do these before starting anything long. Each one has killed a paid run before.

```bash
tmux new -s dprag
git clone <repo-url> /workspace/DPRAG && cd /workspace/DPRAG
bash scripts/bootstrap_gpu.sh
```

`bootstrap_gpu.sh` installs dependencies, clones and patches the two baseline
repos, downloads the weights to `/workspace/hf-cache`, and then hard-fails if any
of these are wrong:

- torch cannot see the GPU (wrong CUDA build),
- the model loaded in fp32 (would not fit alongside activations),
- the first generated token is still `<think>` (would invalidate every membership score).

In every new shell afterwards:

```bash
cd /workspace/DPRAG && source .env.gpu
```

Then confirm the optimisations behave on the real model — these are the changes
that fail silently rather than raising:

```bash
uv run python scripts/verify_optimizations.py --generator Qwen/Qwen3.5-9B
```

It checks that Grislain's shared prompt prefix is rewound correctly between
candidates, that Koga's left-padded voter batch reads the same positions as
one-at-a-time forwards, and — the one that only becomes meaningful here — that the
first generated token is not `<think>`. It exits non-zero on any failure. Do not
start the matrix until it passes.

`pyproject.toml` requires `transformers>=5.8`, which registers the `qwen3_5`
architecture — the model loads without any special install. If it still reports
*unrecognized model type*, the resolved version is older than the floor; check with
`uv run python -c "import transformers; print(transformers.__version__)"` before
reaching for a git install.

**Commit `uv.lock` after the first `uv sync`.** The repository ships without one, so
this run resolves dependencies fresh; committing the lock it produces is what makes
every later run reproduce this exact environment.

---

## 1. Calibration — do this before the full matrix

The GPU-hour estimate in the plan is an estimate. Measure one cell, then decide.

```bash
uv run python scripts/run_attack.py \
  --generator Qwen/Qwen3.5-9B --output-dir runs/calib \
  --pipelines grislain,koga --variants original --variant-grid base \
  --epsilons inf --modes unified --n-tokens 32 --delta 1e-6 \
  --members 1000 --non-members 1000 --targets-per-class 25 --seeds 42
```

Read `seconds` from the printed summary and scale: a full cell is
`targets-per-class 250`, i.e. **10x** this. Multiply by the cell counts in §2–§4
(≈560 MIA cells, plus the utility and canary runs). If the total is
more than ~2x off, cut `--targets-per-class` to 150 or drop a seed **before**
starting, not halfway through.

Watch VRAM in a second pane (`watch -n5 nvidia-smi`). Headroom below ~4 GB means
`--n-split` or `--top-k` needs lowering; the logit tensor scales with both.

---

## 2. Group 1 + 2 — baseline (ε=∞) and the fixed-ε sweep

One command covers both: `inf` in the epsilon list selects each baseline's
non-private code path (plain plurality vote for Koga, `_non_dp_call` and a non-DP
retrieval threshold for Grislain), and `naive` is automatically pinned to the
`inf` cell instead of being re-run per epsilon.

```bash
uv run python scripts/run_attack.py \
  --generator Qwen/Qwen3.5-9B --output-dir runs/mia_main \
  --variant-grid base --epsilons inf,0.5,1,2,4,8,16,32 \
  --n-tokens 32 --delta 1e-6 --modes unified,upper_bound \
  --members 1000 --non-members 1000 --targets-per-class 250 \
  --seeds 42,43,44
```

`--epsilons` are **target total budgets for one 32-token answer**, including
retrieval; each pipeline solves its own parameters to meet them. The grid spans
both systems' ranges: Grislain covers 0.5–8, Koga 4–32 (§4a). Cells a mechanism
cannot reach are recorded `INFEASIBLE` and skipped in milliseconds, so one command
costs no more than the feasible cells.

Voter sizing is automatic (`--auto-n-split-cap 128`, the default); the count each
cell used is recorded in the `n_split` column.

`report.md` in the output directory is the readable summary.

### Koga's black-box pass

Black-box runs a full generation loop per query and is roughly 10x the cost, so it
is a separate, narrower command rather than a third mode in the main sweep — and
restricted to budgets Koga can actually reach:

```bash
uv run python scripts/run_attack.py \
  --generator Qwen/Qwen3.5-9B --output-dir runs/mia_main \
  --pipelines koga --variant-grid base --epsilons inf,8,16,32 \
  --n-tokens 32 --delta 1e-6 --modes black_box \
  --members 1000 --non-members 1000 \
  --targets-per-class 250 --seeds 42,43,44
```

Same `--output-dir` on purpose: it appends to the same checkpoint and report.

---

## 3. Group 3 — duplication strength

Epsilon held at two representative values while duplication strength varies. This
is the RQ1 axis and the largest group (~26 variant configs).

```bash
uv run python scripts/run_attack.py \
  --generator Qwen/Qwen3.5-9B --output-dir runs/mia_duplication \
  --variant-grid duplication --epsilons inf,4,16 \
  --n-tokens 32 --delta 1e-6 --modes unified \
  --members 1000 --non-members 1000 --targets-per-class 250 \
  --seeds 42,43,44
```

The grid expands to: `full_duplicate` at rates 0.1–0.9 (one extra copy each) and
at 1/2/3/5/10 copies (rate 0.3); `partial_overlap` at overlap rates 0.1–0.9;
`canary` at 1/2/5/10 repeats. Override any axis with `--dup-rates`,
`--dup-copies`, `--overlap-rates`, `--canary-repeats`.

**This group dominates the budget** — ~26 configs x 3 seeds x the epsilon values,
and Koga's cells at ε=4 use 128 voters each. Two levers, in this order:

1. Run Koga's part of this group at **ε=8 only** (69 voters, ~1.7x cheaper per cell
   than ε=4). RQ1 is a trend *across duplication levels at fixed ε*, so one
   well-chosen ε answers it; the ε-dependence comes from group 2.
2. Drop to `--targets-per-class 100` here. The duplication effect is read from the
   trend across ~26 configs, so per-cell precision matters less than in group 2 —
   but note TPR@FPR=1% becomes unusable below 100 targets, so do not go lower.

Decide both **before** starting, using the §1 calibration numbers.

---

## 4. Canary extraction

Separate script, same sweep machinery. `n_repeats` is the number of copies of the
*same secret*, which is the most direct measurement of the thesis question.

```bash
uv run python scripts/run_canary.py \
  --generator Qwen/Qwen3.5-9B --output-dir runs/canary_main \
  --epsilons inf,4,8,16,32 --n-tokens 32 --delta 1e-6 \
  --canary-repeats 1,2,5,10 \
  --corpus-size 1000 --n-canaries 20 --n-distractors 255 \
  --modes exposure --seeds 42,43,44
```

Cost driver is `--n-distractors`: exposure mode scores `n_distractors + 1`
candidates per canary. Grislain scores the whole candidate set against one shared
prompt prefix (the prefix KV cache is computed once), so 255 distractors there
cost far less than 256 independent queries. Naive and Koga have no such batching —
if the cell times look bad, they are the reason; drop to `--n-distractors 63`
(exposure ceiling falls from 8 to 6 bits) rather than cutting canaries.

---

## 4a. Koga's reachable ε range

Koga's LD-Gumbel selection only returns a token once some candidate clears a
Limited-Domain cutoff. With fewer voters than that cutoff, **no** vote distribution
can clear it: every query fails, every target scores 0, and AUC lands on exactly
0.5 — easy to misread as "DP defeated the attack" when it means the configuration
was infeasible.

Voters required at δ=1e-6, where ε is the total budget for one answer:

| ε_total | n_tokens=1 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| 0.5 | 35 | 143 | 287 | 555 | 890 |
| 1 | 18 | 71 | 143 | 285 | 462 |
| 2 | 10 | 36 | 72 | 145 | 242 |
| 4 | 6 | 19 | 37 | 74 | 128 |
| 8 | 4 | 11 | 20 | 38 | 69 |

Regenerate with `uv run python scripts/report_feasibility.py`.

### Why the default is `--auto-n-split-cap 128`

A fixed voter count is wrong at both ends. At `--n-tokens 32`, a flat `n_split=50`
reaches only ε≥16 — and since Grislain's grid stops at 8, **the two systems would
have no common point at all**. A flat 128 would instead spend 128 prompts per query
at ε=32, where 21 suffice.

The default sizes the count per cell to exactly what the target needs, capped at
128:

| ε | voters used | cell cost vs flat 50 |
|---|---|---|
| 4 | 128 | 2.6x |
| 8 | 69 | 1.4x |
| 16 | 38 | 0.8x |
| 32 | 21 | 0.4x |

Averaged over Koga's grid that is ~1.3x the cost of a flat 50 — and unlike a flat
50 it actually produces results.

**The cap is 128, not higher.** Raising it to 256 would add ε=2 (242 voters), but
that cell costs ~80 min against 42 at ε=4, and 242 voters pull a quarter of a
1000-document corpus into context, at which point retrieval is no longer a
selection. The pipeline warns when the voter count exceeds 25% of the corpus.

**Resulting ranges at `--n-tokens 32`:** Koga {∞, 4, 8, 16, 32}, Grislain
{∞, 0.5, 1, 2, 4, 8}. Compare the two systems directly only on the overlap,
**ε ∈ {4, 8}**; outside it the honest statement is "not reachable at this voter
count", not a number.

To widen the overlap you must either pay (raise the cap) or shorten `--n-tokens` —
and the latter has to be applied to the utility runs too, where short answers make
token-F1 hard to interpret.

**Do not lower `--n-split` to save time** — it silently invalidates the low-ε
points. Cut `--targets-per-class` instead.

---

## 5. Interpreting the Koga columns

Koga appears with up to three scores; they are not redundant and must not share a
line on a plot.

| mode | what it measures | responds to ε? | use it for |
|---|---|---|---|
| `unified` | P(the deployed system answers 'Yes'), sampled from the real LD-Gumbel vote | **yes** | the headline result, ε-curves, TPR@FPR |
| `upper_bound` | raw voter logits, before the noise | no, flat by construction | adversarial upper bound |
| `black_box` | parsed Yes/No text | yes | external validity — what an API actually leaks |

`unified` is the only mode comparable across pipelines: it is the same estimand
everywhere, computed exactly for naive and Grislain (their output distributions are
closed-form) and by sampling for Koga. `upper_bound − unified` is how much leakage
the noise removes; `unified − black_box` is how much the attacker loses without
access to the output distribution. A flat `upper_bound` line across ε is the
expected, correct result.

Both pipelines are calibrated to the **same** total (ε, δ) per answer, so the axis
is shared. Plot against the recorded `realized_epsilon`. Full definitions in
[`docs/METHODOLOGY.md`](METHODOLOGY.md).

---

## 5a. Utility

Same axes, same configuration — this is what makes the privacy-utility curve valid:

```bash
uv run python scripts/run_utility.py \
  --generator Qwen/Qwen3.5-9B --output-dir runs/utility_main \
  --variant-grid base --epsilons inf,0.5,1,2,4,8,16,32 \
  --n-tokens 32 --delta 1e-6 \
  --corpus-size 1000 --n-queries 200 --max-answer-chars 176   --seeds 42,43,44
```

`--n-tokens` and `--delta` **must match the attack runs exactly**.

`--max-answer-chars 176` is `5.5 x n_tokens`: a system allowed 32 tokens cannot
recall a 70-word reference, so leaving the default 400 caps F1 at ~0.6 for even a
perfect answer and that cap would read as a privacy cost. The runner warns if the
two are mismatched.

Token F1 is the primary metric; exact match will read ~0 for every system including
the non-private baseline, which reflects the free-text corpus rather than DP.

---

## 6. Recovery

| symptom | action |
|---|---|
| SSH dropped | `tmux attach -t dprag`; the sweep kept running |
| pod restarted | re-run `bootstrap_gpu.sh` (weights survive on `/workspace`), then re-issue the same command — finished cells are skipped |
| CUDA OOM | lower `--n-split`, then `--top-k`, then `--max-doc-chars`. The logit tensor is `batch x vocab (248k)`, so batch size dominates |
| one cell errors | it is recorded with `status=ERROR` and the sweep continues; the traceback is in the console log and `error` in the CSV |
| need to redo a cell | delete its row from `results.json` (keyed by `cell`) and re-run the same command |

Never delete `results.json` to "start clean" mid-matrix — that discards every
finished cell. To rerun from scratch, use a new `--output-dir`.

---

## 7. Before releasing the box

```bash
tar czf /workspace/dprag-results.tar.gz runs/
```

Download it, verify locally that `results.csv` opens and row counts match, **then**
destroy the pod. A snapshot of the volume is cheaper than the GPU, but only the
`runs/` directory is irreplaceable — everything else is reproducible from git.
