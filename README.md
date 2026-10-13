# dp-rag-eval

A unified evaluation framework for Differentially Private retrieval-augmented generation (DP-RAG). It runs two published DP-RAG mechanisms — the token voting of Koga et al. and the logit aggregation of Grislain — behind one interface, calibrates both to a common privacy budget, and measures how **duplicated content in the retrieval corpus weakens their formal (ε, δ) guarantees** with privacy attacks and utility metrics under controlled corpus conditions.

The code accompanies the Master's thesis *A Unified Attack Evaluation Framework for Differentially Private RAG Systems* (Karina Malyshkina, University of Vienna, 2026).

## What the framework does

| Component | Where | What it does |
|---|---|---|
| Pipelines | `src/dp_rag_eval/pipelines/` | Naive RAG (ε = ∞ reference), Koga DPVoteRAG and Grislain DP-RAG behind a common interface; the DP mechanisms are the original code, only model loading and indexing are ours |
| Privacy calibration | `src/dp_rag_eval/privacy/` | Solves each mechanism's parameters, with its own accountant, so that one complete answer of `n_tokens` tokens (retrieval included) costs a target total (ε, δ); infeasible targets are reported, not run |
| Corpus variants | `src/dp_rag_eval/datasets/` | `original`, `full_duplicate` (verbatim copies), `partial_overlap` (shared fragments), `canary` (injected secrets) |
| Attacks | `src/dp_rag_eval/attacks/` | Membership inference (Anderson et al., 2024) with one estimand for every pipeline; canary extraction (Secret Sharer, Carlini et al., 2019) |
| Utility | `src/dp_rag_eval/evaluation/metrics.py` | Token F1, ROUGE-L, exact match, empty-answer count |
| Sweeps | `src/dp_rag_eval/runner/`, `scripts/run_*.py` | Grid over seed × corpus variant × ε × pipeline; every finished cell is written at once, so an interrupted sweep resumes where it stopped |
| Analysis | `src/dp_rag_eval/evaluation/reporter.py`, `scripts/report_degradation.py` | Duplicated-vs-control differences at matched pipeline, ε and seed, pooled over seeds |
| Monitoring | `scripts/watch_progress.py`, `scripts/notify.py`, `scripts/telegram_bot.py` | Progress, ETA and stall alerts for long GPU sweeps |

## Baselines

### Koga et al. — DPVoteRAG
- **Paper:** [arXiv:2412.04697](https://arxiv.org/abs/2412.04697)
- **Original repo:** [tacchan7412/DPRAG](https://github.com/tacchan7412/DPRAG), pinned to commit `28c79e5`
- **Mechanism:** retrieved documents are split across voters; each output token is chosen by a Limited-Domain Gumbel vote; accounting under RDP
- **Here:** the voting runs on HuggingFace Transformers (all voters as one batch) instead of vLLM; the voter count is sized per budget so that the target is feasible

### Grislain — DP-RAG
- **Paper:** [arXiv:2412.19291](https://arxiv.org/abs/2412.19291)
- **Original repo:** [sarus-tech/dp-rag](https://github.com/sarus-tech/dp-rag), pinned to commit `b9d1dea`
- **Mechanism:** DP retrieval threshold via the exponential mechanism, then clipped aggregation of per-document logits; accounting under PLD (`dp-accounting`)

## Models and data

| Role | Model / dataset |
|---|---|
| Generator (all experiments) | `Qwen/Qwen3.5-9B` |
| Second generator (replication, Grislain) | `google/gemma-4-12B-it` |
| Retrieval embedder | `sentence-transformers/all-MiniLM-L6-v2` |
| Corpus | HealthCareMagic (`RafaelMPereira/HealthCareMagic-100k-Chat-Format-en`): doctor replies as documents, patient questions as queries |

The generator is chosen per run with `--generator`; reasoning mode is disabled through the chat template.

## Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/) (dependency versions are pinned in `uv.lock`)
- CPU is enough for the test suite and the demo
- A GPU for the experiments; the reported runs used one NVIDIA RTX A6000 (48 GB). `pyproject.toml` resolves torch from the CUDA 12.8 index

## Setup

```bash
git clone https://github.com/KarinaMalyshkina/DP_RAG.git
cd DP_RAG
uv sync --extra dev
uv run python setup_baselines.py   # clone both baselines at the pinned commits and apply patches/
```

On a fresh GPU machine use the bootstrap script instead; it also downloads the weights and fails loudly on a CPU-only torch build, fp32 loading or an enabled reasoning mode:

```bash
bash scripts/bootstrap_gpu.sh
uv run python scripts/verify_optimizations.py --generator Qwen/Qwen3.5-9B
```

The full operating procedure for the experiment matrix — calibration, the sweeps, interpretation, recovery — is [`docs/RUNBOOK_GPU.md`](docs/RUNBOOK_GPU.md).

## Running experiments

```bash
# membership inference on the clean corpus across the budget axis
uv run python scripts/run_attack.py \
  --generator Qwen/Qwen3.5-9B --output-dir runs/mia_main \
  --variant-grid base --epsilons inf,0.5,1,2,4,8 \
  --n-tokens 32 --delta 1e-6 --modes unified \
  --members 1000 --non-members 1000 --targets-per-class 250 \
  --seeds 42,43,44

# the duplication grid, written to the same directory so every cell has its control
uv run python scripts/run_attack.py \
  --generator Qwen/Qwen3.5-9B --output-dir runs/mia_main \
  --variant-grid duplication --epsilons inf,1,5 \
  --n-tokens 32 --delta 1e-6 --modes unified \
  --members 1000 --non-members 1000 --targets-per-class 250 \
  --seeds 42,43,44

# canary extraction
uv run python scripts/run_canary.py \
  --generator Qwen/Qwen3.5-9B --output-dir runs/canary_main \
  --epsilons inf,0.5,1,2,5 --n-tokens 32 --delta 1e-6 \
  --canary-repeats 1,2,5,10 --variants original \
  --corpus-size 1000 --n-canaries 20 --n-distractors 255 \
  --modes exposure --seeds 42,43,44

# utility on the same configurations
uv run python scripts/run_utility.py --generator Qwen/Qwen3.5-9B --output-dir runs/utility_main \
  --epsilons inf,1,5 --n-tokens 32 --delta 1e-6 --seeds 42,43,44

# duplicated vs. control, pooled over seeds
uv run python scripts/report_degradation.py --results runs/mia_main \
  --metric auc --metric tpr_at_fpr_05 --output-dir runs/analysis
```

Each sweep writes `results.json`, `results.csv` and `report.md` to its output directory and refreshes them after every cell. `scripts/report_feasibility.py` prints the voter counts Koga needs for each budget; `scripts/probe_generator.py` checks a new generator before a sweep.

Results of the main runs on `Qwen/Qwen3.5-9B` are in `runs/` (`mia_main`, `calib`, `calib2`); the remaining folders there are small smoke runs on CPU.

## Tests

```bash
uv run pytest tests/ -v
```

245 tests on CPU, with no GPU and no model downloads. Tests of the Koga and Grislain integrations import the original code, so run `setup_baselines.py` first.

```bash
uv run python demo.py   # the DP mechanisms of both baselines on toy inputs, CPU only
```

## Baseline patches

`baselines/` holds clones of the external repositories and is not tracked. Local changes to them are stored in `patches/koga.patch` and `patches/grislain.patch` and applied by `setup_baselines.py`; a patch that does not apply stops the setup. The patches leave both mechanisms unchanged: they disable the generator's reasoning mode, encode the corpus in chunks, make device selection work without a GPU, change the upstream example checkpoints, and draw Grislain's retrieval threshold in log space so that a fully duplicated corpus does not underflow.

After editing a clone, refresh its patch:

```bash
(cd baselines/koga && git diff) > patches/koga.patch
(cd baselines/grislain && git diff) > patches/grislain.patch
```

## Documentation

| File | Content |
|---|---|
| [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) | Definitions: privacy unit, common budget axis, calibration, estimands, metrics |
| [`docs/RUNBOOK_GPU.md`](docs/RUNBOOK_GPU.md) | Operating procedure for the experiment matrix on a GPU machine |
| [`docs/decisions-needed.md`](docs/decisions-needed.md) | Open design decisions |
| [`docs/empirical-coverage-report.md`](docs/empirical-coverage-report.md) | What the code measures against what the thesis claims |
| [`docs/Literature_Attacks.txt`](docs/Literature_Attacks.txt) | Attack literature |

## Project structure

```
DP_RAG/
├── baselines/                 # not tracked: cloned and patched by setup_baselines.py
├── patches/                   # koga.patch, grislain.patch
├── src/dp_rag_eval/
│   ├── core/                  # Document and other shared types, protocols
│   ├── datasets/              # HealthCareMagic loader, corpus variants, JSONL io
│   ├── pipelines/             # naive, Koga, Grislain; shared model loading
│   ├── privacy/               # calibration to a common (ε, δ) per answer
│   ├── attacks/               # membership inference, canary extraction
│   ├── evaluation/            # utility metrics, degradation reporter, progress reader
│   ├── runner/                # grid expansion and resumable result writing
│   ├── wrappers/              # imports from baselines/koga and baselines/grislain
│   └── notifications.py       # Telegram transport for sweep alerts
├── scripts/                   # run_attack, run_canary, run_utility, report_*, bootstrap_gpu, ...
├── tests/                     # 245 CPU tests
├── runs/                      # experiment results
├── docs/
├── setup_baselines.py
├── demo.py
├── pyproject.toml
└── uv.lock
```
