# dp-rag-eval

Evaluation framework for Differentially Private RAG systems. It integrates two original DP-RAG baselines (Koga et al. and Grislain) and provides a dataset module that builds controlled corpus variants for studying how **document duplication weakens formal DP guarantees**.

What exists today:
- **Baseline wrappers** — thin re-exports of both baselines' DP mechanisms (`src/dp_rag_eval/wrappers/`).
- **Dataset framework** — a loader plus four controlled corpus variants (`src/dp_rag_eval/datasets/`).
- **Patch mechanism** — local edits to the cloned baselines (e.g. the model switch) are stored as patches and re-applied automatically.

## Baselines

### Koga et al. — DPVoteRAG
- **Paper:** [arXiv:2412.04697](https://arxiv.org/abs/2412.04697)
- **Original repo:** [tacchan7412/DPRAG](https://github.com/tacchan7412/DPRAG)
- **Mechanism:** Token-by-token DP voting with the Limited Domain Gumbel Mechanism
- **Generator model:** `allenai/OLMo-2-1124-7B-Instruct` (via vLLM) — *upstream default `facebook/opt-1.3b`*

### Grislain — DP-RAG
- **Paper:** [arXiv:2412.19291](https://arxiv.org/abs/2412.19291)
- **Original repo:** [sarus-tech/dp-rag](https://github.com/sarus-tech/dp-rag)
- **Mechanism:** DP logit aggregation with clipping (custom `LogitsProcessor`)
- **Generator model:** `allenai/OLMo-2-1124-7B-Instruct` (via HuggingFace Transformers) — *upstream default `microsoft/Phi-3.5-mini-instruct`*

Both baselines were switched to the same generator (OLMo-2-7B-Instruct) so that privacy–utility comparisons hold the model constant. The switch is applied through `patches/` over the upstream clones (see *Baseline patches* below).

## Requirements

- **Python 3.11+** (the framework does not run on 3.8 — `datasets`/`faker` and 3.11 syntax are required)
- CPU is sufficient for the test suite, the dataset framework, and the DP-mechanism demo
- GPU with 16+ GB VRAM is needed for full LLM inference (the OLMo-2-7B generator)

## Setup

```bash
git clone https://github.com/KarinaMalyshkina/DP_RAG.git
cd DP_RAG

# Create a Python 3.11 virtual environment
python3.11 -m venv .venv        # Windows: py -3.11 -m venv .venv

# Activate
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# Install
pip install -e ".[dev]"

# Clone the original baseline repos AND apply our patches (model switch, etc.)
python setup_baselines.py
```

## Dataset Preparation

The framework loads a public corpus and builds four controlled evaluation settings:

| Variant | What it does |
|---|---|
| `original` | Control — all documents unique, unchanged |
| `full_duplicate` | A subset of documents repeated **verbatim** |
| `partial_overlap` | A subset of documents **share a sentence/paragraph** |
| `canary` | Unique secret strings injected as **extraction-attack targets** (with ground truth) |

Default corpus: **HealthCareMagic** (`RafaelMPereira/HealthCareMagic-100k-Chat-Format-en`); each document is a doctor's answer, the patient question is kept in `metadata["query"]`.

```bash
python scripts/prepare_dataset.py --n-docs 1000
# key options: --dup-rate --dup-copies --overlap-rate --overlap-hosts
#              --n-canaries --canary-repeats --seed --output-dir
```

Outputs to `data/variants/` (gitignored): `original.jsonl`, `full_duplicate.jsonl`, `partial_overlap.jsonl`, `canary.jsonl`, `canary_ground_truth.json`, `summary.json`.

## Running Tests

```bash
pytest tests/ -v
```

28 CPU tests — **no GPU, no network, no model downloads** (the HuggingFace loader is mocked):
- `test_core_types.py` — `Document` (roundtrip, copy independence)
- `test_dataset_variants.py` — all 4 variants: determinism, counts, provenance, input immutability, verbatim duplicates, shared fragments, canary ground truth
- `test_dataset_io.py` — JSONL roundtrip, `to_texts`
- `test_loader.py` — chat parsing and the HealthCareMagic loader (mocked)

## Demo

```bash
python demo.py
```

Exercises the DP mechanisms of both baselines (CPU only) — majority voting, Limited Domain Gumbel, exponential mechanism, Grislain per-token ε and logit aggregation.

## Baseline patches

`baselines/` is gitignored (clones of external repos), so local edits there are not tracked by git and would be lost on a re-clone. They are stored as patch files and re-applied by `setup_baselines.py`:

- `patches/koga.patch`, `patches/grislain.patch` — currently the OLMo-2-7B generator switch.
- Re-application is idempotent (a re-run reports `already applied, skipping`).
- If you edit a clone by hand, refresh its patch:
  ```bash
  (cd baselines/koga && git diff) > patches/koga.patch
  (cd baselines/grislain && git diff) > patches/grislain.patch
  ```

## Project Structure

```
DP_RAG/
├── baselines/                     # gitignored, cloned + patched by setup_baselines.py
│   ├── koga/                      # tacchan7412/DPRAG
│   └── grislain/                  # sarus-tech/dp-rag
├── patches/                       # edits re-applied to the clones (model switch)
│   ├── koga.patch
│   └── grislain.patch
├── src/dp_rag_eval/
│   ├── core/
│   │   ├── types.py               # Document
│   │   └── protocols.py           # DatasetLoader, DatasetVariant
│   ├── datasets/
│   │   ├── loaders.py             # HealthCareMagicLoader
│   │   ├── variants.py            # Original / FullDuplicate / PartialOverlap / CanaryInjection
│   │   └── io.py                  # JSONL save/load, to_texts
│   └── wrappers/
│       ├── koga_imports.py        # imports from baselines/koga/
│       └── grislain_imports.py    # imports from baselines/grislain/
├── scripts/
│   └── prepare_dataset.py         # build the 4 corpus variants
├── tests/                         # 28 CPU tests (no GPU/network)
├── data/                          # gitignored: prepared variants
├── setup_baselines.py
├── demo.py
└── pyproject.toml
```
