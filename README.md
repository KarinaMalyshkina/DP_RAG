# dp-rag-eval

Evaluation framework for Differentially Private RAG systems. Wraps two original DP-RAG baselines (Koga et al. and Grislain) behind a common interface for comparative analysis.

## Baselines

### Koga et al. — DPVoteRAG
- **Paper:** [arXiv:2412.04697](https://arxiv.org/abs/2412.04697)
- **Original repo:** [tacchan7412/DPRAG](https://github.com/tacchan7412/DPRAG)
- **Mechanism:** Token-by-token DP voting with Limited Domain Gumbel Mechanism
- **Default model:** `facebook/opt-1.3b` (via vLLM)

### Grislain — DP-RAG
- **Paper:** [arXiv:2412.19291](https://arxiv.org/abs/2412.19291)
- **Original repo:** [sarus-tech/dp-rag](https://github.com/sarus-tech/dp-rag)
- **Mechanism:** DP logit aggregation with clipping (custom LogitsProcessor)
- **Default model:** `microsoft/Phi-3.5-mini-instruct` (via HuggingFace Transformers)

## Requirements

- Python 3.11+
- CPU is sufficient for tests and DP mechanism demos
- GPU with 16+ GB VRAM needed for full LLM inference

## Setup

```bash
git clone https://github.com/KarinaMalyshkina/DP_RAG.git
cd DP_RAG

# Create virtual environment with Python 3.11
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate
# Activate (Linux/Mac)
source .venv/bin/activate

# Install
pip install -e ".[dev]"

# Clone the original baseline repos
python setup_baselines.py
```

## Running Tests

```bash
pytest tests/ -v
```

14 CPU tests covering DP mechanisms of both baselines:
- `test_koga.py` — LDGumbelMechanism, ExpMechanism, majority_vote, RDP composition
- `test_grislain.py` — DPGenerationConfig, DPLogitsAggregator, PUPVectorStoreConfig

## Demo

```bash
python demo.py
```

## Project Structure

```
DP_RAG/
├── baselines/                     # gitignored, cloned by setup_baselines.py
│   ├── koga/                      # tacchan7412/DPRAG
│   └── grislain/                  # sarus-tech/dp-rag
├── src/dp_rag_eval/
│   ├── __init__.py
│   └── wrappers/
│       ├── koga_imports.py        # imports from baselines/koga/
│       └── grislain_imports.py    # imports from baselines/grislain/
├── tests/
│   ├── conftest.py
│   ├── test_koga.py
│   └── test_grislain.py
├── setup_baselines.py
├── demo.py
└── pyproject.toml
```
