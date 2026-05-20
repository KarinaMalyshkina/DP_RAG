# dp-rag-eval

A unified evaluation framework for Differentially Private RAG systems. Tests how document duplication weakens formal DP guarantees through privacy attacks across controlled corpus settings.

## Requirements

- **Python 3.11+**
- **GPU with 16+ GB VRAM** (for real experiments with LLMs)
- CPU-only mode works for tests and development

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd dp-rag-eval

# Create a virtual environment
python -m venv .venv

# Activate (Linux/Mac)
source .venv/bin/activate
# Activate (Windows)
.venv\Scripts\activate

# Install the package with dev dependencies
pip install -e ".[dev]"
```

### Downloading NQ data (for Koga pipeline)

```bash
bash scripts/download_data.sh
```

## Running Tests

All tests run on CPU without downloading large models.

```bash
# Run all tests
PYTHONPATH=src pytest tests/ -v

# Run specific test files
PYTHONPATH=src pytest tests/test_pipelines.py -v
PYTHONPATH=src pytest tests/test_attacks.py -v
PYTHONPATH=src pytest tests/test_datasets.py -v
PYTHONPATH=src pytest tests/test_dp_mechanisms.py -v

# Run a specific test
PYTHONPATH=src pytest tests/test_pipelines.py::TestKogaDPMechanisms::test_ld_gumbel_basic -v
```

### Linting and Formatting

```bash
# Check for lint errors
ruff check src/ tests/

# Auto-fix lint issues
ruff check --fix src/ tests/

# Check formatting
ruff format --check src/ tests/

# Apply formatting
ruff format src/ tests/
```

## Running Experiments

### Single Experiment

```bash
python scripts/run_experiment.py --config configs/koga_nq.yaml
```

With overrides:

```bash
python scripts/run_experiment.py --config configs/koga_nq.yaml --privacy.epsilon 2.0 --experiment.seed 123
```

### Sweep over Parameters

```bash
python scripts/run_sweep.py \
  --config configs/koga_nq.yaml \
  --sweep "privacy.epsilon=[0.5,1,2,4,8]" \
  --sweep "experiment.seed=[42,43,44]"
```

### Manual Usage from Python

```python
from dp_rag_eval.datasets import SyntheticMedicalLoader, create_variant
from dp_rag_eval.attacks import CanaryExtractionAttack
from dp_rag_eval.evaluation import exact_match, f1_score

# 1. Load data
docs = SyntheticMedicalLoader(n_docs=200, seed=42).load()

# 2. Apply a corpus variant (e.g., inject canaries)
variant = create_variant("canary", {"n_canaries": 50, "copies_per_canary": 3})
docs = variant.apply(docs)

# 3. Create and index a pipeline (example with mock)
from tests.conftest import MockPipeline
pipeline = MockPipeline()
pipeline.index(docs)

# 4. Ask a question
result = pipeline.answer("What is the most common diagnosis?", eps=1.0, delta=1e-5)
print(result.text)
print(f"Privacy spent: {result.privacy_spent}")

# 5. Run a privacy attack
attack = CanaryExtractionAttack()
attack.setup(pipeline, docs)
attack_result = attack.run(n_queries=50, eps=1.0, delta=1e-5)
print(f"Canary extraction success rate: {attack_result.success_rate:.2%}")

# 6. Evaluate utility
em = exact_match("Paris", ["Paris", "paris"])
f1 = f1_score("the capital is Paris", ["Paris is the capital"])
print(f"EM={em}, F1={f1:.2f}")
```

## Pipelines

### Koga (DPVoteRAG)

- **Model:** `facebook/opt-1.3b` (via vLLM)
- **Mechanism:** Token-by-token DP voting with the Limited Domain Gumbel Mechanism
- **Retriever:** DPR (facebook/dpr-*-nq-base) — no DP noise at retrieval
- **Config:** `configs/koga_nq.yaml`

### Grislain (DP-RAG)

- **Model:** `microsoft/Phi-3.5-mini-instruct` (via HuggingFace Transformers)
- **Mechanism:** DP logit aggregation with clipping (custom LogitsProcessor)
- **Retriever:** PUP Vector Store with Exponential Mechanism (Snowflake Arctic embeddings)
- **Config:** `configs/grislain_medical.yaml`

## Changing the Model

Edit the `model` field in the YAML config:

```yaml
pipeline:
  type: "koga"                    # or "grislain"
  model: "facebook/opt-1.3b"     # any HuggingFace causal LM
  model_revision: null            # pin a specific revision (optional)
  device: "cuda"                  # "cuda" or "cpu"
```

### Compatible models

| Model | Size | VRAM | Notes |
|-------|------|------|-------|
| `sshleifer/tiny-gpt2` | 0.6M | <1 GB | For testing only |
| `facebook/opt-350m` | 350M | ~2 GB | Lightweight, fast iteration |
| `facebook/opt-1.3b` | 1.3B | ~4 GB | Koga default |
| `microsoft/Phi-3.5-mini-instruct` | 3.8B | ~8 GB | Grislain default |
| `meta-llama/Llama-2-7b-hf` | 7B | ~16 GB | Larger, better quality |

Any HuggingFace `AutoModelForCausalLM`-compatible model works. For Grislain, the model must support `.generate()` with custom `LogitsProcessor` (raw logit access required). For Koga, vLLM must support the model.

To change model in code:

```python
from dp_rag_eval.llm import LocalHFBackend, LocalVLLMBackend

# For Grislain (HuggingFace Transformers)
backend = LocalHFBackend(model_name="microsoft/Phi-3.5-mini-instruct", device="cuda")

# For Koga (vLLM)
backend = LocalVLLMBackend(model_name="facebook/opt-1.3b", device="cuda")
```

## Dataset Variants

| Variant | Description | Config key |
|---------|-------------|------------|
| Original | No changes | `original` |
| Full Duplicate | Exact copies of a fraction of documents | `full_duplicate` |
| Partial Overlap | Near-duplicates with minor text changes | `partial_overlap` |
| Canary Injection | Inject documents with known secrets | `canary` |

```yaml
dataset:
  loader: "synthetic_medical"     # or "nq"
  variant: "full_duplicate"
  variant_params:
    duplication_rate: 0.3
    copies: 5
```

## Privacy Attacks

| Attack | What it does |
|--------|-------------|
| Canary Extraction | Inserts secret strings, queries to extract them |
| Membership Inference | Determines if a document was in the corpus |
| Property Inference | Infers aggregate properties (e.g., most common diagnosis) |

## Output

Results are saved to `runs/<experiment_name>/`:

- `meta.json` — hardware/software metadata
- `run_eps<X>_seed<Y>.json` — per-run results
- `summary.json` — aggregated results across all runs

## Project Structure

```
dp-rag-eval/
├── src/dp_rag_eval/
│   ├── core/          # Types, protocols, config
│   ├── pipelines/     # Koga + Grislain adapters
│   ├── attacks/       # Canary, MIA, property inference
│   ├── datasets/      # Loaders + variants
│   ├── evaluation/    # Metrics + reporter
│   ├── llm/           # HF + vLLM backends
│   └── runner/        # Experiment orchestrator
├── configs/           # YAML experiment configs
├── scripts/           # CLI entry points
├── tests/             # Pytest test suite
└── pyproject.toml
```
