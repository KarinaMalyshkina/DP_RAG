#!/usr/bin/env bash
# One-shot setup for a fresh GPU box (RunPod / any Ubuntu + CUDA image).
#
#   bash scripts/bootstrap_gpu.sh
#
# Idempotent: safe to re-run after a pod restart. Everything that must survive a
# restart (HF weights, the baseline clones, the venv) is placed under $WORKSPACE,
# which should be a mounted network volume — the container filesystem is wiped
# when a pod is recreated, and re-downloading ~20 GB of weights each time is the
# single easiest way to waste rented GPU hours.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
GENERATOR="${GENERATOR:-Qwen/Qwen3.5-9B}"
EMBEDDER="${EMBEDDER:-sentence-transformers/all-MiniLM-L6-v2}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== repo: $REPO_ROOT"
echo "=== workspace: $WORKSPACE"

if [ ! -d "$WORKSPACE" ]; then
    echo "WARNING: $WORKSPACE does not exist — falling back to \$HOME."
    echo "         Model weights will NOT survive a pod restart."
    WORKSPACE="$HOME"
fi

# HF cache on the persistent volume, and the fast downloader (~3x on big weights).
export HF_HOME="$WORKSPACE/hf-cache"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME"

cat > "$REPO_ROOT/.env.gpu" <<EOF
# Sourced by the runbook commands: keeps the HF cache on the persistent volume.
export HF_HOME="$HF_HOME"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
EOF
echo "=== wrote $REPO_ROOT/.env.gpu (source it in every new shell)"

echo
echo "=== 1/6  GPU check"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. This is not a GPU machine, or the driver is missing."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo
echo "=== 2/6  Disk space"
# Weights alone are ~18 GB for a 9B model in bf16, plus ~5 GB of venv (torch is most
# of it) and the dataset cache. Running out at 90% of an 18 GB download wastes both
# the transfer and the rental time it took.
AVAIL_GB=$(df -Pk "$WORKSPACE" 2>/dev/null | awk 'NR==2 {printf "%d", $4/1024/1024}')
# An unparseable df would otherwise make the numeric test below abort the script
# with a shell error instead of a useful message.
[ -n "$AVAIL_GB" ] || AVAIL_GB=0
echo "free on $WORKSPACE: ${AVAIL_GB} GiB"
if [ "$AVAIL_GB" -lt 40 ]; then
    echo "ERROR: need ~40 GiB free (weights ~18, venv ~5, dataset + results the rest)."
    echo "       Attach a larger volume, or set WORKSPACE to one that has room."
    exit 1
fi

echo
echo "=== 3/6  Python dependencies"
if ! command -v uv >/dev/null 2>&1; then
    echo "installing uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # The installer picks its own location depending on the image; take whichever
    # exists rather than assuming ~/.local/bin.
    for candidate in "$HOME/.local/bin" "$HOME/.cargo/bin" /usr/local/bin; do
        [ -x "$candidate/uv" ] && export PATH="$candidate:$PATH"
    done
    # Some installers write an env file instead of touching PATH.
    [ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env"
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv installed but is not on PATH. Add its directory to PATH and re-run."
    exit 1
fi
uv --version
cd "$REPO_ROOT"
HAD_LOCK=$([ -f "$REPO_ROOT/uv.lock" ] && echo yes || echo no)
uv sync --extra dev
if [ "$HAD_LOCK" = "no" ] && [ -f "$REPO_ROOT/uv.lock" ]; then
    echo
    echo ">>> uv.lock was just created by resolving pyproject.toml fresh."
    echo ">>> COMMIT IT — without it, a rerun months from now may resolve different"
    echo ">>> versions and the results will not reproduce."
fi

echo
echo "=== 4/6  Baseline repositories (Koga, Grislain) at pinned commits + patches"
# Fails hard if a patch does not apply: the pipelines depend on those edits, and
# running without them silently changes what is computed.
uv run python setup_baselines.py

echo
echo "=== 5/6  Pre-downloading model weights and the dataset into $HF_HOME"
# The dataset is fetched here too, not just the models. Otherwise the first
# experiment discovers a network problem hours into a paid run.
uv run python - "$GENERATOR" "$EMBEDDER" <<'PY'
import sys
from huggingface_hub import snapshot_download

for model_id in sys.argv[1:]:
    print(f"--- {model_id}")
    try:
        # Skip the duplicate GGUF/consolidated copies some repos ship alongside safetensors.
        path = snapshot_download(model_id, ignore_patterns=["*.gguf", "*.pth", "original/*"])
    except Exception as exc:  # noqa: BLE001 - turn a stack trace into an actionable message
        name = type(exc).__name__
        print(f"    FAILED: {name}: {exc}", file=sys.stderr)
        if "Gated" in name or "401" in str(exc) or "403" in str(exc):
            print(
                "    This repo is gated. Accept its licence on huggingface.co, then:\n"
                "        export HF_TOKEN=hf_...   (or: uv run huggingface-cli login)\n"
                "    and re-run this script.",
                file=sys.stderr,
            )
        raise SystemExit(1)
    print(f"    -> {path}")

# The corpus. Pulling it now means a network or gating problem surfaces in the first
# minute rather than three hours into a paid sweep.
print("--- RafaelMPereira/HealthCareMagic-100k-Chat-Format-en")
try:
    from dp_rag_eval.datasets.loaders import HealthCareMagicLoader

    docs = HealthCareMagicLoader(n_docs=8, seed=0).load()
    assert docs, "loader returned no documents"
    assert docs[0].metadata.get("query"), "documents carry no metadata['query'] — utility task would be empty"
    print(f"    -> ok, {len(docs)} sample documents, query field present")
except Exception as exc:  # noqa: BLE001
    print(f"    FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
    raise SystemExit(1)
PY

echo
echo "=== 6/6  Sanity check: dtype, VRAM, thinking mode"
uv run python - "$GENERATOR" <<'PY'
import sys
import torch
from dp_rag_eval.pipelines.loading import chat_prompt, last_token_logits, load_generator, supports_thinking

model_id = sys.argv[1]
print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("ERROR: torch cannot see the GPU — wrong torch build for this CUDA version.")
print(f"device: {torch.cuda.get_device_name(0)}")
print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")

model, tokenizer = load_generator(model_id, "cuda")
print(f"loaded dtype: {model.dtype}")
if model.dtype == torch.float32:
    raise SystemExit("ERROR: model loaded in fp32 — a 9B model will not fit alongside activations.")

messages = [{"role": "user", "content": "Does this appear in the context? Answer with Yes or No."}]
prompt = chat_prompt(tokenizer, messages, tokenize=False, add_generation_prompt=True)
print(f"thinking-capable template: {supports_thinking(tokenizer)}")
print(f"prompt tail: ...{prompt[-120:]!r}")

ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
# Via the framework helper, so this also exercises the logits_to_keep path the
# whole matrix depends on for memory.
first = int(last_token_logits(model, input_ids=ids)[0].argmax())
print(f"first generated token: {first} -> {tokenizer.decode([first])!r}")
if "think" in tokenizer.decode([first]).lower():
    raise SystemExit(
        "ERROR: the model still wants to emit a <think> token. Gray-box scores would be "
        "meaningless — check that enable_thinking=False reached apply_chat_template."
    )

allocated = torch.cuda.memory_allocated() / 1024**3
total = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"VRAM: {allocated:.1f} GiB used by weights / {total:.1f} GiB total")
print("OK")
PY

echo
echo "=== bootstrap complete"
echo
echo "In every new shell:  cd $REPO_ROOT && source .env.gpu"
echo
echo "Before starting the matrix, run the equivalence checks — they catch the"
echo "optimisations that fail silently rather than raising:"
echo "    uv run python scripts/verify_optimizations.py --generator $GENERATOR"
echo
echo "Then follow docs/RUNBOOK_GPU.md (start with the calibration cell in section 1)."
