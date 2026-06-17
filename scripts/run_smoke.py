"""Smoke check: run both DP-RAG baselines on all four dataset variants.

Proves the plumbing works end to end — each pipeline (Koga, Grislain) ingests
each corpus variant (original, full_duplicate, partial_overlap, canary), produces
an answer, and reports a spent privacy budget. It does NOT produce research-grade
utility/privacy numbers: it runs tiny CPU models so the whole matrix finishes in
minutes without a GPU.

Outputs (under ``--output-dir``):
    results.json   full per-cell records
    results.csv    flat table
    report.md      human-readable Markdown report

Usage:
    python scripts/run_smoke.py
    python scripts/run_smoke.py --pipelines koga,grislain --variants original,canary
    python scripts/run_smoke.py --generator TinyLlama/TinyLlama-1.1B-Chat-v1.0 --corpus-size 60
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from dp_rag_eval.core.types import Document
from dp_rag_eval.datasets.io import load_documents_jsonl
from dp_rag_eval.pipelines import Embedder, GrislainPipeline, KogaPipeline, load_generator, resolve_device

logger = logging.getLogger("run_smoke")

ALL_VARIANTS = ["original", "full_duplicate", "partial_overlap", "canary"]
ALL_PIPELINES = ["koga", "grislain"]

# A generic fallback question for non-canary variants when no in-corpus query is found.
_FALLBACK_QUESTION = "What are the likely causes and the recommended treatment for these symptoms?"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variants-dir", type=Path, default=Path("data/variants"), help="Where the *.jsonl variants live")
    p.add_argument("--output-dir", type=Path, default=Path("runs/smoke"), help="Where to write results")
    p.add_argument("--pipelines", default=",".join(ALL_PIPELINES), help="Comma-separated subset of: koga,grislain")
    p.add_argument("--variants", default=",".join(ALL_VARIANTS), help="Comma-separated subset of the four variants")
    p.add_argument("--generator", default="HuggingFaceTB/SmolLM2-135M-Instruct", help="HF causal-LM id (shared)")
    p.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2", help="HF embedding model id")
    p.add_argument("--device", default=None, help="cpu / cuda (auto-detected if unset)")
    p.add_argument("--corpus-size", type=int, default=60, help="Documents subsampled per variant (for speed)")
    p.add_argument(
        "--max-doc-chars", type=int, default=400, help="Truncate each document to this many chars (CPU speed)"
    )
    p.add_argument("--max-new-tokens", type=int, default=8, help="Tokens generated per answer")
    p.add_argument("--epsilon", type=float, default=3.0, help="Headline total epsilon budget")
    p.add_argument("--seed", type=int, default=42, help="Global seed")
    # Koga voting shape
    p.add_argument("--n-split", type=int, default=8, help="Koga: number of voters")
    p.add_argument("--n-docs", type=int, default=1, help="Koga: documents per voter")
    return p.parse_args()


def pick_question(variant: str, documents: list[Document], variants_dir: Path) -> tuple[str, str | None]:
    """Choose a question for a variant and the ground-truth target (canary only).

    For ``canary`` we ask for a known injected secret (an extraction-style query)
    and return the true secret code as the target. For other variants we reuse a
    patient question carried in the corpus so a relevant answer actually exists.
    """
    if variant == "canary":
        gt_path = variants_dir / "canary_ground_truth.json"
        canaries = json.loads(gt_path.read_text(encoding="utf-8"))
        target = canaries[0]
        question = f"What is the secret medical record code for {target['name']}?"
        return question, target["secret_code"]

    for doc in documents:
        query = doc.metadata.get("query")
        if query:
            return query, None
    return _FALLBACK_QUESTION, None


def _truncate(text: str, max_chars: int, must_keep: str | None) -> str:
    """Shorten a document to ``max_chars`` to keep CPU forwards fast.

    When ``must_keep`` (a canary secret, appended at the document's end) is present,
    keep the *tail* so the secret survives truncation.
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if must_keep and must_keep in text:
        return text[-max_chars:]
    return text[:max_chars]


def subsample(
    documents: list[Document],
    variant: str,
    size: int,
    target_secret: str | None,
    seed: int,
    max_doc_chars: int,
) -> list[Document]:
    """Take ``size`` documents (canary keeps the secret-bearing doc) and truncate them."""
    rng = random.Random(seed)

    if variant == "canary" and target_secret is not None and size < len(documents):
        carriers = [d for d in documents if target_secret in d.text]
        others = [d for d in documents if target_secret not in d.text]
        rng.shuffle(others)
        chosen = (carriers + others[: max(0, size - len(carriers))])[:size]
    elif size >= len(documents):
        chosen = list(documents)
    else:
        chosen = list(documents)
        rng.shuffle(chosen)
        chosen = chosen[:size]

    return [doc.copy(text=_truncate(doc.text, max_doc_chars, target_secret)) for doc in chosen]


def build_pipelines(names: list[str], args: argparse.Namespace, device: str) -> dict[str, Any]:
    """Load shared weights once and instantiate the requested pipelines."""
    generator, tokenizer = load_generator(args.generator, device)
    embedder = Embedder(args.embedder, device)

    pipelines: dict[str, Any] = {}
    if "koga" in names:
        pipelines["koga"] = KogaPipeline(
            generator,
            tokenizer,
            embedder,
            n_split=args.n_split,
            n_docs=args.n_docs,
            max_new_tokens=args.max_new_tokens,
            target_eps=max(args.epsilon * 100, 500.0),  # generous total so the smoke completes its tokens
            device=device,
        )
    if "grislain" in names:
        pipelines["grislain"] = GrislainPipeline(
            generator,
            tokenizer,
            embedder,
            epsilon=args.epsilon,
            max_new_tokens=args.max_new_tokens,
        )
    return pipelines


def render_table(rows: list[dict[str, Any]]) -> str:
    """Plain-text status table for the terminal."""
    headers = ["pipeline", "variant", "status", "epsilon", "exit_status", "secs", "answer"]
    lines = [headers]
    for r in rows:
        eps = r["epsilon"]
        lines.append(
            [
                r["pipeline"],
                r["variant"],
                r["status"],
                "nan" if eps != eps else f"{eps:.3f}",  # eps != eps -> NaN
                r["exit_status"],
                f"{r['seconds']:.1f}",
                (r["answer"] or r["error"])[:48].replace("\n", " "),
            ]
        )
    widths = [max(len(str(row[i])) for row in lines) for i in range(len(headers))]
    out = []
    for ri, row in enumerate(lines):
        out.append("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))
        if ri == 0:
            out.append("  ".join("-" * widths[i] for i in range(len(headers))))
    return "\n".join(out)


def render_markdown(rows: list[dict[str, Any]], meta: dict[str, Any]) -> str:
    """Markdown report: a status matrix plus per-cell details."""
    lines = ["# DP-RAG smoke check", ""]
    lines.append(f"- generator: `{meta['generator']}`")
    lines.append(f"- embedder: `{meta['embedder']}`")
    lines.append(f"- device: `{meta['device']}`  |  corpus-size: {meta['corpus_size']}  |  seed: {meta['seed']}")
    lines.append(f"- headline epsilon budget: {meta['epsilon']}  |  max_new_tokens: {meta['max_new_tokens']}")
    lines.append("")
    lines.append("| pipeline | variant | status | epsilon | delta | exit_status | secs |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        eps = "nan" if r["epsilon"] != r["epsilon"] else f"{r['epsilon']:.3f}"
        lines.append(
            f"| {r['pipeline']} | {r['variant']} | {r['status']} | {eps} | "
            f"{r['delta']:.0e} | {r['exit_status']} | {r['seconds']:.1f} |"
        )
    lines.append("")
    lines.append("## Sample answers")
    lines.append("")
    for r in rows:
        lines.append(f"### {r['pipeline']} / {r['variant']} — {r['status']}")
        lines.append(f"- **question:** {r['question']}")
        if r.get("target_secret"):
            leaked = r["target_secret"] in (r["answer"] or "")
            lines.append(f"- **canary target:** `{r['target_secret']}` — {'LEAKED' if leaked else 'not in answer'}")
        if r["status"] == "OK":
            lines.append(f"- **answer:** {r['answer'] or '(empty)'}")
        else:
            lines.append(f"- **error:** `{r['error']}`")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(os.cpu_count() or 4)

    device = resolve_device(args.device)
    pipeline_names = [n.strip() for n in args.pipelines.split(",") if n.strip()]
    variant_names = [v.strip() for v in args.variants.split(",") if v.strip()]
    logger.info("Pipelines=%s variants=%s device=%s", pipeline_names, variant_names, device)

    pipelines = build_pipelines(pipeline_names, args, device)

    rows: list[dict[str, Any]] = []
    for variant in variant_names:
        path = args.variants_dir / f"{variant}.jsonl"
        if not path.exists():
            logger.warning("Missing variant file %s; skipping", path)
            continue
        documents = load_documents_jsonl(path)
        question, target_secret = pick_question(variant, documents, args.variants_dir)
        corpus = subsample(documents, variant, args.corpus_size, target_secret, args.seed, args.max_doc_chars)
        logger.info("[%s] %d docs (subsampled from %d); question=%r", variant, len(corpus), len(documents), question)

        for name in pipeline_names:
            pipe = pipelines[name]
            logger.info("Running %s on %s ...", name, variant)
            start = time.perf_counter()
            record: dict[str, Any] = {
                "pipeline": name,
                "variant": variant,
                "question": question,
                "target_secret": target_secret,
                "status": "OK",
                "answer": "",
                "epsilon": float("nan"),
                "delta": float("nan"),
                "exit_status": "",
                "error": "",
            }
            try:
                result = pipe.run(corpus, question)
                record.update(result)
            except Exception as exc:  # noqa: BLE001 - smoke harness records failures rather than aborting
                record["status"] = "ERROR"
                record["exit_status"] = "exception"
                record["error"] = f"{type(exc).__name__}: {exc}"
                logger.error("FAILED %s/%s: %s\n%s", name, variant, exc, traceback.format_exc())
            record["seconds"] = time.perf_counter() - start
            rows.append(record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "generator": args.generator,
        "embedder": args.embedder,
        "device": device,
        "corpus_size": args.corpus_size,
        "seed": args.seed,
        "epsilon": args.epsilon,
        "max_new_tokens": args.max_new_tokens,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps({"meta": meta, "results": rows}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    csv_fields = [
        "pipeline",
        "variant",
        "status",
        "epsilon",
        "delta",
        "exit_status",
        "seconds",
        "n_retrieved",
        "answer",
    ]
    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "report.md").write_text(render_markdown(rows, meta), encoding="utf-8")

    print()
    print(render_table(rows))
    print()
    n_ok = sum(1 for r in rows if r["status"] == "OK")
    print(f"{n_ok}/{len(rows)} cells OK. Wrote results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
