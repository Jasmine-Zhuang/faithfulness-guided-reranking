#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm.auto import tqdm

from fgr.io import read_jsonl, resolve_candidate_jsonl, write_jsonl
from fgr.metrics import NLIConfig, NLIFaithfulnessScorer, compute_rouge, keyword_precision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 2: Evaluate baseline top-1 with ROUGE + faithfulness metrics.")
    parser.add_argument("--input", type=str, default=None, help="Path to *_candidates.jsonl")
    parser.add_argument("--dataset", choices=["cnn_dailymail", "xsum"], default=None)
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--split", default="validation", help="Used with --dataset when --input is omitted.")
    parser.add_argument("--beam-size", type=int, default=5, help="Used with --dataset when --input is omitted.")
    parser.add_argument("--nli-model", type=str, default="facebook/bart-large-mnli")
    parser.add_argument("--nli-batch-size", type=int, default=32)
    parser.add_argument("--nli-max-source-sentences", type=int, default=20)
    parser.add_argument("--nli-max-summary-sentences", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve_candidate_jsonl(
        input_path=args.input,
        dataset=args.dataset,
        outdir=args.outdir,
        split=args.split,
        beam_size=args.beam_size,
    )
    rows = list(read_jsonl(input_path))
    if not rows:
        raise ValueError(f"No rows found in {input_path}")

    predictions = [r["top1"] for r in rows]
    references = [r["reference"] for r in rows]

    rouge_scores = compute_rouge(predictions, references)

    nli = NLIFaithfulnessScorer(
        cfg=NLIConfig(
            model_name=args.nli_model,
            batch_size=args.nli_batch_size,
            max_source_sentences=args.nli_max_source_sentences,
            max_summary_sentences=args.nli_max_summary_sentences,
        )
    )

    per_example = []
    for r in tqdm(rows, desc="Scoring faithfulness"):
        nli_score = nli.score(r["source"], r["top1"])
        kw_score = keyword_precision(r["source"], r["top1"])
        per_example.append(
            {
                "example_id": r["example_id"],
                "nli_support": nli_score,
                "keyword_precision": kw_score,
            }
        )

    mean_nli = sum(x["nli_support"] for x in per_example) / len(per_example)
    mean_kw = sum(x["keyword_precision"] for x in per_example) / len(per_example)

    summary = {
        "num_examples": len(rows),
        "rouge": rouge_scores,
        "faithfulness": {
            "nli_support": mean_nli,
            "keyword_precision": mean_kw,
        },
    }

    dataset_name = rows[0].get("dataset", input_path.parent.name)
    stem = input_path.stem.replace("_candidates", "")
    run_dir = Path(args.outdir) / dataset_name / f"baseline_{stem}"
    run_dir.mkdir(parents=True, exist_ok=True)

    per_example_out = run_dir / "per_example_faithfulness.jsonl"
    summary_out = run_dir / "summary_metrics.json"
    write_jsonl(per_example_out, per_example)
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved summary: {summary_out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
