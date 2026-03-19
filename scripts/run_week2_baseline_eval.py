#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from fgr.baseline import BaselineEvalConfig, run_baseline_eval


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
    result = run_baseline_eval(
        BaselineEvalConfig(
            input_path=args.input,
            dataset=args.dataset,
            outdir=args.outdir,
            split=args.split,
            beam_size=args.beam_size,
            nli_model=args.nli_model,
            nli_batch_size=args.nli_batch_size,
            nli_max_source_sentences=args.nli_max_source_sentences,
            nli_max_summary_sentences=args.nli_max_summary_sentences,
        )
    )
    print(f"Saved summary: {result['summary_out']}")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
