#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from fgr.qags import QAGSConfig, run_qags_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline top-1 summaries with a QAGS-style QA consistency metric."
    )
    parser.add_argument("--input", type=str, default=None, help="Path to *_candidates.jsonl")
    parser.add_argument("--dataset", choices=["cnn_dailymail", "xsum"], default=None)
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--split", default="validation", help="Used with --dataset when --input is omitted.")
    parser.add_argument("--beam-size", type=int, default=5, help="Used with --dataset when --input is omitted.")
    parser.add_argument("--qg-model", type=str, default="iarfmoose/t5-base-question-generator")
    parser.add_argument("--qa-model", type=str, default="deepset/roberta-base-squad2")
    parser.add_argument("--num-questions", type=int, default=3, help="Maximum questions generated per summary.")
    parser.add_argument("--max-answer-sentences", type=int, default=5, help="Maximum summary sentences to use as answer candidates.")
    parser.add_argument("--qg-max-source-length", type=int, default=384)
    parser.add_argument("--qg-max-target-length", type=int, default=64)
    parser.add_argument("--qa-batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cpu or cuda")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    result = run_qags_eval(
        QAGSConfig(
            input_path=args.input,
            dataset=args.dataset,
            outdir=args.outdir,
            split=args.split,
            beam_size=args.beam_size,
            qg_model=args.qg_model,
            qa_model=args.qa_model,
            num_questions=args.num_questions,
            max_answer_sentences=args.max_answer_sentences,
            qg_max_source_length=args.qg_max_source_length,
            qg_max_target_length=args.qg_max_target_length,
            qa_batch_size=args.qa_batch_size,
            device=args.device,
        )
    )
    print(f"Saved summary: {result['summary_out']}")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
