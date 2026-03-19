#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from fgr.factcc import FactCCConfig, run_factcc_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline top-1 summaries with FactCC without modifying the Week 2 pipeline."
    )
    parser.add_argument("--input", type=str, default=None, help="Path to *_candidates.jsonl")
    parser.add_argument("--dataset", choices=["cnn_dailymail", "xsum"], default=None)
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--split", default="validation", help="Used with --dataset when --input is omitted.")
    parser.add_argument("--beam-size", type=int, default=5, help="Used with --dataset when --input is omitted.")
    parser.add_argument("--model-name", type=str, default="manueldeprada/FactCC")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cpu or cuda")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    result = run_factcc_eval(
        FactCCConfig(
            input_path=args.input,
            dataset=args.dataset,
            outdir=args.outdir,
            split=args.split,
            beam_size=args.beam_size,
            model_name=args.model_name,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
        )
    )
    print(f"Saved summary: {result['summary_out']}")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
