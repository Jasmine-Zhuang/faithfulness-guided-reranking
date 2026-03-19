#!/usr/bin/env python3
from __future__ import annotations

import argparse

from fgr.generation_pipeline import GenerationPipelineConfig, run_generation_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 1: Generate top-1 and n-best BART summaries.")
    parser.add_argument("--dataset", choices=["cnn_dailymail", "xsum"], required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--num-examples", type=int, default=300)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--min-new-tokens", type=int, default=20)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--outdir", type=str, default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_generation_pipeline(
        GenerationPipelineConfig(
            dataset=args.dataset,
            split=args.split,
            num_examples=args.num_examples,
            beam_size=args.beam_size,
            batch_size=args.batch_size,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            length_penalty=args.length_penalty,
            model_name=args.model_name,
            outdir=args.outdir,
        )
    )
    print(f"Wrote {result['num_examples']} examples to: {result['output_path']}")


if __name__ == "__main__":
    main()
