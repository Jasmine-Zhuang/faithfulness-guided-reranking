#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from fgr.data import load_split
from fgr.generation import BartCandidateGenerator, GenerationConfig, resolve_model_name
from fgr.io import write_jsonl


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
    spec, ds = load_split(args.dataset, args.split)

    n = min(args.num_examples, len(ds))
    ds = ds.select(range(n))
    sources = [row[spec.text_field] for row in ds]
    references = [row[spec.summary_field] for row in ds]

    cfg = GenerationConfig(
        model_name=resolve_model_name(spec, args.model_name),
        beam_size=args.beam_size,
        batch_size=args.batch_size,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        length_penalty=args.length_penalty,
    )

    generator = BartCandidateGenerator(cfg)
    candidates = generator.generate_candidates(sources)

    rows = []
    for i, (src, ref, cand_list) in enumerate(zip(sources, references, candidates, strict=True)):
        rows.append(
            {
                "example_id": i,
                "dataset": args.dataset,
                "split": args.split,
                "source": src,
                "reference": ref,
                "top1": cand_list[0],
                "candidates": cand_list,
            }
        )

    outpath = Path(args.outdir) / args.dataset / f"{args.split}_k{args.beam_size}_candidates.jsonl"
    write_jsonl(outpath, rows)
    print(f"Wrote {len(rows)} examples to: {outpath}")


if __name__ == "__main__":
    main()
