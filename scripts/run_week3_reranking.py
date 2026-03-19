#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from fgr.reranking import Week3RerankingConfig, run_week3_reranking


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Week 3: Run single-metric and agreement-gated reranking using SummaC and FactCC."
    )
    parser.add_argument("--input", type=str, default=None, help="Path to *_candidates.jsonl from Week 1")
    parser.add_argument("--dataset", choices=["cnn_dailymail", "xsum"], default=None)
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--split", default="validation", help="Used with --dataset when --input is omitted.")
    parser.add_argument("--beam-size", type=int, default=5, help="Used with --dataset when --input is omitted.")
    parser.add_argument("--num-examples", type=int, default=None, help="Optionally limit evaluation to the first N examples.")
    parser.add_argument(
        "--fallback-strategy",
        choices=["weighted_sum", "top1"],
        default="weighted_sum",
        help="Selection used when the agreement gate is not satisfied.",
    )
    parser.add_argument("--weight-summac", type=float, default=1.0)
    parser.add_argument("--weight-factcc", type=float, default=1.0)

    parser.add_argument("--summac-model-type", choices=["conv", "zs"], default="conv")
    parser.add_argument("--summac-model-name", type=str, default="vitc")
    parser.add_argument("--summac-granularity", choices=["document", "paragraph", "sentence", "2sents"], default="sentence")
    parser.add_argument("--summac-batch-size", type=int, default=8)
    parser.add_argument("--summac-cache-dir", type=str, default=".cache/summac")

    parser.add_argument("--factcc-model-name", type=str, default="manueldeprada/FactCC")
    parser.add_argument("--factcc-batch-size", type=int, default=8)
    parser.add_argument("--factcc-max-length", type=int, default=512)

    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cpu or cuda")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    result = run_week3_reranking(
        Week3RerankingConfig(
            input_path=args.input,
            dataset=args.dataset,
            outdir=args.outdir,
            split=args.split,
            beam_size=args.beam_size,
            num_examples=args.num_examples,
            fallback_strategy=args.fallback_strategy,
            weight_summac=args.weight_summac,
            weight_factcc=args.weight_factcc,
            summac_model_type=args.summac_model_type,
            summac_model_name=args.summac_model_name,
            summac_granularity=args.summac_granularity,
            summac_batch_size=args.summac_batch_size,
            summac_cache_dir=args.summac_cache_dir,
            factcc_model_name=args.factcc_model_name,
            factcc_batch_size=args.factcc_batch_size,
            factcc_max_length=args.factcc_max_length,
            device=args.device,
        )
    )

    print(f"Saved reranked examples: {result['reranked_out']}")
    print(f"Saved strategy metrics: {result['metrics_out']}")
    print(json.dumps(result["strategy_metrics"], indent=2))


if __name__ == "__main__":
    main()
