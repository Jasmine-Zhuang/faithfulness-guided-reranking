#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev

from tqdm.auto import tqdm

from fgr.io import read_jsonl, write_jsonl
from fgr.metrics import NLIConfig, NLIFaithfulnessScorer, compute_rouge, keyword_precision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Week 3: Run single-metric and agreement-gated reranking over k-best summaries."
    )
    parser.add_argument("--input", type=str, required=True, help="Path to *_candidates.jsonl from Week 1")
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--nli-model", type=str, default="facebook/bart-large-mnli")
    parser.add_argument("--nli-batch-size", type=int, default=32)
    parser.add_argument("--nli-max-source-sentences", type=int, default=20)
    parser.add_argument("--nli-max-summary-sentences", type=int, default=5)
    parser.add_argument(
        "--fallback-strategy",
        choices=["weighted_sum", "top1"],
        default="weighted_sum",
        help="Selection used when agreement gate is not satisfied.",
    )
    parser.add_argument(
        "--weight-nli",
        type=float,
        default=1.0,
        help="Weight for normalized NLI support in weighted-sum fallback.",
    )
    parser.add_argument(
        "--weight-keyword",
        type=float,
        default=1.0,
        help="Weight for normalized keyword precision in weighted-sum fallback.",
    )
    return parser.parse_args()


def zscore(values: list[float]) -> list[float]:
    if not values:
        return []
    if len(values) == 1:
        return [0.0]
    std = pstdev(values)
    if std == 0.0:
        return [0.0 for _ in values]
    mu = mean(values)
    return [(v - mu) / std for v in values]


def argmax(values: list[float]) -> int:
    if not values:
        raise ValueError("Cannot select from an empty list.")
    return max(range(len(values)), key=values.__getitem__)


def compute_candidate_scores(
    source: str,
    candidates: list[str],
    nli_scorer: NLIFaithfulnessScorer,
) -> list[dict[str, float | int | str]]:
    rows = []
    for idx, candidate in enumerate(candidates):
        rows.append(
            {
                "candidate_idx": idx,
                "summary": candidate,
                "nli_support": nli_scorer.score(source, candidate),
                "keyword_precision": keyword_precision(source, candidate),
            }
        )

    nli_z = zscore([row["nli_support"] for row in rows])
    kw_z = zscore([row["keyword_precision"] for row in rows])
    for row, nli_norm, kw_norm in zip(rows, nli_z, kw_z, strict=True):
        row["nli_support_z"] = nli_norm
        row["keyword_precision_z"] = kw_norm
    return rows


def select_strategies(
    candidate_rows: list[dict[str, float | int | str]],
    *,
    weight_nli: float,
    weight_keyword: float,
    fallback_strategy: str,
) -> dict[str, dict[str, int | float | str]]:
    nli_scores = [float(row["nli_support"]) for row in candidate_rows]
    kw_scores = [float(row["keyword_precision"]) for row in candidate_rows]
    nli_z = [float(row["nli_support_z"]) for row in candidate_rows]
    kw_z = [float(row["keyword_precision_z"]) for row in candidate_rows]

    weighted_sum_scores = [
        weight_nli * nli_score + weight_keyword * kw_score
        for nli_score, kw_score in zip(nli_z, kw_z, strict=True)
    ]

    nli_idx = argmax(nli_scores)
    kw_idx = argmax(kw_scores)
    weighted_idx = argmax(weighted_sum_scores)
    top1_idx = 0

    votes = Counter([nli_idx, kw_idx])
    agreed_idx, agreed_votes = votes.most_common(1)[0]
    use_gate = agreed_votes >= 2

    if use_gate:
        agreement_idx = agreed_idx
        agreement_reason = "metric_agreement"
    elif fallback_strategy == "weighted_sum":
        agreement_idx = weighted_idx
        agreement_reason = "fallback_weighted_sum"
    else:
        agreement_idx = top1_idx
        agreement_reason = "fallback_top1"

    def build_selection(name: str, idx: int, extra: dict[str, int | float | str] | None = None) -> dict[str, int | float | str]:
        selected = dict(candidate_rows[idx])
        selected["strategy"] = name
        if extra:
            selected.update(extra)
        return selected

    return {
        "top1": build_selection("top1", top1_idx),
        "single_metric_nli": build_selection("single_metric_nli", nli_idx),
        "single_metric_keyword": build_selection("single_metric_keyword", kw_idx),
        "weighted_sum": build_selection("weighted_sum", weighted_idx),
        "agreement_gated": build_selection(
            "agreement_gated",
            agreement_idx,
            {
                "gate_passed": int(use_gate),
                "gate_reason": agreement_reason,
                "agreed_candidate_idx": agreed_idx,
                "agreement_votes": agreed_votes,
            },
        ),
    }


def summarize_strategy(rows: list[dict], strategy_name: str) -> dict[str, object]:
    selected = [row["strategies"][strategy_name] for row in rows]
    predictions = [str(row["summary"]) for row in selected]
    references = [row["reference"] for row in rows]

    rouge_scores = compute_rouge(predictions, references)
    mean_nli = mean(float(row["nli_support"]) for row in selected)
    mean_kw = mean(float(row["keyword_precision"]) for row in selected)

    summary: dict[str, object] = {
        "num_examples": len(rows),
        "rouge": rouge_scores,
        "faithfulness": {
            "nli_support": mean_nli,
            "keyword_precision": mean_kw,
        },
    }

    if strategy_name == "agreement_gated":
        gate_passes = sum(int(row.get("gate_passed", 0)) for row in selected)
        reasons = Counter(str(row.get("gate_reason", "unknown")) for row in selected)
        summary["agreement_gate"] = {
            "pass_rate": gate_passes / len(selected),
            "gate_reason_counts": dict(reasons),
        }

    return summary


def main() -> None:
    args = parse_args()
    rows = list(read_jsonl(args.input))
    if not rows:
        raise ValueError(f"No rows found in {args.input}")

    nli_scorer = NLIFaithfulnessScorer(
        cfg=NLIConfig(
            model_name=args.nli_model,
            batch_size=args.nli_batch_size,
            max_source_sentences=args.nli_max_source_sentences,
            max_summary_sentences=args.nli_max_summary_sentences,
        )
    )

    reranked_rows = []
    for row in tqdm(rows, desc="Week 3 reranking"):
        candidates = row["candidates"]
        candidate_rows = compute_candidate_scores(row["source"], candidates, nli_scorer)
        strategies = select_strategies(
            candidate_rows,
            weight_nli=args.weight_nli,
            weight_keyword=args.weight_keyword,
            fallback_strategy=args.fallback_strategy,
        )
        reranked_rows.append(
            {
                "example_id": row["example_id"],
                "dataset": row.get("dataset"),
                "split": row.get("split"),
                "source": row["source"],
                "reference": row["reference"],
                "original_top1": row.get("top1", candidates[0]),
                "candidate_scores": candidate_rows,
                "strategies": strategies,
            }
        )

    strategy_metrics = {
        strategy_name: summarize_strategy(reranked_rows, strategy_name)
        for strategy_name in ["top1", "single_metric_nli", "single_metric_keyword", "weighted_sum", "agreement_gated"]
    }

    input_path = Path(args.input)
    dataset_name = rows[0].get("dataset", input_path.parent.name)
    stem = input_path.stem.replace("_candidates", "")
    run_dir = Path(args.outdir) / dataset_name / f"week3_{stem}"
    run_dir.mkdir(parents=True, exist_ok=True)

    reranked_out = run_dir / "reranked_examples.jsonl"
    metrics_out = run_dir / "strategy_metrics.json"
    config_out = run_dir / "run_config.json"

    write_jsonl(reranked_out, reranked_rows)
    metrics_out.write_text(json.dumps(strategy_metrics, indent=2), encoding="utf-8")
    config_out.write_text(
        json.dumps(
            {
                "input": args.input,
                "fallback_strategy": args.fallback_strategy,
                "weight_nli": args.weight_nli,
                "weight_keyword": args.weight_keyword,
                "nli_model": args.nli_model,
                "nli_batch_size": args.nli_batch_size,
                "nli_max_source_sentences": args.nli_max_source_sentences,
                "nli_max_summary_sentences": args.nli_max_summary_sentences,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved reranked examples: {reranked_out}")
    print(f"Saved strategy metrics: {metrics_out}")
    print(json.dumps(strategy_metrics, indent=2))


if __name__ == "__main__":
    main()
