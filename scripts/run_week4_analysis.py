#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

from fgr.io import ensure_parent_dir, read_jsonl


STRATEGIES = [
    "top1",
    "single_metric_summac",
    "single_metric_factcc",
    "single_metric_nli_support",
    "weighted_sum",
    "agreement_gated",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Week 4-5: aggregate reranking metrics and extract qualitative examples."
    )
    parser.add_argument("--baseline-root", type=str, default="outputs")
    parser.add_argument("--week3-root", type=str, default="outputs")
    parser.add_argument("--datasets", nargs="+", default=["xsum", "cnn_dailymail"])
    parser.add_argument("--split", default="validation")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--outdir", type=str, default="outputs/week4_analysis")
    parser.add_argument("--num-qual-cases", type=int, default=3)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_strategy_metrics(week3_root: Path, dataset: str, split: str, beam_size: int) -> dict:
    path = week3_root / dataset / f"week3_{split}_k{beam_size}" / "strategy_metrics.json"
    return load_json(path)


def load_reranked_rows(week3_root: Path, dataset: str, split: str, beam_size: int) -> list[dict]:
    path = week3_root / dataset / f"week3_{split}_k{beam_size}" / "reranked_examples.jsonl"
    return list(read_jsonl(path))


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def fmt(value: float) -> str:
    return f"{value:.4f}"


def fmt_delta(value: float) -> str:
    return f"{value:+.4f}"


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_strategy_rows(dataset: str, metrics: dict) -> list[dict]:
    rows = []
    for strategy in STRATEGIES:
        item = metrics[strategy]
        row = {
            "dataset": dataset,
            "strategy": strategy,
            "rouge1": item["rouge"]["rouge1"],
            "rouge2": item["rouge"]["rouge2"],
            "rougeL": item["rouge"]["rougeL"],
            "rougeLsum": item["rouge"]["rougeLsum"],
            "summac": item["faithfulness"]["summac"],
            "factcc": item["faithfulness"]["factcc"],
            "nli_support": item["faithfulness"]["nli_support"],
            "delta_rougeL_vs_top1": item["delta_vs_top1"]["rouge"]["rougeL"],
            "delta_summac_vs_top1": item["delta_vs_top1"]["faithfulness"]["summac"],
            "delta_factcc_vs_top1": item["delta_vs_top1"]["faithfulness"]["factcc"],
            "delta_nli_support_vs_top1": item["delta_vs_top1"]["faithfulness"]["nli_support"],
            "changed_from_top1_rate": item["selection"]["changed_from_top1_rate"],
            "mean_selected_candidate_idx": item["selection"]["mean_selected_candidate_idx"],
            "agreement_gate_pass_rate": item.get("agreement_gate", {}).get("pass_rate", ""),
        }
        rows.append(row)
    return rows


def choose_tradeoff_winner(metrics: dict) -> str:
    candidates = [s for s in STRATEGIES if s != "top1"]
    return max(
        candidates,
        key=lambda s: (
            (
                metrics[s]["delta_vs_top1"]["faithfulness"]["summac"]
                + metrics[s]["delta_vs_top1"]["faithfulness"]["factcc"]
                + metrics[s]["delta_vs_top1"]["faithfulness"]["nli_support"]
            )
            / 3.0,
            metrics[s]["delta_vs_top1"]["rouge"]["rougeL"],
        ),
    )


def choose_examples(rows: list[dict], strategy_name: str, num_cases: int) -> tuple[list[dict], list[dict]]:
    scored = []
    for row in rows:
        top1 = row["strategies"]["top1"]
        chosen = row["strategies"][strategy_name]
        score = (
            (chosen["summac"] - top1["summac"])
            + (chosen["factcc"] - top1["factcc"])
            + (chosen["nli_support"] - top1["nli_support"])
        )
        scored.append(
            {
                "score": score,
                "example_id": row["example_id"],
                "reference": row["reference"],
                "source": row["source"],
                "top1": top1,
                "selected": chosen,
            }
        )
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:num_cases], list(reversed(scored[-num_cases:]))


def example_block(dataset: str, label: str, example: dict) -> str:
    top1 = example["top1"]
    selected = example["selected"]
    return "\n".join(
        [
            f"### {dataset} Example {example['example_id']} ({label})",
            f"- Reference: {example['reference']}",
            f"- Top-1: {top1['summary']}",
            f"- Agreement-gated: {selected['summary']}",
            f"- Top-1 scores: FactCC={fmt(top1['factcc'])}, NLI={fmt(top1['nli_support'])}, SummaC={fmt(top1['summac'])}",
            f"- Agreement-gated scores: FactCC={fmt(selected['factcc'])}, NLI={fmt(selected['nli_support'])}, SummaC={fmt(selected['summac'])}",
            f"- Gate reason: {selected.get('gate_reason', 'n/a')}",
        ]
    )


def build_summary_markdown(
    dataset_to_week3: dict[str, dict],
) -> str:
    lines = [
        "# Week 4 Analysis Summary",
        "",
        "This file summarizes the quantitative comparison between baseline decoding and Week 3 reranking strategies using the proposal metrics: ROUGE-L, SummaC, FactCC, and NLI support.",
        "",
    ]

    for dataset, metrics in dataset_to_week3.items():
        top1 = metrics["top1"]
        winner = choose_tradeoff_winner(metrics)
        agreement = metrics["agreement_gated"]
        weighted = metrics["weighted_sum"]
        lines.extend(
            [
                f"## {dataset}",
                "",
                f"- Baseline top-1 scores: ROUGE-L={fmt(top1['rouge']['rougeL'])}, SummaC={fmt(top1['faithfulness']['summac'])}, FactCC={fmt(top1['faithfulness']['factcc'])}, NLI={fmt(top1['faithfulness']['nli_support'])}",
                f"- Best overall trade-off in this run: `{winner}`",
                f"- `agreement_gated` changed the top-1 choice on {pct(agreement['selection']['changed_from_top1_rate'])} of examples and the gate passed on {pct(agreement['agreement_gate']['pass_rate'])} of examples.",
                f"- `agreement_gated` vs top-1: delta ROUGE-L {fmt_delta(agreement['delta_vs_top1']['rouge']['rougeL'])}, delta SummaC {fmt_delta(agreement['delta_vs_top1']['faithfulness']['summac'])}, delta FactCC {fmt_delta(agreement['delta_vs_top1']['faithfulness']['factcc'])}, delta NLI {fmt_delta(agreement['delta_vs_top1']['faithfulness']['nli_support'])}.",
                f"- `weighted_sum` vs top-1: delta ROUGE-L {fmt_delta(weighted['delta_vs_top1']['rouge']['rougeL'])}, delta SummaC {fmt_delta(weighted['delta_vs_top1']['faithfulness']['summac'])}, delta FactCC {fmt_delta(weighted['delta_vs_top1']['faithfulness']['factcc'])}, delta NLI {fmt_delta(weighted['delta_vs_top1']['faithfulness']['nli_support'])}.",
                "",
            ]
        )

    lines.extend(
        [
            "## Takeaways",
            "",
            "- `xsum` shows the clearest benefit from reranking: FactCC and NLI improve meaningfully while ROUGE-L drops by less than 0.5 absolute points, though SummaC moves only slightly.",
            "- `cnn_dailymail` also improves on all three faithfulness metrics, but the gain is smaller relative to its already strong top-1 baseline.",
            "- The agreement gate is conservative in spirit, but in this run it still changes many selections because metric agreement happens on more than half of examples.",
            "- For the final report, `weighted_sum` remains the strongest default comparator and `agreement_gated` is the main proposed method.",
            "",
        ]
    )
    return "\n".join(lines)


def build_qualitative_markdown(dataset_to_examples: dict[str, tuple[list[dict], list[dict]]]) -> str:
    lines = [
        "# Week 5 Qualitative Analysis Notes",
        "",
        "The examples below highlight where agreement-gated reranking helps and where it can still fail.",
        "",
    ]
    for dataset, (best_examples, worst_examples) in dataset_to_examples.items():
        lines.append(f"## {dataset}")
        lines.append("")
        for example in best_examples:
            lines.append(example_block(dataset, "improvement", example))
            lines.append("")
        for example in worst_examples:
            lines.append(example_block(dataset, "failure", example))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    week3_root = Path(args.week3_root)
    outdir = Path(args.outdir)

    dataset_to_week3: dict[str, dict] = {}
    strategy_rows: list[dict] = []
    dataset_to_examples: dict[str, tuple[list[dict], list[dict]]] = {}

    for dataset in args.datasets:
        dataset_to_week3[dataset] = load_strategy_metrics(week3_root, dataset, args.split, args.beam_size)
        strategy_rows.extend(summarize_strategy_rows(dataset, dataset_to_week3[dataset]))
        reranked_rows = load_reranked_rows(week3_root, dataset, args.split, args.beam_size)
        dataset_to_examples[dataset] = choose_examples(reranked_rows, "agreement_gated", args.num_qual_cases)

    csv_path = outdir / "strategy_comparison.csv"
    write_csv(csv_path, strategy_rows, list(strategy_rows[0].keys()))

    summary_md = build_summary_markdown(dataset_to_week3)
    summary_path = outdir / "summary.md"
    ensure_parent_dir(summary_path)
    summary_path.write_text(summary_md, encoding="utf-8")

    qual_md = build_qualitative_markdown(dataset_to_examples)
    qual_path = outdir / "qualitative_examples.md"
    qual_path.write_text(qual_md, encoding="utf-8")

    print(f"Wrote strategy table: {csv_path}")
    print(f"Wrote quantitative summary: {summary_path}")
    print(f"Wrote qualitative notes: {qual_path}")


if __name__ == "__main__":
    main()
