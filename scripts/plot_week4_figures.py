#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


FOCUS_STRATEGIES = ["top1", "weighted_sum", "agreement_gated"]
DISPLAY_NAMES = {
    "top1": "Top-1",
    "single_metric_summac": "SummaC",
    "single_metric_factcc": "FactCC",
    "single_metric_nli_support": "NLI Support",
    "weighted_sum": "Weighted Sum",
    "agreement_gated": "Agreement Gated",
}
COLORS = {
    "top1": "#264653",
    "single_metric_summac": "#8ab17d",
    "single_metric_factcc": "#e9c46a",
    "single_metric_nli_support": "#577590",
    "weighted_sum": "#2a9d8f",
    "agreement_gated": "#e76f51",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Week 4/5 figures from strategy comparison outputs.")
    parser.add_argument("--input", default="outputs/week4_analysis/strategy_comparison.csv")
    parser.add_argument("--outdir", default="outputs/week4_analysis/figures")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float_rows(rows: list[dict]) -> list[dict]:
    float_keys = {
        "rouge1",
        "rouge2",
        "rougeL",
        "rougeLsum",
        "summac",
        "factcc",
        "nli_support",
        "delta_rougeL_vs_top1",
        "delta_summac_vs_top1",
        "delta_factcc_vs_top1",
        "delta_nli_support_vs_top1",
        "changed_from_top1_rate",
        "mean_selected_candidate_idx",
    }
    converted = []
    for row in rows:
        item = dict(row)
        for key in float_keys:
            item[key] = float(item[key])
        if row.get("agreement_gate_pass_rate", "") == "":
            item["agreement_gate_pass_rate"] = None
        else:
            item["agreement_gate_pass_rate"] = float(row["agreement_gate_pass_rate"])
        converted.append(item)
    return converted


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def filter_rows(rows: list[dict], dataset: str) -> list[dict]:
    by_strategy = {row["strategy"]: row for row in rows if row["dataset"] == dataset}
    return [by_strategy[strategy] for strategy in FOCUS_STRATEGIES]


def plot_metric_bars(rows: list[dict], outdir: Path) -> Path:
    datasets = sorted({row["dataset"] for row in rows})
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    metric_specs = [
        ("rougeL", "ROUGE-L"),
        ("factcc", "FactCC"),
        ("nli_support", "NLI Support"),
    ]

    for row_idx, dataset in enumerate(datasets):
        dataset_rows = filter_rows(rows, dataset)
        labels = [DISPLAY_NAMES[row["strategy"]] for row in dataset_rows]
        colors = [COLORS[row["strategy"]] for row in dataset_rows]
        for col_idx, (metric_key, metric_title) in enumerate(metric_specs):
            ax = axes[row_idx][col_idx]
            values = [row[metric_key] for row in dataset_rows]
            bars = ax.bar(labels, values, color=colors, width=0.65)
            ax.set_title(f"{dataset}: {metric_title}")
            ax.set_ylim(0, max(values) * 1.18)
            ax.tick_params(axis="x", rotation=15)
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + max(values) * 0.02,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    path = outdir / "figure1_metric_bars.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_tradeoff(rows: list[dict], outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
    datasets = sorted({row["dataset"] for row in rows})
    markers = {"xsum": "o", "cnn_dailymail": "s"}

    for dataset in datasets:
        dataset_rows = [row for row in rows if row["dataset"] == dataset and row["strategy"] != "top1"]
        for row in dataset_rows:
            x = row["delta_rougeL_vs_top1"]
            y = row["delta_factcc_vs_top1"] + row["delta_nli_support_vs_top1"]
            ax.scatter(
                x,
                y,
                s=130,
                marker=markers[dataset],
                color=COLORS[row["strategy"]],
                edgecolor="black",
                linewidth=0.7,
                alpha=0.9,
            )
            ax.text(
                x + 0.0001,
                y + 0.001,
                f"{dataset}\n{DISPLAY_NAMES[row['strategy']]}",
                fontsize=8.5,
            )

    ax.axvline(0.0, color="#444444", linestyle="--", linewidth=1)
    ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1)
    ax.set_xlabel("Delta ROUGE-L vs Top-1")
    ax.set_ylabel("Delta Faithfulness (FactCC + NLI)")
    ax.set_title("Faithfulness-ROUGE Trade-off")

    path = outdir / "figure2_tradeoff_scatter.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_gate_behavior(rows: list[dict], outdir: Path) -> Path:
    datasets = sorted({row["dataset"] for row in rows})
    dataset_rows = [next(row for row in rows if row["dataset"] == dataset and row["strategy"] == "agreement_gated") for dataset in datasets]

    labels = datasets
    changed = [row["changed_from_top1_rate"] * 100.0 for row in dataset_rows]
    passed = [(row["agreement_gate_pass_rate"] or 0.0) * 100.0 for row in dataset_rows]

    x = range(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    bars1 = ax.bar([i - width / 2 for i in x], changed, width=width, color="#457b9d", label="Changed from Top-1")
    bars2 = ax.bar([i + width / 2 for i in x], passed, width=width, color="#f4a261", label="Gate Passed")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Rate (%)")
    ax.set_title("Agreement Gate Behavior")
    ax.legend(frameon=False)
    ax.set_ylim(0, max(changed + passed) * 1.2)

    for bars in (bars1, bars2):
        for bar in bars:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1.0, f"{value:.1f}", ha="center", va="bottom", fontsize=9)

    path = outdir / "figure3_gate_behavior.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    rows = to_float_rows(load_rows(Path(args.input)))
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    outputs = [
        plot_metric_bars(rows, outdir),
        plot_tradeoff(rows, outdir),
        plot_gate_behavior(rows, outdir),
    ]
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
