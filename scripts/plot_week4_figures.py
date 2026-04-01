#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


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
DATASET_ORDER = ["xsum", "cnn_dailymail"]
DATASET_LABELS = {
    "xsum": "XSum",
    "cnn_dailymail": "CNN/DailyMail",
}
STRATEGY_TICK_LABELS = {
    "top1": "Top-1",
    "weighted_sum": "Weighted\nSum",
    "agreement_gated": "Agreement\nGate",
}
SCATTER_LABELS = {
    "single_metric_summac": "SummaC",
    "single_metric_factcc": "FactCC",
    "single_metric_nli_support": "NLI Support",
    "weighted_sum": "Weighted Sum",
    "agreement_gated": "Agreement Gate",
}
SCATTER_MARKERS = {"xsum": "o", "cnn_dailymail": "s"}
SCATTER_LABEL_OFFSETS = {
    ("xsum", "single_metric_summac"): (12, 6),
    ("xsum", "single_metric_factcc"): (10, 6),
    ("xsum", "single_metric_nli_support"): (10, 10),
    ("xsum", "weighted_sum"): (14, 8),
    ("xsum", "agreement_gated"): (10, 6),
    ("cnn_dailymail", "single_metric_summac"): (10, 4),
    ("cnn_dailymail", "single_metric_factcc"): (10, 4),
    ("cnn_dailymail", "single_metric_nli_support"): (14, -2),
    ("cnn_dailymail", "weighted_sum"): (12, 4),
    ("cnn_dailymail", "agreement_gated"): (12, 4),
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


def ordered_datasets(rows: list[dict]) -> list[str]:
    present = {row["dataset"] for row in rows}
    ordered = [dataset for dataset in DATASET_ORDER if dataset in present]
    remaining = sorted(present - set(ordered))
    return ordered + remaining


def filter_rows(rows: list[dict], dataset: str) -> list[dict]:
    by_strategy = {row["strategy"]: row for row in rows if row["dataset"] == dataset}
    return [by_strategy[strategy] for strategy in FOCUS_STRATEGIES]


def plot_metric_bars(rows: list[dict], outdir: Path) -> Path:
    datasets = ordered_datasets(rows)
    metric_specs = [
        ("rougeL", "ROUGE-L"),
        ("summac", "SummaC"),
        ("factcc", "FactCC"),
        ("nli_support", "NLI Support"),
    ]
    fig, axes = plt.subplots(
        len(datasets),
        len(metric_specs),
        figsize=(16, 8),
        constrained_layout=True,
        squeeze=False,
    )

    for row_idx, dataset in enumerate(datasets):
        dataset_rows = filter_rows(rows, dataset)
        labels = [STRATEGY_TICK_LABELS[row["strategy"]] for row in dataset_rows]
        colors = [COLORS[row["strategy"]] for row in dataset_rows]
        for col_idx, (metric_key, metric_title) in enumerate(metric_specs):
            ax = axes[row_idx][col_idx]
            values = [row[metric_key] for row in dataset_rows]
            bars = ax.bar(labels, values, color=colors, width=0.65)
            ax.set_title(f"{DATASET_LABELS.get(dataset, dataset)}: {metric_title}")
            ax.set_ylim(0, max(values) * 1.18)
            ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.6)
            ax.set_axisbelow(True)
            ax.tick_params(axis="x", labelsize=10)
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
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    datasets = ordered_datasets(rows)
    x_values = []
    y_values = []

    for dataset in datasets:
        dataset_rows = [row for row in rows if row["dataset"] == dataset and row["strategy"] != "top1"]
        for row in dataset_rows:
            x = row["delta_rougeL_vs_top1"]
            y = (
                row["delta_summac_vs_top1"]
                + row["delta_factcc_vs_top1"]
                + row["delta_nli_support_vs_top1"]
            ) / 3.0
            x_values.append(x)
            y_values.append(y)
            ax.scatter(
                x,
                y,
                s=130,
                marker=SCATTER_MARKERS[dataset],
                color=COLORS[row["strategy"]],
                edgecolor="black",
                linewidth=0.7,
                alpha=0.9,
                zorder=3,
            )
            dx, dy = SCATTER_LABEL_OFFSETS[(dataset, row["strategy"])]
            ax.annotate(
                SCATTER_LABELS[row["strategy"]],
                xy=(x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=8.5,
                ha="left" if dx >= 0 else "right",
                va="bottom" if dy >= 0 else "top",
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "white",
                    "edgecolor": "#dddddd",
                    "linewidth": 0.6,
                    "alpha": 0.9,
                },
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#666666",
                    "linewidth": 0.7,
                    "shrinkA": 0,
                    "shrinkB": 6,
                },
                zorder=4,
            )

    ax.axvline(0.0, color="#444444", linestyle="--", linewidth=1)
    ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1)
    ax.set_xlabel("Delta ROUGE-L vs Top-1")
    ax.set_ylabel("Mean Delta Faithfulness (SummaC, FactCC, NLI)")
    ax.set_title("Faithfulness-ROUGE Trade-off")
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)

    x_pad = max((max(x_values) - min(x_values)) * 0.12, 0.0004)
    y_pad = max((max(y_values) - min(y_values)) * 0.12, 0.004)
    ax.set_xlim(min(x_values) - x_pad, max(x_values) + x_pad * 2.8)
    ax.set_ylim(min(0.0, min(y_values) - y_pad * 0.25), max(y_values) + y_pad)

    strategy_handles = [
        Patch(facecolor=COLORS[strategy], edgecolor="none", label=DISPLAY_NAMES[strategy])
        for strategy in ("single_metric_summac", "single_metric_factcc", "single_metric_nli_support", "weighted_sum", "agreement_gated")
    ]
    dataset_handles = [
        Line2D(
            [0],
            [0],
            marker=SCATTER_MARKERS[dataset],
            color="black",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=8,
            linewidth=0,
            label=DATASET_LABELS.get(dataset, dataset),
        )
        for dataset in datasets
    ]
    strategy_legend = ax.legend(
        handles=strategy_handles,
        title="Strategy",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
    )
    dataset_legend = ax.legend(
        handles=dataset_handles,
        title="Dataset",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.47),
        frameon=False,
    )
    ax.add_artist(strategy_legend)

    path = outdir / "figure2_tradeoff_scatter.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_gate_behavior(rows: list[dict], outdir: Path) -> Path:
    datasets = ordered_datasets(rows)
    dataset_rows = [next(row for row in rows if row["dataset"] == dataset and row["strategy"] == "agreement_gated") for dataset in datasets]

    labels = [DATASET_LABELS.get(dataset, dataset) for dataset in datasets]
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
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 100)

    for bars in (bars1, bars2):
        for bar in bars:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1.0, f"{value:.1f}%", ha="center", va="bottom", fontsize=9)

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
