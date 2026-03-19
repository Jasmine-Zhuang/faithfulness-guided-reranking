from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .factcc import chunked, resolve_correct_label_id, resolve_device
from .io import read_jsonl, resolve_candidate_jsonl, write_jsonl
from .metrics import compute_rouge
from .summac import (
    SummaCConfig,
    build_summac_model,
    ensure_nltk_punkt,
    patch_legacy_tokenizer_api,
    patch_summac_loaders,
)

STRATEGY_NAMES = [
    "top1",
    "single_metric_summac",
    "single_metric_factcc",
    "weighted_sum",
    "agreement_gated",
]
METRIC_NAMES = ["summac", "factcc"]


@dataclass(frozen=True)
class Week3RerankingConfig:
    input_path: str | None = None
    dataset: str | None = None
    outdir: str = "outputs"
    split: str = "validation"
    beam_size: int = 5
    num_examples: int | None = None
    fallback_strategy: str = "weighted_sum"
    weight_summac: float = 1.0
    weight_factcc: float = 1.0
    summac_model_type: str = "conv"
    summac_model_name: str = "vitc"
    summac_granularity: str = "sentence"
    summac_batch_size: int = 8
    summac_cache_dir: str = ".cache/summac"
    factcc_model_name: str = "manueldeprada/FactCC"
    factcc_batch_size: int = 8
    factcc_max_length: int = 512
    device: str | None = None


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


def score_summac_batch(model, source: str, summaries: list[str], batch_size: int) -> list[float]:
    scores: list[float] = []
    for summary_batch in chunked(summaries, batch_size):
        result = model.score([source] * len(summary_batch), summary_batch)
        scores.extend(float(x) for x in result["scores"])
    return scores


def score_factcc_batch(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    source: str,
    summaries: list[str],
    batch_size: int,
    max_length: int,
    device: str,
) -> tuple[list[float], list[str]]:
    correct_label_id = resolve_correct_label_id(model)
    scores: list[float] = []
    labels: list[str] = []

    model.eval()
    for summary_batch in chunked(summaries, batch_size):
        encoded = tokenizer(
            [source] * len(summary_batch),
            summary_batch,
            max_length=max_length,
            padding=True,
            truncation="only_first",
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)
            pred_ids = logits.argmax(dim=-1)

        scores.extend(float(score) for score in probs[:, correct_label_id].detach().cpu().tolist())
        labels.extend(str(model.config.id2label[int(idx)]).upper() for idx in pred_ids.detach().cpu().tolist())

    return scores, labels


def compute_candidate_scores(
    source: str,
    candidates: list[str],
    *,
    cfg: Week3RerankingConfig,
    device: str,
    summac_model,
    factcc_model: AutoModelForSequenceClassification,
    factcc_tokenizer: AutoTokenizer,
) -> list[dict[str, float | int | str]]:
    if not candidates:
        raise ValueError("Each example must contain at least one candidate summary.")

    summac_scores = score_summac_batch(summac_model, source, candidates, cfg.summac_batch_size)
    factcc_scores, factcc_labels = score_factcc_batch(
        factcc_model,
        factcc_tokenizer,
        source,
        candidates,
        cfg.factcc_batch_size,
        cfg.factcc_max_length,
        device,
    )

    rows = []
    for idx, candidate in enumerate(candidates):
        rows.append(
            {
                "candidate_idx": idx,
                "summary": candidate,
                "summac": float(summac_scores[idx]),
                "factcc": float(factcc_scores[idx]),
                "factcc_label": factcc_labels[idx],
            }
        )

    for metric_name in METRIC_NAMES:
        metric_z = zscore([float(row[metric_name]) for row in rows])
        for row, metric_value in zip(rows, metric_z, strict=True):
            row[f"{metric_name}_z"] = metric_value

    for row in rows:
        row["weighted_sum_score"] = (
            cfg.weight_summac * float(row["summac_z"])
            + cfg.weight_factcc * float(row["factcc_z"])
        )

    return rows


def select_strategies(
    candidate_rows: list[dict[str, float | int | str]],
    *,
    fallback_strategy: str,
) -> dict[str, dict[str, int | float | str]]:
    top1_idx = 0
    summac_idx = argmax([float(row["summac"]) for row in candidate_rows])
    factcc_idx = argmax([float(row["factcc"]) for row in candidate_rows])
    weighted_idx = argmax([float(row["weighted_sum_score"]) for row in candidate_rows])

    votes = Counter([summac_idx, factcc_idx])
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

    def build_selection(name: str, idx: int, extra: dict[str, int | float | str] | None = None):
        selected = dict(candidate_rows[idx])
        selected["strategy"] = name
        if extra:
            selected.update(extra)
        return selected

    return {
        "top1": build_selection("top1", top1_idx),
        "single_metric_summac": build_selection("single_metric_summac", summac_idx),
        "single_metric_factcc": build_selection("single_metric_factcc", factcc_idx),
        "weighted_sum": build_selection("weighted_sum", weighted_idx),
        "agreement_gated": build_selection(
            "agreement_gated",
            agreement_idx,
            {
                "gate_passed": int(use_gate),
                "gate_reason": agreement_reason,
                "agreed_candidate_idx": agreed_idx,
                "agreement_votes": agreed_votes,
                "metric_votes": {
                    "summac": summac_idx,
                    "factcc": factcc_idx,
                },
            },
        ),
    }


def build_relative_improvements(summary: dict[str, object], baseline: dict[str, object]) -> dict[str, object]:
    summary_rouge = dict(summary["rouge"])
    baseline_rouge = dict(baseline["rouge"])
    summary_faithfulness = dict(summary["faithfulness"])
    baseline_faithfulness = dict(baseline["faithfulness"])
    return {
        "rouge": {
            metric: float(summary_rouge[metric]) - float(baseline_rouge[metric])
            for metric in summary_rouge
        },
        "faithfulness": {
            metric: float(summary_faithfulness[metric]) - float(baseline_faithfulness[metric])
            for metric in summary_faithfulness
        },
    }


def summarize_strategy(rows: list[dict[str, Any]], strategy_name: str) -> dict[str, object]:
    selected = [row["strategies"][strategy_name] for row in rows]
    predictions = [str(row["summary"]) for row in selected]
    references = [row["reference"] for row in rows]
    rouge_scores = compute_rouge(predictions, references)

    summary: dict[str, object] = {
        "num_examples": len(rows),
        "rouge": rouge_scores,
        "faithfulness": {
            metric_name: mean(float(row[metric_name]) for row in selected)
            for metric_name in METRIC_NAMES
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


def summarize_selection_behavior(rows: list[dict[str, Any]], strategy_name: str) -> dict[str, float]:
    selected = [row["strategies"][strategy_name] for row in rows]
    changed = [int(item["candidate_idx"]) != 0 for item in selected]
    return {
        "changed_from_top1_rate": sum(changed) / len(selected),
        "mean_selected_candidate_idx": mean(int(item["candidate_idx"]) for item in selected),
    }


def build_strategy_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, object]]:
    metrics = {
        strategy_name: summarize_strategy(rows, strategy_name)
        for strategy_name in STRATEGY_NAMES
    }
    baseline = metrics["top1"]
    for strategy_name in STRATEGY_NAMES:
        metrics[strategy_name]["selection"] = summarize_selection_behavior(rows, strategy_name)
        metrics[strategy_name]["delta_vs_top1"] = build_relative_improvements(metrics[strategy_name], baseline)
    return metrics


def run_week3_reranking(cfg: Week3RerankingConfig) -> dict[str, Any]:
    input_path = resolve_candidate_jsonl(
        input_path=cfg.input_path,
        dataset=cfg.dataset,
        outdir=cfg.outdir,
        split=cfg.split,
        beam_size=cfg.beam_size,
    )
    rows = list(read_jsonl(input_path))
    if not rows:
        raise ValueError(f"No rows found in {input_path}")
    if cfg.num_examples is not None:
        if cfg.num_examples <= 0:
            raise ValueError("--num-examples must be a positive integer.")
        rows = rows[: cfg.num_examples]

    device = resolve_device(cfg.device)
    ensure_nltk_punkt()

    summac_model = build_summac_model(
        SummaCConfig(
            model_type=cfg.summac_model_type,
            model_name=cfg.summac_model_name,
            granularity=cfg.summac_granularity,
            batch_size=cfg.summac_batch_size,
            device=device,
            cache_dir=cfg.summac_cache_dir,
        ),
        device,
    )
    patch_summac_loaders(summac_model)
    patch_legacy_tokenizer_api(summac_model)

    factcc_tokenizer = AutoTokenizer.from_pretrained(cfg.factcc_model_name)
    factcc_model = AutoModelForSequenceClassification.from_pretrained(cfg.factcc_model_name).to(device)

    reranked_rows = []
    for row in tqdm(rows, desc="Week 3 reranking"):
        candidates = row["candidates"]
        candidate_rows = compute_candidate_scores(
            row["source"],
            candidates,
            cfg=cfg,
            device=device,
            summac_model=summac_model,
            factcc_model=factcc_model,
            factcc_tokenizer=factcc_tokenizer,
        )
        strategies = select_strategies(
            candidate_rows,
            fallback_strategy=cfg.fallback_strategy,
        )
        reranked_rows.append(
            {
                "example_id": row["example_id"],
                "dataset": row.get("dataset"),
                "split": row.get("split"),
                "source": row["source"],
                "reference": row["reference"],
                "original_top1": row.get("top1", candidates[0]),
                "num_candidates": len(candidates),
                "candidate_scores": candidate_rows,
                "strategies": strategies,
            }
        )

    strategy_metrics = build_strategy_metrics(reranked_rows)

    dataset_name = rows[0].get("dataset", input_path.parent.name)
    stem = input_path.stem.replace("_candidates", "")
    run_dir = Path(cfg.outdir) / dataset_name / f"week3_{stem}"
    run_dir.mkdir(parents=True, exist_ok=True)

    reranked_out = run_dir / "reranked_examples.jsonl"
    metrics_out = run_dir / "strategy_metrics.json"
    config_out = run_dir / "run_config.json"

    write_jsonl(reranked_out, reranked_rows)
    metrics_out.write_text(json.dumps(strategy_metrics, indent=2), encoding="utf-8")
    config_out.write_text(
        json.dumps(
            {
                "input": cfg.input_path,
                "resolved_input": str(input_path),
                "device": device,
                "fallback_strategy": cfg.fallback_strategy,
                "weight_summac": cfg.weight_summac,
                "weight_factcc": cfg.weight_factcc,
                "num_examples": len(rows),
                "requested_num_examples": cfg.num_examples,
                "strategy_names": STRATEGY_NAMES,
                "metric_names": METRIC_NAMES,
                "summac_config": {
                    "model_type": cfg.summac_model_type,
                    "model_name": cfg.summac_model_name,
                    "granularity": cfg.summac_granularity,
                    "batch_size": cfg.summac_batch_size,
                    "cache_dir": cfg.summac_cache_dir,
                },
                "factcc_config": {
                    "model_name": cfg.factcc_model_name,
                    "batch_size": cfg.factcc_batch_size,
                    "max_length": cfg.factcc_max_length,
                },
                "notes": "Week 3 reranking currently uses SummaC and FactCC only.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "input_path": input_path,
        "run_dir": run_dir,
        "reranked_out": reranked_out,
        "metrics_out": metrics_out,
        "config_out": config_out,
        "strategy_metrics": strategy_metrics,
    }
