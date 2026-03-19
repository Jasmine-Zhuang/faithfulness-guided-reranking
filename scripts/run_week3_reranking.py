#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from urllib.request import urlretrieve

import nltk
import torch
from summac.model_summac import SummaCConv, SummaCZS
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from fgr.io import read_jsonl, resolve_candidate_jsonl, write_jsonl
from fgr.metrics import compute_rouge

DEFAULT_CONV_CHECKPOINT_URL = "https://github.com/tingofurro/summac/raw/master/summac_conv_vitc_sent_perc_e.bin"

STRATEGY_NAMES = [
    "top1",
    "single_metric_summac",
    "single_metric_factcc",
    "weighted_sum",
    "agreement_gated",
]
METRIC_NAMES = ["summac", "factcc"]


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


def resolve_device(device: str | None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def chunked(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


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


def ensure_nltk_punkt() -> None:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab/english")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


def ensure_conv_checkpoint(args: argparse.Namespace) -> str:
    cache_dir = Path(args.summac_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = cache_dir / "summac_conv_vitc_sent_perc_e.bin"
    if not checkpoint_path.exists():
        urlretrieve(DEFAULT_CONV_CHECKPOINT_URL, checkpoint_path)
    return str(checkpoint_path)


def build_summac_model(args: argparse.Namespace, device: str):
    if args.summac_model_type == "conv":
        start_file = ensure_conv_checkpoint(args)
        return SummaCConv(
            models=[args.summac_model_name],
            bins="percentile",
            granularity=args.summac_granularity,
            nli_labels="e",
            device=device,
            start_file=start_file,
            agg="mean",
        )
    return SummaCZS(
        granularity=args.summac_granularity,
        model_name=args.summac_model_name,
        device=device,
    )


def patch_legacy_tokenizer_api(model) -> None:
    imagers = getattr(model, "imagers", [])
    for imager in imagers:
        tokenizer = getattr(imager, "tokenizer", None)
        if tokenizer is None or "batch_encode_plus" in getattr(tokenizer, "__dict__", {}):
            continue

        def batch_encode_plus(batch_text_or_text_pairs, **kwargs):
            truncation_strategy = kwargs.pop("truncation_strategy", None)
            if truncation_strategy == "only_first":
                kwargs["truncation"] = "only_first"
            return tokenizer(batch_text_or_text_pairs, **kwargs)

        tokenizer.batch_encode_plus = batch_encode_plus


def patch_summac_loaders(model) -> None:
    imagers = getattr(model, "imagers", [])
    for imager in imagers:
        original_load_nli = getattr(imager, "load_nli", None)
        if original_load_nli is None or getattr(imager, "_patched_load_nli", False):
            continue

        def load_nli_with_patch(*args, _original_load_nli=original_load_nli, _imager=imager, **kwargs):
            result = _original_load_nli(*args, **kwargs)
            tokenizer = getattr(_imager, "tokenizer", None)
            if tokenizer is not None and "batch_encode_plus" not in getattr(tokenizer, "__dict__", {}):
                def batch_encode_plus(batch_text_or_text_pairs, **tokenizer_kwargs):
                    truncation_strategy = tokenizer_kwargs.pop("truncation_strategy", None)
                    if truncation_strategy == "only_first":
                        tokenizer_kwargs["truncation"] = "only_first"
                    return tokenizer(batch_text_or_text_pairs, **tokenizer_kwargs)

                tokenizer.batch_encode_plus = batch_encode_plus
            return result

        imager.load_nli = load_nli_with_patch
        imager._patched_load_nli = True


def score_summac_batch(model, source: str, summaries: list[str], batch_size: int) -> list[float]:
    scores: list[float] = []
    for summary_batch in chunked(summaries, batch_size):
        result = model.score([source] * len(summary_batch), summary_batch)
        scores.extend(float(x) for x in result["scores"])
    return scores


def resolve_correct_label_id(model: AutoModelForSequenceClassification) -> int:
    label2id = {str(label).upper(): idx for label, idx in model.config.label2id.items()}
    for label in ("CORRECT", "CONSISTENT", "ENTAILMENT", "LABEL_1"):
        if label in label2id:
            return int(label2id[label])

    id2label = {int(idx): str(label).upper() for idx, label in model.config.id2label.items()}
    for idx, label in id2label.items():
        if label in {"CORRECT", "CONSISTENT", "ENTAILMENT"}:
            return idx

    raise ValueError(
        f"Could not infer positive FactCC label from config.label2id={model.config.label2id} "
        f"and config.id2label={model.config.id2label}"
    )


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
    args: argparse.Namespace,
    device: str,
    summac_model,
    factcc_model: AutoModelForSequenceClassification,
    factcc_tokenizer: AutoTokenizer,
) -> list[dict[str, float | int | str]]:
    if not candidates:
        raise ValueError("Each example must contain at least one candidate summary.")

    summac_scores = score_summac_batch(summac_model, source, candidates, args.summac_batch_size)
    factcc_scores, factcc_labels = score_factcc_batch(
        factcc_model,
        factcc_tokenizer,
        source,
        candidates,
        args.factcc_batch_size,
        args.factcc_max_length,
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
            args.weight_summac * float(row["summac_z"])
            + args.weight_factcc * float(row["factcc_z"])
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


def summarize_strategy(rows: list[dict], strategy_name: str) -> dict[str, object]:
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


def summarize_selection_behavior(rows: list[dict], strategy_name: str) -> dict[str, float]:
    selected = [row["strategies"][strategy_name] for row in rows]
    changed = [int(item["candidate_idx"]) != 0 for item in selected]
    return {
        "changed_from_top1_rate": sum(changed) / len(selected),
        "mean_selected_candidate_idx": mean(int(item["candidate_idx"]) for item in selected),
    }


def build_strategy_metrics(rows: list[dict]) -> dict[str, dict[str, object]]:
    metrics = {
        strategy_name: summarize_strategy(rows, strategy_name)
        for strategy_name in STRATEGY_NAMES
    }
    baseline = metrics["top1"]
    for strategy_name in STRATEGY_NAMES:
        metrics[strategy_name]["selection"] = summarize_selection_behavior(rows, strategy_name)
        metrics[strategy_name]["delta_vs_top1"] = build_relative_improvements(metrics[strategy_name], baseline)
    return metrics


def main() -> None:
    args = parse_args()
    input_path = resolve_candidate_jsonl(
        input_path=args.input,
        dataset=args.dataset,
        outdir=args.outdir,
        split=args.split,
        beam_size=args.beam_size,
    )
    rows = list(read_jsonl(input_path))
    if not rows:
        raise ValueError(f"No rows found in {input_path}")
    if args.num_examples is not None:
        if args.num_examples <= 0:
            raise ValueError("--num-examples must be a positive integer.")
        rows = rows[: args.num_examples]

    device = resolve_device(args.device)
    ensure_nltk_punkt()

    summac_model = build_summac_model(args, device)
    patch_summac_loaders(summac_model)
    patch_legacy_tokenizer_api(summac_model)

    factcc_tokenizer = AutoTokenizer.from_pretrained(args.factcc_model_name)
    factcc_model = AutoModelForSequenceClassification.from_pretrained(args.factcc_model_name).to(device)

    reranked_rows = []
    for row in tqdm(rows, desc="Week 3 reranking"):
        candidates = row["candidates"]
        candidate_rows = compute_candidate_scores(
            row["source"],
            candidates,
            args=args,
            device=device,
            summac_model=summac_model,
            factcc_model=factcc_model,
            factcc_tokenizer=factcc_tokenizer,
        )
        strategies = select_strategies(
            candidate_rows,
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
                "num_candidates": len(candidates),
                "candidate_scores": candidate_rows,
                "strategies": strategies,
            }
        )

    strategy_metrics = build_strategy_metrics(reranked_rows)

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
                "resolved_input": str(input_path),
                "device": device,
                "fallback_strategy": args.fallback_strategy,
                "weight_summac": args.weight_summac,
                "weight_factcc": args.weight_factcc,
                "num_examples": len(rows),
                "requested_num_examples": args.num_examples,
                "strategy_names": STRATEGY_NAMES,
                "metric_names": METRIC_NAMES,
                "summac_config": {
                    "model_type": args.summac_model_type,
                    "model_name": args.summac_model_name,
                    "granularity": args.summac_granularity,
                    "batch_size": args.summac_batch_size,
                    "cache_dir": args.summac_cache_dir,
                },
                "factcc_config": {
                    "model_name": args.factcc_model_name,
                    "batch_size": args.factcc_batch_size,
                    "max_length": args.factcc_max_length,
                },
                "notes": "Week 3 reranking currently uses SummaC and FactCC only.",
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
