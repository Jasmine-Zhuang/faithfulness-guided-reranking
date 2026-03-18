#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from fgr.io import read_jsonl, write_jsonl
from fgr.metrics import compute_rouge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline top-1 summaries with FactCC without modifying the Week 2 pipeline."
    )
    parser.add_argument("--input", type=str, required=True, help="Path to *_candidates.jsonl")
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--model-name", type=str, default="manueldeprada/FactCC")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cpu or cuda")
    return parser.parse_args()


def resolve_device(device: str | None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


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


def chunked(seq: list[str], size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def score_factcc(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    sources: list[str],
    summaries: list[str],
    batch_size: int,
    max_length: int,
    device: str,
) -> tuple[list[float], list[str]]:
    correct_label_id = resolve_correct_label_id(model)
    scores: list[float] = []
    labels: list[str] = []

    model.eval()
    for source_batch, summary_batch in tqdm(
        zip(chunked(sources, batch_size), chunked(summaries, batch_size), strict=True),
        total=(len(sources) + batch_size - 1) // batch_size,
        desc="Scoring FactCC",
    ):
        encoded = tokenizer(
            source_batch,
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

        batch_scores = probs[:, correct_label_id].detach().cpu().tolist()
        batch_labels = [str(model.config.id2label[int(idx)]).upper() for idx in pred_ids.detach().cpu().tolist()]
        scores.extend(float(score) for score in batch_scores)
        labels.extend(batch_labels)

    return scores, labels


def main() -> None:
    args = parse_args()
    rows = list(read_jsonl(args.input))
    if not rows:
        raise ValueError(f"No rows found in {args.input}")

    predictions = [r["top1"] for r in rows]
    references = [r["reference"] for r in rows]
    sources = [r["source"] for r in rows]

    rouge_scores = compute_rouge(predictions, references)

    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name).to(device)
    factcc_scores, factcc_labels = score_factcc(
        model=model,
        tokenizer=tokenizer,
        sources=sources,
        summaries=predictions,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )

    per_example = []
    for row, score, label in zip(rows, factcc_scores, factcc_labels, strict=True):
        per_example.append(
            {
                "example_id": row["example_id"],
                "factcc": score,
                "factcc_label": label,
            }
        )

    summary = {
        "num_examples": len(rows),
        "rouge": rouge_scores,
        "faithfulness": {
            "factcc": sum(factcc_scores) / len(factcc_scores),
        },
        "factcc_config": {
            "model_name": args.model_name,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "device": device,
        },
    }

    input_path = Path(args.input)
    dataset_name = rows[0].get("dataset", input_path.parent.name)
    stem = input_path.stem.replace("_candidates", "")
    run_dir = Path(args.outdir) / dataset_name / f"factcc_{stem}"
    run_dir.mkdir(parents=True, exist_ok=True)

    per_example_out = run_dir / "per_example_factcc.jsonl"
    summary_out = run_dir / "summary_metrics.json"
    write_jsonl(per_example_out, per_example)
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved summary: {summary_out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
