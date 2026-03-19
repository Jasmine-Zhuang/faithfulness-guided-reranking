from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .io import read_jsonl, resolve_candidate_jsonl, write_jsonl
from .metrics import compute_rouge


@dataclass(frozen=True)
class FactCCConfig:
    input_path: str | None = None
    dataset: str | None = None
    outdir: str = "outputs"
    split: str = "validation"
    beam_size: int = 5
    model_name: str = "manueldeprada/FactCC"
    batch_size: int = 8
    max_length: int = 512
    device: str | None = None


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


def run_factcc_eval(cfg: FactCCConfig) -> dict[str, Any]:
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

    predictions = [r["top1"] for r in rows]
    references = [r["reference"] for r in rows]
    sources = [r["source"] for r in rows]

    rouge_scores = compute_rouge(predictions, references)

    device = resolve_device(cfg.device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name).to(device)
    factcc_scores, factcc_labels = score_factcc(
        model=model,
        tokenizer=tokenizer,
        sources=sources,
        summaries=predictions,
        batch_size=cfg.batch_size,
        max_length=cfg.max_length,
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
            "model_name": cfg.model_name,
            "batch_size": cfg.batch_size,
            "max_length": cfg.max_length,
            "device": device,
        },
    }

    dataset_name = rows[0].get("dataset", input_path.parent.name)
    stem = input_path.stem.replace("_candidates", "")
    run_dir = Path(cfg.outdir) / dataset_name / f"factcc_{stem}"
    run_dir.mkdir(parents=True, exist_ok=True)

    per_example_out = run_dir / "per_example_factcc.jsonl"
    summary_out = run_dir / "summary_metrics.json"
    write_jsonl(per_example_out, per_example)
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "input_path": input_path,
        "run_dir": run_dir,
        "per_example_out": per_example_out,
        "summary_out": summary_out,
        "summary": summary,
    }
