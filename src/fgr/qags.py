from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM, AutoTokenizer

from .io import read_jsonl, resolve_candidate_jsonl, write_jsonl
from .metrics import compute_rouge, split_sentences

_ARTICLE_RE = re.compile(r"\b(a|an|the)\b")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class QAGSConfig:
    input_path: str | None = None
    dataset: str | None = None
    outdir: str = "outputs"
    split: str = "validation"
    beam_size: int = 5
    qg_model: str = "iarfmoose/t5-base-question-generator"
    qa_model: str = "deepset/roberta-base-squad2"
    num_questions: int = 3
    max_answer_sentences: int = 5
    qg_max_source_length: int = 384
    qg_max_target_length: int = 64
    qa_batch_size: int = 8
    device: str | None = None


def resolve_device(device: str | None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = _ARTICLE_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    ref_counts: dict[str, int] = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1

    overlap = 0
    for token in pred_tokens:
        count = ref_counts.get(token, 0)
        if count > 0:
            overlap += 1
            ref_counts[token] = count - 1

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def select_answer_candidates(summary: str, max_sentences: int, num_questions: int) -> list[str]:
    candidates = []
    for sentence in split_sentences(summary)[:max_sentences]:
        cleaned = sentence.strip()
        if len(cleaned.split()) < 4:
            continue
        candidates.append(cleaned)
        if len(candidates) >= num_questions:
            break
    return candidates


def generate_questions(
    *,
    qg_model: AutoModelForSeq2SeqLM,
    qg_tokenizer: AutoTokenizer,
    summary: str,
    answer_candidates: list[str],
    max_source_length: int,
    max_target_length: int,
    device: str,
) -> list[str]:
    if not answer_candidates:
        return []

    prompts = [f"answer: {answer} context: {summary}" for answer in answer_candidates]
    encoded = qg_tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_source_length,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = qg_model.generate(
            **encoded,
            max_length=max_target_length,
            num_beams=4,
            early_stopping=True,
        )

    questions = qg_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    cleaned_questions = []
    for question in questions:
        question = question.strip()
        if question and not question.endswith("?"):
            question += "?"
        cleaned_questions.append(question)
    return cleaned_questions


def chunked(seq: list[Any], size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def answer_questions(
    *,
    qa_model: AutoModelForQuestionAnswering,
    qa_tokenizer: AutoTokenizer,
    questions: list[str],
    contexts: list[str],
    batch_size: int,
    device: str,
) -> list[dict[str, float | str]]:
    results: list[dict[str, float | str]] = []
    qa_model.eval()
    for question_batch, context_batch in zip(chunked(questions, batch_size), chunked(contexts, batch_size), strict=True):
        encoded_batch = qa_tokenizer(
            question_batch,
            context_batch,
            max_length=384,
            truncation="only_second",
            padding=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offset_mapping = encoded_batch.pop("offset_mapping")
        sequence_ids_batch = []
        for batch_idx in range(len(question_batch)):
            sequence_ids_batch.append([s if s is not None else -1 for s in encoded_batch.sequence_ids(batch_idx)])
        encoded = {k: v.to(device) for k, v in encoded_batch.items()}

        with torch.no_grad():
            outputs = qa_model(**encoded)

        start_logits = outputs.start_logits.detach().cpu()
        end_logits = outputs.end_logits.detach().cpu()
        attention_mask = encoded["attention_mask"].detach().cpu()
        for batch_idx, context in enumerate(context_batch):
            offsets = offset_mapping[batch_idx].tolist()
            seq_ids = sequence_ids_batch[batch_idx]
            valid_positions = [
                idx for idx, (seq_id, mask) in enumerate(zip(seq_ids, attention_mask[batch_idx].tolist(), strict=True))
                if seq_id == 1 and mask == 1
            ]

            best_score = None
            best_span = (0, 0)
            for start_idx in valid_positions:
                max_end = min(start_idx + 30, len(offsets) - 1)
                for end_idx in range(start_idx, max_end + 1):
                    if seq_ids[end_idx] != 1:
                        continue
                    score = float(start_logits[batch_idx, start_idx] + end_logits[batch_idx, end_idx])
                    if best_score is None or score > best_score:
                        best_score = score
                        best_span = (start_idx, end_idx)

            start_char = offsets[best_span[0]][0]
            end_char = offsets[best_span[1]][1]
            answer = context[start_char:end_char].strip() if end_char > start_char else ""
            results.append({"answer": answer, "score": float(best_score or 0.0)})

    return results


def compute_qags_score(
    *,
    qg_model: AutoModelForSeq2SeqLM,
    qg_tokenizer: AutoTokenizer,
    qa_model: AutoModelForQuestionAnswering,
    qa_tokenizer: AutoTokenizer,
    source: str,
    summary: str,
    cfg: QAGSConfig,
    device: str,
) -> tuple[float, list[dict[str, Any]]]:
    answer_candidates = select_answer_candidates(
        summary=summary,
        max_sentences=cfg.max_answer_sentences,
        num_questions=cfg.num_questions,
    )
    questions = generate_questions(
        qg_model=qg_model,
        qg_tokenizer=qg_tokenizer,
        summary=summary,
        answer_candidates=answer_candidates,
        max_source_length=cfg.qg_max_source_length,
        max_target_length=cfg.qg_max_target_length,
        device=device,
    )
    questions = questions[: cfg.num_questions]
    if not questions:
        return 0.0, []

    source_answers = answer_questions(
        qa_model=qa_model,
        qa_tokenizer=qa_tokenizer,
        questions=questions,
        contexts=[source] * len(questions),
        batch_size=cfg.qa_batch_size,
        device=device,
    )
    summary_answers = answer_questions(
        qa_model=qa_model,
        qa_tokenizer=qa_tokenizer,
        questions=questions,
        contexts=[summary] * len(questions),
        batch_size=cfg.qa_batch_size,
        device=device,
    )

    details = []
    f1_scores = []
    for question, source_answer, summary_answer in zip(questions, source_answers, summary_answers, strict=True):
        source_text = str(source_answer.get("answer", "")).strip()
        summary_text = str(summary_answer.get("answer", "")).strip()
        score = token_f1(source_text, summary_text)
        f1_scores.append(score)
        details.append(
            {
                "question": question,
                "source_answer": source_text,
                "summary_answer": summary_text,
                "source_answer_score": float(source_answer.get("score", 0.0)),
                "summary_answer_score": float(summary_answer.get("score", 0.0)),
                "answer_f1": score,
            }
        )

    return sum(f1_scores) / len(f1_scores), details


def run_qags_eval(cfg: QAGSConfig) -> dict[str, Any]:
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
    rouge_scores = compute_rouge(predictions, references)

    device = resolve_device(cfg.device)
    qg_tokenizer = AutoTokenizer.from_pretrained(cfg.qg_model)
    qg_model = AutoModelForSeq2SeqLM.from_pretrained(cfg.qg_model).to(device)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(cfg.qa_model).to(device)
    qa_tokenizer = AutoTokenizer.from_pretrained(cfg.qa_model)

    per_example = []
    for row in tqdm(rows, desc="Scoring QAGS"):
        qags_score, qa_pairs = compute_qags_score(
            qg_model=qg_model,
            qg_tokenizer=qg_tokenizer,
            qa_model=qa_model,
            qa_tokenizer=qa_tokenizer,
            source=row["source"],
            summary=row["top1"],
            cfg=cfg,
            device=device,
        )
        per_example.append(
            {
                "example_id": row["example_id"],
                "qags": qags_score,
                "qa_pairs": qa_pairs,
            }
        )

    summary = {
        "num_examples": len(rows),
        "rouge": rouge_scores,
        "faithfulness": {
            "qags": sum(row["qags"] for row in per_example) / len(per_example),
        },
        "qags_config": {
            "qg_model": cfg.qg_model,
            "qa_model": cfg.qa_model,
            "num_questions": cfg.num_questions,
            "max_answer_sentences": cfg.max_answer_sentences,
            "qg_max_source_length": cfg.qg_max_source_length,
            "qg_max_target_length": cfg.qg_max_target_length,
            "qa_batch_size": cfg.qa_batch_size,
            "device": device,
        },
        "notes": "This is a QAGS-style implementation using Hugging Face question generation and QA models, not the original QAGS codebase.",
    }

    dataset_name = rows[0].get("dataset", input_path.parent.name)
    stem = input_path.stem.replace("_candidates", "")
    run_dir = Path(cfg.outdir) / dataset_name / f"qags_{stem}"
    run_dir.mkdir(parents=True, exist_ok=True)

    per_example_out = run_dir / "per_example_qags.jsonl"
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
