from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

import evaluate
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in", "is", "it", "its", "of", "on", "that",
    "the", "to", "was", "were", "will", "with", "this", "these", "those", "or", "if", "than", "then", "their", "there", "here", "she",
    "they", "we", "you", "your", "i", "my", "our", "about", "after", "before", "during", "into", "over", "under", "up", "down", "out",
}


def split_sentences(text: str) -> list[str]:
    chunks = [s.strip() for s in _SENT_SPLIT_RE.split(text.strip()) if s.strip()]
    return chunks if chunks else [text.strip()] if text.strip() else []


def tokenize_keywords(text: str) -> set[str]:
    words = {w.lower() for w in _WORD_RE.findall(text)}
    return {w for w in words if len(w) >= 3 and w not in _STOPWORDS}


def keyword_precision(source: str, summary: str) -> float:
    summary_keywords = tokenize_keywords(summary)
    if not summary_keywords:
        return 1.0
    source_keywords = tokenize_keywords(source)
    supported = summary_keywords & source_keywords
    return len(supported) / len(summary_keywords)


@dataclass(frozen=True)
class NLIConfig:
    model_name: str = "facebook/bart-large-mnli"
    batch_size: int = 32
    max_length: int = 256
    max_source_sentences: int = 20
    max_summary_sentences: int = 5


class NLIFaithfulnessScorer:
    def __init__(self, cfg: NLIConfig | None = None):
        self.cfg = cfg or NLIConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.cfg.model_name).to(self.device)
        self.model.eval()
        self.entailment_idx = self._resolve_entailment_idx()

    def _resolve_entailment_idx(self) -> int:
        label2id = self.model.config.label2id
        for label, idx in label2id.items():
            if label.lower() == "entailment":
                return int(idx)
        raise ValueError(f"Could not find entailment label in {label2id}")

    def _batched_entailment_probs(self, premises: Sequence[str], hypotheses: Sequence[str]) -> np.ndarray:
        probs = []
        bs = self.cfg.batch_size
        for i in range(0, len(premises), bs):
            batch_prem = list(premises[i : i + bs])
            batch_hyp = list(hypotheses[i : i + bs])
            encoded = self.tokenizer(
                batch_prem,
                batch_hyp,
                return_tensors="pt",
                truncation=True,
                max_length=self.cfg.max_length,
                padding=True,
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                logits = self.model(**encoded).logits
                batch_probs = torch.softmax(logits, dim=-1)[:, self.entailment_idx]
            probs.append(batch_probs.detach().cpu().numpy())
        return np.concatenate(probs, axis=0) if probs else np.array([])

    def score(self, source: str, summary: str) -> float:
        source_sents = split_sentences(source)[: self.cfg.max_source_sentences]
        summary_sents = split_sentences(summary)[: self.cfg.max_summary_sentences]

        if not source_sents or not summary_sents:
            return 0.0

        premises = []
        hypotheses = []
        for hyp in summary_sents:
            for prem in source_sents:
                premises.append(prem)
                hypotheses.append(hyp)

        ent_probs = self._batched_entailment_probs(premises, hypotheses)
        if ent_probs.size == 0:
            return 0.0

        matrix = ent_probs.reshape(len(summary_sents), len(source_sents))
        per_summary_support = matrix.max(axis=1)
        return float(np.mean(per_summary_support))


def compute_rouge(predictions: Sequence[str], references: Sequence[str]) -> dict[str, float]:
    rouge = evaluate.load("rouge")
    scores = rouge.compute(predictions=list(predictions), references=list(references), use_stemmer=True)
    return {k: float(v) for k, v in scores.items()}
