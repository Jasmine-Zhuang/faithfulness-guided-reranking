from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .data import DatasetSpec


@dataclass(frozen=True)
class GenerationConfig:
    model_name: str
    beam_size: int = 5
    max_input_tokens: int = 1024
    max_new_tokens: int = 128
    min_new_tokens: int = 20
    length_penalty: float = 1.0
    batch_size: int = 4


class BartCandidateGenerator:
    def __init__(self, cfg: GenerationConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name).to(self.device)
        self.model.eval()

    def generate_candidates(self, sources: Sequence[str]) -> list[list[str]]:
        candidates: list[list[str]] = []
        batch_size = self.cfg.batch_size

        for start in tqdm(range(0, len(sources), batch_size), desc="Generating summaries"):
            batch_sources = sources[start : start + batch_size]
            encoded = self.tokenizer(
                list(batch_sources),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.cfg.max_input_tokens,
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                output_ids = self.model.generate(
                    **encoded,
                    num_beams=self.cfg.beam_size,
                    num_return_sequences=self.cfg.beam_size,
                    max_new_tokens=self.cfg.max_new_tokens,
                    min_new_tokens=self.cfg.min_new_tokens,
                    length_penalty=self.cfg.length_penalty,
                    early_stopping=True,
                )

            decoded = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            for i in range(len(batch_sources)):
                start_idx = i * self.cfg.beam_size
                end_idx = (i + 1) * self.cfg.beam_size
                candidates.append([cand.strip() for cand in decoded[start_idx:end_idx]])

        return candidates


def resolve_model_name(spec: DatasetSpec, model_name: str | None) -> str:
    return model_name or spec.default_model
