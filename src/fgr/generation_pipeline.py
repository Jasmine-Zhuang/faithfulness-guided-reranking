from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data import load_split
from .generation import BartCandidateGenerator, GenerationConfig, resolve_model_name
from .io import write_jsonl


@dataclass(frozen=True)
class GenerationPipelineConfig:
    dataset: str
    split: str = "validation"
    num_examples: int = 300
    beam_size: int = 5
    batch_size: int = 4
    max_input_tokens: int = 1024
    max_new_tokens: int = 128
    min_new_tokens: int = 20
    length_penalty: float = 1.0
    model_name: str | None = None
    outdir: str = "outputs"


def run_generation_pipeline(cfg: GenerationPipelineConfig) -> dict[str, Any]:
    spec, ds = load_split(cfg.dataset, cfg.split)

    n = min(cfg.num_examples, len(ds))
    ds = ds.select(range(n))
    sources = [row[spec.text_field] for row in ds]
    references = [row[spec.summary_field] for row in ds]

    generation_cfg = GenerationConfig(
        model_name=resolve_model_name(spec, cfg.model_name),
        beam_size=cfg.beam_size,
        batch_size=cfg.batch_size,
        max_input_tokens=cfg.max_input_tokens,
        max_new_tokens=cfg.max_new_tokens,
        min_new_tokens=cfg.min_new_tokens,
        length_penalty=cfg.length_penalty,
    )

    generator = BartCandidateGenerator(generation_cfg)
    candidates = generator.generate_candidates(sources)

    rows = []
    for i, (src, ref, cand_list) in enumerate(zip(sources, references, candidates, strict=True)):
        rows.append(
            {
                "example_id": i,
                "dataset": cfg.dataset,
                "split": cfg.split,
                "source": src,
                "reference": ref,
                "top1": cand_list[0],
                "candidates": cand_list,
            }
        )

    outpath = Path(cfg.outdir) / cfg.dataset / f"{cfg.split}_k{cfg.beam_size}_candidates.jsonl"
    write_jsonl(outpath, rows)

    return {
        "num_examples": len(rows),
        "output_path": outpath,
    }
