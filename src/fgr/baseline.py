from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from .io import read_jsonl, resolve_candidate_jsonl, write_jsonl
from .metrics import NLIConfig, NLIFaithfulnessScorer, compute_rouge, keyword_precision


@dataclass(frozen=True)
class BaselineEvalConfig:
    input_path: str | None = None
    dataset: str | None = None
    outdir: str = "outputs"
    split: str = "validation"
    beam_size: int = 5
    nli_model: str = "facebook/bart-large-mnli"
    nli_batch_size: int = 32
    nli_max_source_sentences: int = 20
    nli_max_summary_sentences: int = 5


def run_baseline_eval(cfg: BaselineEvalConfig) -> dict[str, Any]:
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

    nli = NLIFaithfulnessScorer(
        cfg=NLIConfig(
            model_name=cfg.nli_model,
            batch_size=cfg.nli_batch_size,
            max_source_sentences=cfg.nli_max_source_sentences,
            max_summary_sentences=cfg.nli_max_summary_sentences,
        )
    )

    per_example = []
    for row in tqdm(rows, desc="Scoring faithfulness"):
        nli_score = nli.score(row["source"], row["top1"])
        kw_score = keyword_precision(row["source"], row["top1"])
        per_example.append(
            {
                "example_id": row["example_id"],
                "nli_support": nli_score,
                "keyword_precision": kw_score,
            }
        )

    mean_nli = sum(x["nli_support"] for x in per_example) / len(per_example)
    mean_kw = sum(x["keyword_precision"] for x in per_example) / len(per_example)

    summary = {
        "num_examples": len(rows),
        "rouge": rouge_scores,
        "faithfulness": {
            "nli_support": mean_nli,
            "keyword_precision": mean_kw,
        },
    }

    dataset_name = rows[0].get("dataset", input_path.parent.name)
    stem = input_path.stem.replace("_candidates", "")
    run_dir = Path(cfg.outdir) / dataset_name / f"baseline_{stem}"
    run_dir.mkdir(parents=True, exist_ok=True)

    per_example_out = run_dir / "per_example_faithfulness.jsonl"
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
