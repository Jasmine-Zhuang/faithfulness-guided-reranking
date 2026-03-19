from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import nltk
import torch
from summac.model_summac import SummaCConv, SummaCZS
from tqdm.auto import tqdm

from .io import read_jsonl, resolve_candidate_jsonl, write_jsonl
from .metrics import compute_rouge

DEFAULT_CONV_CHECKPOINT_URL = "https://github.com/tingofurro/summac/raw/master/summac_conv_vitc_sent_perc_e.bin"


@dataclass(frozen=True)
class SummaCConfig:
    input_path: str | None = None
    dataset: str | None = None
    outdir: str = "outputs"
    split: str = "validation"
    beam_size: int = 5
    model_type: str = "conv"
    model_name: str = "vitc"
    granularity: str = "sentence"
    batch_size: int = 8
    device: str | None = None
    cache_dir: str = ".cache/summac"


def resolve_device(device: str | None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def ensure_nltk_punkt() -> None:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab/english")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


def ensure_conv_checkpoint(cache_dir: str) -> str:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = cache_path / "summac_conv_vitc_sent_perc_e.bin"
    if not checkpoint_path.exists():
        urlretrieve(DEFAULT_CONV_CHECKPOINT_URL, checkpoint_path)
    return str(checkpoint_path)


def build_summac_model(cfg: SummaCConfig, device: str):
    if cfg.model_type == "conv":
        start_file = ensure_conv_checkpoint(cfg.cache_dir)
        return SummaCConv(
            models=[cfg.model_name],
            bins="percentile",
            granularity=cfg.granularity,
            nli_labels="e",
            device=device,
            start_file=start_file,
            agg="mean",
        )
    return SummaCZS(
        granularity=cfg.granularity,
        model_name=cfg.model_name,
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


def chunked(seq: list[str], size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def run_summac_eval(cfg: SummaCConfig) -> dict[str, Any]:
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
    ensure_nltk_punkt()

    device = resolve_device(cfg.device)
    model = build_summac_model(cfg, device)
    patch_summac_loaders(model)
    patch_legacy_tokenizer_api(model)

    summac_scores: list[float] = []
    for source_batch, summary_batch in tqdm(
        zip(chunked(sources, cfg.batch_size), chunked(predictions, cfg.batch_size), strict=True),
        total=(len(rows) + cfg.batch_size - 1) // cfg.batch_size,
        desc="Scoring SummaC",
    ):
        result = model.score(source_batch, summary_batch)
        summac_scores.extend(float(x) for x in result["scores"])

    per_example = []
    for row, score in zip(rows, summac_scores, strict=True):
        per_example.append(
            {
                "example_id": row["example_id"],
                "summac": score,
            }
        )

    summary = {
        "num_examples": len(rows),
        "rouge": rouge_scores,
        "faithfulness": {
            "summac": sum(summac_scores) / len(summac_scores),
        },
        "summac_config": {
            "model_type": cfg.model_type,
            "model_name": cfg.model_name,
            "granularity": cfg.granularity,
            "batch_size": cfg.batch_size,
            "device": device,
        },
    }

    dataset_name = rows[0].get("dataset", input_path.parent.name)
    stem = input_path.stem.replace("_candidates", "")
    run_dir = Path(cfg.outdir) / dataset_name / f"summac_{stem}"
    run_dir.mkdir(parents=True, exist_ok=True)

    per_example_out = run_dir / "per_example_summac.jsonl"
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
