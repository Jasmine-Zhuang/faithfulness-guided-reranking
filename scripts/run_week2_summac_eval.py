#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlretrieve

import nltk
import torch
from tqdm.auto import tqdm

from fgr.io import read_jsonl, resolve_candidate_jsonl, write_jsonl
from fgr.metrics import compute_rouge
from summac.model_summac import SummaCConv, SummaCZS

DEFAULT_CONV_CHECKPOINT_URL = "https://github.com/tingofurro/summac/raw/master/summac_conv_vitc_sent_perc_e.bin"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline top-1 summaries with SummaC without modifying the Week 2 pipeline."
    )
    parser.add_argument("--input", type=str, default=None, help="Path to *_candidates.jsonl")
    parser.add_argument("--dataset", choices=["cnn_dailymail", "xsum"], default=None)
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--split", default="validation", help="Used with --dataset when --input is omitted.")
    parser.add_argument("--beam-size", type=int, default=5, help="Used with --dataset when --input is omitted.")
    parser.add_argument("--model-type", choices=["conv", "zs"], default="conv")
    parser.add_argument("--model-name", type=str, default="vitc")
    parser.add_argument("--granularity", choices=["document", "paragraph", "sentence", "2sents"], default="sentence")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cpu or cuda")
    parser.add_argument("--cache-dir", type=str, default=".cache/summac")
    return parser.parse_args()


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


def ensure_conv_checkpoint(args: argparse.Namespace) -> str:
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = cache_dir / "summac_conv_vitc_sent_perc_e.bin"
    if not checkpoint_path.exists():
        urlretrieve(DEFAULT_CONV_CHECKPOINT_URL, checkpoint_path)
    return str(checkpoint_path)


def build_summac_model(args: argparse.Namespace):
    device = resolve_device(args.device)
    if args.model_type == "conv":
        start_file = ensure_conv_checkpoint(args)
        return SummaCConv(
            models=[args.model_name],
            bins="percentile",
            granularity=args.granularity,
            nli_labels="e",
            device=device,
            start_file=start_file,
            agg="mean",
        )
    return SummaCZS(
        granularity=args.granularity,
        model_name=args.model_name,
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

    predictions = [r["top1"] for r in rows]
    references = [r["reference"] for r in rows]
    sources = [r["source"] for r in rows]

    rouge_scores = compute_rouge(predictions, references)
    ensure_nltk_punkt()
    model = build_summac_model(args)
    patch_summac_loaders(model)
    patch_legacy_tokenizer_api(model)

    summac_scores: list[float] = []
    for source_batch, summary_batch in tqdm(
        zip(chunked(sources, args.batch_size), chunked(predictions, args.batch_size), strict=True),
        total=(len(rows) + args.batch_size - 1) // args.batch_size,
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
            "model_type": args.model_type,
            "model_name": args.model_name,
            "granularity": args.granularity,
            "batch_size": args.batch_size,
            "device": resolve_device(args.device),
        },
    }

    dataset_name = rows[0].get("dataset", input_path.parent.name)
    stem = input_path.stem.replace("_candidates", "")
    run_dir = Path(args.outdir) / dataset_name / f"summac_{stem}"
    run_dir.mkdir(parents=True, exist_ok=True)

    per_example_out = run_dir / "per_example_summac.jsonl"
    summary_out = run_dir / "summary_metrics.json"
    write_jsonl(per_example_out, per_example)
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved summary: {summary_out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
