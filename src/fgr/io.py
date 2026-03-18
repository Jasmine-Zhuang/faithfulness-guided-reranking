from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator


def candidate_jsonl_name(split: str, beam_size: int) -> str:
    return f"{split}_k{beam_size}_candidates.jsonl"


def candidate_jsonl_paths(*, dataset: str, outdir: str | Path = "outputs", split: str = "validation", beam_size: int = 5) -> list[Path]:
    outdir = Path(outdir)
    filename = candidate_jsonl_name(split, beam_size)
    return [
        outdir / dataset / filename,
        outdir / f"{dataset}_{filename}",
        outdir / filename,
    ]


def resolve_candidate_jsonl(
    *,
    input_path: str | Path | None,
    dataset: str | None,
    outdir: str | Path = "outputs",
    split: str = "validation",
    beam_size: int = 5,
) -> Path:
    if input_path is not None:
        return Path(input_path)

    if dataset is None:
        raise ValueError("Provide either --input or --dataset.")

    candidate_paths = candidate_jsonl_paths(
        dataset=dataset,
        outdir=outdir,
        split=split,
        beam_size=beam_size,
    )
    for path in candidate_paths:
        if path.exists():
            return path

    searched = ", ".join(str(path) for path in candidate_paths)
    raise FileNotFoundError(f"Could not find candidate JSONL for dataset '{dataset}'. Tried: {searched}")


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    ensure_parent_dir(path)
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
