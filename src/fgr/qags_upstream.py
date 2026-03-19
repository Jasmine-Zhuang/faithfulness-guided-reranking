from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import read_jsonl, resolve_candidate_jsonl, write_jsonl
from .metrics import compute_rouge


@dataclass(frozen=True)
class QAGSUpstreamCommonConfig:
    input_path: str | None = None
    dataset: str | None = None
    outdir: str = "outputs"
    split: str = "validation"
    beam_size: int = 5
    qags_repo: str = ""


@dataclass(frozen=True)
class QAGSUpstreamPrepareConfig(QAGSUpstreamCommonConfig):
    n_ans: int = 10


@dataclass(frozen=True)
class QAGSUpstreamFormatQAConfig(QAGSUpstreamCommonConfig):
    workdir_name: str | None = None
    gen_qst_file: str = ""
    gen_prob_file: str = ""
    gen_ans_file: str | None = None
    gen_prd_file: str | None = None
    src_w_trg_txt_file: str | None = None
    n_ans_per_doc: int = 10
    n_gen_qsts: int = 10
    n_qsts_per_doc: int = 5
    use_all_qsts: bool = False
    use_act_anss: bool = False
    use_exp_anss: bool = False


@dataclass(frozen=True)
class QAGSUpstreamScoreConfig(QAGSUpstreamCommonConfig):
    workdir_name: str | None = None
    source_ans_file: str = ""
    target_ans_file: str = ""
    ans_similarity_fn: str = "f1"
    n_qsts_per_doc: int = 5


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_upstream_modules(qags_repo: Path, *, need_qg: bool, need_qa: bool):
    sys.path.insert(0, str(qags_repo))
    try:
        load_module("utils", qags_repo / "utils.py")
        qg_utils = load_module("qags_qg_utils", qags_repo / "qg_utils.py") if need_qg else None
        qa_utils = load_module("qags_qa_utils", qags_repo / "qa_utils.py") if need_qa else None
    finally:
        sys.path.pop(0)
    return qg_utils, qa_utils


def resolve_input_path(cfg: QAGSUpstreamCommonConfig) -> Path:
    return resolve_candidate_jsonl(
        input_path=cfg.input_path,
        dataset=cfg.dataset,
        outdir=cfg.outdir,
        split=cfg.split,
        beam_size=cfg.beam_size,
    )


def get_rows(cfg: QAGSUpstreamCommonConfig) -> tuple[Path, list[dict[str, Any]]]:
    input_path = resolve_input_path(cfg)
    rows = list(read_jsonl(input_path))
    if not rows:
        raise ValueError(f"No rows found in {input_path}")
    return input_path, rows


def get_run_dir(cfg: QAGSUpstreamCommonConfig, input_path: Path, rows: list[dict[str, Any]], workdir_name: str | None = None) -> Path:
    dataset_name = rows[0].get("dataset", input_path.parent.name)
    stem = input_path.stem.replace("_candidates", "")
    run_name = workdir_name or f"qags_upstream_{stem}"
    return Path(cfg.outdir) / dataset_name / run_name


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def run_qags_upstream_prepare(cfg: QAGSUpstreamPrepareConfig) -> dict[str, Any]:
    input_path, rows = get_rows(cfg)
    run_dir = get_run_dir(cfg, input_path, rows)
    qags_repo = Path(cfg.qags_repo)
    qg_utils, _ = load_upstream_modules(qags_repo, need_qg=True, need_qa=False)

    source_lines = [str(row["source"]).replace("\n", " ").strip() for row in rows]
    summary_lines = [str(row["top1"]).replace("\n", " ").strip() for row in rows]
    example_ids = [str(row["example_id"]) for row in rows]

    inputs_dir = run_dir / "inputs"
    qg_dir = run_dir / "qg"
    fairseq_dir = run_dir / "fairseq_input"

    source_txt = inputs_dir / "source.txt"
    summary_txt = inputs_dir / "summary.txt"
    example_id_txt = inputs_dir / "example_ids.txt"
    write_lines(source_txt, source_lines)
    write_lines(summary_txt, summary_lines)
    write_lines(example_id_txt, example_ids)

    qg_utils.prepare_ans_conditional_data(
        data_file=str(summary_txt),
        out_dir=str(qg_dir),
        out_prefix="summary",
        n_ans_per_txt=cfg.n_ans,
    )

    summary_w_ans = qg_dir / f"summary_w_{cfg.n_ans}ans.txt"
    test_src = fairseq_dir / "test.src"
    test_trg = fairseq_dir / "test.trg"
    test_src.parent.mkdir(parents=True, exist_ok=True)
    test_src.write_text(summary_w_ans.read_text(encoding="utf-8"), encoding="utf-8")
    test_trg.write_text(summary_w_ans.read_text(encoding="utf-8"), encoding="utf-8")

    manifest = {
        "input_candidates": str(input_path),
        "qags_repo": str(qags_repo.resolve()),
        "num_examples": len(rows),
        "n_ans": cfg.n_ans,
        "files": {
            "source_txt": str(source_txt),
            "summary_txt": str(summary_txt),
            "example_ids_txt": str(example_id_txt),
            "summary_answers_txt": str(qg_dir / f"summary_{cfg.n_ans}ans.txt"),
            "summary_conditioned_txt": str(summary_w_ans),
            "fairseq_test_src": str(test_src),
            "fairseq_test_trg": str(test_trg),
        },
        "next_steps": [
            "Run upstream fairseq preprocessing on fairseq_input/test.src and fairseq_input/test.trg.",
            "Run upstream question generation to produce gens.txt and probs.txt.",
            "Use this script's format-qa command to build src.json and gen.json.",
            "Run the upstream QA model twice: once on src.json and once on gen.json.",
            "Use this script's score command to compute final QAGS metrics.",
        ],
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "run_dir": run_dir,
        "manifest_path": manifest_path,
        "manifest": manifest,
    }


def run_qags_upstream_format_qa(cfg: QAGSUpstreamFormatQAConfig) -> dict[str, Any]:
    input_path, rows = get_rows(cfg)
    run_dir = get_run_dir(cfg, input_path, rows, cfg.workdir_name)
    qags_repo = Path(cfg.qags_repo)
    _, qa_utils = load_upstream_modules(qags_repo, need_qg=False, need_qa=True)

    inputs_dir = run_dir / "inputs"
    qa_dir = run_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)

    source_txt = inputs_dir / "source.txt"
    summary_txt = inputs_dir / "summary.txt"
    if not source_txt.exists() or not summary_txt.exists():
        raise FileNotFoundError(f"Missing staged inputs in {inputs_dir}. Run the prepare command first.")

    qa_utils.aggregate_questions_from_txt(
        out_dir=str(qa_dir),
        src_txt_file=str(source_txt),
        gen_txt_file=str(summary_txt),
        gen_qst_file=cfg.gen_qst_file,
        gen_prob_file=cfg.gen_prob_file,
        gen_ans_file=cfg.gen_ans_file,
        gen_prd_file=cfg.gen_prd_file,
        src_w_trg_txt_file=cfg.src_w_trg_txt_file,
        use_all_qsts=cfg.use_all_qsts,
        use_act_anss=cfg.use_act_anss,
        use_exp_anss=cfg.use_exp_anss,
        n_gen_qsts=cfg.n_gen_qsts,
        n_ans=cfg.n_ans_per_doc,
        n_qsts=cfg.n_qsts_per_doc,
    )

    manifest = {
        "qags_repo": str(qags_repo.resolve()),
        "qa_dir": str(qa_dir),
        "source_json": str(qa_dir / "src.json"),
        "summary_json": str(qa_dir / "gen.json"),
        "notes": [
            "Run the upstream QA model on src.json using article/source context.",
            "Run the upstream QA model on gen.json using summary context.",
            "Then use the score subcommand with the two prediction JSON files.",
        ],
    }
    manifest_path = qa_dir / "qa_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "qa_dir": qa_dir,
        "manifest_path": manifest_path,
        "manifest": manifest,
    }


def run_qags_upstream_score(cfg: QAGSUpstreamScoreConfig) -> dict[str, Any]:
    input_path, rows = get_rows(cfg)
    run_dir = get_run_dir(cfg, input_path, rows, cfg.workdir_name)
    qags_repo = Path(cfg.qags_repo)
    _, qa_utils = load_upstream_modules(qags_repo, need_qg=False, need_qa=True)

    qags_scores = qa_utils.get_qags_scores(
        src_ans_file=cfg.source_ans_file,
        trg_ans_file=cfg.target_ans_file,
        metric_name=cfg.ans_similarity_fn,
        n_qsts_per_doc=cfg.n_qsts_per_doc,
    )
    if len(qags_scores) != len(rows):
        raise ValueError(f"Expected {len(rows)} QAGS scores, got {len(qags_scores)}")

    rouge_scores = compute_rouge([row["top1"] for row in rows], [row["reference"] for row in rows])
    per_example = [
        {
            "example_id": row["example_id"],
            "qags_upstream": float(score),
        }
        for row, score in zip(rows, qags_scores, strict=True)
    ]
    summary = {
        "num_examples": len(rows),
        "rouge": rouge_scores,
        "faithfulness": {
            "qags_upstream": float(sum(qags_scores) / len(qags_scores)),
        },
        "qags_upstream_config": {
            "qags_repo": str(qags_repo.resolve()),
            "ans_similarity_fn": cfg.ans_similarity_fn,
            "n_qsts_per_doc": cfg.n_qsts_per_doc,
            "source_ans_file": cfg.source_ans_file,
            "target_ans_file": cfg.target_ans_file,
        },
        "notes": "Scores are computed using upstream qa_utils.get_qags_scores from W4ngatang/qags.",
    }

    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    per_example_out = results_dir / "per_example_qags_upstream.jsonl"
    summary_out = results_dir / "summary_metrics.json"
    write_jsonl(per_example_out, per_example)
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "results_dir": results_dir,
        "per_example_out": per_example_out,
        "summary_out": summary_out,
        "summary": summary,
    }
