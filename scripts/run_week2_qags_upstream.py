#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from fgr.qags_upstream import (
    QAGSUpstreamFormatQAConfig,
    QAGSUpstreamPrepareConfig,
    QAGSUpstreamScoreConfig,
    run_qags_upstream_format_qa,
    run_qags_upstream_prepare,
    run_qags_upstream_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use the official W4ngatang/qags code for staged QAGS preparation and scoring."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--input", type=str, default=None, help="Path to *_candidates.jsonl")
    common.add_argument("--dataset", choices=["cnn_dailymail", "xsum"], default=None)
    common.add_argument("--outdir", type=str, default="outputs")
    common.add_argument("--split", default="validation", help="Used with --dataset when --input is omitted.")
    common.add_argument("--beam-size", type=int, default=5, help="Used with --dataset when --input is omitted.")
    common.add_argument("--qags-repo", type=str, required=True, help="Path to a local checkout of https://github.com/W4ngatang/qags")

    prepare = subparsers.add_parser(
        "prepare",
        parents=[common],
        help="Write upstream-compatible text files and run upstream answer extraction.",
    )
    prepare.add_argument("--n-ans", type=int, default=10, help="Number of answer candidates per summary for upstream QG prep.")

    format_qa = subparsers.add_parser(
        "format-qa",
        parents=[common],
        help="Use upstream qa_utils.py to build SQuAD-format QA inputs from generated questions.",
    )
    format_qa.add_argument("--workdir-name", type=str, default=None, help="Override staged work directory name.")
    format_qa.add_argument("--gen-qst-file", type=str, required=True, help="Question file from upstream question generation.")
    format_qa.add_argument("--gen-prob-file", type=str, required=True, help="Question probability file from upstream question generation.")
    format_qa.add_argument("--gen-ans-file", type=str, default=None, help="Expected answers file for answer-consistency filtering.")
    format_qa.add_argument("--gen-prd-file", type=str, default=None, help="QA predictions file for answer-consistency filtering.")
    format_qa.add_argument("--src-w-trg-txt-file", type=str, default=None, help="Optional upstream XSum-style concatenated source+target file.")
    format_qa.add_argument("--n-ans-per-doc", type=int, default=10)
    format_qa.add_argument("--n-gen-qsts", type=int, default=10)
    format_qa.add_argument("--n-qsts-per-doc", type=int, default=5)
    format_qa.add_argument("--use-all-qsts", action="store_true")
    format_qa.add_argument("--use-act-anss", action="store_true")
    format_qa.add_argument("--use-exp-anss", action="store_true")

    score = subparsers.add_parser(
        "score",
        parents=[common],
        help="Use upstream qa_utils.py scoring logic on QA prediction files and save repo-style metrics.",
    )
    score.add_argument("--workdir-name", type=str, default=None, help="Override staged work directory name.")
    score.add_argument("--source-ans-file", type=str, required=True, help="QA predictions using article/source context.")
    score.add_argument("--target-ans-file", type=str, required=True, help="QA predictions using summary context.")
    score.add_argument("--ans-similarity-fn", choices=["em", "f1"], default="f1")
    score.add_argument("--n-qsts-per-doc", type=int, default=5)

    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        result = run_qags_upstream_prepare(
            QAGSUpstreamPrepareConfig(
                input_path=args.input,
                dataset=args.dataset,
                outdir=args.outdir,
                split=args.split,
                beam_size=args.beam_size,
                qags_repo=args.qags_repo,
                n_ans=args.n_ans,
            )
        )
        print(f"Prepared upstream QAGS inputs in {result['run_dir']}")
        print(json.dumps(result["manifest"], indent=2))
    elif args.command == "format-qa":
        result = run_qags_upstream_format_qa(
            QAGSUpstreamFormatQAConfig(
                input_path=args.input,
                dataset=args.dataset,
                outdir=args.outdir,
                split=args.split,
                beam_size=args.beam_size,
                qags_repo=args.qags_repo,
                workdir_name=args.workdir_name,
                gen_qst_file=args.gen_qst_file,
                gen_prob_file=args.gen_prob_file,
                gen_ans_file=args.gen_ans_file,
                gen_prd_file=args.gen_prd_file,
                src_w_trg_txt_file=args.src_w_trg_txt_file,
                n_ans_per_doc=args.n_ans_per_doc,
                n_gen_qsts=args.n_gen_qsts,
                n_qsts_per_doc=args.n_qsts_per_doc,
                use_all_qsts=args.use_all_qsts,
                use_act_anss=args.use_act_anss,
                use_exp_anss=args.use_exp_anss,
            )
        )
        print(f"Formatted upstream QA data in {result['qa_dir']}")
        print(json.dumps(result["manifest"], indent=2))
    elif args.command == "score":
        result = run_qags_upstream_score(
            QAGSUpstreamScoreConfig(
                input_path=args.input,
                dataset=args.dataset,
                outdir=args.outdir,
                split=args.split,
                beam_size=args.beam_size,
                qags_repo=args.qags_repo,
                workdir_name=args.workdir_name,
                source_ans_file=args.source_ans_file,
                target_ans_file=args.target_ans_file,
                ans_similarity_fn=args.ans_similarity_fn,
                n_qsts_per_doc=args.n_qsts_per_doc,
            )
        )
        print(f"Saved summary: {result['summary_out']}")
        print(json.dumps(result["summary"], indent=2))
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
