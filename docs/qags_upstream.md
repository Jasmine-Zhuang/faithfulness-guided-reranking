# Upstream QAGS Integration

This repo now includes [scripts/run_week2_qags_upstream.py](/Users/jasminezhuang/faithfulness-guided-reranking/scripts/run_week2_qags_upstream.py), which is meant to use the official `W4ngatang/qags` Python code where that is practical.

It does not replace the entire upstream stack. The original repo still expects:

- a local checkout of `https://github.com/W4ngatang/qags`
- `spacy` plus the `en_core_web_lg` model for answer extraction
- `editdistance` for QA answer comparison
- the upstream frozen `fairseq` setup for question generation
- the upstream QA checkpoint flow from `finetune_pt_squad.py`

## What This Wrapper Uses From Upstream

The wrapper directly imports and uses:

- `qg_utils.prepare_ans_conditional_data(...)`
- `qa_utils.aggregate_questions_from_txt(...)`
- `qa_utils.get_qags_scores(...)`

That matters because the upstream CLI is not especially portable as-is:

- it assumes older environments and checkpoints
- some commands are easier to use safely by importing the functions directly
- this repo needs to map scores back to `example_id` and existing output directories

## Recommended Workflow

### 1. Prepare staged inputs

```bash
PYTHONPATH=src python3 scripts/run_week2_qags_upstream.py prepare \
  --dataset cnn_dailymail \
  --qags-repo /path/to/qags
```

This writes:

- `outputs/<dataset>/qags_upstream_<split>_k5/inputs/source.txt`
- `outputs/<dataset>/qags_upstream_<split>_k5/inputs/summary.txt`
- `outputs/<dataset>/qags_upstream_<split>_k5/qg/summary_w_10ans.txt`
- `outputs/<dataset>/qags_upstream_<split>_k5/fairseq_input/test.src`
- `outputs/<dataset>/qags_upstream_<split>_k5/fairseq_input/test.trg`

The `qg/` files are produced by upstream `qg_utils.py` logic. The `fairseq_input/` files are staged for the upstream question generation flow.

### 2. Run upstream question generation

Use the official QAGS repo for this stage. In practice, point its preprocessing and generation commands at:

- `fairseq_input/test.src`
- `fairseq_input/test.trg`

You need the original QAGS question-generation checkpoint and the upstream frozen `fairseq` environment for this to behave like the paper code.

After generation, you should have question and probability files such as:

- `gens.txt`
- `probs.txt`

Depending on how you run upstream generation, you may need to use upstream `qg_utils.py --command extract_gen` first to extract those from the fairseq log.

### 3. Format QA data with upstream code

```bash
PYTHONPATH=src python3 scripts/run_week2_qags_upstream.py format-qa \
  --dataset cnn_dailymail \
  --qags-repo /path/to/qags \
  --gen-qst-file /path/to/gens.txt \
  --gen-prob-file /path/to/probs.txt \
  --use-all-qsts
```

This uses upstream `qa_utils.aggregate_questions_from_txt(...)` and writes:

- `outputs/<dataset>/qags_upstream_<split>_k5/qa/src.json`
- `outputs/<dataset>/qags_upstream_<split>_k5/qa/gen.json`

If you want answer-consistency filtering closer to the original QAGS pipeline, also provide:

- `--gen-ans-file`
- `--gen-prd-file`
- `--use-exp-anss`
- optionally `--use-act-anss`

## 4. Run upstream QA twice

Run the upstream QA model:

- once with article/source context on `qa/src.json`
- once with summary context on `qa/gen.json`

That should produce two prediction JSON files.

## 5. Compute final QAGS scores

```bash
PYTHONPATH=src python3 scripts/run_week2_qags_upstream.py score \
  --dataset cnn_dailymail \
  --qags-repo /path/to/qags \
  --source-ans-file /path/to/src_predictions.json \
  --target-ans-file /path/to/gen_predictions.json \
  --ans-similarity-fn f1
```

This uses upstream `qa_utils.get_qags_scores(...)` and writes:

- `outputs/<dataset>/qags_upstream_<split>_k5/results/per_example_qags_upstream.jsonl`
- `outputs/<dataset>/qags_upstream_<split>_k5/results/summary_metrics.json`

## Practical Advice

If your goal is:

- faster local experimentation: use [scripts/run_week2_qags_eval.py](/Users/jasminezhuang/faithfulness-guided-reranking/scripts/run_week2_qags_eval.py)
- closer reproduction of the ACL 2020 setup: use the upstream wrapper plus the original QAGS checkpoints and environment

The wrapper is most useful when you want upstream-style scoring logic without manually wiring the repo’s intermediate files into this project every time.
