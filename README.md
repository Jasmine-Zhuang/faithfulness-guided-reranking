# faithfulness-guided-reranking
Faithfulness-guided reranking for abstractive summarization using transformer models and automatic factual consistency metrics.

## Week 1-2 Bootstrap

This repository now includes the initial experiment pipeline for:
- Week 1 (Mar 3-7): setup + dataset loading + BART top-1 and n-best generation (`k=5`)
- Week 2 (Mar 10-14): baseline scoring with ROUGE and faithfulness metrics
- Week 3 (Mar 17-21): reranking with single-metric and agreement-gated selection

### 1) Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Generate top-1 and n-best summaries

CNN/DailyMail example:

```bash
PYTHONPATH=src python3 scripts/run_week1_generation.py \
  --dataset cnn_dailymail \
  --split validation \
  --num-examples 300 \
  --beam-size 5 \
  --batch-size 4
```

XSum example:

```bash
PYTHONPATH=src python3 scripts/run_week1_generation.py \
  --dataset xsum \
  --split validation \
  --num-examples 300 \
  --beam-size 5 \
  --batch-size 4
```

Output file format:
- `outputs/<dataset>/<split>_k5_candidates.jsonl`
- each row contains `source`, `reference`, `top1`, and full `candidates` list

### 3) Evaluate Week 2 baseline (top-1)

```bash
PYTHONPATH=src python3 scripts/run_week2_baseline_eval.py \
  --input outputs/cnn_dailymail/validation_k5_candidates.jsonl
```

Or let the script resolve the candidate file from the dataset name:

```bash
PYTHONPATH=src python3 scripts/run_week2_baseline_eval.py \
  --dataset cnn_dailymail
```

This computes:
- ROUGE (`rouge1`, `rouge2`, `rougeL`, `rougeLsum`)
- `nli_support`: sentence-level support score using `facebook/bart-large-mnli`
- `keyword_precision`: lexical support proxy (summary content words supported by source)

Output:
- `outputs/<dataset>/baseline_<split>_k5/summary_metrics.json`
- `outputs/<dataset>/baseline_<split>_k5/per_example_faithfulness.jsonl`

### 4) Run Week 3 reranking on the k-best list

CNN/DailyMail example:

```bash
PYTHONPATH=src python3 scripts/run_week3_reranking.py \
  --input outputs/cnn_dailymail/validation_k5_candidates.jsonl
```

XSum example:

```bash
PYTHONPATH=src python3 scripts/run_week3_reranking.py \
  --input outputs/xsum/validation_k5_candidates.jsonl
```

Dataset-driven resolution is also supported:

```bash
PYTHONPATH=src python3 scripts/run_week3_reranking.py \
  --dataset xsum
```

Small-scale smoke test example:

```bash
PYTHONPATH=src python3 scripts/run_week3_reranking.py \
  --dataset xsum \
  --num-examples 20
```

This computes candidate-level faithfulness scores for each n-best list and compares:
- `top1`
- `single_metric_summac`
- `single_metric_factcc`
- `weighted_sum` (equal-weight z-score normalization over `summac` and `factcc` by default)
- `agreement_gated` (selects a candidate only when at least two faithfulness metrics pick the same best candidate; otherwise falls back to `weighted_sum`)

Optional flags:
- `--fallback-strategy top1` to fall back to the original top-1 instead of `weighted_sum`
- `--weight-summac` and `--weight-factcc` to change weighted-sum reranking weights
- `--num-examples 20` to run a small subset first
- `--device cpu|cuda` to force inference onto a specific device

Output:
- `outputs/<dataset>/week3_<split>_k5/reranked_examples.jsonl`
- `outputs/<dataset>/week3_<split>_k5/strategy_metrics.json`
- `outputs/<dataset>/week3_<split>_k5/run_config.json`

### 5) Evaluate top-1 with SummaC, FactCC, QAGS-style QA consistency, or upstream QAGS

SummaC example:

```bash
PYTHONPATH=src python3 scripts/run_week2_summac_eval.py \
  --input outputs/cnn_dailymail/validation_k5_candidates.jsonl
```

FactCC example:

```bash
PYTHONPATH=src python3 scripts/run_week2_factcc_eval.py \
  --input outputs/cnn_dailymail/validation_k5_candidates.jsonl
```

QAGS-style example:

```bash
PYTHONPATH=src python3 scripts/run_week2_qags_eval.py \
  --input outputs/cnn_dailymail/validation_k5_candidates.jsonl
```

This script is a local QAGS-style approximation. It uses Hugging Face question generation and QA models and writes to:
- `outputs/<dataset>/qags_<split>_k5/summary_metrics.json`
- `outputs/<dataset>/qags_<split>_k5/per_example_qags.jsonl`

Upstream QAGS prep example:

```bash
PYTHONPATH=src python3 scripts/run_week2_qags_upstream.py prepare \
  --dataset cnn_dailymail \
  --qags-repo /path/to/qags
```

The upstream wrapper is separate from the local QAGS-style approximation. It stages inputs for the original `W4ngatang/qags` workflow and writes under:
- `outputs/<dataset>/qags_upstream_<split>_k5/...`

All Week 2/3 scripts accept either:
- `--input <path>`
- or `--dataset <name>` plus optional `--split` / `--beam-size`

When resolving from `--dataset`, scripts look for candidates in this order:
- `outputs/<dataset>/<split>_k<beam>_candidates.jsonl`
- `outputs/<dataset>_<split>_k<beam>_candidates.jsonl`
- `outputs/<split>_k<beam>_candidates.jsonl`

Outputs:
- `outputs/<dataset>/summac_<split>_k5/summary_metrics.json`
- `outputs/<dataset>/summac_<split>_k5/per_example_summac.jsonl`
- `outputs/<dataset>/factcc_<split>_k5/summary_metrics.json`
- `outputs/<dataset>/factcc_<split>_k5/per_example_factcc.jsonl`
- `outputs/<dataset>/qags_<split>_k5/summary_metrics.json`
- `outputs/<dataset>/qags_<split>_k5/per_example_qags.jsonl`
- `outputs/<dataset>/qags_upstream_<split>_k5/...`

See [docs/qags_upstream.md](/Users/jasminezhuang/faithfulness-guided-reranking/docs/qags_upstream.md) for the staged upstream workflow and dependencies.

### 6) Run upstream QAGS on Kaggle

If you want to run the upstream QAGS pipeline on Kaggle instead of configuring the environment locally, use:

- [notebooks/qags_upstream_kaggle.ipynb](/Users/jasminezhuang/faithfulness-guided-reranking/notebooks/qags_upstream_kaggle.ipynb)

What the notebook expects:
- a Kaggle GPU notebook
- this repo cloned from GitHub inside the notebook
- a Kaggle dataset containing `validation_k5_candidates.jsonl` (or another `*_candidates.jsonl` file)
- a Kaggle dataset containing the downloaded upstream QAGS checkpoint folder with:
  - `qa/`
  - `qg/`
  - `dict.txt`

The notebook auto-discovers:
- `validation_k5_candidates.jsonl` under `/kaggle/input`
- the QAGS checkpoint root directory by looking for a folder that contains `qa/`, `qg/`, and `dict.txt`

It is configured for the upstream checkpoint file names:
- `qg/qg_best.pt`
- `qg/best_pretrained_bert.pt`

Recommended Kaggle runtime:
- GPU enabled
- internet enabled if you want the notebook to clone GitHub repositories directly

The notebook writes final upstream QAGS results to:
- `outputs/<dataset>/qags_upstream_<split>_k5/results/summary_metrics.json`

## Project Structure

```text
src/fgr/baseline.py      # Baseline ROUGE + NLI/keyword evaluation pipeline
src/fgr/data.py          # dataset specs + loading
src/fgr/factcc.py        # FactCC evaluation pipeline
src/fgr/generation.py    # BART generation for top-1 + n-best
src/fgr/metrics.py       # ROUGE + faithfulness metrics
src/fgr/io.py            # JSONL utilities
src/fgr/generation_pipeline.py # Week 1 generation pipeline
src/fgr/qags.py          # Local QAGS-style evaluation pipeline
src/fgr/qags_upstream.py # Upstream QAGS staging/scoring wrapper
src/fgr/summac.py        # SummaC evaluation pipeline
scripts/run_week1_generation.py
scripts/run_week2_baseline_eval.py
scripts/run_week2_factcc_eval.py
scripts/run_week2_qags_eval.py
scripts/run_week2_qags_upstream.py
scripts/run_week2_summac_eval.py
scripts/run_week3_reranking.py
notebooks/qags_upstream_kaggle.ipynb
```
