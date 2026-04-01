# Faithfulness-Guided Reranking

This repository studies faithfulness-guided reranking for abstractive summarization. It generates `k` candidate summaries with BART, scores them with automatic factual consistency metrics, and compares reranking strategies against the original top-1 beam output.

The codebase was developed across course project milestones:
- Week 1: candidate generation
- Week 2: baseline evaluation and standalone faithfulness metrics
- Week 3: reranking
- Week 4-5: aggregate analysis and figures

## Pipeline Overview

The end-to-end workflow is:

1. Generate `k` candidate summaries for `cnn_dailymail` or `xsum`.
2. Evaluate the original top-1 summary with ROUGE and lightweight faithfulness signals.
3. Score every candidate with SummaC, FactCC, and NLI support.
4. Compare reranking strategies such as single-metric selection, weighted sum, and agreement gating.
5. Aggregate quantitative and qualitative analysis artifacts.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

Optional:
- `scripts/plot_week4_figures.py` also requires `matplotlib`.

## Quickstart

### 1. Generate Candidate Summaries

CNN/DailyMail:

```bash
python3 scripts/run_week1_generation.py \
  --dataset cnn_dailymail \
  --split validation \
  --num-examples 300 \
  --beam-size 5 \
  --batch-size 4
```

XSum:

```bash
python3 scripts/run_week1_generation.py \
  --dataset xsum \
  --split validation \
  --num-examples 300 \
  --beam-size 5 \
  --batch-size 4
```

Output:
- `outputs/<dataset>/<split>_k<beam-size>_candidates.jsonl`

Each row contains:
- `example_id`
- `dataset`
- `split`
- `source`
- `reference`
- `top1`
- `candidates`

### 2. Evaluate the Top-1 Baseline

Using an explicit candidate file:

```bash
python3 scripts/run_week2_baseline_eval.py \
  --input outputs/cnn_dailymail/validation_k5_candidates.jsonl
```

Using dataset-based resolution:

```bash
python3 scripts/run_week2_baseline_eval.py \
  --dataset cnn_dailymail
```

This computes:
- ROUGE: `rouge1`, `rouge2`, `rougeL`, `rougeLsum`
- `nli_support`: sentence-level support via `facebook/bart-large-mnli`
- `keyword_precision`: lexical support proxy

Outputs:
- `outputs/<dataset>/baseline_<split>_k<beam-size>/summary_metrics.json`
- `outputs/<dataset>/baseline_<split>_k<beam-size>/per_example_faithfulness.jsonl`

### 3. Run Reranking

Using an explicit candidate file:

```bash
python3 scripts/run_week3_reranking.py \
  --input outputs/xsum/validation_k5_candidates.jsonl
```

Using dataset-based resolution:

```bash
python3 scripts/run_week3_reranking.py \
  --dataset xsum
```

Small smoke test:

```bash
python3 scripts/run_week3_reranking.py \
  --dataset xsum \
  --num-examples 20
```

Compared strategies:
- `top1`
- `single_metric_summac`
- `single_metric_factcc`
- `single_metric_nli_support`
- `weighted_sum`
- `agreement_gated`

`weighted_sum` uses equal-weight z-score normalization over `summac`, `factcc`, and `nli_support` by default.

`agreement_gated` selects the candidate chosen by at least two of the three faithfulness metrics. If no pair agrees, it falls back to `weighted_sum` by default.

Useful flags:
- `--fallback-strategy top1`
- `--weight-summac`, `--weight-factcc`, `--weight-nli-support`
- `--nli-model-name` and related `--nli-*` flags
- `--device cpu|cuda`

Outputs:
- `outputs/<dataset>/week3_<split>_k<beam-size>/reranked_examples.jsonl`
- `outputs/<dataset>/week3_<split>_k<beam-size>/strategy_metrics.json`
- `outputs/<dataset>/week3_<split>_k<beam-size>/run_config.json`

Kaggle notebook used for the Week 3 reranking run:
- https://www.kaggle.com/code/runxinzhuang/657d-project-week3?scriptVersionId=305947620

### 4. Run Standalone Faithfulness Metrics

SummaC:

```bash
python3 scripts/run_week2_summac_eval.py \
  --input outputs/cnn_dailymail/validation_k5_candidates.jsonl
```

FactCC:

```bash
python3 scripts/run_week2_factcc_eval.py \
  --input outputs/cnn_dailymail/validation_k5_candidates.jsonl
```

QAGS-style QA consistency:

```bash
python3 scripts/run_week2_qags_eval.py \
  --input outputs/cnn_dailymail/validation_k5_candidates.jsonl
```

Outputs:
- `outputs/<dataset>/summac_<split>_k<beam-size>/summary_metrics.json`
- `outputs/<dataset>/summac_<split>_k<beam-size>/per_example_summac.jsonl`
- `outputs/<dataset>/factcc_<split>_k<beam-size>/summary_metrics.json`
- `outputs/<dataset>/factcc_<split>_k<beam-size>/per_example_factcc.jsonl`
- `outputs/<dataset>/qags_<split>_k<beam-size>/summary_metrics.json`
- `outputs/<dataset>/qags_<split>_k<beam-size>/per_example_qags.jsonl`

Note:
- The QAGS script is a local approximation built from Hugging Face question generation and QA models.

### 5. Aggregate Week 4-5 Analysis

```bash
python3 scripts/run_week4_analysis.py
```

By default, the script reads:
- reranking outputs from `outputs/<dataset>/week3_<split>_k<beam-size>/`

The generated summary uses the proposal-aligned metric set from Week 3 reranking outputs:
- `ROUGE-L`
- `SummaC`
- `FactCC`
- `NLI support`

It writes:
- `outputs/week4_analysis/strategy_comparison.csv`
- `outputs/week4_analysis/summary.md`
- `outputs/week4_analysis/qualitative_examples.md`

### 6. Plot Comparison Figures

```bash
python3 scripts/plot_week4_figures.py
```

Outputs:
- `outputs/week4_analysis/figures/figure1_metric_bars.png`
- `outputs/week4_analysis/figures/figure2_tradeoff_scatter.png`
- `outputs/week4_analysis/figures/figure3_gate_behavior.png`

Figure details:
- `figure1_metric_bars.png` compares `ROUGE-L`, `SummaC`, `FactCC`, and `NLI support` for the main comparison strategies.
- `figure2_tradeoff_scatter.png` plots ROUGE-L change against the mean faithfulness change across `SummaC`, `FactCC`, and `NLI support`.

## Input Resolution Rules

The Week 2 and Week 3 scripts accept either:
- `--input <path>`
- `--dataset <name>` with optional `--split` and `--beam-size`

When resolving from `--dataset`, the scripts look for candidate files in this order:
- `outputs/<dataset>/<split>_k<beam-size>_candidates.jsonl`
- `outputs/<dataset>_<split>_k<beam-size>_candidates.jsonl`
- `outputs/<split>_k<beam-size>_candidates.jsonl`

## Repository Layout

```text
src/fgr/
  baseline.py             Baseline ROUGE + NLI/keyword evaluation
  data.py                 Dataset specs and loading
  factcc.py               FactCC evaluation pipeline
  generation.py           BART generation utilities
  generation_pipeline.py  Candidate generation pipeline
  io.py                   JSONL helpers and path resolution
  metrics.py              ROUGE and faithfulness metrics
  qags.py                 Local QAGS-style evaluation pipeline
  reranking.py            Week 3 reranking pipeline
  summac.py               SummaC evaluation pipeline

scripts/
  run_week1_generation.py
  run_week2_baseline_eval.py
  run_week2_factcc_eval.py
  run_week2_qags_eval.py
  run_week2_summac_eval.py
  run_week3_reranking.py
  run_week4_analysis.py
  plot_week4_figures.py

notebooks/
  kaggle_week3_reranking.ipynb

report/
  final_report.tex
```
