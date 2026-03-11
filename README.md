# faithfulness-guided-reranking
Faithfulness-guided reranking for abstractive summarization using transformer models and automatic factual consistency metrics.

## Week 1-2 Bootstrap

This repository now includes the initial experiment pipeline for:
- Week 1 (Mar 3-7): setup + dataset loading + BART top-1 and n-best generation (`k=5`)
- Week 2 (Mar 10-14): baseline scoring with ROUGE and two faithfulness metrics

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

This computes:
- ROUGE (`rouge1`, `rouge2`, `rougeL`, `rougeLsum`)
- `nli_support`: sentence-level support score using `facebook/bart-large-mnli`
- `keyword_precision`: lexical support proxy (summary content words supported by source)

Output:
- `outputs/<dataset>/baseline_<split>_k5/summary_metrics.json`
- `outputs/<dataset>/baseline_<split>_k5/per_example_faithfulness.jsonl`

## Project Structure

```text
src/fgr/data.py          # dataset specs + loading
src/fgr/generation.py    # BART generation for top-1 + n-best
src/fgr/metrics.py       # ROUGE + faithfulness metrics
src/fgr/io.py            # JSONL utilities
scripts/run_week1_generation.py
scripts/run_week2_baseline_eval.py
```
