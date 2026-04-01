# Week 4 Analysis Summary

This file summarizes the quantitative comparison between baseline decoding and Week 3 reranking strategies using the proposal metrics: ROUGE-L, SummaC, FactCC, and NLI support.

## xsum

- Baseline top-1 scores: ROUGE-L=0.3843, SummaC=0.2247, FactCC=0.2219, NLI=0.0863
- Best overall trade-off in this run: `single_metric_factcc`
- `agreement_gated` changed the top-1 choice on 70.0% of examples and the gate passed on 59.7% of examples.
- `agreement_gated` vs top-1: delta ROUGE-L -0.0042, delta SummaC +0.0009, delta FactCC +0.0653, delta NLI +0.0242.
- `weighted_sum` vs top-1: delta ROUGE-L -0.0025, delta SummaC +0.0010, delta FactCC +0.0692, delta NLI +0.0298.

## cnn_dailymail

- Baseline top-1 scores: ROUGE-L=0.2807, SummaC=0.2633, FactCC=0.4937, NLI=0.7579
- Best overall trade-off in this run: `weighted_sum`
- `agreement_gated` changed the top-1 choice on 81.7% of examples and the gate passed on 58.7% of examples.
- `agreement_gated` vs top-1: delta ROUGE-L -0.0029, delta SummaC +0.0219, delta FactCC +0.0883, delta NLI +0.0520.
- `weighted_sum` vs top-1: delta ROUGE-L -0.0029, delta SummaC +0.0229, delta FactCC +0.1005, delta NLI +0.0565.

## Takeaways

- `xsum` shows the clearest benefit from reranking: FactCC and NLI improve meaningfully while ROUGE-L drops by less than 0.5 absolute points, though SummaC moves only slightly.
- `cnn_dailymail` also improves on all three faithfulness metrics, but the gain is smaller relative to its already strong top-1 baseline.
- The agreement gate is conservative in spirit, but in this run it still changes many selections because metric agreement happens on more than half of examples.
- For the final report, `weighted_sum` remains the strongest default comparator and `agreement_gated` is the main proposed method.
