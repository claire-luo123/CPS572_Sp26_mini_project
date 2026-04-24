# Experiment Report

- **Run timestamp:** 2026-04-24T08:06:22.476717Z
- **Base model:** meta-llama/Llama-3.1-8B
- **Training steps:** 100
- **Checkpoint cadence:** every 25 steps

## Baseline Scores (Base Model)

- **IFEval baseline:** 0.1682
- **GSM8K baseline:** 0.0970
- **HumanEval baseline:** 0.0000

## Quick Checkpoint Ranking

| Checkpoint | Step | Avg Score | IFEval | GSM8K | HumanEval |
| --- | ---: | ---: | ---: | ---: | ---: |
| sanity_1b_step_50 | 50 | 0.4778 | 0.2333 | 0.5000 | 0.7000 |
| sanity_1b_step_75 | 75 | 0.4667 | 0.3000 | 0.4667 | 0.6333 |
| sanity_1b_step_25 | 25 | 0.4333 | 0.3000 | 0.3333 | 0.6667 |
| sanity_1b | 100 | 0.4333 | 0.2333 | 0.4667 | 0.6000 |
| sanity_1b_step_100 | 100 | 0.4111 | 0.2333 | 0.4000 | 0.6000 |

## Selected Best Checkpoint

- **Name:** sanity_1b_step_50
- **Step:** 50
- **Path:** `tinker://c77c35b7-de48-5d8c-afe2-5c44378fb8c9:train:0/sampler_weights/sanity_1b_step_50`

## Top Checkpoints Full Evaluations

| Checkpoint | Step | IFEval | GSM8K | HumanEval |
| --- | ---: | ---: | ---: | ---: |
| sanity_1b_step_50 | 50 | 0.2458 | 0.5989 | 0.5000 |
| sanity_1b_step_75 | 75 | 0.2976 | 0.5481 | 0.4878 |
| sanity_1b_step_25 | 25 | 0.3179 | 0.5155 | 0.5061 |

## Best Checkpoint Full Evaluation

- **IFEval score:** 0.2458
- **GSM8K accuracy:** 0.5989
- **HumanEval score:** 0.5000
