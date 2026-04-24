# Experiment Report

- **Run timestamp:** 2026-04-24T17:57:12.529127Z
- **Base model:** meta-llama/Llama-3.1-8B
- **Training steps:** 100
- **Checkpoint cadence:** every 25 steps

## Baseline Scores (Base Model)

- **IFEval baseline:** 0.1590
- **GSM8K baseline:** 0.0902
- **HumanEval baseline:** 0.0000

## Quick Checkpoint Ranking

| Checkpoint | Step | Avg Score | IFEval | GSM8K | HumanEval |
| --- | ---: | ---: | ---: | ---: | ---: |
| sanity_1b_step_50 | 50 | 0.4889 | 0.2333 | 0.5667 | 0.6667 |
| sanity_1b_step_75 | 75 | 0.4444 | 0.3000 | 0.3667 | 0.6667 |
| sanity_1b | 100 | 0.4444 | 0.2667 | 0.4333 | 0.6333 |
| sanity_1b_step_100 | 100 | 0.4333 | 0.2333 | 0.4333 | 0.6333 |
| sanity_1b_step_25 | 25 | 0.4333 | 0.2667 | 0.3667 | 0.6667 |

## Selected Best Checkpoint

- **Name:** sanity_1b
- **Step:** 100
- **Path:** `tinker://1bc68e71-0a9a-5b7a-849b-540fe01fbb63:train:0/sampler_weights/sanity_1b`

## Top Checkpoints Full Evaluations

| Checkpoint | Step | IFEval | GSM8K | HumanEval |
| --- | ---: | ---: | ---: | ---: |
| sanity_1b_step_50 | 50 | 0.3494 | 0.5580 | 0.4878 |
| sanity_1b_step_75 | 75 | 0.3179 | 0.5489 | 0.4939 |
| sanity_1b | 100 | 0.3253 | 0.5914 | 0.5000 |

## Best Checkpoint Full Evaluation

- **IFEval score:** 0.3253
- **GSM8K accuracy:** 0.5914
- **HumanEval score:** 0.5000
