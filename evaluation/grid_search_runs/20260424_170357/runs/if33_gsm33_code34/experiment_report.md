# Experiment Report

- **Run timestamp:** 2026-04-24T17:30:53.697181Z
- **Base model:** meta-llama/Llama-3.1-8B
- **Training steps:** 100
- **Checkpoint cadence:** every 25 steps

## Baseline Scores (Base Model)

- **IFEval baseline:** 0.1701
- **GSM8K baseline:** 0.0978
- **HumanEval baseline:** 0.0000

## Quick Checkpoint Ranking

| Checkpoint | Step | Avg Score | IFEval | GSM8K | HumanEval |
| --- | ---: | ---: | ---: | ---: | ---: |
| sanity_1b_step_50 | 50 | 0.4667 | 0.3000 | 0.4333 | 0.6667 |
| sanity_1b_step_75 | 75 | 0.4444 | 0.1667 | 0.4667 | 0.7000 |
| sanity_1b_step_100 | 100 | 0.4111 | 0.2333 | 0.3667 | 0.6333 |
| sanity_1b | 100 | 0.3889 | 0.2333 | 0.3000 | 0.6333 |
| sanity_1b_step_25 | 25 | 0.3778 | 0.2333 | 0.3333 | 0.5667 |

## Selected Best Checkpoint

- **Name:** sanity_1b_step_50
- **Step:** 50
- **Path:** `tinker://f6bc0ecf-39c6-50a7-ac54-95fd6d9945fb:train:0/sampler_weights/sanity_1b_step_50`

## Top Checkpoints Full Evaluations

| Checkpoint | Step | IFEval | GSM8K | HumanEval |
| --- | ---: | ---: | ---: | ---: |
| sanity_1b_step_50 | 50 | 0.2514 | 0.4951 | 0.4512 |
| sanity_1b_step_75 | 75 | 0.2551 | 0.4428 | 0.4451 |
| sanity_1b_step_100 | 100 | 0.2440 | 0.3662 | 0.4573 |

## Best Checkpoint Full Evaluation

- **IFEval score:** 0.2514
- **GSM8K accuracy:** 0.4951
- **HumanEval score:** 0.4512
