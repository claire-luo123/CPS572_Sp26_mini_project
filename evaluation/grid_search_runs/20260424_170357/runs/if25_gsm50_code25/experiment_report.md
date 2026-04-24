# Experiment Report

- **Run timestamp:** 2026-04-24T18:24:25.221389Z
- **Base model:** meta-llama/Llama-3.1-8B
- **Training steps:** 100
- **Checkpoint cadence:** every 25 steps

## Baseline Scores (Base Model)

- **IFEval baseline:** 0.1553
- **GSM8K baseline:** 0.1039
- **HumanEval baseline:** 0.0000

## Quick Checkpoint Ranking

| Checkpoint | Step | Avg Score | IFEval | GSM8K | HumanEval |
| --- | ---: | ---: | ---: | ---: | ---: |
| sanity_1b_step_75 | 75 | 0.5000 | 0.3000 | 0.5667 | 0.6333 |
| sanity_1b_step_25 | 25 | 0.4889 | 0.3000 | 0.4667 | 0.7000 |
| sanity_1b_step_50 | 50 | 0.4111 | 0.1333 | 0.4667 | 0.6333 |
| sanity_1b_step_100 | 100 | 0.3778 | 0.2000 | 0.2667 | 0.6667 |
| sanity_1b | 100 | 0.3556 | 0.2000 | 0.2667 | 0.6000 |

## Selected Best Checkpoint

- **Name:** sanity_1b_step_75
- **Step:** 75
- **Path:** `tinker://3c0eafd9-007f-54aa-a38e-e77593f66ad6:train:0/sampler_weights/sanity_1b_step_75`

## Top Checkpoints Full Evaluations

| Checkpoint | Step | IFEval | GSM8K | HumanEval |
| --- | ---: | ---: | ---: | ---: |
| sanity_1b_step_75 | 75 | 0.2569 | 0.5413 | 0.5061 |
| sanity_1b_step_25 | 25 | 0.2643 | 0.5368 | 0.4878 |
| sanity_1b_step_50 | 50 | 0.2107 | 0.4018 | 0.4878 |

## Best Checkpoint Full Evaluation

- **IFEval score:** 0.2569
- **GSM8K accuracy:** 0.5413
- **HumanEval score:** 0.5061
