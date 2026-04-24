# Experiment Report

- **Run timestamp:** 2026-04-23T21:50:38.575810Z
- **Base model:** meta-llama/Llama-3.2-1B
- **Training steps:** 100
- **Checkpoint cadence:** every 25 steps

## Baseline Scores (Base Model)

- **IFEval baseline:** 0.1497
- **GSM8K baseline:** 0.0258
- **HumanEval baseline:** 0.0000

## Quick Checkpoint Ranking

| Checkpoint | Step | Avg Score | IFEval | GSM8K | HumanEval |
| --- | ---: | ---: | ---: | ---: | ---: |
| sanity_1b_step_25 | 25 | 0.0667 | 0.1667 | 0.0333 | 0.0000 |
| sanity_1b_step_75 | 75 | 0.0556 | 0.1333 | 0.0333 | 0.0000 |
| sanity_1b_step_50 | 50 | 0.0444 | 0.1333 | 0.0000 | 0.0000 |
| sanity_1b_step_100 | 100 | 0.0444 | 0.1000 | 0.0333 | 0.0000 |
| sanity_1b | 100 | 0.0444 | 0.1000 | 0.0333 | 0.0000 |

## Selected Best Checkpoint

- **Name:** sanity_1b_step_25
- **Step:** 25
- **Path:** `tinker://7d63e8c6-a02a-5923-8c08-b6ce69e64952:train:0/sampler_weights/sanity_1b_step_25`

## Final Full Evaluation Scores

- **IFEval score:** 0.1590
- **GSM8K accuracy:** 0.0842
- **HumanEval score:** 0.0061
