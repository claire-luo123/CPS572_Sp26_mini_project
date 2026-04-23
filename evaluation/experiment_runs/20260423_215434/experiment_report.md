# Experiment Report

- **Run timestamp:** 2026-04-23T22:24:13.048947Z
- **Base model:** meta-llama/Llama-3.2-3B
- **Training steps:** 200
- **Checkpoint cadence:** every 25 steps

## Baseline Scores (Base Model)

- **IFEval baseline:** 0.1627
- **GSM8K baseline:** 0.0910
- **HumanEval baseline:** 0.0061

## Quick Checkpoint Ranking

| Checkpoint | Step | Avg Score | IFEval | GSM8K | HumanEval |
| --- | ---: | ---: | ---: | ---: | ---: |
| sanity_1b_step_100 | 100 | 0.1444 | 0.1000 | 0.3333 | 0.0000 |
| sanity_1b_step_150 | 150 | 0.1444 | 0.1000 | 0.3333 | 0.0000 |
| sanity_1b_step_50 | 50 | 0.1333 | 0.1667 | 0.1667 | 0.0667 |
| sanity_1b_step_200 | 200 | 0.1333 | 0.1000 | 0.2667 | 0.0333 |
| sanity_1b_step_175 | 175 | 0.1222 | 0.1333 | 0.2333 | 0.0000 |
| sanity_1b | 200 | 0.1111 | 0.1000 | 0.2333 | 0.0000 |
| sanity_1b_step_25 | 25 | 0.0889 | 0.0667 | 0.1667 | 0.0333 |
| sanity_1b_step_125 | 125 | 0.0889 | 0.0667 | 0.1667 | 0.0333 |
| sanity_1b_step_75 | 75 | 0.0778 | 0.0667 | 0.1667 | 0.0000 |

## Selected Best Checkpoint

- **Name:** sanity_1b_step_100
- **Step:** 100
- **Path:** `tinker://13881810-6ef7-51ce-b383-21a956931b8b:train:0/sampler_weights/sanity_1b_step_100`

## Top Checkpoints Full Evaluations

| Checkpoint | Step | IFEval | GSM8K | HumanEval |
| --- | ---: | ---: | ---: | ---: |
| sanity_1b_step_100 | 100 | 0.1682 | 0.2987 | 0.0000 |
| sanity_1b_step_150 | 150 | 0.1941 | 0.2980 | 0.0000 |
| sanity_1b_step_50 | 50 | 0.1793 | 0.2350 | 0.0183 |

## Best Checkpoint Full Evaluation

- **IFEval score:** 0.1682
- **GSM8K accuracy:** 0.2987
- **HumanEval score:** 0.0000
