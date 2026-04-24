# Experiment Report

- **Run timestamp:** 2026-04-24T20:56:31.154350Z
- **Base model:** meta-llama/Llama-3.1-8B
- **Training steps:** 200
- **Checkpoint cadence:** every 25 steps

## Baseline Scores (Base Model)

- **IFEval baseline:** 0.1627
- **GSM8K baseline:** 0.0933
- **HumanEval baseline:** 0.0000

## Quick Checkpoint Ranking

| Checkpoint | Step | Avg Score | IFEval | GSM8K | HumanEval |
| --- | ---: | ---: | ---: | ---: | ---: |
| sanity_1b | 200 | 0.5067 | 0.2800 | 0.6000 | 0.6400 |
| sanity_1b_step_75 | 75 | 0.4933 | 0.2200 | 0.6400 | 0.6200 |
| sanity_1b_step_25 | 25 | 0.4867 | 0.2800 | 0.5400 | 0.6400 |
| sanity_1b_step_200 | 200 | 0.4867 | 0.2600 | 0.5200 | 0.6800 |
| sanity_1b_step_50 | 50 | 0.4600 | 0.2600 | 0.4400 | 0.6800 |
| sanity_1b_step_125 | 125 | 0.4600 | 0.2000 | 0.5800 | 0.6000 |
| sanity_1b_step_100 | 100 | 0.4467 | 0.2800 | 0.4400 | 0.6200 |
| sanity_1b_step_150 | 150 | 0.4467 | 0.1800 | 0.5200 | 0.6400 |
| sanity_1b_step_175 | 175 | 0.4200 | 0.1800 | 0.4800 | 0.6000 |

## Selected Best Checkpoint

- **Name:** sanity_1b
- **Step:** 200
- **Path:** `tinker://5ad4c1e8-78b1-5a4d-8046-9d3b97a39d07:train:0/sampler_weights/sanity_1b`

## Top Checkpoints Full Evaluations

| Checkpoint | Step | IFEval | GSM8K | HumanEval |
| --- | ---: | ---: | ---: | ---: |
| sanity_1b | 200 | 0.3179 | 0.5186 | 0.4939 |
| sanity_1b_step_75 | 75 | 0.2754 | 0.5800 | 0.4695 |
| sanity_1b_step_25 | 25 | 0.2643 | 0.4291 | 0.4329 |

## Best Checkpoint Full Evaluation

- **IFEval score:** 0.3179
- **GSM8K accuracy:** 0.5186
- **HumanEval score:** 0.4939
