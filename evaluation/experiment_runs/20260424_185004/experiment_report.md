# Experiment Report

- **Run timestamp:** 2026-04-24T19:36:27.842997Z
- **Base model:** meta-llama/Llama-3.1-8B
- **Training steps:** 200
- **Checkpoint cadence:** every 25 steps

## Baseline Scores (Base Model)

- **IFEval baseline:** 0.1516
- **GSM8K baseline:** 0.0970
- **HumanEval baseline:** 0.0000

## Quick Checkpoint Ranking

| Checkpoint | Step | Avg Score | IFEval | GSM8K | HumanEval |
| --- | ---: | ---: | ---: | ---: | ---: |
| sanity_1b | 200 | 0.5333 | 0.3667 | 0.5667 | 0.6667 |
| sanity_1b_step_200 | 200 | 0.5222 | 0.3667 | 0.5333 | 0.6667 |
| sanity_1b_step_25 | 25 | 0.4667 | 0.2667 | 0.4667 | 0.6667 |
| sanity_1b_step_150 | 150 | 0.4889 | 0.2333 | 0.5333 | 0.7000 |
| sanity_1b_step_125 | 125 | 0.4667 | 0.2333 | 0.5000 | 0.6667 |
| sanity_1b_step_175 | 175 | 0.4667 | 0.2333 | 0.4667 | 0.7000 |
| sanity_1b_step_50 | 50 | 0.4556 | 0.2000 | 0.5333 | 0.6333 |
| sanity_1b_step_100 | 100 | 0.4333 | 0.2333 | 0.4333 | 0.6333 |
| sanity_1b_step_75 | 75 | 0.4667 | 0.1667 | 0.5000 | 0.7333 |

## Selected Best Checkpoint

- **Name:** sanity_1b_step_200
- **Step:** 200
- **Path:** `tinker://a929a1f9-e027-5a08-a9bf-a142a4ae38e4:train:0/sampler_weights/sanity_1b_step_200`

## Top Checkpoints Full Evaluations

| Checkpoint | Step | IFEval | GSM8K | HumanEval |
| --- | ---: | ---: | ---: | ---: |
| sanity_1b | 200 | 0.3678 | 0.5944 | 0.4451 |
| sanity_1b_step_200 | 200 | 0.3752 | 0.6027 | 0.4573 |
| sanity_1b_step_25 | 25 | 0.2514 | 0.5610 | 0.4939 |

## Best Checkpoint Full Evaluation

- **IFEval score:** 0.3752
- **GSM8K accuracy:** 0.6027
- **HumanEval score:** 0.4573
