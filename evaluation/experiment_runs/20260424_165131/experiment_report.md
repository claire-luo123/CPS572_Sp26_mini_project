# Experiment Report

- **Run timestamp:** 2026-04-24T18:17:36.320866Z
- **Base model:** meta-llama/Llama-3.1-8B
- **Training steps:** 500
- **Checkpoint cadence:** every 25 steps

## Baseline Scores (Base Model)

- **IFEval baseline:** 0.1590
- **GSM8K baseline:** 0.0986
- **HumanEval baseline:** 0.0000

## Quick Checkpoint Ranking

| Checkpoint | Step | Avg Score | IFEval | GSM8K | HumanEval |
| --- | ---: | ---: | ---: | ---: | ---: |
| sanity_1b | 500 | 0.5556 | 0.4000 | 0.6000 | 0.6667 |
| sanity_1b_step_500 | 500 | 0.5444 | 0.3667 | 0.6333 | 0.6333 |
| sanity_1b_step_225 | 225 | 0.5222 | 0.3333 | 0.5333 | 0.7000 |
| sanity_1b_step_300 | 300 | 0.5222 | 0.3000 | 0.6000 | 0.6667 |
| sanity_1b_step_375 | 375 | 0.5222 | 0.2333 | 0.6667 | 0.6667 |
| sanity_1b_step_275 | 275 | 0.5111 | 0.2667 | 0.4667 | 0.8000 |
| sanity_1b_step_400 | 400 | 0.5111 | 0.3667 | 0.5333 | 0.6333 |
| sanity_1b_step_350 | 350 | 0.5000 | 0.3000 | 0.5000 | 0.7000 |
| sanity_1b_step_425 | 425 | 0.4889 | 0.3000 | 0.5000 | 0.6667 |
| sanity_1b_step_325 | 325 | 0.4778 | 0.2000 | 0.4667 | 0.7667 |
| sanity_1b_step_175 | 175 | 0.4778 | 0.2333 | 0.5667 | 0.6333 |
| sanity_1b_step_450 | 450 | 0.4778 | 0.2667 | 0.5333 | 0.6333 |
| sanity_1b_step_250 | 250 | 0.4778 | 0.2333 | 0.5333 | 0.6667 |
| sanity_1b_step_25 | 25 | 0.4667 | 0.2667 | 0.4667 | 0.6667 |
| sanity_1b_step_100 | 100 | 0.4667 | 0.2333 | 0.5000 | 0.6667 |
| sanity_1b_step_50 | 50 | 0.4556 | 0.2333 | 0.4667 | 0.6667 |
| sanity_1b_step_475 | 475 | 0.4556 | 0.3000 | 0.4000 | 0.6667 |
| sanity_1b_step_125 | 125 | 0.4333 | 0.1333 | 0.5333 | 0.6333 |
| sanity_1b_step_150 | 150 | 0.4333 | 0.1667 | 0.5000 | 0.6333 |
| sanity_1b_step_75 | 75 | 0.4111 | 0.2000 | 0.4000 | 0.6333 |
| sanity_1b_step_200 | 200 | 0.4111 | 0.2333 | 0.4333 | 0.5667 |

## Selected Best Checkpoint

- **Name:** sanity_1b
- **Step:** 500
- **Path:** `tinker://e2b08588-608a-59da-83f9-f956fe2e0965:train:0/sampler_weights/sanity_1b`

## Top Checkpoints Full Evaluations

| Checkpoint | Step | IFEval | GSM8K | HumanEval |
| --- | ---: | ---: | ---: | ---: |
| sanity_1b | 500 | 0.3475 | 0.6080 | 0.4817 |
| sanity_1b_step_500 | 500 | 0.3512 | 0.6111 | 0.4573 |
| sanity_1b_step_225 | 225 | 0.2902 | 0.5542 | 0.4817 |

## Best Checkpoint Full Evaluation

- **IFEval score:** 0.3475
- **GSM8K accuracy:** 0.6080
- **HumanEval score:** 0.4817
