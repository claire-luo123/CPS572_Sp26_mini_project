# Experiment Report

- **Run timestamp:** 2026-04-24T04:06:23.054820Z
- **Base model:** meta-llama/Llama-3.1-8B
- **Training steps:** 100
- **Checkpoint cadence:** every 25 steps

## Baseline Scores (Base Model)

- **IFEval baseline:** 0.1553
- **GSM8K baseline:** 0.0925
- **HumanEval baseline:** 0.0000

## Quick Checkpoint Ranking

| Checkpoint | Step | Avg Score | IFEval | GSM8K | HumanEval |
| --- | ---: | ---: | ---: | ---: | ---: |
| sanity_1b_step_50 | 50 | 0.5556 | 0.4000 | 0.5333 | 0.7333 |
| sanity_1b_step_250 | 250 | 0.5222 | 0.2667 | 0.5000 | 0.8000 |
| sanity_1b_step_375 | 375 | 0.5222 | 0.3667 | 0.5667 | 0.6333 |
| sanity_1b_step_275 | 275 | 0.4889 | 0.3000 | 0.5667 | 0.6000 |
| sanity_1b_step_300 | 300 | 0.4889 | 0.2000 | 0.5333 | 0.7333 |
| sanity_1b_step_325 | 325 | 0.4778 | 0.3000 | 0.5333 | 0.6000 |
| sanity_1b_step_75 | 75 | 0.4667 | 0.3000 | 0.3667 | 0.7333 |
| sanity_1b_step_400 | 400 | 0.4667 | 0.3667 | 0.3667 | 0.6667 |
| sanity_1b_step_25 | 25 | 0.4444 | 0.2333 | 0.5000 | 0.6000 |
| sanity_1b_step_100 | 100 | 0.4444 | 0.2667 | 0.4667 | 0.6000 |
| sanity_1b_step_125 | 125 | 0.4333 | 0.3333 | 0.3333 | 0.6333 |
| sanity_1b_step_225 | 225 | 0.4333 | 0.2333 | 0.4000 | 0.6667 |
| sanity_1b_step_350 | 350 | 0.4222 | 0.2667 | 0.4000 | 0.6000 |
| sanity_1b_step_450 | 450 | 0.4111 | 0.2667 | 0.2333 | 0.7333 |
| sanity_1b_step_475 | 475 | 0.4111 | 0.2667 | 0.3333 | 0.6333 |
| sanity_1b_step_425 | 425 | 0.3778 | 0.2667 | 0.2000 | 0.6667 |
| sanity_1b_step_500 | 500 | 0.3778 | 0.1667 | 0.2667 | 0.7000 |
| sanity_1b_step_175 | 175 | 0.3667 | 0.2000 | 0.2000 | 0.7000 |
| sanity_1b | 500 | 0.3667 | 0.1333 | 0.2667 | 0.7000 |
| sanity_1b_step_200 | 200 | 0.3556 | 0.2667 | 0.1333 | 0.6667 |
| sanity_1b_step_150 | 150 | 0.3444 | 0.2000 | 0.2333 | 0.6000 |

## Selected Best Checkpoint

- **Name:** sanity_1b_step_375
- **Step:** 375
- **Path:** `tinker://ce353178-5516-5f06-89b8-25dd28bcc9ed:train:0/sampler_weights/sanity_1b_step_375`

## Top Checkpoints Full Evaluations

| Checkpoint | Step | IFEval | GSM8K | HumanEval |
| --- | ---: | ---: | ---: | ---: |
| sanity_1b_step_50 | 50 | 0.3623 | 0.4655 | 0.4817 |
| sanity_1b_step_250 | 250 | 0.2514 | 0.4685 | 0.4390 |
| sanity_1b_step_375 | 375 | 0.4085 | 0.5459 | 0.3963 |

## Best Checkpoint Full Evaluation

- **IFEval score:** 0.4085
- **GSM8K accuracy:** 0.5459
- **HumanEval score:** 0.3963
