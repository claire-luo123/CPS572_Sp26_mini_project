# Experiment Report

- **Run timestamp:** 2026-04-24T23:55:17.161884Z
- **Base model:** meta-llama/Llama-3.1-8B
- **Training steps:** 500
- **Checkpoint cadence:** every 25 steps

## Baseline Scores (Base Model)

- **IFEval baseline:** 0.1682
- **GSM8K baseline:** 0.1077
- **HumanEval baseline:** 0.0000

## Quick Checkpoint Ranking

| Checkpoint | Step | Avg Score | IFEval | GSM8K | HumanEval |
| --- | ---: | ---: | ---: | ---: | ---: |
| sanity_1b_step_450 | 450 | 0.5778 | 0.5333 | 0.5000 | 0.7000 |
| sanity_1b_step_200 | 200 | 0.5556 | 0.3667 | 0.6000 | 0.7000 |
| sanity_1b_step_425 | 425 | 0.5444 | 0.4667 | 0.4667 | 0.7000 |
| sanity_1b_step_225 | 225 | 0.5222 | 0.4333 | 0.4667 | 0.6667 |
| sanity_1b_step_250 | 250 | 0.5222 | 0.4667 | 0.4333 | 0.6667 |
| sanity_1b_step_350 | 350 | 0.5222 | 0.5000 | 0.3667 | 0.7000 |
| sanity_1b_step_400 | 400 | 0.5222 | 0.4667 | 0.4667 | 0.6333 |
| sanity_1b_step_100 | 100 | 0.5222 | 0.2333 | 0.6000 | 0.7333 |
| sanity_1b | 500 | 0.5222 | 0.3000 | 0.5333 | 0.7333 |
| sanity_1b_step_500 | 500 | 0.5111 | 0.3000 | 0.5333 | 0.7000 |
| sanity_1b_step_25 | 25 | 0.5000 | 0.2667 | 0.5000 | 0.7333 |
| sanity_1b_step_275 | 275 | 0.4889 | 0.4000 | 0.4333 | 0.6333 |
| sanity_1b_step_325 | 325 | 0.4889 | 0.4000 | 0.4333 | 0.6333 |
| sanity_1b_step_375 | 375 | 0.4889 | 0.4000 | 0.4333 | 0.6333 |
| sanity_1b_step_300 | 300 | 0.4889 | 0.3333 | 0.4333 | 0.7000 |
| sanity_1b_step_75 | 75 | 0.4778 | 0.3333 | 0.4667 | 0.6333 |
| sanity_1b_step_150 | 150 | 0.4778 | 0.2333 | 0.5000 | 0.7000 |
| sanity_1b_step_175 | 175 | 0.4778 | 0.3667 | 0.4333 | 0.6333 |
| sanity_1b_step_50 | 50 | 0.4444 | 0.3333 | 0.4000 | 0.6000 |
| sanity_1b_step_475 | 475 | 0.4222 | 0.2667 | 0.3000 | 0.7000 |
| sanity_1b_step_125 | 125 | 0.3889 | 0.2333 | 0.3333 | 0.6000 |

## Selected Best Checkpoint

- **Name:** sanity_1b_step_425
- **Step:** 425
- **Path:** `tinker://1a2515ff-a5c7-52d7-a329-d94a6eb2e6dd:train:0/sampler_weights/sanity_1b_step_425`

## Top Checkpoints Full Evaluations

| Checkpoint | Step | IFEval | GSM8K | HumanEval |
| --- | ---: | ---: | ---: | ---: |
| sanity_1b_step_450 | 450 | 0.4325 | 0.5853 | 0.4817 |
| sanity_1b_step_200 | 200 | 0.3845 | 0.5193 | 0.4634 |
| sanity_1b_step_425 | 425 | 0.4362 | 0.5944 | 0.4695 |

## Best Checkpoint Full Evaluation

- **IFEval score:** 0.4362
- **GSM8K accuracy:** 0.5944
- **HumanEval score:** 0.4695
