# Experiment Report

- **Run timestamp:** 2026-04-24T07:08:47.177725Z
- **Base model:** meta-llama/Llama-3.1-8B
- **Training steps:** 100
- **Checkpoint cadence:** every 25 steps

## Baseline Scores (Base Model)

- **IFEval baseline:** 0.1571
- **GSM8K baseline:** 0.0993
- **HumanEval baseline:** 0.0000

## Quick Checkpoint Ranking

| Checkpoint | Step | Avg Score | IFEval | GSM8K | HumanEval |
| --- | ---: | ---: | ---: | ---: | ---: |
| sanity_1b_step_340 | 340 | 0.5889 | 0.3667 | 0.6667 | 0.7333 |
| sanity_1b | 500 | 0.5556 | 0.4000 | 0.6000 | 0.6667 |
| sanity_1b_step_500 | 500 | 0.5444 | 0.4333 | 0.5667 | 0.6333 |
| sanity_1b_step_180 | 180 | 0.5333 | 0.2667 | 0.6333 | 0.7000 |
| sanity_1b_step_320 | 320 | 0.5333 | 0.4000 | 0.5667 | 0.6333 |
| sanity_1b_step_360 | 360 | 0.5333 | 0.3667 | 0.5333 | 0.7000 |
| sanity_1b_step_440 | 440 | 0.5333 | 0.4000 | 0.5667 | 0.6333 |
| sanity_1b_step_460 | 460 | 0.5111 | 0.3333 | 0.5667 | 0.6333 |
| sanity_1b_step_260 | 260 | 0.5000 | 0.2667 | 0.5333 | 0.7000 |
| sanity_1b_step_480 | 480 | 0.5000 | 0.3333 | 0.5333 | 0.6333 |
| sanity_1b_step_400 | 400 | 0.4889 | 0.2667 | 0.6000 | 0.6000 |
| sanity_1b_step_420 | 420 | 0.4889 | 0.3667 | 0.4333 | 0.6667 |
| sanity_1b_step_380 | 380 | 0.4889 | 0.3667 | 0.4667 | 0.6333 |
| sanity_1b_step_240 | 240 | 0.4778 | 0.2333 | 0.5333 | 0.6667 |
| sanity_1b_step_40 | 40 | 0.4667 | 0.3000 | 0.4333 | 0.6667 |
| sanity_1b_step_120 | 120 | 0.4667 | 0.2667 | 0.4667 | 0.6667 |
| sanity_1b_step_160 | 160 | 0.4667 | 0.2333 | 0.5000 | 0.6667 |
| sanity_1b_step_200 | 200 | 0.4667 | 0.2667 | 0.5000 | 0.6333 |
| sanity_1b_step_220 | 220 | 0.4667 | 0.3000 | 0.4333 | 0.6667 |
| sanity_1b_step_60 | 60 | 0.4556 | 0.1667 | 0.5333 | 0.6667 |
| sanity_1b_step_80 | 80 | 0.4444 | 0.2333 | 0.4667 | 0.6333 |
| sanity_1b_step_140 | 140 | 0.4444 | 0.3667 | 0.3333 | 0.6333 |
| sanity_1b_step_280 | 280 | 0.4333 | 0.2667 | 0.4000 | 0.6333 |
| sanity_1b_step_20 | 20 | 0.4111 | 0.2000 | 0.3667 | 0.6667 |
| sanity_1b_step_100 | 100 | 0.4111 | 0.3000 | 0.3667 | 0.5667 |
| sanity_1b_step_300 | 300 | 0.3778 | 0.1667 | 0.3667 | 0.6000 |

## Selected Best Checkpoint

- **Name:** sanity_1b_step_500
- **Step:** 500
- **Path:** `tinker://2e906d1d-b2e9-59fb-9c1b-8acd97691780:train:0/sampler_weights/sanity_1b_step_500`

## Top Checkpoints Full Evaluations

| Checkpoint | Step | IFEval | GSM8K | HumanEval |
| --- | ---: | ---: | ---: | ---: |
| sanity_1b_step_340 | 340 | 0.3771 | 0.5951 | 0.4939 |
| sanity_1b | 500 | 0.3697 | 0.6156 | 0.5061 |
| sanity_1b_step_500 | 500 | 0.3752 | 0.6232 | 0.4939 |

## Best Checkpoint Full Evaluation

- **IFEval score:** 0.3752
- **GSM8K accuracy:** 0.6232
- **HumanEval score:** 0.4939
