# Experiment Report

- **Run timestamp:** 2026-04-25T04:24:37.789249Z
- **Base model:** meta-llama/Llama-3.1-8B
- **Training steps:** 400
- **Checkpoint cadence:** every 50 steps

## Baseline Scores (Base Model)

- **IFEval baseline:** 0.2324
- **GSM8K baseline:** 0.1008
- **HumanEval baseline:** 0.0000

## Quick Checkpoint Ranking

| Checkpoint | Step | Avg Score | IFEval | GSM8K | HumanEval |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3_grpo_seed_step_400 | 400 | 0.5212 | 0.5511 | 0.5750 | 0.4375 |
| v3_grpo_seed_step_300 | 300 | 0.5155 | 0.5090 | 0.5750 | 0.4625 |
| v3_grpo_seed_step_250 | 250 | 0.5099 | 0.5046 | 0.5500 | 0.4750 |
| v3_grpo_seed_step_350 | 350 | 0.4947 | 0.5090 | 0.5625 | 0.4125 |
| v3_grpo_seed | 400 | 0.4880 | 0.5016 | 0.5125 | 0.4500 |
| v3_grpo_seed_step_100 | 100 | 0.4846 | 0.5038 | 0.5125 | 0.4375 |
| v3_grpo_seed_step_150 | 150 | 0.4727 | 0.4807 | 0.5125 | 0.4250 |
| v3_grpo_seed_step_200 | 200 | 0.4570 | 0.4460 | 0.5000 | 0.4250 |
| v3_grpo_seed_step_50 | 50 | 0.4442 | 0.4077 | 0.4375 | 0.4875 |

## Selected Best Checkpoint

- **Name:** v3_grpo_seed_step_400
- **Step:** 400
- **Path:** `tinker://d5c7e388-0792-5e3e-9420-20c985168ec1:train:0/sampler_weights/v3_grpo_seed_step_400`

## Top Checkpoints Full Evaluations

| Checkpoint | Step | IFEval | GSM8K | HumanEval |
| --- | ---: | ---: | ---: | ---: |
| v3_grpo_seed_step_400 | 400 | 0.5413 | 0.6262 | 0.4573 |
| v3_grpo_seed_step_300 | 300 | 0.5140 | 0.5625 | 0.4390 |

## Best Checkpoint Full Evaluation

- **IFEval score:** 0.5413
- **GSM8K accuracy:** 0.6262
- **HumanEval score:** 0.4573
