# Experiment Report

- **Run timestamp:** 2026-04-24T22:16:28.104239Z
- **Base model:** meta-llama/Llama-3.1-8B
- **Training steps:** 3
- **Checkpoint cadence:** every 0 steps

## Baseline Scores (Base Model)

- **IFEval baseline:** 0.3333
- **GSM8K baseline:** 0.0000
- **HumanEval baseline:** 0.0000

## Quick Checkpoint Ranking

| Checkpoint | Step | Avg Score | IFEval | GSM8K | HumanEval |
| --- | ---: | ---: | ---: | ---: | ---: |
| smoke_exp | 3 | 0.2222 | 0.3333 | 0.3333 | 0.0000 |

## Selected Best Checkpoint

- **Name:** smoke_exp
- **Step:** 3
- **Path:** `tinker://26dc0c0e-20b0-5f50-b824-681b88c2a88d:train:0/sampler_weights/smoke_exp`

## Top Checkpoints Full Evaluations

| Checkpoint | Step | IFEval | GSM8K | HumanEval |
| --- | ---: | ---: | ---: | ---: |
| smoke_exp | 3 | 0.3333 | 0.3333 | 0.0000 |

## Best Checkpoint Full Evaluation

- **IFEval score:** 0.3333
- **GSM8K accuracy:** 0.3333
- **HumanEval score:** 0.0000
