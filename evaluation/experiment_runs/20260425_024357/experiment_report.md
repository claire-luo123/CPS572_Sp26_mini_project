# Experiment Report

- **Run timestamp:** 2026-04-25T03:12:36.847889Z
- **Base model:** meta-llama/Llama-3.1-8B
- **Training steps:** 200
- **Checkpoint cadence:** every 50 steps

## Baseline Scores (Base Model)

- **IFEval baseline:** 0.2185
- **GSM8K baseline:** 0.0826
- **HumanEval baseline:** 0.0000

## Quick Checkpoint Ranking

| Checkpoint | Step | Avg Score | IFEval | GSM8K | HumanEval |
| --- | ---: | ---: | ---: | ---: | ---: |
| run_curr_8b_step_100 | 100 | 0.5064 | 0.4193 | 0.6125 | 0.4875 |
| run_curr_8b | 200 | 0.5056 | 0.4669 | 0.5375 | 0.5125 |
| run_curr_8b_step_200 | 200 | 0.5035 | 0.4479 | 0.5500 | 0.5125 |
| run_curr_8b_step_50 | 50 | 0.4990 | 0.3719 | 0.6250 | 0.5000 |
| run_curr_8b_step_150 | 150 | 0.4866 | 0.4097 | 0.5875 | 0.4625 |

## Selected Best Checkpoint

- **Name:** run_curr_8b
- **Step:** 200
- **Path:** `tinker://53cd6253-e7dc-5092-b589-56fb32452103:train:0/sampler_weights/run_curr_8b`

## Top Checkpoints Full Evaluations

| Checkpoint | Step | IFEval | GSM8K | HumanEval |
| --- | ---: | ---: | ---: | ---: |
| run_curr_8b_step_100 | 100 | 0.4563 | 0.5550 | 0.4451 |
| run_curr_8b | 200 | 0.4648 | 0.5876 | 0.4512 |
| run_curr_8b_step_200 | 200 | 0.4592 | 0.5861 | 0.4390 |

## Best Checkpoint Full Evaluation

- **IFEval score:** 0.4648
- **GSM8K accuracy:** 0.5876
- **HumanEval score:** 0.4512
