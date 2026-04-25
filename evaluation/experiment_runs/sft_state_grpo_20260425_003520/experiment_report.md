# Experiment Report

- **Run timestamp:** 2026-04-25T05:11:34.918678Z
- **Base model:** meta-llama/Llama-3.1-8B
- **Training steps:** 400
- **Checkpoint cadence:** every 50 steps

## Baseline Scores (Base Model)

- **IFEval baseline:** 0.2225
- **GSM8K baseline:** 0.1001
- **HumanEval baseline:** 0.0000

## Quick Checkpoint Ranking

| Checkpoint | Step | Avg Score | IFEval | GSM8K | HumanEval |
| --- | ---: | ---: | ---: | ---: | ---: |
| v4_with_state_step_100 | 100 | 0.5316 | 0.5449 | 0.6000 | 0.4500 |
| v4_with_state_step_400 | 400 | 0.5260 | 0.5406 | 0.5750 | 0.4625 |
| v4_with_state_step_300 | 300 | 0.5222 | 0.5291 | 0.5875 | 0.4500 |
| v4_with_state_step_250 | 250 | 0.5181 | 0.5542 | 0.5375 | 0.4625 |
| v4_with_state_step_350 | 350 | 0.5128 | 0.5258 | 0.5625 | 0.4500 |
| v4_with_state_step_200 | 200 | 0.5087 | 0.4386 | 0.6000 | 0.4875 |
| v4_with_state | 400 | 0.5086 | 0.5133 | 0.5625 | 0.4500 |
| v4_with_state_step_150 | 150 | 0.4788 | 0.4364 | 0.5750 | 0.4250 |
| v4_with_state_step_50 | 50 | 0.4598 | 0.3920 | 0.4875 | 0.5000 |

## Selected Best Checkpoint

- **Name:** v4_with_state_step_400
- **Step:** 400
- **Path:** `tinker://3d1619c2-fea4-576b-8b6d-5b3f8cfa38d9:train:0/sampler_weights/v4_with_state_step_400`
- **State path (for GRPO):** `tinker://3d1619c2-fea4-576b-8b6d-5b3f8cfa38d9:train:0/weights/v4_with_state_step_400`

## Top Checkpoints Full Evaluations

| Checkpoint | Step | IFEval | GSM8K | HumanEval |
| --- | ---: | ---: | ---: | ---: |
| v4_with_state_step_100 | 100 | 0.5222 | 0.5193 | 0.4512 |
| v4_with_state_step_400 | 400 | 0.5300 | 0.6103 | 0.4573 |

## Best Checkpoint Full Evaluation

- **IFEval score:** 0.5300
- **GSM8K accuracy:** 0.6103
- **HumanEval score:** 0.4573

## Appendix: What we changed in SFT (and how it fits with post-SFT GRPO)

### SFT training improvements (implemented in `evaluation/train_and_publish.py`)

- **Cosine LR + warmup (`--use_cosine_lr`, `--lr_warmup_frac`, `--lr_min_frac`)**  
  Instead of a constant learning rate for the whole run, the LR ramps up during warmup and then decays on a cosine schedule down to a floor (`lr * lr_min_frac`). This is meant to reduce late-training instability (especially on instruction-following) while still allowing early learning.

- **Two-stage curriculum sampling (`--stage2_fraction`, `--stage2_if_weight`)**  
  For the final fraction of steps, each batch slot is sampled with probability `stage2_if_weight` from the instruction-following pool and otherwise from the “other tasks” pool (GSM8K/code/etc.). In this run’s recorded args (`experiment_report.json`), that was **`stage2_fraction=0.4`** and **`stage2_if_weight=0.7`** (last 40% of steps biased ~70% IF per slot).

- **Task tagging + task-mix reporting**  
  While loading JSONL, each example is tagged as **`gsm8k` / `ifeval` / `code` / `other`** using simple heuristics (e.g., GSM8K answers containing `####`, code answers starting like code). Those tags drive the curriculum pools. After tokenization, the script prints a **task-count summary** so you can verify the effective mix you are actually training on.

- **Saving both sampler weights and training state (`save_weights_for_sampler` + `save_state`)**  
  SFT checkpoints used for Inspect eval are sampler-weights paths. GRPO recipes typically need a **state** checkpoint to continue training from the exact optimizer/training client state. This run records a **`state_path`** for GRPO continuation.

### How this connects to post-SFT GRPO (reinforcement learning)

- **What GRPO is doing here**: after SFT, we run GSM8K-focused RL using the cookbook recipe (`evaluation/train_grpo_gsm8k.py` → `tinker_cookbook.recipes.math_rl.train`). The reward is **verifiable correctness** on GSM8K-style numeric answers, so the optimizer gets a strong learning signal without human preference labels.

- **Why it “fits”**: SFT establishes a competent base policy (format + rough capabilities). GRPO then specializes the policy distribution toward **higher reward on GSM8K**, which often produces large GSM8K gains because the objective aligns tightly with the benchmark metric.

### Results: SFT full-eval vs post-GRPO full-eval (why terminal scores can jump)

This experiment report’s **best SFT checkpoint full-eval** (from the table above) is:

- **IFEval (headline metric used by the pipeline):** **0.5300**
- **GSM8K:** **0.6103**
- **HumanEval:** **0.4573**

We also saved a **post-GRPO** full core eval at `evaluation/grpo_run_grpo_full_eval.json` (checkpoint: `tinker://ca052725-3804-5d33-a437-9917a66412e7:train:0/sampler_weights/final`). Headline-style numbers from that file:

- **IFEval `final_acc`:** **0.5872** *(Δ vs SFT headline IFEval: +0.0572)*
- **IFEval `prompt_strict_acc`:** **0.5139** *(Δ vs SFT headline IFEval: -0.0161 — note this is a different IFEval sub-metric than the headline “0.5300” number)*
- **GSM8K accuracy:** **0.7551** *(Δ: +0.1448)*
- **HumanEval accuracy:** **0.5244** *(Δ: +0.0671)*

**Why GSM8K can jump a lot after GRPO:** RL is explicitly optimizing a **GSM8K-aligned reward**, so it is normal to see a large accuracy increase even when SFT was already decent. **Why IFEval can look confusing if you compare the wrong columns:** IFEval reports multiple accuracies (`prompt_strict`, `prompt_loose`, `inst_*`, `final_acc`). The pipeline “headline” IFEval score is whichever metric key matches first in the picker; GRPO is **not** trained on IFEval reward, so IFEval can move up or down depending on which sub-metric you compare and side-effects of the updated policy (format/verbosity/exploration).

**Why terminal numbers can look “much higher” than a quick table:** quick-eval uses `--limit` and is only a proxy; full-eval is the apples-to-apples comparison. Also, any time you change **checkpoint identity** (SFT sampler weights vs GRPO `final`) you should expect score shifts consistent with the new training phase.
