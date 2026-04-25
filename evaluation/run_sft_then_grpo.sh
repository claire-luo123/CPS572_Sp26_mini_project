#!/usr/bin/env bash
# Full SFT (with save_state) via run_full_experiment, then GRPO on GSM8K from
# best_checkpoint.state_path in the experiment report.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

RUN_DIR="evaluation/experiment_runs/sft_state_grpo_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

echo "==> 1/2: SFT + quick/full eval. Artifacts: $RUN_DIR"
.venv/bin/python evaluation/run_full_experiment.py \
  --output_dir "$RUN_DIR" \
  --model meta-llama/Llama-3.1-8B \
  --rank 32 --batch_size 4 --max_seq_len 4096 \
  --lr 1e-4 --use_cosine_lr --lr_warmup_frac 0.05 --lr_min_frac 0.1 \
  --num_steps 400 --checkpoint_every 50 \
  --stage2_fraction 0.4 --stage2_if_weight 0.7 \
  --quick_eval_limit 80 --quick_eval_sample_shuffle_seed 42 \
  --num_best_checkpoints 2 \
  --checkpoint_name v4_with_state

REPORT="$RUN_DIR/experiment_report.json"
STATE="$(
.venv/bin/python - <<PY
import json
from pathlib import Path
p = Path("$REPORT")
r = json.loads(p.read_text())
bc = r.get("best_checkpoint") or {}
sp = bc.get("state_path")
if not sp:
    for row in r.get("top_full_evaluations") or []:
        if row.get("state_path"):
            sp = row["state_path"]
            break
print(sp or "")
PY
)"

if [ -z "$STATE" ]; then
  echo "ERROR: No state_path in $REPORT — GRPO cannot start (check save_state in train output)."
  exit 1
fi

echo ""
echo "==> 2/2: GRPO (GSM8K) from state: $STATE"
.venv/bin/python evaluation/train_grpo_gsm8k.py \
  --base_model meta-llama/Llama-3.1-8B \
  --load_checkpoint_path "$STATE" \
  --lora_rank 32 \
  --learning_rate 2e-5 \
  --group_size 8 --groups_per_batch 32 \
  --max_tokens 512 --temperature 1.0 \
  --max_steps 150 --save_every 25 --eval_every 25 \
  --log_path /tmp/tinker-examples/math_rl/v4_sft_then_grpo

echo ""
echo "Done. SFT+eval: $RUN_DIR | GRPO logs: /tmp/tinker-examples/math_rl/v4_sft_then_grpo"
