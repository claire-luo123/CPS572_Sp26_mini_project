"""
Train a model (minimal SFT), save checkpoint, and publish it.

NOTE: This is a TOY EXAMPLE that trains for a few steps on dummy data
to verify the full workflow end-to-end. You should replace the training
data and training logic with your own implementation.

TODO:
  - Replace DEMO_CONVERSATIONS with your task-specific training data
  - Tune hyperparameters (learning rate, batch size, number of steps, LoRA rank)
  - Add validation / early stopping as needed

Usage:
    python evaluation/train_and_publish.py
    python evaluation/train_and_publish.py --num_steps 20
    python evaluation/train_and_publish.py --no_publish   # skip publishing
"""

import argparse
import asyncio
import json
import math
import os
from datetime import datetime

import numpy as np
import tinker
from tinker import types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer


def cosine_lr(step, total_steps, base_lr, warmup_frac=0.05, min_lr_frac=0.1):
    """Cosine LR schedule with linear warmup. Prevents the late-step IF collapse
    we see at constant LR (e.g. step 475 IFEval drop in 20260424_223215)."""
    warmup = max(1, int(round(warmup_frac * total_steps)))
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    progress = min(max(progress, 0.0), 1.0)
    return base_lr * (min_lr_frac + (1.0 - min_lr_frac) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def classify_conversation(convo):
    """Return one of 'gsm8k', 'ifeval', 'code', 'other' so we can sample by task
    in the two-stage curriculum."""
    user_text = ""
    assistant_text = ""
    for msg in convo:
        role = msg.get("role") if isinstance(msg, dict) else None
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        if role == "user" and not user_text:
            user_text = content or ""
        elif role == "assistant" and not assistant_text:
            assistant_text = content or ""
    a = (assistant_text or "").strip()
    if "####" in a and any(ch.isdigit() for ch in a[-40:]):
        return "gsm8k"
    if "def " in a or a.lstrip().startswith(("def ", "import ", "from ", "class ", "#")):
        return "code"
    if user_text:
        return "ifeval"
    return "other"

# Default submission model; override with --model.
MODEL = "meta-llama/Llama-3.1-8B"

# Default max sequence length for tokenization (override with --max_seq_len).
# Use 1024 for IF-heavy focus; 2048 if you need longer code / multi-turn contexts (more VRAM).
DEFAULT_MAX_SEQ_LEN = 1024

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _pick_metric(metrics, preferred_substrings):
    numeric_items = [(k, v) for k, v in metrics.items() if _is_number(v)]
    if not numeric_items:
        return None
    for needle in preferred_substrings:
        for key, value in numeric_items:
            if needle in key.lower():
                return float(value)
    return float(numeric_items[0][1])


def _extract_headline_scores(task_results):
    ifeval_metrics = task_results.get("ifeval", {}).get("metrics", {})
    gsm8k_metrics = task_results.get("gsm8k", {}).get("metrics", {})
    humaneval_metrics = task_results.get("humaneval", {}).get("metrics", {})

    return {
        "ifeval_score": _pick_metric(
            ifeval_metrics,
            [
                "final_acc",
                "prompt_level_strict",
                "prompt_strict",
                "prompt_level_loose",
                "prompt_loose",
                "inst_level_strict",
                "inst_strict",
                "inst_level_loose",
                "inst_loose",
                "accuracy",
                "acc",
            ],
        ),
        "gsm8k_accuracy": _pick_metric(gsm8k_metrics, ["accuracy", "acc", "exact_match", "correct"]),
        "humaneval_score": _pick_metric(humaneval_metrics, ["pass@1", "accuracy", "acc", "correct"]),
    }


def _average_available(values):
    present = [v for v in values if v is not None]
    if not present:
        return None
    return float(sum(present) / len(present))


def _format_score(value):
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def main():
    parser = argparse.ArgumentParser(description="Train, save, and publish a checkpoint")
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL,
        help="Base model to train from (e.g. meta-llama/Llama-3.2-3B or meta-llama/Llama-3.1-8B)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=200,
        help="Training steps (try 200 first, extend toward 500 if eval still improving)",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (try 8 if memory allows)")
    parser.add_argument(
        "--lr",
        type=float,
        default=8e-5,
        help="Learning rate (8e-5 is often safer for instruction fidelity than 1e-4)",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=64,
        help="LoRA rank (higher can help format / constraint following)",
    )
    parser.add_argument("--checkpoint_name", type=str, default="mixed_sft_1b", help="Checkpoint name")
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=25,
        help="Save intermediate checkpoint every N steps (0 disables periodic saves)",
    )
    parser.add_argument(
        "--evaluate_top_k",
        type=int,
        default=3,
        help="After training, evaluate top-K checkpoints by lowest train loss (0 disables)",
    )
    parser.add_argument(
        "--eval_limit",
        type=int,
        default=30,
        help="Sample limit per task for candidate checkpoint evaluation",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data shuffling")
    parser.add_argument(
        "--shuffle_data",
        dest="shuffle_data",
        action="store_true",
        help="Shuffle training examples each pass through the dataset (default: enabled)",
    )
    parser.add_argument(
        "--no_shuffle_data",
        dest="shuffle_data",
        action="store_false",
        help="Disable data shuffling and use deterministic sequential order",
    )
    parser.add_argument("--no_publish", action="store_true", help="Skip publishing")
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to training JSONL (default: data/my_mixed_training_data.jsonl)",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=DEFAULT_MAX_SEQ_LEN,
        help="Max sequence length for tokenization (1024 IF focus; 2048 longer code/context)",
    )
    parser.add_argument(
        "--use_cosine_lr",
        dest="use_cosine_lr",
        action="store_true",
        help="Use cosine LR schedule with linear warmup (default: enabled).",
    )
    parser.add_argument(
        "--no_cosine_lr",
        dest="use_cosine_lr",
        action="store_false",
        help="Use constant learning rate (legacy behavior).",
    )
    parser.add_argument("--lr_warmup_frac", type=float, default=0.05,
                        help="Fraction of steps used for linear warmup (default 0.05)")
    parser.add_argument("--lr_min_frac", type=float, default=0.1,
                        help="Cosine schedule floor (final LR = base_lr * lr_min_frac)")
    parser.add_argument("--stage2_fraction", type=float, default=0.0,
                        help="Two-stage curriculum: fraction of steps at the END that "
                             "should sample IFEval-heavy batches (e.g. 0.25 = last 25%%). "
                             "0.0 disables the curriculum.")
    parser.add_argument("--stage2_if_weight", type=float, default=0.8,
                        help="In stage 2, probability each batch slot is filled from the "
                             "IF pool (default 0.8 = 80%% IF, 20%% other tasks).")
    parser.set_defaults(shuffle_data=True, use_cosine_lr=True)
    args = parser.parse_args()
    model_name = args.model
    max_seq_len = int(args.max_seq_len)

    # Setup
    print(f"Model: {model_name}")
    print(f"Seed: {args.seed} | Shuffle data: {args.shuffle_data}")
    tokenizer = get_tokenizer(model_name)
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    print(f"Renderer: {renderer_name}")

    # Load custom training data
    print("Loading custom training data from JSONL...")
    all_conversations = []
    
    data_path = args.data_path or os.path.join(EVAL_DIR, "..", "data", "my_mixed_training_data.jsonl")
    data_path = os.path.abspath(data_path)
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Training data not found: {data_path}")

    print(f"Training data path: {data_path}")
    conversation_tasks = []
    with open(data_path, "r") as f:
        for line in f:
            row = json.loads(line)
            convo = None
            task = None
            if "question" in row and "answer" in row:
                convo = [
                    {"role": "user", "content": row["question"]},
                    {"role": "assistant", "content": row["answer"]},
                ]
                task = "gsm8k"
            elif "messages" in row:
                convo = row["messages"]
                task = (row.get("task") if isinstance(row, dict) else None) or "ifeval"
            elif "instruction" in row and "output" in row:
                convo = [
                    {"role": "user", "content": row["instruction"]},
                    {"role": "assistant", "content": row["output"]},
                ]
                task = "code"
            if convo is None:
                continue
            all_conversations.append(convo)
            conversation_tasks.append(task)

    # Tokenize. Drop rows that fail conversation_to_datum, but keep the task tag aligned.
    print(f"Tokenizing training data (max_length={max_seq_len})...")
    all_data = []
    all_tasks = []
    for convo, task in zip(all_conversations, conversation_tasks):
        try:
            datum = conversation_to_datum(
                convo,
                renderer,
                max_length=max_seq_len,
                train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES,
            )
        except Exception:
            continue
        all_data.append(datum)
        all_tasks.append(task)

    task_counts = {t: all_tasks.count(t) for t in set(all_tasks)}
    print(f"  {len(all_data)} training examples prepared and tokenized!")
    print(f"  Task mix: {task_counts}")

    # Per-task index pools for the two-stage curriculum sampler.
    task_pools = {"ifeval": [], "gsm8k": [], "code": [], "other": []}
    for idx, t in enumerate(all_tasks):
        task_pools.setdefault(t, []).append(idx)
    if not task_pools.get("ifeval"):
        # Fallback so curriculum doesn't crash on a code/gsm8k-only file.
        task_pools["ifeval"] = list(range(len(all_data)))
    other_pool = task_pools.get("gsm8k", []) + task_pools.get("code", []) + task_pools.get("other", [])
    if not other_pool:
        other_pool = list(range(len(all_data)))

    # Create training client
    print(f"Creating LoRA training client (rank={args.rank})...")
    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=model_name, rank=args.rank)
    print("  Training client ready")

    # Train
    print(f"\nTraining for {args.num_steps} steps (batch_size={args.batch_size}, base_lr={args.lr})...")
    if args.use_cosine_lr:
        print(f"  LR schedule: cosine, warmup_frac={args.lr_warmup_frac}, min_lr_frac={args.lr_min_frac}")
    else:
        print(f"  LR schedule: constant lr={args.lr}")
    if args.stage2_fraction and args.stage2_fraction > 0:
        stage2_start_step = int(round(args.num_steps * (1.0 - args.stage2_fraction)))
        print(
            f"  Curriculum: stage 1 = steps 0..{stage2_start_step}, "
            f"stage 2 = steps {stage2_start_step}..{args.num_steps} "
            f"(IF weight = {args.stage2_if_weight})"
        )
    else:
        stage2_start_step = args.num_steps + 1  # never trigger
        print("  Curriculum: disabled (stage2_fraction=0)")
    if args.checkpoint_every > 0:
        print(f"  Intermediate checkpoints every {args.checkpoint_every} steps.")

    checkpoints = []
    rng = np.random.default_rng(args.seed)
    py_rng = np.random.default_rng(args.seed + 1)
    order = np.arange(len(all_data))
    if args.shuffle_data:
        rng.shuffle(order)
    order_cursor = 0

    if_pool_arr = np.array(task_pools["ifeval"], dtype=np.int64) if task_pools.get("ifeval") else np.array([], dtype=np.int64)
    other_pool_arr = np.array(other_pool, dtype=np.int64)

    for step in range(args.num_steps):
        in_stage2 = step >= stage2_start_step

        if in_stage2 and len(if_pool_arr) > 0 and args.stage2_if_weight > 0:
            # Curriculum batch: each slot drawn from IF pool with prob stage2_if_weight,
            # else from the rest. Sampling with replacement avoids edge cases on small pools.
            picks = py_rng.random(args.batch_size) < args.stage2_if_weight
            batch_indices = [
                int(if_pool_arr[py_rng.integers(0, len(if_pool_arr))]) if pick
                else int(other_pool_arr[py_rng.integers(0, len(other_pool_arr))])
                for pick in picks
            ]
        else:
            # Normal stage-1 batches: walk the (optionally shuffled) order.
            batch_indices = []
            for _ in range(args.batch_size):
                if order_cursor >= len(order):
                    order_cursor = 0
                    if args.shuffle_data:
                        rng.shuffle(order)
                batch_indices.append(int(order[order_cursor]))
                order_cursor += 1

        batch = [all_data[i] for i in batch_indices]

        # Cosine LR (or constant if --no_cosine_lr): build AdamParams per step.
        if args.use_cosine_lr:
            current_lr = cosine_lr(
                step,
                args.num_steps,
                args.lr,
                warmup_frac=args.lr_warmup_frac,
                min_lr_frac=args.lr_min_frac,
            )
        else:
            current_lr = args.lr
        adam_params = types.AdamParams(learning_rate=current_lr, beta1=0.9, beta2=0.95, eps=1e-8)

        fwd_bwd_future = tc.forward_backward(batch, loss_fn="cross_entropy")
        optim_future = tc.optim_step(adam_params)

        fwd_bwd_result = fwd_bwd_future.result()
        optim_future.result()

        # Compute loss
        logprobs = np.concatenate([o["logprobs"].tolist() for o in fwd_bwd_result.loss_fn_outputs])
        weights = np.concatenate([d.loss_fn_inputs["weights"].tolist() for d in batch])
        loss = -np.dot(logprobs, weights) / max(float(weights.sum()), 1.0)

        if step % 10 == 0 or step == args.num_steps - 1:
            stage_tag = "S2" if in_stage2 else "S1"
            print(
                f"  Step {step+1}/{args.num_steps} [{stage_tag}] | "
                f"lr={current_lr:.2e} | Loss: {loss:.4f}"
            )

        # Save periodic checkpoint if requested.
        # We save BOTH a sampler-weights checkpoint (for inference / eval) AND
        # a state checkpoint (the only kind that the cookbook RL recipes can
        # load via `create_training_client_from_state`). Without state saving
        # there's no way to seed GRPO from this SFT run.
        step_num = step + 1
        if args.checkpoint_every > 0 and step_num % args.checkpoint_every == 0:
            periodic_name = f"{args.checkpoint_name}_step_{step_num}"
            print(f"\nSaving periodic checkpoint '{periodic_name}'...")
            ckpt = tc.save_weights_for_sampler(name=periodic_name).result()
            periodic_path = ckpt.path
            print(f"  Periodic sampler ckpt: {periodic_path}")
            try:
                state_ckpt = tc.save_state(name=periodic_name).result()
                state_path = state_ckpt.path
                print(f"  Periodic state ckpt:   {state_path}")
            except Exception as e:
                state_path = None
                print(f"  WARNING: save_state failed for {periodic_name}: {e}")
            checkpoints.append(
                {
                    "name": periodic_name,
                    "path": periodic_path,
                    "state_path": state_path,
                    "step": step_num,
                    "type": "periodic",
                    "train_loss": float(loss),
                }
            )

    # Save final checkpoint (both flavors).
    print(f"\nSaving checkpoint '{args.checkpoint_name}'...")
    ckpt = tc.save_weights_for_sampler(name=args.checkpoint_name).result()
    checkpoint_path = ckpt.path
    print(f"  Final sampler ckpt: {checkpoint_path}")
    try:
        state_ckpt = tc.save_state(name=args.checkpoint_name).result()
        final_state_path = state_ckpt.path
        print(f"  Final state ckpt:   {final_state_path}")
    except Exception as e:
        final_state_path = None
        print(f"  WARNING: save_state failed for {args.checkpoint_name}: {e}")
    checkpoints.append(
        {
            "name": args.checkpoint_name,
            "path": checkpoint_path,
            "state_path": final_state_path,
            "step": args.num_steps,
            "type": "final",
            "train_loss": float(loss),
        }
    )

    candidate_eval = None
    if args.evaluate_top_k > 0:
        from evaluation.eval_all import run_core

        # Pick the lowest-loss checkpoints as candidates to evaluate on real tasks.
        sorted_by_loss = sorted(
            checkpoints,
            key=lambda cp: cp.get("train_loss", float("inf")),
        )
        candidates = sorted_by_loss[: min(args.evaluate_top_k, len(sorted_by_loss))]
        print("\nEvaluating candidate checkpoints...")
        print(
            "  Candidates (lowest train loss): "
            + ", ".join(
                f"{cp['name']}@{cp['step']} (loss={cp.get('train_loss', float('nan')):.4f})"
                for cp in candidates
            )
        )

        ranked_results = []
        for cp in candidates:
            print(f"\nRunning eval on candidate: {cp['name']} ({cp['path']})")
            _, task_results = asyncio.run(
                run_core(
                    base_model=model_name,
                    checkpoint_path=cp["path"],
                    renderer_name=renderer_name,
                    temperature=0.0,
                    top_p=1.0,
                    limit=args.eval_limit,
                    log_dir=None,
                    verbose=False,
                )
            )
            headline_scores = _extract_headline_scores(task_results)
            avg_score = _average_available(
                [
                    headline_scores["ifeval_score"],
                    headline_scores["gsm8k_accuracy"],
                    headline_scores["humaneval_score"],
                ]
            )
            ranked_results.append(
                {
                    "name": cp["name"],
                    "path": cp["path"],
                    "step": cp["step"],
                    "train_loss": cp.get("train_loss"),
                    "headline_scores": headline_scores,
                    "average_score": avg_score,
                }
            )

        ranked_results = sorted(
            ranked_results,
            key=lambda r: r["average_score"] if r["average_score"] is not None else -1.0,
            reverse=True,
        )

        candidate_eval = {
            "top_k": args.evaluate_top_k,
            "eval_limit": args.eval_limit,
            "candidates": [cp["name"] for cp in candidates],
            "ranked_results": ranked_results,
            "best_candidate": ranked_results[0] if ranked_results else None,
        }

        print("\nCandidate checkpoint ranking (higher is better):")
        for i, row in enumerate(ranked_results, start=1):
            scores = row["headline_scores"]
            print(
                f"  {i}. {row['name']} | avg={_format_score(row['average_score'])} "
                f"| ifeval={_format_score(scores['ifeval_score'])} "
                f"| gsm8k={_format_score(scores['gsm8k_accuracy'])} "
                f"| humaneval={_format_score(scores['humaneval_score'])}"
            )
    else:
        print("\nSkipping candidate checkpoint evaluation (--evaluate_top_k=0).")

    # Publish
    if not args.no_publish:
        print("\nPublishing checkpoint...")
        rest_client = sc.create_rest_client()
        rest_client.publish_checkpoint_from_tinker_path(checkpoint_path).result()
        print("  Published successfully!")
    else:
        print("\nSkipping publish (--no_publish).")

    # Save checkpoint info
    info = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "checkpoint_path": checkpoint_path,
        "state_path": final_state_path,
        "checkpoints": checkpoints,
        "base_model": model_name,
        "renderer_name": renderer_name,
        "training": {
            "data_path": data_path,
            "num_steps": args.num_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "lora_rank": args.rank,
            "max_seq_len": max_seq_len,
            "checkpoint_every": args.checkpoint_every,
            "seed": args.seed,
            "shuffle_data": args.shuffle_data,
            "use_cosine_lr": args.use_cosine_lr,
            "lr_warmup_frac": args.lr_warmup_frac,
            "lr_min_frac": args.lr_min_frac,
            "stage2_fraction": args.stage2_fraction,
            "stage2_if_weight": args.stage2_if_weight,
            "task_counts": task_counts,
        },
        "candidate_evaluation": candidate_eval,
        "published": not args.no_publish,
    }
    info_path = os.path.join(EVAL_DIR, "checkpoint_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nCheckpoint info saved to {info_path}")
    print(f"\nNext: evaluate your checkpoint with")
    print(f"  PYTHONPATH=. python evaluation/eval_all.py --checkpoint_path \"{checkpoint_path}\" --base_model {model_name}")

if __name__ == "__main__":
    main()