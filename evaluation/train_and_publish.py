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
import os
from datetime import datetime

import numpy as np
import tinker
from tinker import types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer

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
                "prompt_level_strict",
                "prompt_level_loose",
                "inst_level_strict",
                "inst_level_loose",
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
    parser.set_defaults(shuffle_data=True)
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
    with open(data_path, "r") as f:
        for line in f:
            row = json.loads(line)
            # Map GSM8K format
            if "question" in row and "answer" in row:
                all_conversations.append([
                    {"role": "user", "content": row["question"]},
                    {"role": "assistant", "content": row["answer"]}
                ])
            # Map Tulu format
            elif "messages" in row:
                all_conversations.append(row["messages"])
            # Map Code format (Instruction/Output)
            elif "instruction" in row and "output" in row:
                all_conversations.append([
                    {"role": "user", "content": row["instruction"]},
                    {"role": "assistant", "content": row["output"]}
                ])

    # Prepare training data
    print(f"Tokenizing training data (max_length={max_seq_len})...")
    all_data = []
    for convo in all_conversations:
        try:
            datum = conversation_to_datum(
                convo,
                renderer,
                max_length=max_seq_len,
                train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES,
            )
            all_data.append(datum)
        except Exception as e:
            continue # Skip any malformed data rows
            
    print(f"  {len(all_data)} training examples prepared and tokenized!")

    # Create training client
    print(f"Creating LoRA training client (rank={args.rank})...")
    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=model_name, rank=args.rank)
    print("  Training client ready")

    # Train
    adam_params = types.AdamParams(learning_rate=args.lr, beta1=0.9, beta2=0.95, eps=1e-8)
    print(f"\nTraining for {args.num_steps} steps (batch_size={args.batch_size}, lr={args.lr})...")
    if args.checkpoint_every > 0:
        print(f"Intermediate checkpoints enabled every {args.checkpoint_every} steps.")

    checkpoints = []
    rng = np.random.default_rng(args.seed)
    order = np.arange(len(all_data))
    if args.shuffle_data:
        rng.shuffle(order)
    order_cursor = 0

    for step in range(args.num_steps):
        # Build each batch from a reproducible (optionally shuffled) index order.
        batch_indices = []
        for _ in range(args.batch_size):
            if order_cursor >= len(order):
                order_cursor = 0
                if args.shuffle_data:
                    rng.shuffle(order)
            batch_indices.append(int(order[order_cursor]))
            order_cursor += 1
        batch = [all_data[i] for i in batch_indices]

        fwd_bwd_future = tc.forward_backward(batch, loss_fn="cross_entropy")
        optim_future = tc.optim_step(adam_params)

        fwd_bwd_result = fwd_bwd_future.result()
        optim_future.result()

        # Compute loss
        logprobs = np.concatenate([o["logprobs"].tolist() for o in fwd_bwd_result.loss_fn_outputs])
        weights = np.concatenate([d.loss_fn_inputs["weights"].tolist() for d in batch])
        loss = -np.dot(logprobs, weights) / max(weights.sum(), 1)
        
        # Only print every 10 steps so it doesn't spam your terminal
        if step % 10 == 0 or step == args.num_steps - 1:
            print(f"  Step {step+1}/{args.num_steps} | Loss: {loss:.4f}")

        # Save periodic checkpoint if requested
        step_num = step + 1
        if args.checkpoint_every > 0 and step_num % args.checkpoint_every == 0:
            periodic_name = f"{args.checkpoint_name}_step_{step_num}"
            print(f"\nSaving periodic checkpoint '{periodic_name}'...")
            ckpt = tc.save_weights_for_sampler(name=periodic_name).result()
            periodic_path = ckpt.path
            print(f"  Periodic checkpoint saved: {periodic_path}")
            checkpoints.append(
                {
                    "name": periodic_name,
                    "path": periodic_path,
                    "step": step_num,
                    "type": "periodic",
                    "train_loss": float(loss),
                }
            )

    # Save checkpoint
    print(f"\nSaving checkpoint '{args.checkpoint_name}'...")
    ckpt = tc.save_weights_for_sampler(name=args.checkpoint_name).result()
    checkpoint_path = ckpt.path
    print(f"  Checkpoint saved: {checkpoint_path}")
    checkpoints.append(
        {
            "name": args.checkpoint_name,
            "path": checkpoint_path,
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
