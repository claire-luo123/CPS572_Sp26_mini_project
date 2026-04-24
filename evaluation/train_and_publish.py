"""
Train and optionally publish a LoRA checkpoint from mixed-task JSONL data.

Responsibilities:
1) Load mixed rows and map to task buckets (gsm8k / ifeval / code).
2) Apply per-task caps and weighted task sampling during training.
3) Save periodic + final checkpoints and checkpoint metadata.
4) Optionally score top checkpoints on core eval tasks.
"""

import argparse
import asyncio
import importlib
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

MODEL = "meta-llama/Llama-3.1-8B"
DEFAULT_MAX_SEQ_LEN = 2048
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_symbol(candidates: List[Tuple[str, str]]) -> Optional[Any]:
    for module_name, attr_name in candidates:
        try:
            module = importlib.import_module(module_name)
            value = getattr(module, attr_name, None)
            if value is not None:
                return value
        except Exception:
            continue
    return None


tinker = _resolve_symbol([("tinker", "ServiceClient")])
types_mod = _resolve_symbol([("tinker", "types"), ("tinker.types", "AdamParams")])
get_renderer_name = _resolve_symbol(
    [
        ("tinker_cookbook.model_info", "get_recommended_renderer_name"),
    ]
)
get_renderer = _resolve_symbol(
    [
        ("tinker_cookbook.renderers", "get_renderer"),
    ]
)
train_on_what = _resolve_symbol(
    [
        ("tinker_cookbook.renderers", "TrainOnWhat"),
    ]
)
conversation_to_datum = _resolve_symbol(
    [
        ("tinker_cookbook.renderers", "conversation_to_datum"),
        ("tinker_cookbook.training", "conversation_to_datum"),
        ("tinker_cookbook.training_utils", "conversation_to_datum"),
    ]
)
get_tokenizer = _resolve_symbol(
    [
        ("tinker_cookbook.renderers", "get_tokenizer"),
        ("tinker_cookbook.training", "get_tokenizer"),
        ("tinker_cookbook.training_utils", "get_tokenizer"),
    ]
)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _pick_metric(metrics: Dict[str, Any], preferred_substrings: List[str]) -> Optional[float]:
    numeric_items = [(k, v) for k, v in metrics.items() if _is_number(v)]
    if not numeric_items:
        return None
    for needle in preferred_substrings:
        for key, value in numeric_items:
            if needle in key.lower():
                return float(value)
    return float(numeric_items[0][1])


def _extract_headline_scores(task_results: Dict[str, Any]) -> Dict[str, Optional[float]]:
    ifeval_metrics = task_results.get("ifeval", {}).get("metrics", {})
    gsm8k_metrics = task_results.get("gsm8k", {}).get("metrics", {})
    humaneval_metrics = task_results.get("humaneval", {}).get("metrics", {})
    return {
        "ifeval_score": _pick_metric(
            ifeval_metrics,
            ["prompt_level_strict", "prompt_level_loose", "inst_level_strict", "accuracy", "acc"],
        ),
        "gsm8k_accuracy": _pick_metric(gsm8k_metrics, ["accuracy", "acc", "exact_match", "correct"]),
        "humaneval_score": _pick_metric(humaneval_metrics, ["pass@1", "accuracy", "acc", "correct"]),
    }


def _average_available(values: List[Optional[float]]) -> Optional[float]:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return float(sum(present) / len(present))


def _format_score(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def _row_to_task_and_conversation(row: Dict[str, Any]) -> Optional[Tuple[str, List[Dict[str, str]]]]:
    if "question" in row and "answer" in row:
        return "gsm8k", [
            {"role": "user", "content": str(row["question"])},
            {"role": "assistant", "content": str(row["answer"])},
        ]
    if "messages" in row and isinstance(row["messages"], list):
        return "ifeval", row["messages"]
    if "instruction" in row and "output" in row:
        return "code", [
            {"role": "user", "content": str(row["instruction"])},
            {"role": "assistant", "content": str(row["output"])},
        ]
    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train, save, and optionally publish a checkpoint")
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--num_steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--checkpoint_name", type=str, default="mixed_sft_1b")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--max_seq_len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--checkpoint_every", type=int, default=25)
    parser.add_argument("--save_every_steps", type=int, default=0)
    parser.add_argument("--evaluate_top_k", type=int, default=3)
    parser.add_argument("--eval_limit", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle_data", dest="shuffle_data", action="store_true")
    parser.add_argument("--no_shuffle_data", dest="shuffle_data", action="store_false")
    parser.add_argument("--no_publish", action="store_true")

    # Backward-compatible knobs used by search scripts.
    parser.add_argument("--weight_gsm8k", type=float, default=1.0)
    parser.add_argument("--weight_ifeval", type=float, default=1.0)
    parser.add_argument("--weight_code", type=float, default=1.0)
    parser.add_argument("--max_gsm8k_examples", type=int, default=0, help="0 means no cap")
    parser.add_argument("--max_ifeval_examples", type=int, default=0, help="0 means no cap")
    parser.add_argument("--max_code_examples", type=int, default=0, help="0 means no cap")
    parser.set_defaults(shuffle_data=True)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    model_name = args.model
    max_seq_len = int(args.max_seq_len)

    if (
        tinker is None
        or types_mod is None
        or get_renderer_name is None
        or get_renderer is None
        or train_on_what is None
        or conversation_to_datum is None
        or get_tokenizer is None
    ):
        raise ModuleNotFoundError(
            "evaluation/train_and_publish.py requires Tinker and tinker-cookbook dependencies. "
            "Install project requirements first."
        )

    data_path = args.data_path or os.path.join(EVAL_DIR, "..", "data", "my_mixed_training_data.jsonl")
    data_path = os.path.abspath(data_path)

    print(f"Model: {model_name}")
    print(f"Seed: {args.seed} | Shuffle data: {args.shuffle_data}")
    print(f"Training data path: {data_path}")

    tokenizer = get_tokenizer(model_name)
    renderer_name = get_renderer_name(model_name)
    renderer = get_renderer(renderer_name, tokenizer)
    print(f"Renderer: {renderer_name}")

    caps = {
        "gsm8k": args.max_gsm8k_examples,
        "ifeval": args.max_ifeval_examples,
        "code": args.max_code_examples,
    }
    task_weights = {
        "gsm8k": max(float(args.weight_gsm8k), 0.0),
        "ifeval": max(float(args.weight_ifeval), 0.0),
        "code": max(float(args.weight_code), 0.0),
    }

    conversations_by_task: Dict[str, List[List[Dict[str, str]]]] = {"gsm8k": [], "ifeval": [], "code": []}
    with open(data_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            mapped = _row_to_task_and_conversation(row)
            if mapped is None:
                continue
            task_name, convo = mapped
            conversations_by_task[task_name].append(convo)

    rng = np.random.default_rng(args.seed)
    for task_name, cap in caps.items():
        rows = conversations_by_task[task_name]
        if cap > 0 and len(rows) > cap:
            indices = np.arange(len(rows))
            rng.shuffle(indices)
            keep = set(int(i) for i in indices[:cap])
            conversations_by_task[task_name] = [rows[i] for i in range(len(rows)) if i in keep]

    print("Loaded rows by task:")
    for task_name in ("gsm8k", "ifeval", "code"):
        print(f"  - {task_name}: {len(conversations_by_task[task_name])}")

    print(f"Tokenizing data (max_length={max_seq_len})...")
    data_by_task: Dict[str, List[Any]] = {"gsm8k": [], "ifeval": [], "code": []}
    dropped = 0
    for task_name, conversations in conversations_by_task.items():
        for convo in conversations:
            try:
                datum = conversation_to_datum(
                    convo,
                    renderer,
                    max_length=max_seq_len,
                    train_on_what=train_on_what.ALL_ASSISTANT_MESSAGES,
                )
                data_by_task[task_name].append(datum)
            except Exception:
                dropped += 1

    active_tasks = [t for t in ("gsm8k", "ifeval", "code") if data_by_task[t] and task_weights[t] > 0]
    if not active_tasks:
        raise RuntimeError("No usable training examples after parsing/tokenization/caps.")

    print("Tokenized rows by task:")
    for task_name in ("gsm8k", "ifeval", "code"):
        print(f"  - {task_name}: {len(data_by_task[task_name])}")
    print(f"Dropped malformed/unusable rows: {dropped}")
    print("Sampling weights:")
    for task_name in ("gsm8k", "ifeval", "code"):
        print(f"  - {task_name}: {task_weights[task_name]:.4f}")

    sampling_probs = np.array([task_weights[t] for t in active_tasks], dtype=float)
    sampling_probs /= sampling_probs.sum()

    print(f"Creating LoRA training client (rank={args.rank})...")
    sc = importlib.import_module("tinker").ServiceClient()
    tc = sc.create_lora_training_client(base_model=model_name, rank=args.rank)
    print("Training client ready")

    adam_params = importlib.import_module("tinker").types.AdamParams(
        learning_rate=args.lr,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
    )
    print(f"\nTraining for {args.num_steps} steps (batch_size={args.batch_size}, lr={args.lr})...")
    if args.checkpoint_every > 0:
        print(f"Periodic checkpoints every {args.checkpoint_every} steps.")
    if args.save_every_steps > 0:
        print(f"Additional compatibility checkpoints every {args.save_every_steps} steps.")

    task_indices = {task: np.arange(len(data_by_task[task])) for task in active_tasks}
    task_cursors = {task: 0 for task in active_tasks}
    running_task_counts = {task: 0 for task in active_tasks}
    if args.shuffle_data:
        for task in active_tasks:
            rng.shuffle(task_indices[task])

    checkpoints: List[Dict[str, Any]] = []
    saved_checkpoints: List[Dict[str, Any]] = []
    loss = float("nan")

    for step in range(args.num_steps):
        batch = []
        batch_task_names = []
        for _ in range(args.batch_size):
            task = str(rng.choice(active_tasks, p=sampling_probs))
            idxs = task_indices[task]
            cursor = task_cursors[task]
            if cursor >= len(idxs):
                cursor = 0
                if args.shuffle_data:
                    rng.shuffle(idxs)
                task_indices[task] = idxs
            datum = data_by_task[task][int(idxs[cursor])]
            task_cursors[task] = cursor + 1
            batch.append(datum)
            batch_task_names.append(task)
            running_task_counts[task] += 1

        fwd_bwd_future = tc.forward_backward(batch, loss_fn="cross_entropy")
        optim_future = tc.optim_step(adam_params)
        fwd_bwd_result = fwd_bwd_future.result()
        optim_future.result()

        logprobs = np.concatenate([o["logprobs"].tolist() for o in fwd_bwd_result.loss_fn_outputs])
        weights = np.concatenate([d.loss_fn_inputs["weights"].tolist() for d in batch])
        loss = float(-np.dot(logprobs, weights) / max(weights.sum(), 1))

        if step % 10 == 0 or step == args.num_steps - 1:
            totals = ", ".join(f"{t}:{running_task_counts[t]}" for t in active_tasks)
            in_batch = ", ".join(batch_task_names[: min(6, len(batch_task_names))])
            print(
                f"  Step {step + 1}/{args.num_steps} | Loss: {loss:.4f} | "
                f"sampled(total)={totals} | batch preview={in_batch}"
            )

        step_num = step + 1
        should_save_compat = (
            args.save_every_steps > 0
            and step_num % args.save_every_steps == 0
            and step_num != args.num_steps
        )
        if should_save_compat:
            name = f"{args.checkpoint_name}_step{step_num}"
            print(f"\nSaving compatibility checkpoint '{name}'...")
            ckpt = tc.save_weights_for_sampler(name=name).result()
            saved_checkpoints.append(
                {
                    "name": name,
                    "path": ckpt.path,
                    "step": step_num,
                    "type": "compat_periodic",
                    "train_loss": loss,
                    "published": False,
                }
            )
            print(f"  Saved: {ckpt.path}")

        if args.checkpoint_every > 0 and step_num % args.checkpoint_every == 0:
            name = f"{args.checkpoint_name}_step_{step_num}"
            print(f"\nSaving periodic checkpoint '{name}'...")
            ckpt = tc.save_weights_for_sampler(name=name).result()
            checkpoints.append(
                {
                    "name": name,
                    "path": ckpt.path,
                    "step": step_num,
                    "type": "periodic",
                    "train_loss": loss,
                }
            )
            print(f"  Saved: {ckpt.path}")

    print(f"\nSaving final checkpoint '{args.checkpoint_name}'...")
    final_ckpt = tc.save_weights_for_sampler(name=args.checkpoint_name).result()
    checkpoint_path = final_ckpt.path
    checkpoints.append(
        {
            "name": args.checkpoint_name,
            "path": checkpoint_path,
            "step": args.num_steps,
            "type": "final",
            "train_loss": loss,
        }
    )
    saved_checkpoints.append(
        {
            "name": args.checkpoint_name,
            "path": checkpoint_path,
            "step": args.num_steps,
            "type": "final",
            "train_loss": loss,
            "published": False,
        }
    )
    print(f"  Saved: {checkpoint_path}")

    candidate_eval = None
    if args.evaluate_top_k > 0 and checkpoints:
        from evaluation.eval_all import run_core

        candidates = sorted(checkpoints, key=lambda cp: cp.get("train_loss", float("inf")))
        candidates = candidates[: min(args.evaluate_top_k, len(candidates))]
        print("\nEvaluating candidate checkpoints...")

        ranked_results = []
        for cp in candidates:
            print(f"  - {cp['name']} ({cp['path']})")
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

        ranked_results.sort(
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

        print("\nCandidate ranking:")
        for i, row in enumerate(ranked_results, start=1):
            s = row["headline_scores"]
            print(
                f"  {i}. {row['name']} | avg={_format_score(row['average_score'])} | "
                f"IFEval={_format_score(s['ifeval_score'])} | "
                f"GSM8K={_format_score(s['gsm8k_accuracy'])} | "
                f"HumanEval={_format_score(s['humaneval_score'])}"
            )
    else:
        print("\nSkipping candidate checkpoint evaluation.")

    published = False
    if not args.no_publish:
        print("\nPublishing final checkpoint...")
        rest_client = sc.create_rest_client()
        rest_client.publish_checkpoint_from_tinker_path(checkpoint_path).result()
        published = True
        if saved_checkpoints:
            saved_checkpoints[-1]["published"] = True
        print("  Published successfully.")
    else:
        print("\nSkipping publish (--no_publish).")

    info = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "checkpoint_path": checkpoint_path,
        "checkpoints": checkpoints,
        "saved_checkpoints": saved_checkpoints,
        "base_model": model_name,
        "renderer_name": renderer_name,
        "data_path": data_path,
        "sampling_weights": task_weights,
        "stage2_sampling_weights": None,
        "task_example_caps": {
            "gsm8k": args.max_gsm8k_examples,
            "ifeval": args.max_ifeval_examples,
            "code": args.max_code_examples,
        },
        "training": {
            "data_path": data_path,
            "num_steps": args.num_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "lora_rank": args.rank,
            "max_seq_len": max_seq_len,
            "checkpoint_every": args.checkpoint_every,
            "save_every_steps": args.save_every_steps,
            "seed": args.seed,
            "shuffle_data": args.shuffle_data,
        },
        "data_stats": {
            "active_tasks": active_tasks,
            "rows_per_task_before_tokenization": {
                "gsm8k": len(conversations_by_task["gsm8k"]),
                "ifeval": len(conversations_by_task["ifeval"]),
                "code": len(conversations_by_task["code"]),
            },
            "rows_per_task_after_tokenization": {
                "gsm8k": len(data_by_task["gsm8k"]),
                "ifeval": len(data_by_task["ifeval"]),
                "code": len(data_by_task["code"]),
            },
            "dropped_rows": dropped,
        },
        "candidate_evaluation": candidate_eval,
        "published": published,
    }

    info_path = os.path.join(EVAL_DIR, "checkpoint_info.json")
    with open(info_path, "w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2)
    print(f"\nCheckpoint info saved to {info_path}")
    print("\nNext: evaluate with")
    print(f'  PYTHONPATH=. python evaluation/eval_all.py --checkpoint_path "{checkpoint_path}" --base_model {model_name}')


if __name__ == "__main__":
    main()
