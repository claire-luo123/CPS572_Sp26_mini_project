"""
Train a multi-task LoRA checkpoint for IFEval + GSM8K + HumanEval.

This script expects a JSONL mix that contains rows in one of these schemas:
  - GSM8K-style: {"question": "...", "answer": "..."}
  - IF-style: {"messages": [{"role": "...", "content": "..."}, ...]}
  - Code-style: {"instruction": "...", "output": "..."}

Compared to the original toy example, this version:
  - tracks per-task examples
  - tokenizes per task
  - trains with weighted task sampling (prevents one task from dominating)
  - exposes practical hyperparameters via CLI flags
"""

import argparse
import json
import os
import re

import numpy as np

try:
    import tinker
    from tinker import types
    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.supervised.data import conversation_to_datum
    from tinker_cookbook.tokenizer_utils import get_tokenizer
except ModuleNotFoundError:
    tinker = None
    types = None
    model_info = None
    renderers = None
    conversation_to_datum = None
    get_tokenizer = None

DEFAULT_MODEL = "meta-llama/Llama-3.2-3B"
DEFAULT_MAX_SEQ_LEN = 1280

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))


def _normalize_message(msg: dict) -> dict | None:
    role = msg.get("role")
    content = msg.get("content")
    if role not in {"system", "user", "assistant"}:
        return None
    if not isinstance(content, str):
        return None
    content = content.strip()
    if not content:
        return None
    return {"role": role, "content": content}


def _strip_code_fences(text: str) -> str:
    """Remove outer markdown code fences while preserving code body."""
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_+-]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s)
    return s.strip()


def _row_to_task_conversation(row: dict) -> tuple[str, list[dict]] | None:
    if "question" in row and "answer" in row:
        q = str(row["question"]).strip()
        a = str(row["answer"]).strip()
        if q and a:
            q = f"Solve step by step, and end with '#### <final answer>'.\n\n{q}"
            return "gsm8k", [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        return None

    if "messages" in row and isinstance(row["messages"], list):
        convo = []
        for m in row["messages"]:
            if not isinstance(m, dict):
                return None
            nm = _normalize_message(m)
            if nm is None:
                return None
            convo.append(nm)

        has_user = any(m["role"] == "user" for m in convo)
        has_assistant = any(m["role"] == "assistant" for m in convo)
        if has_user and has_assistant:
            return "ifeval", convo
        return None

    if "instruction" in row and "output" in row:
        instr = str(row["instruction"]).strip()
        out = _strip_code_fences(str(row["output"]))
        if instr and out:
            return "code", [{"role": "user", "content": instr}, {"role": "assistant", "content": out}]
        return None

    return None


def _load_task_conversations(data_path: str) -> dict[str, list[list[dict]]]:
    task_conversations = {"gsm8k": [], "ifeval": [], "code": []}
    skipped = 0

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                skipped += 1
                continue

            mapped = _row_to_task_conversation(row)
            if mapped is None:
                skipped += 1
                continue

            task, convo = mapped
            task_conversations[task].append(convo)

    total = sum(len(v) for v in task_conversations.values())
    print(f"Loaded {total} conversations from {data_path}")
    print(
        "  Task split:"
        f" gsm8k={len(task_conversations['gsm8k'])},"
        f" ifeval={len(task_conversations['ifeval'])},"
        f" code={len(task_conversations['code'])}"
    )
    if skipped:
        print(f"  Skipped {skipped} malformed/unusable rows")
    return task_conversations


def _prepare_task_data(
    task_conversations: dict[str, list[list[dict]]],
    renderer,
    max_seq_len: int,
    rng: np.random.Generator,
    max_examples_per_task: int,
    max_examples_by_task: dict[str, int],
) -> dict[str, list]:
    task_data: dict[str, list] = {}

    for task, conversations in task_conversations.items():
        if not conversations:
            task_data[task] = []
            continue

        shuffled = list(conversations)
        rng.shuffle(shuffled)

        task_cap = max_examples_by_task.get(task, 0)
        if task_cap > 0:
            shuffled = shuffled[:task_cap]
        elif max_examples_per_task > 0:
            shuffled = shuffled[:max_examples_per_task]

        prepared = []
        dropped = 0
        for convo in shuffled:
            try:
                datum = conversation_to_datum(
                    convo,
                    renderer,
                    max_length=max_seq_len,
                    train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES,
                )
                prepared.append(datum)
            except Exception:
                dropped += 1

        task_data[task] = prepared
        print(f"  {task}: tokenized={len(prepared)}, dropped={dropped}")

    return task_data


def _sample_batch(
    task_data: dict[str, list],
    tasks: list[str],
    probs: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
):
    task_choices = rng.choice(tasks, size=batch_size, p=probs)
    batch = []
    per_task_counts = {t: 0 for t in tasks}
    for task in task_choices:
        pool = task_data[task]
        idx = int(rng.integers(0, len(pool)))
        batch.append(pool[idx])
        per_task_counts[task] += 1
    return batch, per_task_counts


def _weights_to_probs(tasks: list[str], weight_map: dict[str, float]) -> np.ndarray:
    raw_weights = np.array([max(weight_map.get(task, 0.0), 0.0) for task in tasks], dtype=float)
    if raw_weights.sum() <= 0:
        raw_weights = np.ones_like(raw_weights)
    return raw_weights / raw_weights.sum()


def _build_stage_weight_map(
    stage1_weights: dict[str, float],
    gsm8k_weight: float | None,
    ifeval_weight: float | None,
    code_weight: float | None,
) -> dict[str, float]:
    return {
        "gsm8k": stage1_weights["gsm8k"] if gsm8k_weight is None else max(gsm8k_weight, 0.0),
        "ifeval": stage1_weights["ifeval"] if ifeval_weight is None else max(ifeval_weight, 0.0),
        "code": stage1_weights["code"] if code_weight is None else max(code_weight, 0.0),
    }


def main():
    parser = argparse.ArgumentParser(description="Train, save, and publish a multi-task LoRA checkpoint")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Base model")
    parser.add_argument("--data_path", type=str, default=os.path.join(EVAL_DIR, "..", "data", "my_mixed_training_data.jsonl"))
    parser.add_argument("--max_seq_len", type=int, default=DEFAULT_MAX_SEQ_LEN, help="Tokenized max sequence length")
    parser.add_argument("--max_examples_per_task", type=int, default=12000, help="Global cap per task after shuffling; <=0 uses all unless per-task caps are set")
    parser.add_argument("--max_gsm8k_examples", type=int, default=0, help="Per-task cap for GSM8K; >0 overrides --max_examples_per_task")
    parser.add_argument("--max_ifeval_examples", type=int, default=0, help="Per-task cap for IFEval; >0 overrides --max_examples_per_task")
    parser.add_argument("--max_code_examples", type=int, default=0, help="Per-task cap for code; >0 overrides --max_examples_per_task")
    parser.add_argument("--num_steps", type=int, default=6500, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size")
    parser.add_argument("--lr", type=float, default=6e-5, help="Learning rate")
    parser.add_argument("--rank", type=int, default=64, help="LoRA rank")
    parser.add_argument("--weight_gsm8k", type=float, default=1.35, help="Sampling weight for GSM8K task")
    parser.add_argument("--weight_ifeval", type=float, default=1.10, help="Sampling weight for IFEval-style task")
    parser.add_argument("--weight_code", type=float, default=1.2, help="Sampling weight for code task")
    parser.add_argument(
        "--stage2_start",
        type=int,
        default=0,
        help="1-based training step where sampling switches to stage-2 weights; <=0 disables",
    )
    parser.add_argument("--stage2_weight_gsm8k", type=float, default=None, help="Optional stage-2 GSM8K weight")
    parser.add_argument("--stage2_weight_ifeval", type=float, default=None, help="Optional stage-2 IFEval weight")
    parser.add_argument("--stage2_weight_code", type=float, default=None, help="Optional stage-2 code weight")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data shuffling/sampling")
    parser.add_argument(
        "--save_every_steps",
        type=int,
        default=0,
        help="If >0, save an intermediate checkpoint every N steps in addition to the final one",
    )
    parser.add_argument("--checkpoint_name", type=str, default="mixed_sft_3b_tuned_v2", help="Checkpoint name")
    parser.add_argument("--no_publish", action="store_true", help="Skip publishing")
    args = parser.parse_args()

    if tinker is None or types is None or model_info is None or renderers is None or conversation_to_datum is None or get_tokenizer is None:
        raise ModuleNotFoundError(
            "evaluation/train_and_publish.py requires Tinker and tinker-cookbook dependencies. "
            "Install project requirements first."
        )

    rng = np.random.default_rng(args.seed)

    # Setup
    print(f"Model: {args.model}")
    tokenizer = get_tokenizer(args.model)
    renderer_name = model_info.get_recommended_renderer_name(args.model)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    print(f"Renderer: {renderer_name}")

    # Load + tokenize by task
    print("Loading mixed training data...")
    task_conversations = _load_task_conversations(args.data_path)
    print(f"Tokenizing training data by task (max_seq_len={args.max_seq_len})...")
    task_data = _prepare_task_data(
        task_conversations=task_conversations,
        renderer=renderer,
        max_seq_len=args.max_seq_len,
        rng=rng,
        max_examples_per_task=args.max_examples_per_task,
        max_examples_by_task={
            "gsm8k": args.max_gsm8k_examples,
            "ifeval": args.max_ifeval_examples,
            "code": args.max_code_examples,
        },
    )

    active_tasks = [t for t, v in task_data.items() if len(v) > 0]
    if not active_tasks:
        raise RuntimeError("No valid training data was prepared. Check data format/path.")

    task_weights = {
        "gsm8k": max(args.weight_gsm8k, 0.0),
        "ifeval": max(args.weight_ifeval, 0.0),
        "code": max(args.weight_code, 0.0),
    }
    probs = _weights_to_probs(active_tasks, task_weights)
    print(
        "Stage 1 sampling probabilities: "
        + ", ".join(f"{t}={p:.3f}" for t, p in zip(active_tasks, probs))
    )

    stage2_weights = None
    stage2_probs = None
    if args.stage2_start > 0:
        stage2_weights = _build_stage_weight_map(
            stage1_weights=task_weights,
            gsm8k_weight=args.stage2_weight_gsm8k,
            ifeval_weight=args.stage2_weight_ifeval,
            code_weight=args.stage2_weight_code,
        )
        stage2_probs = _weights_to_probs(active_tasks, stage2_weights)
        print(
            "Stage 2 sampling probabilities: "
            + ", ".join(f"{t}={p:.3f}" for t, p in zip(active_tasks, stage2_probs))
        )

    # Create training client
    print(f"Creating LoRA training client (rank={args.rank})...")
    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=args.model, rank=args.rank)
    print("  Training client ready")

    # Train
    adam_params = types.AdamParams(learning_rate=args.lr, beta1=0.9, beta2=0.95, eps=1e-8)
    print(f"\nTraining for {args.num_steps} steps (batch_size={args.batch_size}, lr={args.lr})...")
    running_task_counts = {t: 0 for t in active_tasks}
    stage_task_counts = {
        1: {t: 0 for t in active_tasks},
        2: {t: 0 for t in active_tasks},
    }
    saved_checkpoints: list[dict] = []

    for step in range(args.num_steps):
        in_stage2 = args.stage2_start > 0 and (step + 1) >= args.stage2_start
        current_stage = 2 if in_stage2 else 1
        current_probs = stage2_probs if in_stage2 and stage2_probs is not None else probs
        if (step + 1) == args.stage2_start and args.stage2_start > 0:
            print(f"\nSwitching to stage 2 sampling at step {step + 1}...")

        batch, step_counts = _sample_batch(
            task_data=task_data,
            tasks=active_tasks,
            probs=current_probs,
            batch_size=args.batch_size,
            rng=rng,
        )
        for t, c in step_counts.items():
            running_task_counts[t] += c
            stage_task_counts[current_stage][t] += c

        fwd_bwd_future = tc.forward_backward(batch, loss_fn="cross_entropy")
        optim_future = tc.optim_step(adam_params)

        fwd_bwd_result = fwd_bwd_future.result()
        optim_future.result()

        # Compute loss
        logprobs = np.concatenate([o["logprobs"].tolist() for o in fwd_bwd_result.loss_fn_outputs])
        weights = np.concatenate([d.loss_fn_inputs["weights"].tolist() for d in batch])
        loss = -np.dot(logprobs, weights) / max(weights.sum(), 1)

        # Print periodic training signal + realized task mix.
        if step % 10 == 0 or step == args.num_steps - 1:
            task_mix = ", ".join(f"{t}:{running_task_counts[t]}" for t in active_tasks)
            stage_mix = ", ".join(f"{t}:{stage_task_counts[current_stage][t]}" for t in active_tasks)
            print(
                f"  Step {step+1}/{args.num_steps} | Stage {current_stage} | Loss: {loss:.4f}"
                f" | Total seen -> {task_mix} | Stage seen -> {stage_mix}"
            )

        if args.save_every_steps > 0 and (step + 1) % args.save_every_steps == 0 and step != args.num_steps - 1:
            interval_name = f"{args.checkpoint_name}_step{step + 1}"
            print(f"\nSaving intermediate checkpoint '{interval_name}'...")
            interval_ckpt = tc.save_weights_for_sampler(name=interval_name).result()
            saved_checkpoints.append(
                {
                    "step": step + 1,
                    "name": interval_name,
                    "path": interval_ckpt.path,
                    "published": False,
                    "final": False,
                }
            )
            print(f"  Intermediate checkpoint saved: {interval_ckpt.path}")

    # Save checkpoint
    print(f"\nSaving checkpoint '{args.checkpoint_name}'...")
    ckpt = tc.save_weights_for_sampler(name=args.checkpoint_name).result()
    checkpoint_path = ckpt.path
    print(f"  Checkpoint saved: {checkpoint_path}")
    saved_checkpoints.append(
        {
            "step": args.num_steps,
            "name": args.checkpoint_name,
            "path": checkpoint_path,
            "published": False,
            "final": True,
        }
    )

    # Publish
    if not args.no_publish:
        print("\nPublishing checkpoint...")
        rest_client = sc.create_rest_client()
        rest_client.publish_checkpoint_from_tinker_path(checkpoint_path).result()
        print("  Published successfully!")
        saved_checkpoints[-1]["published"] = True
    else:
        print("\nSkipping publish (--no_publish).")

    # Save checkpoint info
    info = {
        "checkpoint_path": checkpoint_path,
        "base_model": args.model,
        "renderer_name": renderer_name,
        "data_path": args.data_path,
        "sampling_weights": task_weights,
        "stage2_sampling_weights": stage2_weights,
        "task_example_caps": {
            "gsm8k": args.max_gsm8k_examples,
            "ifeval": args.max_ifeval_examples,
            "code": args.max_code_examples,
        },
        "training": {
            "num_steps": args.num_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "lora_rank": args.rank,
            "max_seq_len": args.max_seq_len,
            "max_examples_per_task": args.max_examples_per_task,
            "save_every_steps": args.save_every_steps,
            "stage2_start": args.stage2_start,
            "seed": args.seed,
        },
        "saved_checkpoints": saved_checkpoints,
        "published": not args.no_publish,
    }
    info_path = os.path.join(EVAL_DIR, "checkpoint_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nCheckpoint info saved to {info_path}")
    print(f"\nNext: evaluate your checkpoint with")
    print(f"  PYTHONPATH=. python evaluation/eval_all.py --checkpoint_path \"{checkpoint_path}\" --base_model {args.model}")

if __name__ == "__main__":
    main()
