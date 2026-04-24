import argparse
import itertools
import json
import os
import subprocess
import sys
import tempfile

BASELINES = {
    "ifeval": 0.45,
    "gsm8k": 0.50,
    "humaneval": 0.30,
}

DEFAULT_WEIGHT_TRIPLETS = [
    (1.0, 1.0, 1.0),
    (1.6, 1.0, 1.0),
    (1.4, 1.0, 1.2),
    (1.5, 0.9, 1.2),
    (1.3, 1.1, 1.2),
    (1.2, 1.2, 1.2),
    (1.1, 1.3, 1.3),
    (0.9, 1.3, 1.4),
]

DEFAULT_TASK_CAPS = [
    (7473, 7473, 7473),
    (7473, 10000, 10000),
    (7473, 12000, 12000),
    (7473, 15000, 12000),
    (7473, 12000, 15000),
    (7473, 15000, 15000),
]


def parse_triplets(text):
    triplets = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values = tuple(float(part) for part in chunk.split(":"))
        if len(values) != 3:
            raise ValueError(f"Expected 3 values per weight triplet, got: {chunk}")
        triplets.append(values)
    return triplets


def parse_caps(text):
    caps = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values = tuple(int(part) for part in chunk.split(":"))
        if len(values) != 3:
            raise ValueError(f"Expected 3 values per cap triplet, got: {chunk}")
        caps.append(values)
    return caps


def find_key(data, target_key):
    if isinstance(data, dict):
        if target_key in data:
            return data[target_key]
        for value in data.values():
            result = find_key(value, target_key)
            if result is not None:
                return result
    return None


def parse_metrics(eval_results):
    return {
        "ifeval": float(find_key(eval_results, "google/IFEval/final_acc") or 0.0),
        "gsm8k": float(find_key(eval_results, "openai/gsm8k/accuracy") or 0.0),
        "humaneval": float(find_key(eval_results, "openai/openai_humaneval/accuracy") or 0.0),
    }


def compute_score(metrics):
    ratios = {task: metrics[task] / BASELINES[task] for task in BASELINES}
    avg_ratio = sum(ratios.values()) / len(ratios)
    min_ratio = min(ratios.values())
    score = 0.6 * avg_ratio + 0.4 * min_ratio
    return score, ratios, avg_ratio, min_ratio


def save_results(path, results):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


def build_train_command(args, weights, caps, checkpoint_name):
    w_gsm8k, w_ifeval, w_code = weights
    cap_gsm8k, cap_ifeval, cap_code = caps

    command = [
        sys.executable,
        "evaluation/train_and_publish.py",
        "--model",
        args.model,
        "--data_path",
        args.data_path,
        "--num_steps",
        str(args.num_steps),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--rank",
        str(args.rank),
        "--max_seq_len",
        str(args.max_seq_len),
        "--weight_gsm8k",
        str(w_gsm8k),
        "--weight_ifeval",
        str(w_ifeval),
        "--weight_code",
        str(w_code),
        "--max_gsm8k_examples",
        str(cap_gsm8k),
        "--max_ifeval_examples",
        str(cap_ifeval),
        "--max_code_examples",
        str(cap_code),
        "--checkpoint_name",
        checkpoint_name,
        "--no_publish",
    ]

    if args.save_every_steps > 0:
        command.extend(["--save_every_steps", str(args.save_every_steps)])

    return command


def build_eval_command(args, checkpoint_path, output_path):
    command = [
        sys.executable,
        "evaluation/eval_all.py",
        "--checkpoint_path",
        checkpoint_path,
        "--base_model",
        args.model,
        "--output_path",
        output_path,
        "--temperature",
        str(args.temperature),
        "--top_p",
        str(args.top_p),
    ]
    if args.eval_limit > 0:
        command.extend(["--limit", str(args.eval_limit)])
    return command


def main():
    parser = argparse.ArgumentParser(description="Search only over data-mixing weights and task caps.")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--data_path", type=str, default=os.path.join("data", "my_mixed_training_data.jsonl"))
    parser.add_argument("--num_steps", type=int, default=800)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--max_seq_len", type=int, default=1280)
    parser.add_argument("--eval_limit", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--save_every_steps", type=int, default=0)
    parser.add_argument(
        "--weight_triplets",
        type=str,
        default=None,
        help="Comma-separated weight triplets like '1.0:1.0:1.0,1.5:1.0:1.2'",
    )
    parser.add_argument(
        "--task_caps",
        type=str,
        default=None,
        help="Comma-separated cap triplets like '7473:7473:7473,7473:12000:12000'",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default=os.path.join("evaluation", "mix_search_results.json"),
    )
    args = parser.parse_args()

    weight_triplets = parse_triplets(args.weight_triplets) if args.weight_triplets else DEFAULT_WEIGHT_TRIPLETS
    task_caps = parse_caps(args.task_caps) if args.task_caps else DEFAULT_TASK_CAPS
    combinations = list(itertools.product(weight_triplets, task_caps))

    print(f"Starting mixing-only search over {len(combinations)} combinations...\n")
    repo_root = os.path.abspath(os.path.dirname(__file__))
    all_results = []
    best_result = None

    for index, (weights, caps) in enumerate(combinations, start=1):
        checkpoint_name = (
            f"mixonly_w{weights[0]:.1f}-{weights[1]:.1f}-{weights[2]:.1f}"
            f"_cap{caps[0]}-{caps[1]}-{caps[2]}"
            f"_s{args.num_steps}"
        )
        print("=" * 72)
        print(f"[{index}/{len(combinations)}] Testing {checkpoint_name}")
        print(
            f"weights={weights} | caps={caps} | steps={args.num_steps} | "
            f"lr={args.lr:g} | rank={args.rank} | eval_limit={args.eval_limit}"
        )
        print("=" * 72)

        train_cmd = build_train_command(args, weights, caps, checkpoint_name)
        try:
            subprocess.run(train_cmd, cwd=repo_root, check=True, env=dict(os.environ, PYTHONPATH=repo_root))
        except subprocess.CalledProcessError as exc:
            print(f"Training failed for {checkpoint_name}: {exc}")
            all_results.append({"weights": weights, "caps": caps, "status": "train_failed"})
            save_results(args.results_path, all_results)
            continue

        info_path = os.path.join(repo_root, "evaluation", "checkpoint_info.json")
        try:
            with open(info_path, "r", encoding="utf-8") as handle:
                checkpoint_info = json.load(handle)
            checkpoint_path = checkpoint_info["checkpoint_path"]
        except Exception as exc:
            print(f"Skipping eval for {checkpoint_name}: could not read checkpoint info ({exc})")
            all_results.append({"weights": weights, "caps": caps, "status": "missing_checkpoint_info"})
            save_results(args.results_path, all_results)
            continue

        with tempfile.TemporaryDirectory() as tmpdir:
            eval_output_path = os.path.join(tmpdir, "submission.json")
            eval_cmd = build_eval_command(args, checkpoint_path, eval_output_path)
            try:
                subprocess.run(eval_cmd, cwd=repo_root, check=True, env=dict(os.environ, PYTHONPATH=repo_root))
                with open(eval_output_path, "r", encoding="utf-8") as handle:
                    eval_results = json.load(handle)
            except Exception as exc:
                print(f"Evaluation failed for {checkpoint_name}: {exc}")
                all_results.append(
                    {
                        "weights": weights,
                        "caps": caps,
                        "checkpoint_path": checkpoint_path,
                        "status": "eval_failed",
                    }
                )
                save_results(args.results_path, all_results)
                continue

        metrics = parse_metrics(eval_results)
        score, ratios, avg_ratio, min_ratio = compute_score(metrics)
        result = {
            "weights": weights,
            "caps": caps,
            "checkpoint_path": checkpoint_path,
            "status": "ok",
            "metrics": metrics,
            "ratios_to_baseline": ratios,
            "avg_ratio_to_baseline": avg_ratio,
            "min_ratio_to_baseline": min_ratio,
            "balanced_score": score,
            "training": {
                "model": args.model,
                "num_steps": args.num_steps,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "rank": args.rank,
            },
        }
        all_results.append(result)

        print(
            "scores: "
            f"IFEval={metrics['ifeval'] * 100:.1f}% | "
            f"GSM8K={metrics['gsm8k'] * 100:.1f}% | "
            f"HumanEval={metrics['humaneval'] * 100:.1f}%"
        )
        print(
            "baseline ratios: "
            f"IFEval={ratios['ifeval']:.3f} | GSM8K={ratios['gsm8k']:.3f} | "
            f"HumanEval={ratios['humaneval']:.3f} | "
            f"avg={avg_ratio:.3f} | min={min_ratio:.3f} | balanced_score={score:.3f}"
        )

        if best_result is None:
            best_result = result
        else:
            best_key = (
                best_result["min_ratio_to_baseline"],
                best_result["avg_ratio_to_baseline"],
                best_result["balanced_score"],
            )
            current_key = (min_ratio, avg_ratio, score)
            if current_key > best_key:
                best_result = result

        save_results(args.results_path, all_results)

    print("\n" + "*" * 72)
    print("MIXING-ONLY SEARCH COMPLETE")
    print("*" * 72)
    if best_result is None:
        print("No successful run completed.")
        return

    print(f"Best checkpoint: {best_result['checkpoint_path']}")
    print(f"Best weights: {best_result['weights']}")
    print(f"Best caps: {best_result['caps']}")
    print(
        "Best scores: "
        f"IFEval={best_result['metrics']['ifeval'] * 100:.1f}% | "
        f"GSM8K={best_result['metrics']['gsm8k'] * 100:.1f}% | "
        f"HumanEval={best_result['metrics']['humaneval'] * 100:.1f}%"
    )
    print(
        "Best baseline ratios: "
        f"IFEval={best_result['ratios_to_baseline']['ifeval']:.3f} | "
        f"GSM8K={best_result['ratios_to_baseline']['gsm8k']:.3f} | "
        f"HumanEval={best_result['ratios_to_baseline']['humaneval']:.3f} | "
        f"avg={best_result['avg_ratio_to_baseline']:.3f} | "
        f"min={best_result['min_ratio_to_baseline']:.3f} | "
        f"balanced_score={best_result['balanced_score']:.3f}"
    )
    print(f"Saved detailed results to {args.results_path}")


if __name__ == "__main__":
    main()
