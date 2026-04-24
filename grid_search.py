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
    (1.5, 1.0, 1.2),
    (1.7, 0.9, 1.2),
    (1.3, 1.1, 1.3),
    (1.2, 1.2, 1.2),
]

DEFAULT_TASK_CAPS = [
    (7473, 7473, 7473),
    (7473, 12000, 12000),
    (7473, 15000, 12000),
]

DEFAULT_LEARNING_RATES = [5e-5, 8e-5]
DEFAULT_RANKS = [64]
DEFAULT_NUM_STEPS = [2500, 4000]


def _pick_metric(metrics, preferred_substrings):
    for needle in preferred_substrings:
        for key, value in metrics.items():
            if needle in key.lower() and isinstance(value, (int, float)):
                return float(value)
    for value in metrics.values():
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def parse_metrics(eval_results):
    ifeval_metrics = eval_results.get("ifeval", {}).get("metrics", {})
    gsm8k_metrics = eval_results.get("gsm8k", {}).get("metrics", {})
    humaneval_metrics = eval_results.get("humaneval", {}).get("metrics", {})
    return {
        "ifeval": _pick_metric(ifeval_metrics, ["prompt_level_strict", "accuracy", "acc"]),
        "gsm8k": _pick_metric(gsm8k_metrics, ["accuracy", "acc", "exact_match"]),
        "humaneval": _pick_metric(humaneval_metrics, ["pass@1", "accuracy", "acc"]),
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


def build_train_command(args, config):
    w_gsm8k, w_ifeval, w_code = config["weights"]
    cap_gsm8k, cap_ifeval, cap_code = config["caps"]

    command = [
        sys.executable,
        "evaluation/train_and_publish.py",
        "--model",
        args.model,
        "--num_steps",
        str(config["num_steps"]),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(config["lr"]),
        "--rank",
        str(config["rank"]),
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
        config["checkpoint_name"],
        "--no_publish",
    ]

    if args.data_path:
        command.extend(["--data_path", args.data_path])
    if args.max_seq_len:
        command.extend(["--max_seq_len", str(args.max_seq_len)])
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
    parser = argparse.ArgumentParser(description="Search over task mixing and core SFT hyperparameters.")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--data_path", type=str, default=None, help="Optional prebuilt training mix JSONL")
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--max_seq_len", type=int, default=1280)
    parser.add_argument("--eval_limit", type=int, default=50, help="Per-task eval limit during search; 0 means full eval")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--save_every_steps", type=int, default=0)
    parser.add_argument(
        "--results_path",
        type=str,
        default=os.path.join("evaluation", "grid_search_results.json"),
    )
    args = parser.parse_args()

    combinations = list(
        itertools.product(
            DEFAULT_WEIGHT_TRIPLETS,
            DEFAULT_TASK_CAPS,
            DEFAULT_LEARNING_RATES,
            DEFAULT_RANKS,
            DEFAULT_NUM_STEPS,
        )
    )
    print(f"Starting focused search over {len(combinations)} combinations...\n")

    repo_root = os.path.abspath(os.path.dirname(__file__))
    all_results = []
    best_result = None

    for index, (weights, caps, lr, rank, num_steps) in enumerate(combinations, start=1):
        checkpoint_name = (
            f"mix_w{weights[0]:.1f}-{weights[1]:.1f}-{weights[2]:.1f}"
            f"_cap{caps[0]}-{caps[1]}-{caps[2]}"
            f"_lr{lr:g}_r{rank}_s{num_steps}"
        )
        config = {
            "weights": weights,
            "caps": caps,
            "lr": lr,
            "rank": rank,
            "num_steps": num_steps,
            "checkpoint_name": checkpoint_name,
        }

        print("=" * 72)
        print(f"[{index}/{len(combinations)}] Testing {checkpoint_name}")
        print(
            f"weights={weights} | caps={caps} | lr={lr:g} | rank={rank} | "
            f"steps={num_steps} | eval_limit={args.eval_limit}"
        )
        print("=" * 72)

        train_cmd = build_train_command(args, config)
        try:
            subprocess.run(train_cmd, cwd=repo_root, check=True, env=dict(os.environ, PYTHONPATH=repo_root))
        except subprocess.CalledProcessError as exc:
            print(f"Training failed for {checkpoint_name}: {exc}")
            all_results.append({"config": config, "status": "train_failed"})
            save_results(args.results_path, all_results)
            continue

        info_path = os.path.join(repo_root, "evaluation", "checkpoint_info.json")
        try:
            with open(info_path, "r", encoding="utf-8") as handle:
                checkpoint_info = json.load(handle)
                checkpoint_path = checkpoint_info["checkpoint_path"]
        except Exception as exc:
            print(f"Skipping eval for {checkpoint_name}: could not read checkpoint info ({exc})")
            all_results.append({"config": config, "status": "missing_checkpoint_info"})
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
                        "config": config,
                        "checkpoint_path": checkpoint_path,
                        "status": "eval_failed",
                    }
                )
                save_results(args.results_path, all_results)
                continue

        metrics = parse_metrics(eval_results)
        score, ratios, avg_ratio, min_ratio = compute_score(metrics)
        result = {
            "config": config,
            "checkpoint_path": checkpoint_path,
            "status": "ok",
            "metrics": metrics,
            "ratios_to_baseline": ratios,
            "avg_ratio_to_baseline": avg_ratio,
            "min_ratio_to_baseline": min_ratio,
            "balanced_score": score,
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
    print("SEARCH COMPLETE")
    print("*" * 72)
    if best_result is None:
        print("No successful run completed.")
        return

    print(f"Best checkpoint: {best_result['checkpoint_path']}")
    print(f"Best config: {json.dumps(best_result['config'], indent=2)}")
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
