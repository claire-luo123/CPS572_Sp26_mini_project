"""Compare two checkpoints by running eval_all and printing key metric deltas."""

import argparse
import json
import os
import subprocess
import sys
import tempfile


def run_eval(
    repo_root: str,
    checkpoint_path: str,
    base_model: str,
    limit: int | None,
    temperature: float,
    top_p: float,
    output_path: str,
) -> dict:
    cmd = [
        sys.executable,
        "-m",
        "evaluation.eval_all",
        "--checkpoint_path",
        checkpoint_path,
        "--base_model",
        base_model,
        "--temperature",
        str(temperature),
        "--top_p",
        str(top_p),
        "--output_path",
        output_path,
    ]
    if limit is not None and limit > 0:
        cmd.extend(["--limit", str(limit)])

    env = os.environ.copy()
    env["PYTHONPATH"] = repo_root
    subprocess.run(cmd, cwd=repo_root, check=True, env=env)

    with open(output_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_metrics(submission: dict) -> dict:
    return {
        "IFEval final_acc": submission["ifeval"]["metrics"]["google/IFEval/final_acc"],
        "GSM8K accuracy": submission["gsm8k"]["metrics"]["openai/gsm8k/accuracy"],
        "HumanEval accuracy": submission["humaneval"]["metrics"]["openai/openai_humaneval/accuracy"],
    }


def print_table(old_metrics: dict, new_metrics: dict) -> None:
    print()
    print(f"{'Metric':<20} {'old':>10} {'new':>10} {'delta(new-old)':>16}")
    print("-" * 58)
    for metric in ["IFEval final_acc", "GSM8K accuracy", "HumanEval accuracy"]:
        old_v = old_metrics[metric]
        new_v = new_metrics[metric]
        delta = new_v - old_v
        print(f"{metric:<20} {old_v:>10.6f} {new_v:>10.6f} {delta:>+16.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two checkpoints via eval_all.")
    parser.add_argument("--old_checkpoint_path", required=True)
    parser.add_argument("--new_checkpoint_path", required=True)
    parser.add_argument("--old_base_model", required=True)
    parser.add_argument("--new_base_model", required=True)
    parser.add_argument("--limit", type=int, default=100, help="Max samples per task; 0 means full eval")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    with tempfile.TemporaryDirectory() as tmpdir:
        old_out = os.path.join(tmpdir, "old_submission.json")
        new_out = os.path.join(tmpdir, "new_submission.json")

        old_submission = run_eval(
            repo_root=repo_root,
            checkpoint_path=args.old_checkpoint_path,
            base_model=args.old_base_model,
            limit=args.limit,
            temperature=args.temperature,
            top_p=args.top_p,
            output_path=old_out,
        )
        new_submission = run_eval(
            repo_root=repo_root,
            checkpoint_path=args.new_checkpoint_path,
            base_model=args.new_base_model,
            limit=args.limit,
            temperature=args.temperature,
            top_p=args.top_p,
            output_path=new_out,
        )

    old_metrics = extract_metrics(old_submission)
    new_metrics = extract_metrics(new_submission)
    print_table(old_metrics, new_metrics)


if __name__ == "__main__":
    main()
