"""
Run the full train -> checkpoint selection -> final eval pipeline.

This script automates:
1) Short training run with optional periodic checkpoints
2) Quick evaluation pass on all checkpoints
3) Best-checkpoint selection using 3 headline task scores
4) Full evaluation on the selected checkpoint
5) Structured experiment logging (JSON + Markdown)

Example:
    PYTHONPATH=. python evaluation/run_full_experiment.py --num_steps 100 --checkpoint_every 25
"""

import argparse
import asyncio
import importlib
import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent
EVAL_ENV_CONSTRAINTS = EVAL_DIR / "eval_env_constraints.txt"
REQUIRED_EVAL_VERSIONS = {
    "inspect-ai": "0.3.170",
    "inspect-evals": "0.3.106",
    "openai": "2.30.0",
}


def run_cmd(cmd: List[str], cwd: Path) -> None:
    print(f"\n>>> Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def assert_eval_environment_ok() -> None:
    issues = []
    for pkg, expected in REQUIRED_EVAL_VERSIONS.items():
        try:
            installed = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            issues.append(f"{pkg} is not installed (expected {expected})")
            continue
        if installed != expected:
            issues.append(f"{pkg}=={installed} installed, expected {expected}")

    # Catch partially broken OpenAI installs (seen in prior runs).
    module_checks = [
        "openai",
        "openai.types.shared_params.metadata",
        "inspect_evals._registry",
        "torch",
        "transformers.models.auto.tokenization_auto",
    ]
    for mod in module_checks:
        try:
            importlib.import_module(mod)
        except Exception as exc:
            issues.append(f"cannot import {mod}: {exc}")

    if issues:
        fix_cmd = (
            f'uv pip install --python "{sys.executable}" --force-reinstall -r "{EVAL_ENV_CONSTRAINTS}"'
        )
        raise RuntimeError(
            "Evaluation environment check failed:\n- "
            + "\n- ".join(issues)
            + "\n\nRun this to repair the env:\n"
            + fix_cmd
        )


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def pick_metric(metrics: Dict[str, Any], preferred_substrings: List[str]) -> Optional[float]:
    numeric_items = [(k, v) for k, v in metrics.items() if is_number(v)]
    if not numeric_items:
        return None

    for needle in preferred_substrings:
        for key, value in numeric_items:
            if needle in key.lower():
                return float(value)

    # Fallback: first numeric metric if preferred one is unavailable.
    return float(numeric_items[0][1])


def extract_headline_scores(submission: Dict[str, Any]) -> Dict[str, Optional[float]]:
    ifeval_metrics = submission.get("ifeval", {}).get("metrics", {})
    gsm8k_metrics = submission.get("gsm8k", {}).get("metrics", {})
    humaneval_metrics = submission.get("humaneval", {}).get("metrics", {})

    return {
        "ifeval_score": pick_metric(
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
        "gsm8k_accuracy": pick_metric(gsm8k_metrics, ["accuracy", "acc", "exact_match", "correct"]),
        "humaneval_score": pick_metric(humaneval_metrics, ["pass@1", "accuracy", "acc", "correct"]),
    }


def average_available(values: List[Optional[float]]) -> Optional[float]:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return float(sum(present) / len(present))


def format_score(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def write_markdown_report(path: Path, report: Dict[str, Any]) -> None:
    quick_rows = []
    for row in report["quick_ranking"]:
        quick_rows.append(
            "| {name} | {step} | {avg} | {ifeval} | {gsm8k} | {humaneval} |".format(
                name=row["checkpoint_name"],
                step=row["step"],
                avg=format_score(row["average_score"]),
                ifeval=format_score(row["headline_scores"]["ifeval_score"]),
                gsm8k=format_score(row["headline_scores"]["gsm8k_accuracy"]),
                humaneval=format_score(row["headline_scores"]["humaneval_score"]),
            )
        )

    best = report["best_checkpoint"]
    top_full = report["top_full_evaluations"]
    baseline_scores = report["baseline"]["headline_scores"]
    best_final_scores = report["final_evaluation"]["headline_scores"]

    lines = [
        "# Experiment Report",
        "",
        f"- **Run timestamp:** {report['run_timestamp_utc']}",
        f"- **Base model:** {report['base_model']}",
        f"- **Training steps:** {report['training_args']['num_steps']}",
        f"- **Checkpoint cadence:** every {report['training_args']['checkpoint_every']} steps",
        "",
        "## Baseline Scores (Base Model)",
        "",
        f"- **IFEval baseline:** {format_score(baseline_scores['ifeval_score'])}",
        f"- **GSM8K baseline:** {format_score(baseline_scores['gsm8k_accuracy'])}",
        f"- **HumanEval baseline:** {format_score(baseline_scores['humaneval_score'])}",
        "",
        "## Quick Checkpoint Ranking",
        "",
        "| Checkpoint | Step | Avg Score | IFEval | GSM8K | HumanEval |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        *quick_rows,
        "",
        "## Selected Best Checkpoint",
        "",
        f"- **Name:** {best['name']}",
        f"- **Step:** {best['step']}",
        f"- **Path:** `{best['path']}`",
        "",
        "## Top Checkpoints Full Evaluations",
        "",
        "| Checkpoint | Step | IFEval | GSM8K | HumanEval |",
        "| --- | ---: | ---: | ---: | ---: |",
        *[
            "| {name} | {step} | {ifeval} | {gsm8k} | {humaneval} |".format(
                name=row["name"],
                step=row["step"],
                ifeval=format_score(row["full_headline_scores"]["ifeval_score"]),
                gsm8k=format_score(row["full_headline_scores"]["gsm8k_accuracy"]),
                humaneval=format_score(row["full_headline_scores"]["humaneval_score"]),
            )
            for row in top_full
        ],
        "",
        "## Best Checkpoint Full Evaluation",
        "",
        f"- **IFEval score:** {format_score(best_final_scores['ifeval_score'])}",
        f"- **GSM8K accuracy:** {format_score(best_final_scores['gsm8k_accuracy'])}",
        f"- **HumanEval score:** {format_score(best_final_scores['humaneval_score'])}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_text_summary(path: Path, report: Dict[str, Any]) -> None:
    baseline_scores = report["baseline"]["headline_scores"]
    final_scores = report["final_evaluation"]["headline_scores"]
    top_full = report["top_full_evaluations"]
    quick_ranking = report["quick_ranking"]
    best = report["best_checkpoint"]
    lines = [
        "EXPERIMENT SUMMARY",
        f"Run timestamp (UTC): {report['run_timestamp_utc']}",
        f"Base model: {report['base_model']}",
        f"Training steps: {report['training_args']['num_steps']}",
        f"Checkpoint cadence: every {report['training_args']['checkpoint_every']} steps",
        "",
        "Baseline scores:",
        f"  IFEval:   {format_score(baseline_scores['ifeval_score'])}",
        f"  GSM8K:    {format_score(baseline_scores['gsm8k_accuracy'])}",
        f"  HumanEval:{format_score(baseline_scores['humaneval_score'])}",
        "",
        "Best checkpoint:",
        f"  Name: {best['name']}",
        f"  Step: {best['step']}",
        f"  Path: {best['path']}",
        "",
        "Best checkpoint full-eval scores:",
        f"  IFEval:   {format_score(final_scores['ifeval_score'])}",
        f"  GSM8K:    {format_score(final_scores['gsm8k_accuracy'])}",
        f"  HumanEval:{format_score(final_scores['humaneval_score'])}",
        "",
        "Checkpoint candidates (quick eval):",
        *[
            (
                f"  - {row['checkpoint_name']} (step {row['step']}): "
                f"IFEval={format_score(row['headline_scores']['ifeval_score'])}, "
                f"GSM8K={format_score(row['headline_scores']['gsm8k_accuracy'])}, "
                f"HumanEval={format_score(row['headline_scores']['humaneval_score'])}, "
                f"Avg={format_score(row['average_score'])}"
            )
            for row in quick_ranking[:3]
        ],
        "",
        "Top checkpoint full-eval scores:",
        *[
            (
                f"  - {row['name']} (step {row['step']}): "
                f"IFEval={format_score(row['full_headline_scores']['ifeval_score'])}, "
                f"GSM8K={format_score(row['full_headline_scores']['gsm8k_accuracy'])}, "
                f"HumanEval={format_score(row['full_headline_scores']['humaneval_score'])}"
            )
            for row in top_full
        ],
        "",
        f"Report JSON: {path.parent / 'experiment_report.json'}",
        f"Report Markdown: {path.parent / 'experiment_report.md'}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def extract_scores_from_task_results(task_results: Dict[str, Any]) -> Dict[str, Optional[float]]:
    pseudo_submission = {
        "ifeval": {"metrics": task_results.get("ifeval", {}).get("metrics", {})},
        "gsm8k": {"metrics": task_results.get("gsm8k", {}).get("metrics", {})},
        "humaneval": {"metrics": task_results.get("humaneval", {}).get("metrics", {})},
    }
    return extract_headline_scores(pseudo_submission)


def main() -> None:
    assert_eval_environment_ok()

    parser = argparse.ArgumentParser(description="One-command full training + evaluation pipeline")
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override base model used for training (e.g. meta-llama/Llama-3.1-8B)",
    )
    parser.add_argument("--checkpoint_name", type=str, default="sanity_1b")
    parser.add_argument("--checkpoint_every", type=int, default=25)
    parser.add_argument(
        "--train_data_path",
        type=str,
        default=None,
        help="Optional JSONL training data path passed to train_and_publish.py",
    )
    parser.add_argument(
        "--num_best_checkpoints",
        type=int,
        default=3,
        help="Number of top quick-ranked checkpoints to run full evaluation on",
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training and reuse checkpoints from checkpoint_info.json",
    )
    parser.add_argument(
        "--checkpoint_info_path",
        type=str,
        default=str(EVAL_DIR / "checkpoint_info.json"),
        help="Path to checkpoint_info.json when reusing existing checkpoints",
    )
    parser.add_argument(
        "--quick_eval_limit",
        type=int,
        default=30,
        help="Sample limit per task when ranking checkpoints quickly",
    )
    parser.add_argument(
        "--final_eval_limit",
        type=int,
        default=None,
        help="Sample limit per task for final evaluation (default: full dataset)",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for experiment artifacts (default: evaluation/experiment_runs/<timestamp>)",
    )
    parser.add_argument(
        "--publish_best",
        action="store_true",
        help="Publish only the selected best checkpoint at the end",
    )
    args = parser.parse_args()

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (EVAL_DIR / "experiment_runs" / timestamp).resolve()
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Artifacts will be written to: {run_dir}")

    # 1) Train and save checkpoints (or reuse existing checkpoint metadata).
    if args.skip_training:
        print("Skipping training and reusing existing checkpoints.")
    else:
        train_cmd = [
            sys.executable,
            str(EVAL_DIR / "train_and_publish.py"),
            "--num_steps",
            str(args.num_steps),
            "--batch_size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--rank",
            str(args.rank),
            "--checkpoint_name",
            args.checkpoint_name,
            "--checkpoint_every",
            str(args.checkpoint_every),
            "--no_publish",
        ]
        if args.model:
            train_cmd.extend(["--model", args.model])
        if args.train_data_path:
            train_cmd.extend(["--data_path", args.train_data_path])
        run_cmd(train_cmd, cwd=REPO_ROOT)

    checkpoint_info_path = Path(args.checkpoint_info_path).resolve()
    checkpoint_info = read_json(checkpoint_info_path)
    checkpoints = checkpoint_info.get("checkpoints", [])
    base_model = checkpoint_info.get("base_model")

    if not checkpoints:
        raise RuntimeError("No checkpoints were found in checkpoint_info.json")
    if not base_model:
        raise RuntimeError(f"base_model missing in {checkpoint_info_path}")

    # 2) Baseline evaluation on base model.
    from evaluation.eval_all import run_core

    print("\nEvaluating baseline (base model, no checkpoint)...")
    baseline_metrics, baseline_task_results = asyncio.run(
        run_core(
            base_model=base_model,
            checkpoint_path=None,
            renderer_name=None,
            temperature=args.temperature,
            top_p=args.top_p,
            limit=args.final_eval_limit,
            log_dir=str(run_dir / "inspect_logs_baseline"),
            verbose=False,
        )
    )
    baseline_headline_scores = extract_scores_from_task_results(baseline_task_results)

    # 3) Quick eval for each checkpoint and ranking.
    quick_results = []
    for cp in checkpoints:
        cp_name = cp["name"]
        cp_path = cp["path"]
        quick_out = run_dir / f"quick_eval_{cp_name}.json"

        eval_cmd = [
            sys.executable,
            str(EVAL_DIR / "eval_all.py"),
            "--checkpoint_path",
            cp_path,
            "--base_model",
            base_model,
            "--limit",
            str(args.quick_eval_limit),
            "--temperature",
            str(args.temperature),
            "--top_p",
            str(args.top_p),
            "--output_path",
            str(quick_out),
        ]
        run_cmd(eval_cmd, cwd=REPO_ROOT)

        quick_submission = read_json(quick_out)
        headline_scores = extract_headline_scores(quick_submission)
        avg_score = average_available(
            [
                headline_scores["ifeval_score"],
                headline_scores["gsm8k_accuracy"],
                headline_scores["humaneval_score"],
            ]
        )
        quick_results.append(
            {
                "checkpoint_name": cp_name,
                "checkpoint_path": cp_path,
                "step": cp.get("step"),
                "headline_scores": headline_scores,
                "average_score": avg_score,
                "quick_submission_path": str(quick_out),
            }
        )

    ranked = sorted(
        quick_results,
        key=lambda x: x["average_score"] if x["average_score"] is not None else -1.0,
        reverse=True,
    )
    num_best = max(1, min(args.num_best_checkpoints, len(ranked)))
    top_ranked = ranked[:num_best]
    best = top_ranked[0]
    print(
        "Best checkpoint after quick eval: "
        f"{best['checkpoint_name']} (avg={format_score(best['average_score'])})"
    )

    # 4) Full eval on selected top checkpoints.
    top_full_evaluations = []
    for i, row in enumerate(top_ranked, start=1):
        cp_name = row["checkpoint_name"]
        cp_path = row["checkpoint_path"]
        final_submission_path = run_dir / f"full_eval_top{i}_{cp_name}.json"
        final_eval_cmd = [
            sys.executable,
            str(EVAL_DIR / "eval_all.py"),
            "--checkpoint_path",
            cp_path,
            "--base_model",
            base_model,
            "--temperature",
            str(args.temperature),
            "--top_p",
            str(args.top_p),
            "--output_path",
            str(final_submission_path),
        ]
        if args.final_eval_limit is not None:
            final_eval_cmd.extend(["--limit", str(args.final_eval_limit)])
        run_cmd(final_eval_cmd, cwd=REPO_ROOT)

        full_submission = read_json(final_submission_path)
        full_scores = extract_headline_scores(full_submission)
        top_full_evaluations.append(
            {
                "rank": i,
                "name": cp_name,
                "path": cp_path,
                "step": row["step"],
                "quick_headline_scores": row["headline_scores"],
                "quick_average_score": row["average_score"],
                "full_submission_path": str(final_submission_path),
                "full_headline_scores": full_scores,
                "full_submission": full_submission,
            }
        )
        print(
            f"Top {i} full eval | {cp_name}: "
            f"IFEval={format_score(full_scores['ifeval_score'])}, "
            f"GSM8K={format_score(full_scores['gsm8k_accuracy'])}, "
            f"HumanEval={format_score(full_scores['humaneval_score'])}"
        )

    # Select best checkpoint by full-eval average score across headline metrics.
    for row in top_full_evaluations:
        row["full_average_score"] = average_available(
            [
                row["full_headline_scores"]["ifeval_score"],
                row["full_headline_scores"]["gsm8k_accuracy"],
                row["full_headline_scores"]["humaneval_score"],
            ]
        )

    best_full = sorted(
        top_full_evaluations,
        key=lambda r: r["full_average_score"] if r["full_average_score"] is not None else -1.0,
        reverse=True,
    )[0]

    print(
        "Best checkpoint after full eval (top candidates): "
        f"{best_full['name']} (avg={format_score(best_full['full_average_score'])})"
    )

    final_submission_path = Path(best_full["full_submission_path"])
    final_submission = best_full["full_submission"]
    final_headline_scores = best_full["full_headline_scores"]
    best = {
        "checkpoint_name": best_full["name"],
        "checkpoint_path": best_full["path"],
        "step": best_full["step"],
        "headline_scores": best_full["quick_headline_scores"],
        "average_score": None,
    }

    # 5) Optionally publish the best checkpoint.
    published = False
    if args.publish_best:
        # Publish directly from checkpoint path without retraining.
        publish_code = (
            "import tinker; "
            f"cp={best['checkpoint_path']!r}; "
            "sc=tinker.ServiceClient(); "
            "rc=sc.create_rest_client(); "
            "rc.publish_checkpoint_from_tinker_path(cp).result(); "
            "print('Published:', cp)"
        )
        run_cmd([sys.executable, "-c", publish_code], cwd=REPO_ROOT)
        published = True

    # 6) Write final experiment report artifacts.
    report = {
        "run_timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "base_model": base_model,
        "training_args": {
            "model_override": args.model,
            "train_data_path": args.train_data_path,
            "num_steps": args.num_steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "rank": args.rank,
            "checkpoint_name": args.checkpoint_name,
            "checkpoint_every": args.checkpoint_every,
        },
        "eval_args": {
            "quick_eval_limit": args.quick_eval_limit,
            "final_eval_limit": args.final_eval_limit,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "num_best_checkpoints": num_best,
        },
        "baseline": {
            "metrics": baseline_metrics,
            "headline_scores": baseline_headline_scores,
            "task_results": baseline_task_results,
        },
        "quick_ranking": ranked,
        "best_checkpoint": {
            "name": best_full["name"],
            "path": best_full["path"],
            "step": best_full["step"],
            "quick_headline_scores": best_full["quick_headline_scores"],
            "quick_average_score": best_full["quick_average_score"],
            "full_headline_scores": best_full["full_headline_scores"],
            "full_average_score": best_full["full_average_score"],
        },
        "top_full_evaluations": top_full_evaluations,
        "final_evaluation": {
            "submission_path": str(final_submission_path),
            "headline_scores": final_headline_scores,
            "full_submission": final_submission,
        },
        "published_best_checkpoint": published,
    }

    report_json = run_dir / "experiment_report.json"
    report_md = run_dir / "experiment_report.md"
    report_txt = run_dir / "experiment_summary.txt"
    write_json(report_json, report)
    write_markdown_report(report_md, report)
    write_text_summary(report_txt, report)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Best checkpoint: {best['checkpoint_name']} @ step {best['step']}")
    print(
        "Baseline scores | "
        f"IFEval: {format_score(baseline_headline_scores['ifeval_score'])}, "
        f"GSM8K: {format_score(baseline_headline_scores['gsm8k_accuracy'])}, "
        f"HumanEval: {format_score(baseline_headline_scores['humaneval_score'])}"
    )
    print(
        "Final headline scores | "
        f"IFEval: {format_score(final_headline_scores['ifeval_score'])}, "
        f"GSM8K: {format_score(final_headline_scores['gsm8k_accuracy'])}, "
        f"HumanEval: {format_score(final_headline_scores['humaneval_score'])}"
    )
    print(f"Report JSON: {report_json}")
    print(f"Report Markdown: {report_md}")
    print(f"Summary text: {report_txt}")


if __name__ == "__main__":
    main()
