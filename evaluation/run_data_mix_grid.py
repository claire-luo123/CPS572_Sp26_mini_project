"""
Grid search data-mixture percentages for training.

This script:
1) Reads an existing mixed training JSONL
2) Builds new JSONL files for each IF/GSM8K/Code ratio
3) Runs evaluation/run_full_experiment.py for each mix
4) Writes an aggregated comparison report

Default run settings follow your requested test setup:
- 8B model
- 100 training steps
- checkpoints every 25
"""

import argparse
import json
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_ROOT / "evaluation"


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def classify_row(row: Dict) -> str:
    if "messages" in row:
        return "if"
    if "question" in row and "answer" in row:
        return "gsm8k"
    if "instruction" in row and "output" in row:
        return "code"
    return "other"


def split_rows(rows: List[Dict]) -> Dict[str, List[Dict]]:
    buckets = {"if": [], "gsm8k": [], "code": [], "other": []}
    for row in rows:
        buckets[classify_row(row)].append(row)
    return buckets


def sample_rows(rows: List[Dict], count: int, rng: random.Random) -> List[Dict]:
    if not rows:
        return []
    if count <= len(rows):
        return rng.sample(rows, count)
    # Sample with replacement to satisfy target count for a mix.
    return [rng.choice(rows) for _ in range(count)]


def parse_mix(mix_str: str) -> Tuple[str, Dict[str, float]]:
    # Format: "if=40,gsm8k=30,code=30"
    parts = [p.strip() for p in mix_str.split(",") if p.strip()]
    mix = {}
    for p in parts:
        k, v = [x.strip() for x in p.split("=", 1)]
        mix[k] = float(v)
    keys = {"if", "gsm8k", "code"}
    if set(mix.keys()) != keys:
        raise ValueError(f"Mix must contain exactly {keys}. Got: {mix}")
    total = mix["if"] + mix["gsm8k"] + mix["code"]
    if abs(total - 100.0) > 1e-6:
        raise ValueError(f"Mix percentages must sum to 100. Got: {total}")
    name = f"if{int(mix['if'])}_gsm{int(mix['gsm8k'])}_code{int(mix['code'])}"
    return name, mix


def format_score(v) -> str:
    if v is None:
        return "N/A"
    return f"{float(v):.4f}"


def run_cmd(cmd: List[str]) -> None:
    print("\n>>> Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run data-mixture grid search.")
    parser.add_argument("--input_path", type=str, default="data/my_mixed_training_data.jsonl")
    parser.add_argument("--target_size", type=int, default=30000, help="Rows per generated mix dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mix",
        action="append",
        default=[],
        help="Mix in format if=40,gsm8k=30,code=30. Can be provided multiple times.",
    )
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--checkpoint_every", type=int, default=25)
    parser.add_argument("--quick_eval_limit", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=32)
    args = parser.parse_args()

    default_mixes = [
        "if=33,gsm8k=33,code=34",
        "if=50,gsm8k=25,code=25",
        "if=25,gsm8k=50,code=25",
        "if=25,gsm8k=25,code=50",
        "if=40,gsm8k=30,code=30",
    ]
    mix_strings = args.mix if args.mix else default_mixes
    mixes = [parse_mix(m) for m in mix_strings]

    input_path = (REPO_ROOT / args.input_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input data file not found: {input_path}")

    rows = read_jsonl(input_path)
    buckets = split_rows(rows)
    print(
        "Loaded base data | "
        f"if={len(buckets['if'])}, gsm8k={len(buckets['gsm8k'])}, code={len(buckets['code'])}, other={len(buckets['other'])}"
    )

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    grid_root = (EVAL_DIR / "grid_search_runs" / timestamp).resolve()
    data_dir = grid_root / "mix_data"
    run_dir_root = grid_root / "runs"
    grid_root.mkdir(parents=True, exist_ok=True)

    results = []
    for idx, (mix_name, mix_pct) in enumerate(mixes):
        rng = random.Random(args.seed + idx)
        n_if = int(round(args.target_size * (mix_pct["if"] / 100.0)))
        n_gsm = int(round(args.target_size * (mix_pct["gsm8k"] / 100.0)))
        n_code = args.target_size - n_if - n_gsm

        sampled = []
        sampled.extend(sample_rows(buckets["if"], n_if, rng))
        sampled.extend(sample_rows(buckets["gsm8k"], n_gsm, rng))
        sampled.extend(sample_rows(buckets["code"], n_code, rng))
        rng.shuffle(sampled)

        mix_data_path = data_dir / f"{mix_name}.jsonl"
        write_jsonl(mix_data_path, sampled)

        mix_run_dir = run_dir_root / mix_name
        run_cmd(
            [
                sys.executable,
                str(EVAL_DIR / "run_full_experiment.py"),
                "--model",
                args.model,
                "--num_steps",
                str(args.num_steps),
                "--checkpoint_every",
                str(args.checkpoint_every),
                "--quick_eval_limit",
                str(args.quick_eval_limit),
                "--batch_size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--rank",
                str(args.rank),
                "--train_data_path",
                str(mix_data_path),
                "--output_dir",
                str(mix_run_dir),
            ]
        )

        report_path = mix_run_dir / "experiment_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        final_scores = report["final_evaluation"]["headline_scores"]
        avg = (
            sum(v for v in final_scores.values() if isinstance(v, (int, float)))
            / max(1, sum(1 for v in final_scores.values() if isinstance(v, (int, float))))
        )
        results.append(
            {
                "mix_name": mix_name,
                "mix_percentages": mix_pct,
                "data_path": str(mix_data_path),
                "run_dir": str(mix_run_dir),
                "best_checkpoint": report["best_checkpoint"]["name"],
                "final_scores": final_scores,
                "final_avg_score": avg,
            }
        )

    results_sorted = sorted(results, key=lambda r: r["final_avg_score"], reverse=True)
    summary_json = grid_root / "grid_summary.json"
    summary_md = grid_root / "grid_summary.md"
    summary_json.write_text(json.dumps({"results": results_sorted}, indent=2), encoding="utf-8")

    lines = [
        "# Data Mix Grid Search Summary",
        "",
        f"- Model: `{args.model}`",
        f"- Steps: `{args.num_steps}`",
        f"- Checkpoint every: `{args.checkpoint_every}`",
        f"- Quick eval limit: `{args.quick_eval_limit}`",
        f"- Target rows per mix: `{args.target_size}`",
        "",
        "| Mix | IFEval | GSM8K | HumanEval | Avg | Best checkpoint |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in results_sorted:
        s = r["final_scores"]
        lines.append(
            "| {mix} | {ifeval} | {gsm} | {code} | {avg} | {best} |".format(
                mix=r["mix_name"],
                ifeval=format_score(s.get("ifeval_score")),
                gsm=format_score(s.get("gsm8k_accuracy")),
                code=format_score(s.get("humaneval_score")),
                avg=f"{r['final_avg_score']:.4f}",
                best=r["best_checkpoint"],
            )
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\nGrid search complete.")
    print(f"Summary JSON: {summary_json}")
    print(f"Summary Markdown: {summary_md}")


if __name__ == "__main__":
    main()
