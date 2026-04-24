import argparse
import hashlib
import json
import random
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def classify_row(row: Dict) -> str:
    if "question" in row and "answer" in row:
        return "math"
    if "messages" in row:
        return "if"
    if "instruction" in row and "output" in row:
        return "code"
    return "other"


def is_quality_row(row: Dict) -> bool:
    min_len = 8
    row_type = classify_row(row)
    if row_type == "math":
        return len(normalize_text(row["question"])) >= min_len and len(normalize_text(row["answer"])) >= min_len
    if row_type == "if":
        msgs = row.get("messages", [])
        if not isinstance(msgs, list) or len(msgs) < 2:
            return False
        for m in msgs:
            if not normalize_text(m.get("content", "")):
                return False
        has_user = any(m.get("role") == "user" for m in msgs)
        has_assistant = any(m.get("role") == "assistant" for m in msgs)
        return has_user and has_assistant
    if row_type == "code":
        return len(normalize_text(row["instruction"])) >= min_len and len(normalize_text(row["output"])) >= min_len
    return False


def row_signature(row: Dict) -> str:
    row_type = classify_row(row)
    if row_type == "math":
        payload = f"math::{normalize_text(row['question'])}::{normalize_text(row['answer'])}"
    elif row_type == "if":
        turns = []
        for m in row["messages"]:
            turns.append(f"{m.get('role','')}:{normalize_text(m.get('content',''))}")
        payload = "if::" + "||".join(turns)
    elif row_type == "code":
        payload = f"code::{normalize_text(row['instruction'])}::{normalize_text(row['output'])}"
    else:
        payload = json.dumps(row, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dedupe_and_filter(rows: List[Dict]) -> Tuple[List[Dict], int, int]:
    seen = set()
    kept = []
    dropped_quality = 0
    dropped_dupe = 0
    for row in rows:
        if not is_quality_row(row):
            dropped_quality += 1
            continue
        sig = row_signature(row)
        if sig in seen:
            dropped_dupe += 1
            continue
        seen.add(sig)
        kept.append(row)
    return kept, dropped_quality, dropped_dupe


def build_if_synthetic_examples(count: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    topics = ["study planning", "code review habits", "math exam prep", "healthy routine"]
    banned_words = ["very", "always", "never", "obviously", "literally"]
    examples = []
    for _ in range(count):
        topic = rng.choice(topics)
        n_bullets = rng.choice([3, 4, 5])
        banned = rng.choice(banned_words)
        prompt = (
            f"Give exactly {n_bullets} bullet points about {topic}. "
            f"Each bullet must be fewer than 10 words. "
            f"Do not use the word '{banned}'. "
            "Output bullets only."
        )
        bullets = [f"- {topic.split()[0].capitalize()} action {i + 1}" for i in range(n_bullets)]
        answer = "\n".join(bullets)
        examples.append(
            {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ]
            }
        )
    return examples


def build_code_edge_examples(count: int, seed: int) -> List[Dict]:
    rng = random.Random(seed + 1)
    templates = [
        (
            "Write a Python function `safe_mean(nums)` that returns 0.0 for empty list.",
            "def safe_mean(nums):\n    if not nums:\n        return 0.0\n    return sum(nums) / float(len(nums))",
        ),
        (
            "Write `first_unique(s)` to return first non-repeating char, else ''.",
            "from collections import Counter\n\ndef first_unique(s):\n    c = Counter(s)\n    for ch in s:\n        if c[ch] == 1:\n            return ch\n    return ''",
        ),
        (
            "Write `normalize_scores(xs)` min-max scaling. If all equal, return zeros.",
            "def normalize_scores(xs):\n    if not xs:\n        return []\n    mn, mx = min(xs), max(xs)\n    if mn == mx:\n        return [0.0] * len(xs)\n    return [(x - mn) / (mx - mn) for x in xs]",
        ),
    ]
    out = []
    for _ in range(count):
        instr, ans = rng.choice(templates)
        out.append({"instruction": instr, "output": ans})
    return out


def build_harder_math_examples(math_rows: List[Dict], count: int, seed: int) -> List[Dict]:
    rng = random.Random(seed + 2)

    def complexity(row: Dict) -> int:
        q = row.get("question", "")
        ops = len(re.findall(r"[\+\-\*/%]", q))
        return len(q.split()) + (10 * ops)

    ranked = sorted(math_rows, key=complexity, reverse=True)
    pool = ranked[: max(1000, count * 2)]
    if not pool:
        return []
    sampled = []
    for _ in range(count):
        row = rng.choice(pool)
        sampled.append({"question": row["question"], "answer": row["answer"]})
    return sampled


def split_by_type(rows: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    math_rows = []
    if_rows = []
    code_rows = []
    for row in rows:
        row_type = classify_row(row)
        if row_type == "math":
            math_rows.append(row)
        elif row_type == "if":
            if_rows.append(row)
        elif row_type == "code":
            code_rows.append(row)
    return math_rows, if_rows, code_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Build augmented training JSONL from existing training data.")
    parser.add_argument("--input_path", type=str, default="data/my_mixed_training_data.jsonl")
    parser.add_argument("--output_path", type=str, default="data/my_mixed_training_data_v2.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--if_synth_count", type=int, default=1000)
    parser.add_argument("--code_edge_count", type=int, default=600)
    parser.add_argument("--hard_math_count", type=int, default=1000)
    parser.add_argument(
        "--run_training",
        action="store_true",
        help="After writing augmented data, run evaluation/run_full_experiment.py using it.",
    )
    parser.add_argument(
        "--training_args",
        type=str,
        default="",
        help="Extra args string passed to run_full_experiment.py when --run_training is set.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input data file not found: {input_path}")

    rows = read_jsonl(input_path)
    math_rows, if_rows, code_rows = split_by_type(rows)

    print(
        f"Loaded {len(rows)} records | "
        f"math={len(math_rows)} if={len(if_rows)} code={len(code_rows)}"
    )

    extras = []
    extras.extend(build_if_synthetic_examples(args.if_synth_count, seed=args.seed))
    extras.extend(build_code_edge_examples(args.code_edge_count, seed=args.seed))
    extras.extend(build_harder_math_examples(math_rows, args.hard_math_count, seed=args.seed))

    combined = rows + extras
    combined, dropped_quality, dropped_dupe = dedupe_and_filter(combined)
    rng = random.Random(args.seed)
    rng.shuffle(combined)
    write_jsonl(output_path, combined)

    m2, i2, c2 = split_by_type(combined)
    print(
        f"Wrote {len(combined)} records to {output_path} | "
        f"math={len(m2)} if={len(i2)} code={len(c2)} | "
        f"dropped_quality={dropped_quality} dropped_dupe={dropped_dupe}"
    )

    if args.run_training:
        cmd = [
            "python3",
            "evaluation/run_full_experiment.py",
            "--train_data_path",
            str(output_path),
        ]
        if args.training_args.strip():
            cmd.extend(args.training_args.strip().split())
        print("Running training command:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
