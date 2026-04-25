import argparse
import json
import os
import random
import re
from collections import Counter

try:
    from datasets import load_dataset
except ModuleNotFoundError:
    load_dataset = None

DEFAULT_OUTPUT_PATH = os.path.join("data", "my_mixed_training_data.jsonl")
DEFAULT_IF_MAX = 12000
DEFAULT_CODE_MAX = 12000
DEFAULT_CANDIDATE_MULTIPLIER = 6

CONSTRAINT_KEYWORDS = [
    "exactly",
    "at least",
    "at most",
    "include",
    "mention",
    "begin",
    "end with",
    "only",
    "list",
    "bullet",
    "paragraph",
    "sentence",
    "word",
    "json",
    "yaml",
    "markdown",
    "lowercase",
    "uppercase",
    "title case",
    "format",
    "respond with",
    "output",
    "step by step",
]

REFUSAL_PREFIXES = [
    "i'm sorry",
    "i am sorry",
    "sorry, but",
    "i cannot",
    "i can't",
    "i will not",
    "i won't",
    "as an ai",
    "i do not have the ability",
]

NON_PYTHON_LANGS = [
    "javascript",
    "typescript",
    "java",
    "c++",
    "cpp",
    "rust",
    "golang",
    " go ",
    "kotlin",
    "swift",
    "sql",
    "php",
]


def get_content(row, keys):
    """Helper to find the right key in a dictionary."""
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def normalize_text(text):
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def strip_code_fences(text):
    body = str(text).strip()
    if body.startswith("```"):
        body = re.sub(r"^```[a-zA-Z0-9_+-]*\n?", "", body)
        body = re.sub(r"\n?```$", "", body)
    return body.strip()


def mostly_ascii(text, threshold=0.85):
    chars = [ch for ch in str(text) if not ch.isspace()]
    if not chars:
        return False
    ascii_chars = sum(ord(ch) < 128 for ch in chars)
    return ascii_chars / len(chars) >= threshold


def looks_like_refusal(text):
    lowered = normalize_text(text)
    return any(lowered.startswith(prefix) for prefix in REFUSAL_PREFIXES)


def candidate_pool_size(dataset_len, max_examples, multiplier):
    if max_examples <= 0:
        return dataset_len
    return min(dataset_len, max_examples * max(multiplier, 1))


def choose_subset(ds, seed, limit):
    if limit >= len(ds):
        return ds.shuffle(seed=seed)
    return ds.shuffle(seed=seed).select(range(limit))


def take_single_turn_conversation(messages):
    normalized = []
    for msg in messages:
        if not isinstance(msg, dict):
            return None
        role = msg.get("role")
        content = msg.get("content")
        if role not in {"system", "user", "assistant"}:
            continue
        if not isinstance(content, str):
            continue
        content = content.strip()
        if not content:
            continue
        normalized.append({"role": role, "content": content})

    system_messages = []
    for index, msg in enumerate(normalized):
        if msg["role"] == "system":
            if not system_messages:
                system_messages.append(msg)
            continue
        if msg["role"] != "user":
            continue
        for follow in normalized[index + 1:]:
            if follow["role"] == "assistant":
                return system_messages + [msg, follow], len(normalized)
        break

    return None


def score_if_example(user_text, assistant_text, turn_count):
    prompt = normalize_text(user_text)
    score = 0
    score += min(sum(1 for keyword in CONSTRAINT_KEYWORDS if keyword in prompt), 5)
    if turn_count <= 3:
        score += 2
    if 40 <= len(user_text) <= 1200:
        score += 1
    if 40 <= len(assistant_text) <= 1200:
        score += 1
    if mostly_ascii(user_text):
        score += 1
    if not looks_like_refusal(assistant_text):
        score += 2
    return score


def looks_like_python(prompt, output):
    prompt_lower = f" {normalize_text(prompt)} "
    output_lower = output.lower()

    mentions_other_language = any(lang in prompt_lower for lang in NON_PYTHON_LANGS)
    if mentions_other_language and "python" not in prompt_lower:
        return False

    python_markers = ("def ", "class ", "import ", "from ", "@")
    return any(marker in output_lower for marker in python_markers)


def starts_like_code(output):
    stripped = output.lstrip()
    return bool(re.match(r"^(def |class |from |import |@|#|\"\"\"|''')", stripped))


def score_code_example(prompt, output):
    prompt_lower = normalize_text(prompt)
    output_lower = output.lower()

    score = 0
    if "python" in prompt_lower:
        score += 3
    if any(word in prompt_lower for word in ["function", "implement", "write", "complete"]):
        score += 1
    if any(word in prompt_lower for word in ["docstring", "unit test", "leetcode"]):
        score += 1
    if "def " in output_lower:
        score += 3
    if "return" in output_lower:
        score += 1
    if starts_like_code(output):
        score += 1
    line_count = output.count("\n") + 1
    if 4 <= line_count <= 80:
        score += 1
    return score


def prepare_gsm8k(max_examples, seed):
    if load_dataset is None:
        raise ModuleNotFoundError("prepare_data.py requires the 'datasets' package. Install requirements first.")
    print("Loading math data...")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = choose_subset(ds, seed=seed, limit=max_examples if max_examples > 0 else len(ds))

    records = [{"task": "gsm8k", "question": row["question"], "answer": row["answer"]} for row in ds]
    return records, {
        "dataset": "openai/gsm8k",
        "loaded": len(ds),
        "selected": len(records),
    }


def prepare_ifeval_mix(max_examples, seed, candidate_multiplier):
    if load_dataset is None:
        raise ModuleNotFoundError("prepare_data.py requires the 'datasets' package. Install requirements first.")
    print("Loading instruction-following data...")
    ds = load_dataset("allenai/tulu-3-sft-mixture", split="train")
    subset_size = candidate_pool_size(len(ds), max_examples, candidate_multiplier)
    ds = choose_subset(ds, seed=seed, limit=subset_size)

    rng = random.Random(seed)
    candidates = []
    seen_prompts = set()
    stats = Counter()

    for row in ds:
        stats["candidate_rows"] += 1
        parsed = take_single_turn_conversation(row.get("messages", []))
        if parsed is None:
            stats["rejected_no_single_turn"] += 1
            continue

        convo, turn_count = parsed
        user_text = next(msg["content"] for msg in convo if msg["role"] == "user")
        assistant_text = next(msg["content"] for msg in convo if msg["role"] == "assistant")
        key = normalize_text(user_text)

        if key in seen_prompts:
            stats["rejected_duplicate_prompt"] += 1
            continue
        if len(user_text) < 20 or len(user_text) > 2500:
            stats["rejected_prompt_length"] += 1
            continue
        if len(assistant_text) < 20 or len(assistant_text) > 3500:
            stats["rejected_answer_length"] += 1
            continue
        if not mostly_ascii(user_text):
            stats["rejected_non_ascii_prompt"] += 1
            continue
        if looks_like_refusal(assistant_text):
            stats["rejected_refusal"] += 1
            continue

        score = score_if_example(user_text, assistant_text, turn_count)
        if score < 3:
            stats["rejected_low_score"] += 1
            continue

        seen_prompts.add(key)
        candidates.append(
            (
                score,
                rng.random(),
                {"task": "ifeval", "messages": convo},
            )
        )

    candidates.sort(key=lambda item: (-item[0], item[1]))
    selected = [record for _, _, record in candidates[: max_examples or None]]
    stats["selected"] = len(selected)
    stats["scored_candidates"] = len(candidates)

    return selected, {
        "dataset": "allenai/tulu-3-sft-mixture",
        "candidate_pool": subset_size,
        **dict(stats),
    }


def prepare_code_mix(max_examples, seed, candidate_multiplier):
    if load_dataset is None:
        raise ModuleNotFoundError("prepare_data.py requires the 'datasets' package. Install requirements first.")
    print("Loading code data...")
    ds = load_dataset("nvidia/OpenCodeInstruct", split="train")
    subset_size = candidate_pool_size(len(ds), max_examples, candidate_multiplier)
    ds = choose_subset(ds, seed=seed, limit=subset_size)

    rng = random.Random(seed)
    candidates = []
    seen_prompts = set()
    stats = Counter()

    for row in ds:
        stats["candidate_rows"] += 1
        prompt = get_content(row, ["instruction", "input", "prompt", "question"])
        output = get_content(row, ["output", "response", "answer", "content"])
        if prompt is None or output is None:
            stats["rejected_missing_fields"] += 1
            continue

        prompt = str(prompt).strip()
        output = strip_code_fences(output)
        key = normalize_text(prompt)

        if key in seen_prompts:
            stats["rejected_duplicate_prompt"] += 1
            continue
        if len(prompt) < 20 or len(prompt) > 3000:
            stats["rejected_prompt_length"] += 1
            continue
        if len(output) < 20 or len(output) > 5000:
            stats["rejected_output_length"] += 1
            continue
        if looks_like_refusal(output):
            stats["rejected_refusal"] += 1
            continue
        if not looks_like_python(prompt, output):
            stats["rejected_not_python"] += 1
            continue

        score = score_code_example(prompt, output)
        if score < 4:
            stats["rejected_low_score"] += 1
            continue

        seen_prompts.add(key)
        candidates.append(
            (
                score,
                rng.random(),
                {"task": "code", "instruction": prompt, "output": output},
            )
        )

    candidates.sort(key=lambda item: (-item[0], item[1]))
    selected = [record for _, _, record in candidates[: max_examples or None]]
    stats["selected"] = len(selected)
    stats["scored_candidates"] = len(candidates)

    return selected, {
        "dataset": "nvidia/OpenCodeInstruct",
        "candidate_pool": subset_size,
        **dict(stats),
    }


def stats_path_for(output_path):
    if output_path.endswith(".jsonl"):
        return output_path[:-6] + ".stats.json"
    return output_path + ".stats.json"


def write_jsonl(output_path, records):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build a filtered multi-task training mix.")
    parser.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_gsm8k_examples", type=int, default=0, help="0 uses all GSM8K train examples")
    parser.add_argument("--max_ifeval_examples", type=int, default=DEFAULT_IF_MAX)
    parser.add_argument("--max_code_examples", type=int, default=DEFAULT_CODE_MAX)
    parser.add_argument(
        "--candidate_multiplier",
        type=int,
        default=DEFAULT_CANDIDATE_MULTIPLIER,
        help="How many random candidates to inspect per kept IF/code example",
    )
    parser.add_argument(
        "--extra_jsonl",
        action="append",
        default=[],
        help="Path to an additional pre-built JSONL of {'messages': [...]} rows "
             "to mix into the final training file (use multiple times for multiple files). "
             "Useful for synthetic IFEval data from data/build_synthetic_ifeval.py.",
    )
    parser.add_argument(
        "--extra_repeat",
        type=int,
        default=1,
        help="Repeat extra JSONL rows N times when concatenating (cheap upweight).",
    )
    args = parser.parse_args()

    gsm8k_records, gsm8k_stats = prepare_gsm8k(args.max_gsm8k_examples, args.seed)
    if_records, if_stats = prepare_ifeval_mix(
        max_examples=args.max_ifeval_examples,
        seed=args.seed,
        candidate_multiplier=args.candidate_multiplier,
    )
    code_records, code_stats = prepare_code_mix(
        max_examples=args.max_code_examples,
        seed=args.seed,
        candidate_multiplier=args.candidate_multiplier,
    )

    extra_records = []
    extra_stats = {}
    for path in args.extra_jsonl:
        if not os.path.exists(path):
            print(f"WARNING: extra_jsonl file not found, skipping: {path}")
            continue
        loaded = 0
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if "messages" not in row:
                    continue
                extra_records.append({"messages": row["messages"]})
                loaded += 1
        if args.extra_repeat > 1 and loaded:
            base = list(extra_records[-loaded:])
            for _ in range(args.extra_repeat - 1):
                extra_records.extend(base)
        extra_stats[path] = {"loaded_rows": loaded, "after_repeat": loaded * args.extra_repeat}
        print(f"Loaded {loaded} extra rows from {path} (repeat={args.extra_repeat})")

    all_records = gsm8k_records + if_records + code_records + extra_records
    write_jsonl(args.output_path, all_records)

    summary = {
        "output_path": args.output_path,
        "seed": args.seed,
        "counts": {
            "gsm8k": len(gsm8k_records),
            "ifeval": len(if_records),
            "code": len(code_records),
            "extra": len(extra_records),
            "total": len(all_records),
        },
        "datasets": {
            "gsm8k": gsm8k_stats,
            "ifeval": if_stats,
            "code": code_stats,
            "extra": extra_stats,
        },
    }
    summary_path = stats_path_for(args.output_path)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved {len(all_records)} training rows to {args.output_path}")
    print(json.dumps(summary["counts"], indent=2))
    print(f"Saved data stats to {summary_path}")


if __name__ == "__main__":
    main()
