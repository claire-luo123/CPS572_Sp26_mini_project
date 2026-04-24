#!/usr/bin/env python3
"""
Build a cleaned, capped, and mixture-balanced training JSONL from my_mixed_training_data.jsonl.

Context (this repo today):
- train_and_publish.py: no separate "data cleaning" pass on disk. Rows that fail
  conversation_to_datum() during tokenization are skipped (silent drop). Shuffling:
  default --shuffle_data: each epoch over the index order uses a seeded RNG shuffle;
  batches walk that order linearly (--no_shuffle_data disables reshuffle between epochs).
- prepare_data.py: concatenates GSM8K train + Tulu subsample + OpenCodeInstruct subsample
  into one JSONL with no deduplication.

This script:
1) Loads mixed JSONL, buckets rows (IF / GSM8K / code) the same way as training.
2) Cleans: strip whitespace, drop empty fields, validate structure, optional max char length.
3) Dedupes within each bucket (stable: first occurrence wins).
4) Applies per-bucket caps (defaults: IF=10_000, GSM8K=7_473, code=10_000 from your cap study).
5) Subsamples to a global 50% / 25% / 25% row mix subject to capped pool sizes (your mix study).
6) Shuffles the final file (seeded) so training order is not bucket-blocked.
7) Writes JSONL + a small manifest JSON for the report.

Weights 1.0 / 1.0 / 1.0: this file does not change the loss function; it only builds the
dataset. Uniform SFT implies implicit weight 1 per example.

Usage:
  python3 data/build_clean_cap_mix_training_data.py \\
    --input data/my_mixed_training_data.jsonl \\
    --output data/my_mixed_training_data_clean_cap_mix.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

Bucket = Literal["instruction_following", "gsm8k", "code"]


def classify_row(row: Dict[str, Any]) -> Optional[Bucket]:
    if "messages" in row and isinstance(row["messages"], list):
        return "instruction_following"
    if "question" in row and "answer" in row:
        return "gsm8k"
    if "instruction" in row and "output" in row:
        return "code"
    return None


def _norm_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def dedup_key(bucket: Bucket, row: Dict[str, Any]) -> str:
    if bucket == "gsm8k":
        return "gsm:" + hashlib.sha256(_norm_text(str(row.get("question", ""))).encode()).hexdigest()
    if bucket == "code":
        return "code:" + hashlib.sha256(_norm_text(str(row.get("instruction", ""))).encode()).hexdigest()
    # IF: hash concatenated user contents in order
    parts: List[str] = []
    for m in row.get("messages") or []:
        if isinstance(m, dict) and m.get("role") == "user":
            parts.append(_norm_text(str(m.get("content", ""))))
    return "if:" + hashlib.sha256("||".join(parts).encode()).hexdigest()


def clean_row(bucket: Bucket, row: Dict[str, Any], max_chars: Optional[int]) -> Optional[Dict[str, Any]]:
    if bucket == "gsm8k":
        q = str(row.get("question", "")).strip()
        a = str(row.get("answer", "")).strip()
        if not q or not a:
            return None
        if max_chars is not None and (len(q) + len(a) > max_chars):
            return None
        return {"question": q, "answer": a}

    if bucket == "code":
        ins = str(row.get("instruction", "")).strip()
        out = str(row.get("output", "")).strip()
        if not ins or not out:
            return None
        if max_chars is not None and (len(ins) + len(out) > max_chars):
            return None
        return {"instruction": ins, "output": out}

    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    cleaned: List[Dict[str, str]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "")).strip()
        content = str(m.get("content", "")).strip()
        if not role or content is None:
            continue
        cleaned.append({"role": role, "content": content})
    if len(cleaned) < 2:
        return None
    roles = [m["role"] for m in cleaned]
    if "assistant" not in roles:
        return None
    total = sum(len(m["content"]) for m in cleaned)
    if max_chars is not None and total > max_chars:
        return None
    return {"messages": cleaned}


def apply_cap(rows: List[Dict[str, Any]], cap: int, rng: random.Random) -> List[Dict[str, Any]]:
    if len(rows) <= cap:
        return list(rows)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    return [rows[i] for i in sorted(idx[:cap])]


def max_total_under_mix(
    n_if: int, n_gsm: int, n_code: int, pct_if: int, pct_gsm: int, pct_code: int
) -> int:
    """Largest N with int splits summing to N and each split <= pool size."""
    if pct_if + pct_gsm + pct_code != 100:
        raise ValueError("mix percentages must sum to 100")
    # n_if_need(N) = (N * pct_if + 99) // 100  # ceil alternative; use floor splits then fix remainder
    def splits(n: int) -> Tuple[int, int, int]:
        a = n * pct_if // 100
        b = n * pct_gsm // 100
        c = n - a - b
        return a, b, c

    hi = min(n_if * 100 // pct_if if pct_if else n_if, n_gsm * 100 // pct_gsm if pct_gsm else n_gsm, n_code * 100 // pct_code if pct_code else n_code)
    lo = 0
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        a, b, c = splits(mid)
        if a <= n_if and b <= n_gsm and c <= n_code and a + b + c == mid:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def main() -> None:
    p = argparse.ArgumentParser(description="Clean + cap + mix training JSONL")
    p.add_argument("--input", type=str, default="data/my_mixed_training_data.jsonl")
    p.add_argument("--output", type=str, default="data/my_mixed_training_data_clean_cap_mix.jsonl")
    p.add_argument("--manifest", type=str, default=None, help="JSON stats path (default: <output>.manifest.json)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cap_if", type=int, default=10_000, help="Max IF rows after clean+dedup")
    p.add_argument("--cap_gsm8k", type=int, default=7_473, help="Max GSM8K rows after clean+dedup")
    p.add_argument("--cap_code", type=int, default=10_000, help="Max code rows after clean+dedup")
    p.add_argument("--mix_if", type=int, default=50, help="Percent IF in final mix (default 50)")
    p.add_argument("--mix_gsm8k", type=int, default=25, help="Percent GSM8K (default 25)")
    p.add_argument("--mix_code", type=int, default=25, help="Percent code (default 25)")
    p.add_argument(
        "--max_chars_per_row",
        type=int,
        default=None,
        help="Drop rows whose total relevant text exceeds this (optional)",
    )
    p.add_argument("--no_shuffle_output", action="store_true", help="Keep IF block, GSM block, code block order")
    args = p.parse_args()

    in_path = Path(args.input).resolve()
    out_path = Path(args.output).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else out_path.with_suffix(out_path.suffix + ".manifest.json")

    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    rng = random.Random(args.seed)

    raw_counts = {"instruction_following": 0, "gsm8k": 0, "code": 0, "unknown": 0}
    buckets: Dict[Bucket, List[Dict[str, Any]]] = {"instruction_following": [], "gsm8k": [], "code": []}
    seen_keys: Dict[Bucket, set] = {b: set() for b in buckets}
    dropped_clean = 0
    dropped_dup = 0

    with in_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                dropped_clean += 1
                continue
            b = classify_row(row)
            if b is None:
                raw_counts["unknown"] += 1
                continue
            raw_counts[b] += 1
            cleaned = clean_row(b, row, args.max_chars_per_row)
            if cleaned is None:
                dropped_clean += 1
                continue
            key = dedup_key(b, cleaned)
            if key in seen_keys[b]:
                dropped_dup += 1
                continue
            seen_keys[b].add(key)
            buckets[b].append(cleaned)

    capped_if = apply_cap(buckets["instruction_following"], args.cap_if, rng)
    capped_gsm = apply_cap(buckets["gsm8k"], args.cap_gsm8k, rng)
    capped_code = apply_cap(buckets["code"], args.cap_code, rng)

    n_total = max_total_under_mix(
        len(capped_if),
        len(capped_gsm),
        len(capped_code),
        args.mix_if,
        args.mix_gsm8k,
        args.mix_code,
    )
    need_if = n_total * args.mix_if // 100
    need_gsm = n_total * args.mix_gsm8k // 100
    need_code = n_total - need_if - need_gsm

    rng.shuffle(capped_if)
    rng.shuffle(capped_gsm)
    rng.shuffle(capped_code)

    final_if = capped_if[:need_if]
    final_gsm = capped_gsm[:need_gsm]
    final_code = capped_code[:need_code]

    out_rows: List[Dict[str, Any]] = final_if + final_gsm + final_code
    if not args.no_shuffle_output:
        rng.shuffle(out_rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "input": str(in_path),
        "output": str(out_path),
        "seed": args.seed,
        "caps": {"instruction_following": args.cap_if, "gsm8k": args.cap_gsm8k, "code": args.cap_code},
        "mix_percent": {"instruction_following": args.mix_if, "gsm8k": args.mix_gsm8k, "code": args.mix_code},
        "weights_note": "Uniform SFT (implicit weight 1.0 per row); caps+mix only affect which rows appear.",
        "raw_line_bucket_counts": raw_counts,
        "after_clean_dedup_counts": {k: len(v) for k, v in buckets.items()},
        "dropped_malformed_or_empty": dropped_clean,
        "dropped_duplicate": dropped_dup,
        "after_cap_pool_sizes": {"instruction_following": len(capped_if), "gsm8k": len(capped_gsm), "code": len(capped_code)},
        "final_total_rows": len(out_rows),
        "final_per_bucket": {"instruction_following": len(final_if), "gsm8k": len(final_gsm), "code": len(final_code)},
        "max_chars_per_row": args.max_chars_per_row,
        "output_shuffled": not args.no_shuffle_output,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    print(f"\nWrote: {out_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
