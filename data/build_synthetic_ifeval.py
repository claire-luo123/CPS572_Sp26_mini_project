"""Generate synthetic IFEval-style supervised data using the Google IFEval
constraint registry as a verifier.

Pipeline per attempt:
  1. Sample a base task from a curated seed list.
  2. Sample 1-2 non-conflicting verifiable constraints from
     `instruction_following_eval.instructions_registry.INSTRUCTION_DICT` and
     instantiate them with reasonable random kwargs.
  3. Build a single-turn user prompt:  "<task>. <constraint description 1>
     <constraint description 2>".
  4. Sample N candidate responses from a Tinker sampling client (a base model
     or your own SFT'd checkpoint).
  5. Verify each candidate against ALL constraints with `check_following`.
  6. Keep verified (prompt, response) pairs as JSONL examples that match the
     "messages" schema consumed by `evaluation/train_and_publish.py`.

This is the data-augmentation extension we never built: training on examples
that were *generated* under verifiable constraints and *verified* by the same
graders that score the IFEval test set. Because we never look at any IFEval
test prompt, this does NOT violate the no-train-on-test rule.

Usage (example):
  PYTHONPATH=. python data/build_synthetic_ifeval.py \
      --base_model meta-llama/Llama-3.1-8B \
      --checkpoint_path tinker://...:train:0/sampler_weights/sanity_1b_step_425 \
      --num_prompts 800 --candidates_per_prompt 6 --max_constraints 2 \
      --output data/synthetic_ifeval.jsonl

Tip: start from your best SFT checkpoint, not the bare base, so the model
already produces coherent prose that has a chance of satisfying constraints.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import string
import time
from dataclasses import dataclass
from typing import Any, Iterable

import tinker
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from instruction_following_eval import instructions_registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("synth_ifeval")

# A short, hand-picked list of "safe" constraint keys that work well from a
# single-turn prompt without any cross-constraint coordination. We intentionally
# skip a few that are awkward for synthetic generation:
#   - language:response_language (drives the model to non-English)
#   - combination:two_responses / repeat_prompt (multi-part outputs are noisy)
#   - detectable_format:constrained_response (forces a fixed sentence)
SAFE_CONSTRAINT_KEYS: list[str] = [
    "keywords:existence",
    "keywords:frequency",
    "keywords:forbidden_words",
    "keywords:letter_frequency",
    "length_constraints:number_sentences",
    "length_constraints:number_paragraphs",
    "length_constraints:number_words",
    "detectable_content:number_placeholders",
    "detectable_content:postscript",
    "detectable_format:number_bullet_lists",
    "detectable_format:number_highlighted_sections",
    "detectable_format:multiple_sections",
    "detectable_format:title",
    "detectable_format:json_format",
    "startend:end_checker",
    "change_case:capital_word_frequency",
    "change_case:english_capital",
    "change_case:english_lowercase",
    "punctuation:no_comma",
    "startend:quotation",
]

# Curated seed prompts. Topical breadth matters more than length here; we just
# need open-ended writing tasks the constraints can be applied on top of.
BASE_TASKS: list[str] = [
    "Write a short essay about renewable energy and its impact on the climate.",
    "Explain how a transformer neural network works to a curious high school student.",
    "Describe the process of making sourdough bread from start to finish.",
    "Write a friendly product review for a wireless mechanical keyboard.",
    "Summarize the rules of chess for someone who has never played before.",
    "Write a short biography of a fictional astronaut who landed on Mars.",
    "Compare the pros and cons of remote work versus office work.",
    "Describe an ideal weekend in San Francisco for a first-time visitor.",
    "Write a polite email asking your manager for a one-week vacation.",
    "Explain why exercise is important for mental health.",
    "Describe a fictional new city built on top of a coral reef.",
    "Write a children's bedtime story about a curious little fox.",
    "Explain how a vaccine teaches the immune system to fight a virus.",
    "Write a guide for beginners on how to start journaling daily.",
    "Describe what makes a great science fiction novel.",
    "Write a short blog post about the benefits of cold showers.",
    "Explain how compound interest works using a concrete example.",
    "Describe the lifecycle of a butterfly in plain language.",
    "Write a short cover letter for a junior software engineering position.",
    "Explain what climate adaptation strategies cities can adopt.",
    "Describe the cultural significance of tea in Japan.",
    "Write a positive performance review for a hardworking teammate.",
    "Explain how a search engine ranks web pages.",
    "Describe what makes a good chess opening for beginners.",
    "Write a friendly Slack announcement introducing a new team member.",
    "Explain how solar panels convert sunlight into electricity.",
    "Describe the physics behind how a paper airplane flies.",
    "Write a heartfelt thank-you note to a mentor.",
    "Describe how compost turns kitchen scraps into nutrient-rich soil.",
    "Write a short explainer comparing electric and gas-powered cars.",
]

# Word pools used when a constraint needs random keywords / postscripts / etc.
WORD_POOL: list[str] = [
    "balance", "horizon", "lattice", "nimbus", "echo", "kindle", "fjord",
    "ripple", "ember", "mosaic", "pebble", "willow", "compass", "harbor",
    "cinder", "verdant", "aurora", "crescent", "cascade", "lantern",
    "meadow", "quartz", "bramble", "saffron", "twilight", "cobalt",
    "harvest", "drizzle", "thicket", "anchor",
]

# Common English short words avoided when generating "forbidden_words" lists,
# because they would make almost any natural response fail.
COMMON_WORDS = {
    "the", "a", "an", "and", "or", "but", "of", "in", "on", "at", "to",
    "for", "with", "is", "are", "be", "this", "that", "it", "as",
}


@dataclass
class ConstraintInstance:
    key: str
    instance: Any
    description: str
    kwargs: dict[str, Any]


def _bool_or_str(x):
    if isinstance(x, bool):
        return str(x).lower()
    return x


def _sanitize_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Make kwargs JSON-serialisable for logging / dumping."""
    out = {}
    for k, v in kwargs.items():
        if isinstance(v, (list, tuple, set)):
            out[k] = [_bool_or_str(x) for x in v]
        else:
            out[k] = _bool_or_str(v)
    return out


def sample_constraint_kwargs(key: str, rng: random.Random) -> dict[str, Any] | None:
    """Pick reasonable random kwargs for a constraint. Return None to skip
    constraints we don't want to handle in this batch."""
    if key == "keywords:existence":
        n = rng.randint(1, 2)
        return {"keywords": rng.sample(WORD_POOL, n)}
    if key == "keywords:frequency":
        return {
            "keyword": rng.choice(WORD_POOL),
            "frequency": rng.randint(2, 4),
            "relation": rng.choice(["at least", "less than"]),
        }
    if key == "keywords:forbidden_words":
        # Pick 1-2 unusual words; common English words would tank the yield.
        words = [w for w in WORD_POOL if w not in COMMON_WORDS]
        return {"forbidden_words": rng.sample(words, rng.randint(1, 2))}
    if key == "keywords:letter_frequency":
        return {
            "letter": rng.choice("abcdefghijklmnopqrstuvwxyz"),
            "let_frequency": rng.randint(2, 4),
            "let_relation": rng.choice(["at least", "less than"]),
        }
    if key == "length_constraints:number_sentences":
        return {
            "num_sentences": rng.randint(3, 6),
            "relation": rng.choice(["at least", "less than"]),
        }
    if key == "length_constraints:number_paragraphs":
        return {"num_paragraphs": rng.randint(2, 4)}
    if key == "length_constraints:number_words":
        return {
            "num_words": rng.choice([60, 80, 100, 120, 150, 200]),
            "relation": rng.choice(["at least", "less than"]),
        }
    if key == "detectable_content:number_placeholders":
        return {"num_placeholders": rng.randint(2, 4)}
    if key == "detectable_content:postscript":
        return {"postscript_marker": rng.choice(["P.S.", "P.P.S", "Postscript:"])}
    if key == "detectable_format:number_bullet_lists":
        return {"num_bullets": rng.randint(2, 5)}
    if key == "detectable_format:number_highlighted_sections":
        return {"num_highlights": rng.randint(2, 4)}
    if key == "detectable_format:multiple_sections":
        return {
            "section_spliter": rng.choice(["Section", "PARAGRAPH"]),
            "num_sections": rng.randint(2, 4),
        }
    if key == "detectable_format:title":
        return {}
    if key == "detectable_format:json_format":
        return {}
    if key == "startend:end_checker":
        return {"end_phrase": rng.choice([
            "Have a great day!",
            "Thank you for reading.",
            "Until next time.",
        ])}
    if key == "change_case:capital_word_frequency":
        return {
            "capital_frequency": rng.randint(2, 4),
            "capital_relation": rng.choice(["at least", "less than"]),
        }
    if key == "change_case:english_capital":
        return {}
    if key == "change_case:english_lowercase":
        return {}
    if key == "punctuation:no_comma":
        return {}
    if key == "startend:quotation":
        return {}
    return None


def build_constraints(
    rng: random.Random, max_constraints: int
) -> list[ConstraintInstance]:
    """Sample 1..max_constraints non-conflicting constraints with kwargs."""
    n = rng.randint(1, max_constraints)
    available = list(SAFE_CONSTRAINT_KEYS)
    rng.shuffle(available)
    chosen: list[ConstraintInstance] = []
    blocked: set[str] = set()
    conflicts = instructions_registry.conflict_make(
        instructions_registry.INSTRUCTION_CONFLICTS
    )
    for key in available:
        if len(chosen) >= n:
            break
        if key in blocked:
            continue
        cls = instructions_registry.INSTRUCTION_DICT.get(key)
        if cls is None:
            continue
        kwargs = sample_constraint_kwargs(key, rng)
        if kwargs is None:
            continue
        try:
            inst = cls(key)
            description = inst.build_description(**kwargs)
        except Exception as e:  # bad random pick; just skip
            log.debug("Skipping %s due to build_description error: %s", key, e)
            continue
        chosen.append(ConstraintInstance(
            key=key,
            instance=inst,
            description=description,
            kwargs=_sanitize_kwargs(kwargs),
        ))
        blocked |= conflicts.get(key, {key})
    return chosen


def build_prompt(base_task: str, constraints: list[ConstraintInstance]) -> str:
    constraint_text = " ".join(c.description.strip() for c in constraints)
    return f"{base_task} {constraint_text}".strip()


def verify(response: str, constraints: list[ConstraintInstance]) -> bool:
    if not response or not response.strip():
        return False
    for c in constraints:
        try:
            if not c.instance.check_following(response):
                return False
        except Exception as e:
            log.debug("check_following error on %s: %s", c.key, e)
            return False
    return True


def submit_batch(
    sampling_client,
    prompts_tokens: list[types.ModelInput],
    candidates_per_prompt: int,
    sampling_params: types.SamplingParams,
):
    """Fire futures for every prompt and return them in order."""
    return [
        sampling_client.sample(
            prompt=p,
            num_samples=candidates_per_prompt,
            sampling_params=sampling_params,
        )
        for p in prompts_tokens
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", default="meta-llama/Llama-3.1-8B",
                        help="Tokenizer / renderer base model.")
    parser.add_argument("--checkpoint_path", default=None,
                        help="Tinker sampler-weights URI to sample from. "
                             "If omitted, samples from the bare base model "
                             "(yield will be very low).")
    parser.add_argument("--renderer_name", default="role_colon")
    parser.add_argument("--num_prompts", type=int, default=500,
                        help="How many distinct (base task + constraints) prompts to draw.")
    parser.add_argument("--candidates_per_prompt", type=int, default=6,
                        help="How many independent samples to draw per prompt; "
                             "we keep all that pass verification.")
    parser.add_argument("--max_constraints", type=int, default=2,
                        help="Maximum number of constraints stacked on a single prompt.")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Number of prompts whose futures we issue and "
                             "drain together (controls concurrency).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="data/synthetic_ifeval.jsonl")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Stop early once this many verified examples are written.")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    log.info("Loading tokenizer / renderer for %s ...", args.base_model)
    tokenizer = get_tokenizer(args.base_model)
    renderer = renderers.get_renderer(args.renderer_name, tokenizer=tokenizer)

    log.info("Connecting to Tinker ...")
    sc = tinker.ServiceClient()
    if args.checkpoint_path:
        log.info("Sampling from checkpoint: %s", args.checkpoint_path)
        sampling_client = sc.create_sampling_client(model_path=args.checkpoint_path)
    else:
        log.info("Sampling from base model (no checkpoint).")
        sampling_client = sc.create_sampling_client(base_model=args.base_model)

    stop_sequences = renderer.get_stop_sequences()
    sampling_params = types.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=stop_sequences,
    )
    log.info("Sampling params: %s", sampling_params)
    log.info("Stop sequences: %r", stop_sequences)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    written = 0
    attempted_prompts = 0
    attempted_samples = 0

    # Pre-build all prompt specs so we can stream them in batches.
    specs: list[dict[str, Any]] = []
    for _ in range(args.num_prompts):
        constraints = build_constraints(rng, args.max_constraints)
        if not constraints:
            continue
        base_task = rng.choice(BASE_TASKS)
        prompt_text = build_prompt(base_task, constraints)
        specs.append({
            "base_task": base_task,
            "prompt_text": prompt_text,
            "constraints": constraints,
        })
    log.info("Built %d prompt specs (requested %d).", len(specs), args.num_prompts)

    out_f = open(args.output, "w", encoding="utf-8")
    try:
        for batch_start in range(0, len(specs), args.batch_size):
            batch = specs[batch_start:batch_start + args.batch_size]
            prompts_tokens = [
                renderer.build_generation_prompt(
                    [{"role": "user", "content": s["prompt_text"]}]
                )
                for s in batch
            ]
            attempted_prompts += len(batch)
            attempted_samples += len(batch) * args.candidates_per_prompt
            t0 = time.time()
            futures = submit_batch(
                sampling_client=sampling_client,
                prompts_tokens=prompts_tokens,
                candidates_per_prompt=args.candidates_per_prompt,
                sampling_params=sampling_params,
            )

            for spec, fut in zip(batch, futures):
                try:
                    resp = fut.result()
                except Exception as e:
                    log.warning("Sampling failed for prompt: %s", e)
                    continue
                # Tinker SampleResponse has .sequences (list[SampledSequence]),
                # each with a .tokens int list.  Despite the docstring example
                # in tinker that says ``result.samples``, the attribute is
                # actually ``sequences``.
                for sample in resp.sequences:
                    text = tokenizer.decode(sample.tokens, skip_special_tokens=True)
                    # Some renderers tail a stop string; strip it.
                    for stop in stop_sequences:
                        if stop and text.endswith(stop):
                            text = text[: -len(stop)]
                    text = text.strip()
                    if not verify(text, spec["constraints"]):
                        continue
                    record = {
                        "messages": [
                            {"role": "user", "content": spec["prompt_text"]},
                            {"role": "assistant", "content": text},
                        ],
                        "task": "ifeval_synth",
                        "meta": {
                            "base_task": spec["base_task"],
                            "constraints": [
                                {"key": c.key, "kwargs": c.kwargs,
                                 "description": c.description}
                                for c in spec["constraints"]
                            ],
                        },
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()
                    written += 1
                    if args.max_examples and written >= args.max_examples:
                        log.info("Reached max_examples=%d; stopping.",
                                 args.max_examples)
                        return
            elapsed = time.time() - t0
            yield_pct = 100.0 * written / max(1, attempted_samples)
            log.info(
                "batch %3d-%3d | wrote=%d | yield=%.1f%% | %.1fs | %d prompts so far",
                batch_start, batch_start + len(batch), written, yield_pct,
                elapsed, attempted_prompts,
            )
    finally:
        out_f.close()
        log.info(
            "Done. Verified %d / %d candidates (%.1f%%). Output: %s",
            written, attempted_samples,
            100.0 * written / max(1, attempted_samples),
            args.output,
        )


if __name__ == "__main__":
    main()
