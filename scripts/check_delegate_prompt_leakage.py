#!/usr/bin/env python3
"""Check whether abstention/non-answer words appear in delegate prompts.

This is a lightweight contamination check for the logit-lens result. It renders
the answer-or-delegate prompt without a tokenizer chat template, then searches
for predeclared token families in both the prompt template and rendered
question prompts.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.questions import load_questions
from tasks import ANSWER_OR_DELEGATE_SETUP, ANSWER_OR_DELEGATE_SYSPROMPT, format_answer_or_delegate_prompt


DEFAULT_OUT = Path(
    "downloads/prism_seq_20260503_085433/targeted_next_analyses/results/delegate_prompt_leakage_check.csv"
)

DEFAULT_DATASETS = [
    ("TriviaMC_difficulty_filtered", 500),
    ("PopMC_0_difficulty_filtered", 500),
    ("SimpleMC_difficulty_filtered", 408),
]

SEARCH_TERMS = [
    "none",
    "neither",
    "unknown",
    "unsure",
    "unclear",
    "abstain",
    "cannot answer",
    "can't answer",
    "delegate",
    "confidence",
    "answer",
]


def count_terms(text: str) -> Dict[str, int]:
    lowered = text.lower()
    counts = {}
    for term in SEARCH_TERMS:
        pattern = r"\b" + re.escape(term.lower()) + r"\b"
        counts[term] = len(re.findall(pattern, lowered))
    return counts


def iter_dataset_specs(values: List[str]) -> Iterable[tuple[str, int]]:
    if not values:
        yield from DEFAULT_DATASETS
        return
    for value in values:
        if ":" in value:
            name, count = value.split(":", 1)
            yield name, int(count)
        else:
            yield value, 500


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        action="append",
        help="Dataset spec NAME[:N]. Defaults to Trivia 500, PopMC 500, SimpleMC filtered 408.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows = []

    template_text = ANSWER_OR_DELEGATE_SYSPROMPT + "\n\n" + ANSWER_OR_DELEGATE_SETUP
    template_counts = count_terms(template_text)
    rows.append(
        {
            "dataset": "__template_only__",
            "n_prompts": 1,
            **template_counts,
            "example_hits": "; ".join(k for k, v in template_counts.items() if v),
        }
    )

    for dataset, n in iter_dataset_specs(args.dataset or []):
        try:
            random.seed(args.seed)
            questions = load_questions(dataset, num_questions=n, seed=args.seed)
        except Exception as exc:
            rows.append(
                {
                    "dataset": dataset,
                    "n_prompts": 0,
                    **{term: -1 for term in SEARCH_TERMS},
                    "example_hits": f"LOAD_FAILED: {exc}",
                }
            )
            continue

        total = {term: 0 for term in SEARCH_TERMS}
        example_hits = []
        for i, question in enumerate(questions):
            prompt, _, _ = format_answer_or_delegate_prompt(
                question,
                tokenizer=None,
                trial_index=i,
                alternate_mapping=True,
                use_chat_template=False,
            )
            counts = count_terms(prompt)
            for term, count in counts.items():
                total[term] += count
            if len(example_hits) < 5:
                hits = [term for term, count in counts.items() if count]
                if hits:
                    example_hits.append(f"{i}:{','.join(hits)}")

        rows.append(
            {
                "dataset": dataset,
                "n_prompts": len(questions),
                **total,
                "example_hits": "; ".join(example_hits),
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "n_prompts", *SEARCH_TERMS, "example_hits"]
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote prompt leakage check to {args.out}")
    for row in rows:
        no_nonanswer = all(row.get(term, 0) == 0 for term in ["none", "neither", "unknown", "unsure", "unclear", "abstain"])
        print(
            f"{row['dataset']}: n={row['n_prompts']} non-answer-template-hit={'NO' if no_nonanswer else 'YES'} "
            f"none={row['none']} neither={row['neither']} unknown={row['unknown']} delegate={row['delegate']}"
        )


if __name__ == "__main__":
    main()
