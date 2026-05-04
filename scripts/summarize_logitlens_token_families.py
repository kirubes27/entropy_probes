#!/usr/bin/env python3
"""Summarize logit-lens token families from saved direction-analysis JSONs.

This script is intentionally post-processing only: it does not load the model.
It reads the top-k positive logit-lens tokens already saved by
analyze_directions.py and reports how much of that saved top-k probability mass
falls into answer-token, abstention-token, delegate-token, and uncertainty-token
families.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEFAULT_RESULTS_DIR = Path(
    "downloads/prism_seq_20260503_085433/targeted_next_analyses/results"
)

DATASETS = [
    "TriviaMC_difficulty_filtered",
    "PopMC_0_difficulty_filtered",
]

FOCUS_DIRECTIONS = [
    "mc_entropy_{dataset}/mean_diff",
    "mc_logit_gap_{dataset}/mean_diff",
    "mc_answer_{dataset}/probe",
    "mc_answer_{dataset}/centroid",
    "d_meta_mc_uncert_delegate_{dataset}/mean_diff_entropy",
    "d_meta_mc_uncert_delegate_{dataset}/mean_diff_logit_gap",
    "d_delegate_logit_margin_{dataset}/mean_diff",
]

TOKEN_FAMILIES = {
    "answer": {"a", "b", "c", "d", "A", "B", "C", "D"},
    "abstain": {
        "none",
        "None",
        "NONE",
        "neither",
        "Neither",
        "NEITHER",
        "unknown",
        "Unknown",
        "unsure",
        "Unsure",
        "unclear",
        "Unclear",
    },
    "delegate": {"delegate", "Delegate", "DELEGATE"},
    "uncertainty": {
        "maybe",
        "Maybe",
        "perhaps",
        "Perhaps",
        "cannot",
        "Cannot",
        "can't",
        "uncertain",
        "Uncertain",
        "unknown",
        "Unknown",
        "unsure",
        "Unsure",
        "unclear",
        "Unclear",
    },
}


def normalize_token(token: str) -> str:
    return token.strip().replace("Ġ", "").replace("▁", "")


def token_family(token: str) -> str:
    cleaned = normalize_token(token)
    for family, members in TOKEN_FAMILIES.items():
        if cleaned in members:
            return family
    return "other"


def load_analysis(results_dir: Path, dataset: str) -> Dict:
    path = results_dir / f"{dataset}_direction_analysis_direction_analysis.json"
    with path.open() as f:
        return json.load(f)


def iter_rows(results_dir: Path) -> Iterable[Dict[str, object]]:
    for dataset in DATASETS:
        data = load_analysis(results_dir, dataset)
        per_layer = data["per_layer"]
        for layer_str, layer_data in sorted(per_layer.items(), key=lambda kv: int(kv[0])):
            layer = int(layer_str)
            lens = layer_data.get("logit_lens", {})
            for direction_template in FOCUS_DIRECTIONS:
                direction = direction_template.format(dataset=dataset)
                entry = lens.get(direction)
                if not entry:
                    continue
                tokens = entry["tokens"]
                probs = entry["probs"]
                masses = {family: 0.0 for family in [*TOKEN_FAMILIES.keys(), "other"]}
                family_tokens: Dict[str, List[str]] = {family: [] for family in masses}
                for token, prob in zip(tokens, probs):
                    family = token_family(token)
                    masses[family] += float(prob)
                    if family != "other":
                        family_tokens[family].append(token)

                top_tokens = " | ".join(
                    f"{token!r}:{float(prob):.3f}" for token, prob in zip(tokens[:8], probs[:8])
                )
                row = {
                    "dataset": dataset,
                    "layer": layer,
                    "direction": direction,
                    "top_k_mass": sum(float(p) for p in probs),
                    "answer_mass": masses["answer"],
                    "abstain_mass": masses["abstain"],
                    "delegate_mass": masses["delegate"],
                    "uncertainty_mass": masses["uncertainty"],
                    "other_mass": masses["other"],
                    "answer_tokens": " | ".join(family_tokens["answer"]),
                    "abstain_tokens": " | ".join(family_tokens["abstain"]),
                    "delegate_tokens": " | ".join(family_tokens["delegate"]),
                    "uncertainty_tokens": " | ".join(family_tokens["uncertainty"]),
                    "top_tokens": top_tokens,
                }
                yield row


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "layer",
        "direction",
        "top_k_mass",
        "answer_mass",
        "abstain_mass",
        "delegate_mass",
        "uncertainty_mass",
        "other_mass",
        "answer_tokens",
        "abstain_tokens",
        "delegate_tokens",
        "uncertainty_tokens",
        "top_tokens",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: List[Dict[str, object]], path: Path) -> None:
    focus = [
        row
        for row in rows
        if row["layer"] in {31, 32, 33, 40, 41, 42, 43}
        and (
            "d_meta_mc_uncert_delegate" in str(row["direction"])
            or "mc_answer" in str(row["direction"])
            or "mc_entropy" in str(row["direction"])
        )
    ]
    focus.sort(
        key=lambda r: (
            str(r["dataset"]),
            int(r["layer"]),
            -float(r["abstain_mass"]),
            -float(r["answer_mass"]),
        )
    )

    lines = [
        "# Logit-Lens Token-Family Summary",
        "",
        "Masses are raw softmax probability mass among the saved top-k positive logit-lens tokens.",
        "They are not full-vocabulary family probabilities unless the relevant family tokens appear in the saved top-k list.",
        "",
        "## Highest Abstention-Mass Rows",
        "",
        "| dataset | layer | direction | abstain_mass | answer_mass | abstain_tokens | top tokens |",
        "|---|---:|---|---:|---:|---|---|",
    ]
    top_abstain = sorted(rows, key=lambda r: float(r["abstain_mass"]), reverse=True)[:20]
    for row in top_abstain:
        lines.append(
            "| {dataset} | {layer} | `{direction}` | {abstain_mass:.3f} | {answer_mass:.3f} | {abstain_tokens} | {top_tokens} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Focus Rows",
            "",
            "| dataset | layer | direction | abstain_mass | answer_mass | abstain_tokens | answer_tokens |",
            "|---|---:|---|---:|---:|---|---|",
        ]
    )
    for row in focus:
        lines.append(
            "| {dataset} | {layer} | `{direction}` | {abstain_mass:.3f} | {answer_mass:.3f} | {abstain_tokens} | {answer_tokens} |".format(
                **row
            )
        )

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "logitlens_token_family_summary.csv",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "logitlens_token_family_summary.md",
    )
    args = parser.parse_args()

    rows = list(iter_rows(args.results_dir))
    write_csv(rows, args.out_csv)
    write_markdown(rows, args.out_md)
    print(f"Wrote {len(rows)} rows to {args.out_csv}")
    print(f"Wrote summary to {args.out_md}")


if __name__ == "__main__":
    main()
