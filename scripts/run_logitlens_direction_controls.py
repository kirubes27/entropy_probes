#!/usr/bin/env python3
"""Run logit-lens controls for key uncertainty directions.

Controls:
1. Positive vs negative direction: compare top tokens for d and -d.
2. Random matched directions: estimate how often random vectors produce
   abstention/non-answer tokens in the top-k logit-lens projection.

This script loads only the tokenizer, lm_head, and final norm via
analyze_directions.py helpers. It does not run model forward passes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import analyze_directions as ad  # noqa: E402
from core.config_utils import get_output_path  # noqa: E402
from core.model_utils import get_model_dir_name  # noqa: E402
from core import get_model_short_name  # noqa: E402


MODEL = "meta-llama/Llama-3.3-70B-Instruct"
LOAD_IN_4BIT = True
LOAD_IN_8BIT = False
ADAPTER = None

DATASETS = [
    "TriviaMC_difficulty_filtered",
    "PopMC_0_difficulty_filtered",
]

LAYERS = [41, 42, 43]

TARGETS = [
    {
        "source_template": "d_meta_mc_uncert_delegate_{dataset}",
        "direction": "mean_diff_entropy",
        "label": "delegate_uncertainty_mean_diff_entropy",
    },
    {
        "source_template": "mc_entropy_{dataset}",
        "direction": "mean_diff",
        "label": "direct_entropy_mean_diff",
    },
    {
        "source_template": "mc_answer_{dataset}",
        "direction": "centroid",
        "label": "direct_answer_centroid",
    },
    {
        "source_template": "mc_answer_{dataset}",
        "direction": "probe",
        "label": "direct_answer_probe",
    },
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
        "abstain",
        "Abstain",
    },
    "delegate": {"delegate", "Delegate", "DELEGATE"},
}


def normalize_token(token: str) -> str:
    return token.strip().replace("Ġ", "").replace("▁", "")


def family_mass(tokens: List[str], probs: List[float]) -> Dict[str, float]:
    out = {family: 0.0 for family in [*TOKEN_FAMILIES.keys(), "other"]}
    for token, prob in zip(tokens, probs):
        cleaned = normalize_token(token)
        family = "other"
        for candidate, members in TOKEN_FAMILIES.items():
            if cleaned in members:
                family = candidate
                break
        out[family] += float(prob)
    return out


def format_tokens(tokens: List[str], probs: List[float], n: int = 10) -> str:
    return " | ".join(f"{tok!r}:{float(prob):.3f}" for tok, prob in zip(tokens[:n], probs[:n]))


def logit_lens(direction: np.ndarray, lm_head_weight, tokenizer, norm_weight, top_k: int) -> Dict:
    tokens, probs = ad.logit_lens_for_layer(direction, lm_head_weight, tokenizer, top_k, norm_weight)
    return {
        "tokens": tokens,
        "probs": probs,
        "family_mass": family_mass(tokens, probs),
    }


def random_summary(
    direction: np.ndarray,
    lm_head_weight,
    tokenizer,
    norm_weight,
    top_k: int,
    n_random: int,
    threshold: float,
    seed: int,
) -> Dict:
    rng = np.random.default_rng(seed)
    norm = np.linalg.norm(direction)
    abstain_masses = []
    answer_masses = []
    max_abstain = {"mass": -1.0, "tokens": [], "probs": []}

    for _ in range(n_random):
        rand = rng.normal(size=direction.shape).astype(direction.dtype)
        rand /= np.linalg.norm(rand) + 1e-12
        rand *= norm
        result = logit_lens(rand, lm_head_weight, tokenizer, norm_weight, top_k)
        abstain_mass = result["family_mass"]["abstain"]
        answer_mass = result["family_mass"]["answer"]
        abstain_masses.append(abstain_mass)
        answer_masses.append(answer_mass)
        if abstain_mass > max_abstain["mass"]:
            max_abstain = {
                "mass": abstain_mass,
                "tokens": result["tokens"],
                "probs": result["probs"],
            }

    abstain_arr = np.asarray(abstain_masses)
    answer_arr = np.asarray(answer_masses)
    return {
        "n_random": n_random,
        "threshold": threshold,
        "abstain_mean": float(abstain_arr.mean()),
        "abstain_std": float(abstain_arr.std(ddof=1)) if n_random > 1 else 0.0,
        "abstain_p95": float(np.quantile(abstain_arr, 0.95)),
        "abstain_p99": float(np.quantile(abstain_arr, 0.99)),
        "abstain_ge_threshold": int(np.sum(abstain_arr >= threshold)),
        "answer_mean": float(answer_arr.mean()),
        "answer_p95": float(np.quantile(answer_arr, 0.95)),
        "max_abstain_mass": float(max_abstain["mass"]),
        "max_abstain_tokens": max_abstain["tokens"],
        "max_abstain_probs": max_abstain["probs"],
    }


def load_all_directions(dataset: str) -> Dict:
    ad.MODEL = MODEL
    ad.ADAPTER = ADAPTER
    ad.LOAD_IN_4BIT = LOAD_IN_4BIT
    ad.LOAD_IN_8BIT = LOAD_IN_8BIT
    ad.DATASET_FILTER = dataset

    model_short = get_model_short_name(MODEL, load_in_4bit=LOAD_IN_4BIT, load_in_8bit=LOAD_IN_8BIT)
    model_dir = get_model_dir_name(MODEL, ADAPTER, LOAD_IN_4BIT, LOAD_IN_8BIT)
    direction_files = ad.find_direction_files(
        model_short,
        metric_filter=None,
        dataset_filter=dataset,
        exclude_adapters=True,
        model_dir=model_dir,
    )

    all_directions = {}
    for source, path in direction_files.items():
        all_directions[source] = ad.load_directions(path)
    return all_directions


def write_outputs(rows: List[Dict], payload: Dict, out_prefix: Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix(".json")
    csv_path = out_prefix.with_suffix(".csv")
    json_path.write_text(json.dumps(payload, indent=2))

    fieldnames = [
        "dataset",
        "layer",
        "label",
        "source",
        "direction_name",
        "sign",
        "abstain_mass",
        "answer_mass",
        "delegate_mass",
        "top_tokens",
        "random_n",
        "random_abstain_mean",
        "random_abstain_p95",
        "random_abstain_p99",
        "random_abstain_ge_threshold",
        "random_max_abstain_mass",
        "random_max_abstain_tokens",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--n-random", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args()

    from dotenv import load_dotenv
    from transformers import AutoTokenizer

    load_dotenv()
    print(f"Loading tokenizer: {MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, token=os.environ.get("HF_TOKEN"))

    print("Loading lm_head and final norm")
    lm_head_weight, norm_weight = ad.load_lm_head_and_norm(MODEL)

    rows = []
    details = {
        "config": {
            "model": MODEL,
            "layers": LAYERS,
            "datasets": DATASETS,
            "top_k": args.top_k,
            "n_random": args.n_random,
            "threshold": args.threshold,
            "seed": args.seed,
            "timestamp": datetime.now().isoformat(),
        },
        "results": [],
    }

    for dataset in DATASETS:
        print(f"\nDataset: {dataset}")
        all_directions = load_all_directions(dataset)
        for target in TARGETS:
            source = target["source_template"].format(dataset=dataset)
            direction_name = target["direction"]
            if source not in all_directions:
                print(f"  Missing source: {source}")
                continue
            for layer in LAYERS:
                direction = all_directions[source].get(layer, {}).get(direction_name)
                if direction is None:
                    print(f"  Missing direction: {source}/{direction_name} L{layer}")
                    continue

                pos = logit_lens(direction, lm_head_weight, tokenizer, norm_weight, args.top_k)
                neg = logit_lens(-direction, lm_head_weight, tokenizer, norm_weight, args.top_k)
                rand = random_summary(
                    direction,
                    lm_head_weight,
                    tokenizer,
                    norm_weight,
                    args.top_k,
                    args.n_random,
                    args.threshold,
                    args.seed + layer,
                )

                for sign, result in [("+d", pos), ("-d", neg)]:
                    row = {
                        "dataset": dataset,
                        "layer": layer,
                        "label": target["label"],
                        "source": source,
                        "direction_name": direction_name,
                        "sign": sign,
                        "abstain_mass": result["family_mass"]["abstain"],
                        "answer_mass": result["family_mass"]["answer"],
                        "delegate_mass": result["family_mass"]["delegate"],
                        "top_tokens": format_tokens(result["tokens"], result["probs"]),
                        "random_n": rand["n_random"] if sign == "+d" else "",
                        "random_abstain_mean": rand["abstain_mean"] if sign == "+d" else "",
                        "random_abstain_p95": rand["abstain_p95"] if sign == "+d" else "",
                        "random_abstain_p99": rand["abstain_p99"] if sign == "+d" else "",
                        "random_abstain_ge_threshold": rand["abstain_ge_threshold"] if sign == "+d" else "",
                        "random_max_abstain_mass": rand["max_abstain_mass"] if sign == "+d" else "",
                        "random_max_abstain_tokens": format_tokens(rand["max_abstain_tokens"], rand["max_abstain_probs"]) if sign == "+d" else "",
                    }
                    rows.append(row)

                details["results"].append(
                    {
                        "dataset": dataset,
                        "layer": layer,
                        "label": target["label"],
                        "source": source,
                        "direction_name": direction_name,
                        "positive": pos,
                        "negative": neg,
                        "random": rand,
                    }
                )
                print(
                    f"  L{layer} {target['label']}: "
                    f"+d abstain={pos['family_mass']['abstain']:.3f}, "
                    f"-d answer={neg['family_mass']['answer']:.3f}, "
                    f"random >= {args.threshold:.2f}: {rand['abstain_ge_threshold']}/{args.n_random}"
                )

    model_dir = get_model_dir_name(MODEL, ADAPTER, LOAD_IN_4BIT, LOAD_IN_8BIT)
    default_name = f"logitlens_direction_controls_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_name = args.output_name or default_name
    # Route through a .json filename so config_utils places the outputs in results/.
    out_prefix = get_output_path(f"{output_name}.json", model_dir=model_dir).with_suffix("")
    write_outputs(rows, details, out_prefix)


if __name__ == "__main__":
    main()
