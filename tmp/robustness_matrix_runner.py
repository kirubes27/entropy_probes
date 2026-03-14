#!/usr/bin/env python3
"""
Run robustness experiments across model/dataset combos without editing core scripts.

This wrapper only overrides module-level constants at runtime and calls each script's main().
It is designed so existing experiment files remain unchanged.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


# -----------------------------
# User-configurable settings
# -----------------------------

# Modes:
# - "stage0_exp2": run prerequisite + Exp1/Exp2 only (best default for robustness sweeps)
# - "full": run Stage0 -> Exp1/2 -> Exp3A/B/C -> Exp4
# - "causal_only": run Exp3A/B/C -> Exp4 (assumes Stage0/Exp1/2 artifacts already exist)
PIPELINE_MODE = "stage0_exp2"

MODELS = [
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
]

DATASETS = [
    "TriviaMC_difficulty_filtered",
    # "MMLU",
    # "TruthfulQA",
    # "GPQA",
]

META_TASK = "delegate"
METRICS = ["logit_gap", "top_logit", "entropy"]
SEED = 42
TRAIN_SPLIT = 0.8
LOAD_IN_4BIT = True
LOAD_IN_8BIT = False

PROBE_POSITIONS = ["question_mark", "question_newline", "options_newline", "final"]

# Gate used to select options_newline transfer-qualified layers for Exp3A/Exp3C.
TRANSFER_R2_THRESHOLD = 0.30

# Exp4 steering settings
EXP4_LAYERS = list(range(20, 51))
EXP4_RUN_ROLE = "confirmatory"  # label in wrapper manifest for your own tracking


# -----------------------------
# Internal utilities
# -----------------------------


@dataclass
class ComboRun:
    model: str
    dataset: str
    mode: str
    started_at: str
    finished_at: str | None = None
    status: str = "running"
    steps: List[Dict[str, Any]] | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run_python_code(code: str) -> None:
    subprocess.run([sys.executable, "-u", "-c", code], check=True)


def _run_module(module: str, overrides: Dict[str, Any], label: str) -> None:
    assigns = "; ".join([f"m.{k}={repr(v)}" for k, v in overrides.items()])
    config_preview = {k: overrides[k] for k in sorted(overrides.keys())}
    code = (
        f"import {module} as m; "
        f"{assigns}; "
        f"print('{label}_CONFIG', {repr(config_preview)}); "
        f"m.main()"
    )
    _run_python_code(code)


def _verify_stage0_artifacts(dataset: str, model: str) -> Dict[str, str]:
    from core.config_utils import find_output_file
    from core.model_utils import get_model_dir_name

    model_dir = get_model_dir_name(
        model,
        model_path=None,
        load_in_4bit=LOAD_IN_4BIT,
        load_in_8bit=LOAD_IN_8BIT,
    )
    required = [
        f"{dataset}_mc_results.json",
        f"{dataset}_mc_logit_gap_directions.npz",
        f"{dataset}_mc_top_logit_directions.npz",
        f"{dataset}_mc_entropy_directions.npz",
    ]
    paths: Dict[str, str] = {}
    missing: List[str] = []
    for name in required:
        p = Path(find_output_file(name, model_dir=model_dir))
        paths[name] = str(p)
        if not p.exists():
            missing.append(str(p))
    if missing:
        raise RuntimeError("Missing Stage0 artifacts:\n" + "\n".join(missing))
    return paths


def _verify_exp2_and_gate(dataset: str, model: str) -> Dict[str, Any]:
    from core.config_utils import find_output_file
    from core.model_utils import get_model_dir_name

    model_dir = get_model_dir_name(
        model,
        model_path=None,
        load_in_4bit=LOAD_IN_4BIT,
        load_in_8bit=LOAD_IN_8BIT,
    )

    result_paths: Dict[str, Path] = {}
    for pos in PROBE_POSITIONS:
        p = Path(find_output_file(f"{dataset}_meta_{META_TASK}_transfer_results_{pos}.json", model_dir=model_dir))
        if not p.exists():
            raise RuntimeError(f"Missing Exp1/2 transfer file for position={pos}: {p}")
        result_paths[pos] = p

    with result_paths["options_newline"].open() as f:
        d = json.load(f)

    def pass_count(section: str) -> int:
        per = d.get(section, {}).get("logit_gap", {}).get("per_layer", {})
        c = 0
        for _, v in per.items():
            r2 = v.get("centered_r2", v.get("d2m_centered_r2", None))
            if r2 is not None and r2 >= TRANSFER_R2_THRESHOLD:
                c += 1
        return c

    probe_n = pass_count("transfer")
    md_n = pass_count("mean_diff_transfer")
    if probe_n == 0 and md_n == 0:
        raise RuntimeError(
            f"No options_newline layers pass R^2>={TRANSFER_R2_THRESHOLD} for logit_gap in probe/mean_diff."
        )

    return {
        "transfer_paths": {k: str(v) for k, v in result_paths.items()},
        "options_newline_probe_ge_threshold": probe_n,
        "options_newline_mean_diff_ge_threshold": md_n,
        "threshold": TRANSFER_R2_THRESHOLD,
    }


def _run_stage0(model: str, dataset: str) -> Dict[str, Any]:
    overrides = {
        "MODEL": model,
        "DATASET": dataset,
        "METRICS": METRICS,
        "NUM_QUESTIONS": 500,
        "SEED": SEED,
        "TRAIN_SPLIT": TRAIN_SPLIT,
        "LOAD_IN_4BIT": LOAD_IN_4BIT,
        "LOAD_IN_8BIT": LOAD_IN_8BIT,
        "FIND_ANSWER_DIRECTIONS": True,
    }
    _run_module("identify_mc_correlate", overrides, "STAGE0")
    paths = _verify_stage0_artifacts(dataset, model)
    return {"stage": "stage0", "status": "ok", "artifacts": paths}


def _run_exp12(model: str, dataset: str) -> Dict[str, Any]:
    overrides = {
        "MODEL": model,
        "DATASET": dataset,
        "META_TASK": META_TASK,
        "METRICS": METRICS,
        "PROBE_POSITIONS": PROBE_POSITIONS,
        "SEED": SEED,
        "TRAIN_SPLIT": TRAIN_SPLIT,
        "LOAD_IN_4BIT": LOAD_IN_4BIT,
        "LOAD_IN_8BIT": LOAD_IN_8BIT,
        "DELEGATE_CONFDIR_TARGET": "logit_margin",
        "FIND_CONFIDENCE_DIRECTIONS": False,
        "FIND_MC_UNCERTAINTY_DIRECTIONS": False,
        "FIND_META_MCQ_DIRECTIONS": False,
    }
    _run_module("test_meta_transfer", overrides, "EXP1_EXP2")
    gate = _verify_exp2_and_gate(dataset, model)
    return {"stage": "exp1_exp2", "status": "ok", "gate": gate}


def _run_exp3a(model: str, dataset: str) -> Dict[str, Any]:
    overrides = {
        "MODEL": model,
        "DATASET": dataset,
        "META_TASK": META_TASK,
        "DIRECTION_TYPE": "uncertainty",
        "METRIC": "logit_gap",
        "USE_TRANSFER_SPLIT": True,
        "TRAIN_SPLIT": TRAIN_SPLIT,
        "SEED": SEED,
        "METHODS": ["probe", "mean_diff"],
        "PROBE_POSITIONS": ["options_newline"],
        "CONFIDENCE_SIGNAL": "logit_margin",
        "LOAD_IN_4BIT": LOAD_IN_4BIT,
        "LOAD_IN_8BIT": LOAD_IN_8BIT,
    }
    _run_module("run_ablation_causality", overrides, "EXP3A")
    return {"stage": "exp3a", "status": "ok"}


def _run_exp3b(model: str, dataset: str) -> Dict[str, Any]:
    overrides = {
        "MODEL": model,
        "DATASET": dataset,
        "META_TASK": META_TASK,
        "DIRECTION_TYPE": "uncertainty",
        "METRIC": "logit_gap",
        "USE_TRANSFER_SPLIT": True,
        "TRAIN_SPLIT": TRAIN_SPLIT,
        "SEED": SEED,
        "METHODS": ["probe", "mean_diff"],
        "PROBE_POSITIONS": ["final"],
        "CONFIDENCE_SIGNAL": "logit_margin",
        "PROPAGATION_CAPTURE_STRIDE": 0,
        "LOAD_IN_4BIT": LOAD_IN_4BIT,
        "LOAD_IN_8BIT": LOAD_IN_8BIT,
    }
    _run_module("run_ablation_causality", overrides, "EXP3B")
    return {"stage": "exp3b", "status": "ok"}


def _run_exp3c(model: str, dataset: str) -> Dict[str, Any]:
    overrides = {
        "MODEL": model,
        "DATASET": dataset,
        "META_TASK": META_TASK,
        "DIRECTION_TYPE": "uncertainty",
        "METRIC": "logit_gap",
        "USE_TRANSFER_SPLIT": True,
        "TRAIN_SPLIT": TRAIN_SPLIT,
        "SEED": SEED,
        "METHODS": ["probe", "mean_diff"],
        "PROBE_POSITIONS": ["options_newline"],
        "CONFIDENCE_SIGNAL": "prob",
        "LOAD_IN_4BIT": LOAD_IN_4BIT,
        "LOAD_IN_8BIT": LOAD_IN_8BIT,
    }
    _run_module("run_ablation_causality", overrides, "EXP3C")
    return {"stage": "exp3c", "status": "ok"}


def _run_exp4(model: str, dataset: str) -> Dict[str, Any]:
    overrides = {
        "MODEL": model,
        "DATASET": dataset,
        "META_TASK": META_TASK,
        "DIRECTION_TYPE": "uncertainty",
        "METRIC": "logit_gap",
        "USE_TRANSFER_SPLIT": True,
        "TRAIN_SPLIT": TRAIN_SPLIT,
        "SEED": SEED,
        "METHODS": ["probe", "mean_diff"],
        "PROBE_POSITIONS": ["final"],
        "LAYERS": EXP4_LAYERS,
        "CONFIDENCE_SIGNAL": "logit_margin",
        "LOAD_IN_4BIT": LOAD_IN_4BIT,
        "LOAD_IN_8BIT": LOAD_IN_8BIT,
    }
    _run_module("run_steering_causality", overrides, "EXP4")
    return {"stage": "exp4", "status": "ok", "run_role": EXP4_RUN_ROLE, "layers": [min(EXP4_LAYERS), max(EXP4_LAYERS)]}


def _run_combo(model: str, dataset: str) -> ComboRun:
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = ComboRun(
        model=model,
        dataset=dataset,
        mode=PIPELINE_MODE,
        started_at=now,
        steps=[],
    )
    try:
        if PIPELINE_MODE in {"stage0_exp2", "full"}:
            run.steps.append(_run_stage0(model, dataset))
            run.steps.append(_run_exp12(model, dataset))
        elif PIPELINE_MODE == "causal_only":
            pass
        else:
            raise ValueError(f"Unknown PIPELINE_MODE={PIPELINE_MODE}")

        if PIPELINE_MODE in {"full", "causal_only"}:
            run.steps.append(_run_exp3a(model, dataset))
            run.steps.append(_run_exp3b(model, dataset))
            run.steps.append(_run_exp3c(model, dataset))
            run.steps.append(_run_exp4(model, dataset))

        run.status = "ok"
    except Exception as e:
        run.status = "failed"
        run.steps.append({"stage": "error", "status": "failed", "message": str(e)})
    finally:
        run.finished_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    return run


def main() -> None:
    repo = _repo_root()
    # Ensure relative imports/path behavior matches existing scripts.
    Path(repo).resolve()
    import os
    os.chdir(repo)

    manifest_dir = repo / "analysis" / "robustness_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"robustness_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    all_runs: List[Dict[str, Any]] = []
    print("=== ROBUSTNESS MATRIX RUNNER ===")
    print("mode:", PIPELINE_MODE)
    print("models:", MODELS)
    print("datasets:", DATASETS)
    print("exp4_run_role_label:", EXP4_RUN_ROLE)
    print("exp4_layers:", [min(EXP4_LAYERS), max(EXP4_LAYERS)])

    for model in MODELS:
        for dataset in DATASETS:
            print("\n--- combo start ---")
            print("model:", model)
            print("dataset:", dataset)
            run = _run_combo(model, dataset)
            all_runs.append(run.__dict__)
            # Save incrementally for crash resilience.
            manifest_path.write_text(json.dumps({"runs": all_runs}, indent=2))
            print("status:", run.status)
            print("--- combo end ---")

    print("\nManifest written to:", manifest_path)


if __name__ == "__main__":
    main()

