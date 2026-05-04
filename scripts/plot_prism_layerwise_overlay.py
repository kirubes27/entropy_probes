#!/usr/bin/env python3
"""Create layerwise overlay figures from completed PRISM runs.

This is a post-processing script. It reads existing transfer, ablation, and
steering JSON outputs and writes a normalized overlay figure plus raw CSV values.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "downloads" / "prism_seq_20260503_085433" / "paper_ready"


DATASETS = {
    "TriviaMC": {
        "transfer": ROOT
        / "downloads/prism_seq_20260503_085433/targeted_next_analyses/results/"
        / "TriviaMC_difficulty_filtered_meta_delegate_transfer_results_options_newline.json",
        "ablation": ROOT
        / "downloads/h100_20260314_001833/mirror/results/"
        / "TriviaMC_difficulty_filtered_ablation_delegate_uncertainty_logit_gap_mean_diff_pos-final_logit_margin_results.json",
        "steering": ROOT
        / "downloads/h100_20260314_001833/mirror/results/"
        / "TriviaMC_difficulty_filtered_steering_delegate_uncertainty_logit_gap_mean_diff_pos-final_logit_margin_results.json",
    },
    "PopMC": {
        "transfer": ROOT
        / "downloads/prism_seq_20260503_085433/targeted_next_analyses/results/"
        / "PopMC_0_difficulty_filtered_meta_delegate_transfer_results_options_newline.json",
        "ablation": ROOT
        / "downloads/h200_exp3_20260322_002155/mirror/results/"
        / "PopMC_0_difficulty_filtered_ablation_delegate_uncertainty_logit_gap_mean_diff_pos-final_logit_margin_results.json",
        "steering": ROOT
        / "downloads/h200_exp3_20260322_002155/mirror/results/"
        / "PopMC_0_difficulty_filtered_steering_delegate_uncertainty_logit_gap_mean_diff_pos-final_logit_margin_results.json",
    },
}


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open() as f:
        return json.load(f)


def get_transfer_series(path: Path) -> dict[int, float]:
    data = load_json(path)
    per_layer = data["mean_diff_transfer"]["logit_gap"]["per_layer"]
    return {int(k): float(v["centered_r2"]) for k, v in per_layer.items()}


def get_ablation_series(path: Path) -> dict[int, float]:
    data = load_json(path)
    return {
        int(k): abs(float(v["correlation_change"]))
        for k, v in data["per_layer"].items()
        if "correlation_change" in v
    }


def get_steering_series(path: Path) -> dict[int, float]:
    data = load_json(path)
    return {
        int(k): abs(float(v["introspection_slope"]))
        for k, v in data["per_layer"].items()
        if "introspection_slope" in v
    }


def normalize(values: list[float]) -> list[float]:
    finite = [abs(v) for v in values if v == v]
    mx = max(finite) if finite else 0.0
    if mx <= 0:
        return [0.0 for _ in values]
    return [v / mx for v in values]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    for ax, (name, paths) in zip(axes, DATASETS.items()):
        transfer = get_transfer_series(paths["transfer"])
        ablation = get_ablation_series(paths["ablation"])
        steering = get_steering_series(paths["steering"])

        layers = list(range(80))
        transfer_vals = [transfer.get(layer, float("nan")) for layer in layers]
        ablation_vals = [ablation.get(layer, float("nan")) for layer in layers]
        steering_vals = [steering.get(layer, float("nan")) for layer in layers]

        for layer in layers:
            rows.append(
                {
                    "dataset": name,
                    "layer": layer,
                    "transfer_r2_options_newline_mean_diff_logit_gap": transfer.get(layer, ""),
                    "ablation_abs_delta_r_final_mean_diff_logit_gap": ablation.get(layer, ""),
                    "steering_abs_slope_final_mean_diff_logit_gap": steering.get(layer, ""),
                }
            )

        ax.plot(layers, normalize(transfer_vals), label="D->M transfer R2 (options_newline)", lw=2.2)
        ax.plot(layers, normalize(ablation_vals), label="|ablation delta r| (final)", lw=2.2)
        ax.plot(layers, normalize(steering_vals), label="|steering slope| (final)", lw=2.2)

        ax.axvspan(31, 33, color="#2b6cb0", alpha=0.08, lw=0)
        ax.axvspan(40, 43, color="#c05621", alpha=0.08, lw=0)
        ax.set_title(name)
        ax.set_ylabel("Normalized within-trace")
        ax.grid(True, color="#dddddd", lw=0.7, alpha=0.7)
        ax.set_ylim(-0.03, 1.08)

        best_transfer = max(transfer.items(), key=lambda kv: kv[1])
        best_ablation = max(ablation.items(), key=lambda kv: kv[1])
        best_steering = max(steering.items(), key=lambda kv: kv[1])
        ax.text(
            0.01,
            0.05,
            (
                f"peaks: transfer L{best_transfer[0]}={best_transfer[1]:.3f}; "
                f"ablation L{best_ablation[0]}={best_ablation[1]:.3f}; "
                f"steering L{best_steering[0]}={best_steering[1]:.3f}"
            ),
            transform=ax.transAxes,
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.92},
        )

    axes[-1].set_xlabel("Layer")
    axes[0].legend(loc="upper right", fontsize=8.5)
    fig.suptitle("Layerwise PRISM Overlay: Transfer, Ablation, Steering", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    fig_path = OUT_DIR / "fig_layerwise_overlay_trivia_popmc.png"
    csv_path = OUT_DIR / "layerwise_overlay_values.csv"
    fig.savefig(fig_path, dpi=220)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {fig_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
