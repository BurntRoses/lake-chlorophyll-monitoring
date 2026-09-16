#!/usr/bin/env python3
"""Render Extended Data Fig. 8 from the frozen same-feature source tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "03_figure_source_data/Figure_S8"
DEFAULT_OUT = ROOT / ".build/figures"

TEAL = "#006A78"
GRAPHITE = "#30343B"
TEXT = "#20242A"
NEUTRAL = "#747B83"


def public_identifier(value: str) -> str:
    return (value.replace("multi_horizon", "multi_horizon")
            .replace("candidate", "multi_horizon")
            .replace("baseline", "feature_matched")
            .replace("episode", "event"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output_root.resolve()
    ed_dir = out / "figures/extended_data"
    vector_dir = out / "figures/vector"
    source_dir = out / "figures/source_data/ed_fig8"
    ed_dir.mkdir(parents=True, exist_ok=True)
    vector_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    target = pd.read_csv(SRC / "target_identity.csv")
    draws = pd.read_csv(SRC / "bootstrap_draws.csv")
    identity = pd.read_csv(SRC / "component_identity.csv")
    target = target.rename(columns={
        "multi_horizon_output": "candidate_policy",
        "matched_60_day_base": "baseline_policy",
        "multi_horizon_captured_events": "candidate_captured_events",
        "matched_60_day_base_captured_events": "baseline_captured_events",
        "multi_horizon_event_coverage": "candidate_capture_rate",
        "matched_60_day_base_event_coverage": "baseline_capture_rate",
        "interval_low_pp": "ci_low_pp",
        "interval_high_pp": "ci_high_pp",
    })
    draws = draws.rename(columns={"event_coverage_difference": "capture_rate_difference"})
    row = target.iloc[0]
    assert int(row["candidate_captured_events"]) == 263
    assert int(row["baseline_captured_events"]) == 274
    assert len(draws) == 5000
    public_target = target.rename(columns={
        "candidate_policy": "multi_horizon_output",
        "baseline_policy": "matched_60_day_base",
        "candidate_captured_events": "multi_horizon_captured_events",
        "baseline_captured_events": "matched_60_day_base_captured_events",
        "candidate_capture_rate": "multi_horizon_event_coverage",
        "baseline_capture_rate": "matched_60_day_base_event_coverage",
        "ci_low_pp": "interval_low_pp",
        "ci_high_pp": "interval_high_pp",
    })
    public_target = public_target.rename(columns={column: public_identifier(column) for column in public_target.columns})
    public_target["multi_horizon_output"] = "Multi-horizon"
    public_target["matched_60_day_base"] = "Same-predictor 60-day base model"
    if "feature_set_id" in public_target.columns:
        public_target["feature_set_id"] = "387 numeric predictors plus one ecoregion predictor"
    if "target_contrast" in public_target.columns:
        public_target["target_contrast"] = "Multi-horizon versus same-predictor 60-day base model"
    if "evidence_tier" in public_target.columns:
        public_target["evidence_tier"] = "Feature-matched component comparison"
    public_target.to_csv(source_dir / "ed8_target_identity.csv", index=False)
    public_draws = draws.rename(columns={"capture_rate_difference": "event_coverage_difference"})
    public_draws = public_draws.rename(columns={column: public_identifier(column) for column in public_draws.columns})
    public_draws.to_csv(source_dir / "ed8_bootstrap_draws.csv", index=False)
    identity = identity.rename(columns={
        "multi_horizon_feature_set_id": "multi_horizon_feature_set_id",
        "multi_horizon_n_numeric_predictors": "multi_horizon_numeric_predictors",
        "multi_horizon_n_categorical_predictors": "multi_horizon_categorical_predictors",
    })
    identity = identity.rename(columns={column: public_identifier(column) for column in identity.columns})
    for column in identity.select_dtypes(include=["object"]).columns:
        identity[column] = identity[column].astype(str).str.replace("multi_horizon", "multi-horizon", case=False, regex=False)
    identity.to_csv(source_dir / "ed8_component_identity.csv", index=False)
    draws["difference_pp"] = draws["capture_rate_difference"] * 100.0
    fig, ax = plt.subplots(figsize=(18 / 2.54, 8 / 2.54), dpi=300)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(TEXT)
    ax.tick_params(labelsize=5.5, width=0.5, length=2)
    value = float(row["difference_pp"])
    lo = float(row["ci_low_pp"])
    hi = float(row["ci_high_pp"])
    ax.errorbar([value], [0], xerr=[[value - lo], [hi - value]], fmt="o", color=TEAL,
                markersize=5.5, capsize=2.5, capthick=0.5, linewidth=0.7, zorder=3)
    ax.plot(draws["difference_pp"], np.full(len(draws), -0.18), linestyle="none", marker="|",
            markersize=2.4, markeredgewidth=0.3, color=NEUTRAL, alpha=0.12, zorder=1)
    ax.axvline(0, color=GRAPHITE, linewidth=0.7)
    ax.set_xlim(-3.5, 1.5)
    ax.set_ylim(-0.35, 0.35)
    ax.set_yticks([0], ["Multi-horizon\nSame-predictor 60-day base model"])
    ax.set_xlabel("Difference in event coverage (pp)", fontsize=6.5, labelpad=2)
    ax.set_title("Same-predictor 60-day base model comparison", loc="left", fontsize=7, color=TEXT, pad=10)
    ax.text(0.02, 0.88, "263 vs 274 events\n−1.58 pp\n95% CI: −2.70 to 0.00 pp", transform=ax.transAxes,
            fontsize=5.5, va="top", color=TEXT)
    ax.text(0.98, 0.04, "5,000 whole-lake bootstrap replicates", transform=ax.transAxes,
            fontsize=5.5, ha="right", va="bottom", color=TEXT)
    fig.tight_layout(pad=1.1)
    png = ed_dir / "ed_fig8.png"
    pdf = vector_dir / "ed_fig8.pdf"
    svg = vector_dir / "ed_fig8.svg"
    fig.savefig(png, dpi=600, facecolor="white")
    fig.savefig(pdf, dpi=300, facecolor="white", metadata={"Creator": "clean public figure renderer"})
    fig.savefig(svg, facecolor="white")
    plt.close(fig)
    (source_dir / "source_data_index.csv").write_text(
        "path,claim\ned8_target_identity.csv,feature-matched target identity\ned8_bootstrap_draws.csv,feature-matched output uncertainty\ned8_component_identity.csv,fold-local component identity\n",
        encoding="utf-8",
    )
    print(png)


if __name__ == "__main__":
    main()
