#!/usr/bin/env python3
"""Render Main Fig. 5 from stored capacity and lake-weighting results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "04_analysis_data/sensitivity_analyses"
DEFAULT_OUT = ROOT / ".build/figures"

TEAL = "#006A78"
ORANGE = "#D55E00"
GRAPHITE = "#30343B"
NEUTRAL = "#747B83"
TEXT = "#20242A"


def public_source_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    capacity = pd.read_csv(BASE / "selection_capacity/external_capacity_curve_summary.csv")
    weighting = pd.read_csv(BASE / "lake_equal_weighting/lake_equal_weight_primary_effect.csv")
    lakes = pd.read_csv(BASE / "lake_equal_weighting/lake_equal_weight_lake_level_contrasts.csv")
    capacity = capacity.rename(columns={
        "capacity_fraction": "selection_fraction",
        "difference_pp": "coverage_difference_pp",
        "ci_low_pp": "interval_low_pp",
        "ci_high_pp": "interval_high_pp",
        "multi_horizon_captured_events": "multi_horizon_captured_events",
        "multi_horizon_capture_rate": "multi_horizon_event_coverage",
        "direct_60_day_captured_events": "direct_60_day_captured_events",
        "direct_60_day_capture_rate": "direct_60_day_event_coverage",
        "all_events_denominator": "event_denominator",
    })
    weighting = weighting.rename(columns={
        "difference_pp": "coverage_difference_pp",
        "ci_low_pp": "interval_low_pp",
        "ci_high_pp": "interval_high_pp",
        "multi_horizon_captured": "multi_horizon_captured_events",
        "direct_60_day_captured": "direct_60_day_captured_events",
        "event_episode_denominator": "event_denominator",
    })
    weighting["weighting_scheme"] = weighting["weighting_scheme"].replace({
        "episode_weighted": "Event-weighted",
        "lake_equal": "Lake-equal",
    })
    lakes = lakes.rename(columns={
        "lake_difference_pp": "lake_specific_coverage_difference_pp",
        "multi_horizon_captured_episode_count": "multi_horizon_captured_event_count",
        "direct_60_day_captured_episode_count": "direct_60_day_captured_event_count",
        "multi_horizon_lake_rate": "multi_horizon_lake_event_coverage",
        "direct_60_day_lake_rate": "direct_60_day_lake_event_coverage",
        "episode_count": "event_count",
    })
    for frame in (capacity, weighting, lakes):
        for column in frame.select_dtypes(include=["object"]).columns:
            frame[column] = (frame[column].astype(str)
                             .str.replace("multi_horizon", "multi-horizon", case=False, regex=False)
                             .str.replace("direct", "direct-60-day", case=False, regex=False)
                             .str.replace("external", "temporal-evaluation", case=False, regex=False)
                             .str.replace("episode", "event", case=False, regex=False))
    return capacity, weighting, lakes


def style_axis(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(TEXT)
    ax.spines[["left", "bottom"]].set_linewidth(0.5)
    ax.tick_params(labelsize=5.5, colors=TEXT, length=2, width=0.5)
    ax.grid(False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output_root.resolve()
    main_dir = out / "figures/main"
    vector_dir = out / "figures/vector"
    source_dir = out / "figures/source_data/fig5"
    main_dir.mkdir(parents=True, exist_ok=True)
    vector_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    capacity, weighting, lakes = public_source_tables()
    capacity.to_csv(source_dir / "fig5a_capacity_response.csv", index=False)
    weighting.to_csv(source_dir / "fig5b_weighting_effect.csv", index=False)
    lakes.to_csv(source_dir / "fig5b_lake_level_differences.csv", index=False)
    fig = plt.figure(figsize=(18 / 2.54, 9 / 2.54), dpi=300, facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.12], left=0.09, right=0.98, top=0.88, bottom=0.20, wspace=0.38)

    ax = fig.add_subplot(gs[0, 0])
    x = capacity["selection_fraction"].to_numpy(float) * 100
    y = capacity["coverage_difference_pp"].to_numpy(float)
    lo = capacity["interval_low_pp"].to_numpy(float)
    hi = capacity["interval_high_pp"].to_numpy(float)
    ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o-", color=TEAL, markerfacecolor=TEAL,
                markeredgecolor="white", markeredgewidth=0.5, markersize=4.3,
                linewidth=1.0, elinewidth=0.5, capsize=2.5)
    ax.axhline(0, color=GRAPHITE, linewidth=0.5)
    ax.axvline(10, color=NEUTRAL, linewidth=0.7, linestyle=(0, (3, 2)))
    ax.text(10.8, max(hi) - 0.4, "Primary 10% selection fraction", fontsize=5.5, va="top", color=TEXT)
    ax.set_xticks(x, [f"{value:g}%" for value in x])
    ax.set_xlabel("Selection fraction", fontsize=6.5)
    ax.set_ylabel("Event coverage difference (pp)", fontsize=6.5)
    ax.set_title("a  Capacity response", loc="left", fontsize=7, pad=8)
    for xi, yi in zip(x, y):
        ax.annotate(f"{yi:+.2f}", (xi, yi), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=5.5)
    style_axis(ax)

    sub = gs[0, 1].subgridspec(2, 1, height_ratios=[1.25, 0.75], hspace=0.55)
    ax_ecdf = fig.add_subplot(sub[0, 0])
    vals = np.sort(lakes["lake_specific_coverage_difference_pp"].to_numpy(float))
    ecdf = np.arange(1, len(vals) + 1) / len(vals)
    ax_ecdf.step(vals, ecdf, where="post", color=NEUTRAL, linewidth=1.0)
    ax_ecdf.axvline(0, color=GRAPHITE, linewidth=0.5)
    ax_ecdf.set_xlabel("Per-lake coverage difference (pp)", fontsize=6.5)
    ax_ecdf.set_ylabel("Cumulative proportion", fontsize=6.5)
    ax_ecdf.set_title("b  Lake-level weighting", loc="left", fontsize=7, pad=8)
    ax_ecdf.text(0.98, 0.05, "355 event lakes", transform=ax_ecdf.transAxes, ha="right", fontsize=5.5)
    style_axis(ax_ecdf)

    ax_w = fig.add_subplot(sub[1, 0])
    schemes = ["Event-weighted", "Lake-equal"]
    positions = np.arange(2)
    interval_lows: list[float] = []
    interval_highs: list[float] = []
    interval_points: list[tuple[int, float]] = []
    for pos, scheme, color, marker, filled in zip(positions, schemes, [TEAL, ORANGE], ["o", "s"], [True, False]):
        row = weighting.loc[weighting["weighting_scheme"] == scheme].iloc[0]
        point = float(row["coverage_difference_pp"])
        low = float(row["interval_low_pp"])
        high = float(row["interval_high_pp"])
        interval_lows.append(low)
        interval_highs.append(high)
        interval_points.append((pos, point))
        ax_w.errorbar([point], [pos], xerr=[[point - low], [high - point]], fmt=marker,
                      color=color, markerfacecolor=color if filled else "white", markersize=4.3,
                      linewidth=0.7, elinewidth=0.5, capsize=2.5)
    ax_w.axvline(0, color=GRAPHITE, linewidth=0.5)
    span = max(interval_highs) - min(min(interval_lows), 0.0)
    ax_w.set_xlim(min(min(interval_lows), 0.0) - 0.02 * span,
                  max(interval_highs) + max(4.5, 0.16 * span))
    label_x = ax_w.get_xlim()[1] - 2.0
    for pos, point in interval_points:
        ax_w.annotate(
            f"{point:+.2f} pp",
            (label_x, pos),
            xytext=(0, 4.5),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=5.5,
            clip_on=True,
        )
    ax_w.set_yticks(positions, schemes)
    ax_w.set_xlabel("Event coverage difference (pp)", fontsize=6.5)
    ax_w.set_ylim(-0.6, 1.6)
    style_axis(ax_w)

    png = main_dir / "fig5.png"
    pdf = vector_dir / "fig5.pdf"
    svg = vector_dir / "fig5.svg"
    fig.savefig(png, dpi=600, facecolor="white")
    fig.savefig(pdf, facecolor="white", metadata={"Creator": "clean public figure renderer"})
    fig.savefig(svg, facecolor="white")
    plt.close(fig)
    (source_dir / "source_data_index.csv").write_text(
        "path,claim\nfig5a_capacity_response.csv,capacity response\nfig5b_weighting_effect.csv,weighting effect\nfig5b_lake_level_differences.csv,lake-level distribution\n",
        encoding="utf-8",
    )
    print(png)


if __name__ == "__main__":
    main()
