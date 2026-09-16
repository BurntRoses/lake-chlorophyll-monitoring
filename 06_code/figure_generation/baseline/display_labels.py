"""Public terminology for the multi_horizon figure family.

Policy IDs and source columns use the project registry; renderers call
``public_label`` only at the display boundary.
"""

from __future__ import annotations

import re
from typing import Any

DISPLAY_LABELS: dict[str, str] = {
    "Multi-horizon": "Multi-horizon",
    "Multi-horizon only": "Multi-horizon only",
    "Direct 60-day": "Direct 60-day",
    "Direct 60-day only": "Direct 60-day only",
    "multi_horizon": "Multi-horizon",
    "historical_risk": "Historical risk",
    "matched_60_day_base": "Same-predictor 60-day base model",
    "multi_horizon": "Multi-horizon",
    "historical_risk": "Historical risk",
    "matched_60_day_base": "Same-predictor 60-day base model",
    "multi_horizon-only": "Multi-horizon only",
    "multi_horizon_only": "Multi-horizon only",
    "multi_horizon_only_direct_rank_ecdf": "Multi-horizon-only selections: Direct 60-day rank",
    "direct": "Direct 60-day",
    "direct_60_day": "Direct 60-day",
    "direct_60_day": "Direct 60-day",
    "direct_60_day": "Direct 60-day",
    "direct-only": "Direct 60-day only",
    "direct_only": "Direct 60-day only",
    "direct_60_day_only": "Direct 60-day only",
    "early_30_day_reference": "30-day reference",
    "basic_60_day_reference": "60-day backbone reference",
    "environment_60_day_reference": "60-day environment reference",
    "Episode identities": "Event coverage status",
    "Capacity closure": "Selection capacity over time",
    "Identity transitions": "Selection-set composition",
    "Cross-rank distance": "Cross-policy rank distribution",
    "Queue overlap": "Prioritized-list overlap",
    "CONUS support": "Evaluation domain",
    "Legal origin support": "Eligible prediction-date support",
    "Onset support": "Event-date uncertainty",
    "Separation support": "Event separation",
    "Washout accounting": "Effect of minimum event separation",
    "both": "Shared",
    "neither": "Neither",
    "none": "Neither",
    "episode": "Event",
    "episodes": "Events",
    "Origin": "Prediction date",
    "Origins": "Prediction dates",
    "130 origins": "130 prediction dates",
    "Candidate origins": "Eligible prediction dates per event",
    "severe": "High-chlorophyll-a",
    "External monitoring": "Lakes with field monitoring · 1,679",
    "Formal event lakes": "Event lakes · 355",
    "Equal 10% queue": "Matched top-10% selections",
    "Frozen scores": "Scores established during development",
    "Frozen before outcomes": "Scores established before evaluation",
    "External 2021-2025": "Temporal evaluation 2021–2025",
    "Selected overlap": "Shared selections",
    "Coverage (%)": "Event coverage (%)",
    "Difference (events)": "Difference in covered events",
    "Difference (pp)": "Event coverage difference (pp)",
    "Onset bracket (days)": "Event-date bracket width (days)",
    "Previous severe gap (days)": "Previous high-chlorophyll-a observation gap (days)",
    "Prior nonsevere count": "Prior event count",
    "Washout (days)": "Minimum event separation (days)",
    "Episode count": "Number of events",
    "Episode-separation days": "Minimum event separation (days)",
    "Legal origin support": "Eligible prediction-date support",
    "Legal origins": "Eligible forecast origins",
    "fixed target 696": "696 events",
    "all": "All events",
    "1-60 d": "1–60 days",
    "31-60 d": "31–60 days",
    "primary_31_60": "31–60 days",
    "secondary_1_60": "1–60 days",
    "Tight 60": "60-day boundary",
    "Main": "Baseline",
    "Strict centroid": "Strict 250-m centroid linkage",
    "other": "Other prespecified strategy",
    "primary": "Primary",
    "secondary": "Secondary",
}

# The project registry is the authority for all four strategy labels.
import json
from pathlib import Path
DISPLAY_LABELS.update({x['policy_id']:x['name_en'] for x in json.loads((Path(__file__).resolve().parents[3]/'07_metadata/policy_registry.json').read_text())})

MISSING_REASON_LABELS: dict[str, str] = {
    "0": "Valid observation available",
    "1": "Source record without a valid value",
    "2": "No source record in window",
    "3": "No lake history",
    "4": "Observation not yet available",
}

_REPLACEMENTS = (
    (re.compile(r"\bmulti_horizon-only\b", re.IGNORECASE), DISPLAY_LABELS["multi_horizon-only"]),
    (re.compile(r"\bmulti_horizon_only\b", re.IGNORECASE), DISPLAY_LABELS["multi_horizon_only"]),
    (re.compile(r"\bdirect[-_]event_60_day(?:_biomass)?\b", re.IGNORECASE), DISPLAY_LABELS["direct_60_day"]),
    (re.compile(r"\bdirect-only\b", re.IGNORECASE), DISPLAY_LABELS["direct-only"]),
    (re.compile(r"\bsingle-horizon\b", re.IGNORECASE), DISPLAY_LABELS["direct_60_day"]),
    (re.compile(r"\bmulti_horizon\b", re.IGNORECASE), DISPLAY_LABELS["multi_horizon"]),
    (re.compile(r"\bdirect\b(?!\s*60[-–]day)", re.IGNORECASE), DISPLAY_LABELS["direct"]),
    (re.compile(r"\bwashout\b", re.IGNORECASE), DISPLAY_LABELS["Washout (days)"]),
    (re.compile(r"\bonset\b", re.IGNORECASE), "event-date"),
    (re.compile(r"\bsevere\b", re.IGNORECASE), DISPLAY_LABELS["severe"]),
    (re.compile(r"\bqueue\b", re.IGNORECASE), "prioritized list"),
    (re.compile(r"\blegal origins?\b", re.IGNORECASE), "eligible forecast origins"),
    (re.compile(r"\borigins?\b", re.IGNORECASE), lambda match: "prediction dates" if match.group(0).lower().endswith("s") else "prediction date"),
    (re.compile(r"\bepisodes?\b", re.IGNORECASE), lambda match: "Events" if match.group(0)[0].isupper() else "events"),
    (re.compile(r"\bCode\s*([0-9]+)\b", re.IGNORECASE), lambda match: missing_reason_label(match.group(1))),
)


def public_label(value: Any, default: str | None = None) -> str:
    """Resolve one display token while leaving source IDs untouched."""
    text = "" if value is None else str(value)
    if text in DISPLAY_LABELS:
        return DISPLAY_LABELS[text]
    for pattern, replacement in _REPLACEMENTS:
        text = pattern.sub(replacement, text)
    return text if text else (default or text)


def missing_reason_label(value: Any) -> str:
    """Decode numeric missingness codes at the display boundary."""
    return MISSING_REASON_LABELS.get(str(value), public_label(value, default="Unknown observation support"))


_ADAPTER_INSTALLED = False


def install_label_adapter() -> None:
    """Sanitize text at Matplotlib's display boundary.

    The renderer keeps internal IDs in all joins and assertions. This adapter
    only touches strings handed to labels, titles, annotations and tick labels.
    """
    global _ADAPTER_INSTALLED
    if _ADAPTER_INSTALLED:
        return
    import matplotlib.axes
    import matplotlib.figure

    _ADAPTER_INSTALLED = True
    original = {
        "xlabel": matplotlib.axes.Axes.set_xlabel,
        "ylabel": matplotlib.axes.Axes.set_ylabel,
        "title": matplotlib.axes.Axes.set_title,
        "text": matplotlib.axes.Axes.text,
        "annotate": matplotlib.axes.Axes.annotate,
        "xticks": matplotlib.axes.Axes.set_xticks,
        "yticks": matplotlib.axes.Axes.set_yticks,
        "figure_text": matplotlib.figure.Figure.text,
        "legend": matplotlib.axes.Axes.legend,
    }

    def set_xlabel(self, label, *args, **kwargs):
        return original["xlabel"](self, public_label(label), *args, **kwargs)

    def set_ylabel(self, label, *args, **kwargs):
        return original["ylabel"](self, public_label(label), *args, **kwargs)

    def set_title(self, label, *args, **kwargs):
        return original["title"](self, public_label(label), *args, **kwargs)

    def text(self, x, y, label, *args, **kwargs):
        return original["text"](self, x, y, public_label(label), *args, **kwargs)

    def annotate(self, label, *args, **kwargs):
        return original["annotate"](self, public_label(label), *args, **kwargs)

    def set_xticks(self, ticks, labels=None, *args, **kwargs):
        if labels is not None:
            labels = [public_label(label) for label in labels]
        return original["xticks"](self, ticks, labels=labels, *args, **kwargs)

    def set_yticks(self, ticks, labels=None, *args, **kwargs):
        if labels is not None:
            labels = [public_label(label) for label in labels]
        return original["yticks"](self, ticks, labels=labels, *args, **kwargs)

    def figure_text(self, x, y, label, *args, **kwargs):
        return original["figure_text"](self, x, y, public_label(label), *args, **kwargs)

    def legend(self, *args, **kwargs):
        if "labels" in kwargs and kwargs["labels"] is not None:
            kwargs["labels"] = [public_label(label) for label in kwargs["labels"]]
        elif len(args) >= 2 and args[1] is not None:
            args = list(args)
            args[1] = [public_label(label) for label in args[1]]
            args = tuple(args)
        result = original["legend"](self, *args, **kwargs)
        if result is not None:
            for item in result.get_texts():
                item.set_text(public_label(item.get_text()))
        return result

    matplotlib.axes.Axes.set_xlabel = set_xlabel
    matplotlib.axes.Axes.set_ylabel = set_ylabel
    matplotlib.axes.Axes.set_title = set_title
    matplotlib.axes.Axes.text = text
    matplotlib.axes.Axes.annotate = annotate
    matplotlib.axes.Axes.set_xticks = set_xticks
    matplotlib.axes.Axes.set_yticks = set_yticks
    matplotlib.figure.Figure.text = figure_text
    matplotlib.axes.Axes.legend = legend
