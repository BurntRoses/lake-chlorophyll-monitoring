#!/usr/bin/env python3
"""Render study figures while verifying their scientific coordinates."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Polygon
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import ScaledTranslation
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[2]
BASELINE = Path(__file__).resolve().parent / "baseline"
OUT = ROOT / ".build/final_figures"
TMP = ROOT / ".build/figure_work"
Q, D, INK, GRAY = "#007C91", "#D55E00", "#20242A", "#68717C"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def array_hash(value):
    a = np.ma.asarray(value)
    return hashlib.sha256(np.ascontiguousarray(a.filled(np.nan), dtype="float64").tobytes()).hexdigest()


def geometry(fig):
    """Record all original data marks and ranges, independent of visual style."""
    result = {}
    for ax in fig.axes:
        result[id(ax)] = (ax, tuple(ax.get_xlim()), tuple(ax.get_ylim()), [])
        records = result[id(ax)][3]
        for obj in ax.lines:
            records.append((obj, "line", array_hash(obj.get_xydata())))
        for obj in ax.collections:
            records.append((obj, "collection", (array_hash(obj.get_offsets()),
                            tuple(array_hash(p.vertices) for p in obj.get_paths()))))
        for obj in ax.images:
            records.append((obj, "image", (array_hash(obj.get_array()), tuple(obj.get_extent()))))
        for obj in ax.patches:
            records.append((obj, "patch", array_hash(obj.get_path().vertices)))
    return result


def verify_geometry(before):
    counts = {"axes": 0, "lines": 0, "collections": 0, "images": 0, "patches": 0}
    for ax, xlim, ylim, marks in before.values():
        assert tuple(ax.get_xlim()) == xlim, ("xlim changed", xlim, ax.get_xlim())
        assert tuple(ax.get_ylim()) == ylim, ("ylim changed", ylim, ax.get_ylim())
        counts["axes"] += 1
        for obj, kind, old in marks:
            if kind == "line":
                new = array_hash(obj.get_xydata()); counts["lines"] += 1
            elif kind == "collection":
                new = (array_hash(obj.get_offsets()), tuple(array_hash(p.vertices) for p in obj.get_paths()))
                counts["collections"] += 1
            elif kind == "image":
                new = (array_hash(obj.get_array()), tuple(obj.get_extent())); counts["images"] += 1
            else:
                new = array_hash(obj.get_path().vertices); counts["patches"] += 1
            assert new == old, f"Changed {kind} scientific coordinates"
    return {"data_changed": "NO", "axis_ranges_changed": "NO", "original_artists_verified": counts}


def pos(ax, x, y, w, h):
    """Position an axis using centimetres from the upper left, not data limits."""
    fw, fh = ax.figure.get_size_inches() * 2.54
    ax.set_position([x / fw, 1 - (y + h) / fh, w / fw, h / fh])


def title(fig, letter, text, x, y):
    w, h = fig.get_size_inches() * 2.54
    fig.text(x / w, 1 - y / h, letter, fontsize=9, weight="bold", va="top", color=INK)
    fig.text((x + .55) / w, 1 - y / h, text, fontsize=8, va="top", color=INK)


TERMS = {
    "Shared": "Selected by both", "Both": "Covered by both", "Neither": "Neither selected",
    "Both selected": "Selected by both", "Frozen evaluation": "Evaluation workflow",
    "External support": "Evaluation set over time", "Resample support": "Bootstrap distribution",
    "Whole-lake bootstrap": "Lake-level bootstrap", "5,000 resamples": "5,000 replicates",
    "Number selected": "Selection capacity", "Cumulative proportion": "ECDF",
    "Multi-horizon": "Multi-horizon",
    "Difference in event coverage (pp)": "Event coverage difference (pp)",
    "Joint resamples": "Joint Bootstrap distribution", "Marginal resamples": "Marginal Bootstrap distributions",
    "Development resamples": "Development Bootstrap distributions",
    "Fraction of valid days": "Valid-day fraction",
    "30 d": "30-day window", "90 d": "90-day window",
    "Prioritized-list overlap": "Selection overlap",
}


def polish_text(fig):
    for obj in fig.findobj(Text):
        t = obj.get_text()
        obj.set_text(TERMS.get(t, t).replace("whole-lake bootstrap", "lake-level bootstrap")
                     .replace("Leave-one-state-out: ", "Excluding "))
        obj.set_fontfamily("Arial")
        obj.set_color(INK)
        obj.set_fontsize(6.8)
    for ax in fig.axes:
        ax.tick_params(labelsize=7, width=.5, length=2.5, pad=2)
        ax.xaxis.label.set_fontsize(7.5)
        ax.yaxis.label.set_fontsize(7.5)
        for t in [ax.title, ax._left_title, ax._right_title]:
            t.set_fontsize(7.5)
        for sp in ax.spines.values():
            sp.set_linewidth(.5)
        for line in ax.lines:
            if line.get_linewidth() > 1:
                line.set_linewidth(.8)
        for c in ax.collections:
            if len(c.get_linewidths()):
                c.set_linewidth(np.minimum(c.get_linewidths(), .7))
    for t in fig.texts:
        if re.fullmatch("[abc]", t.get_text()):
            t.set_fontsize(9); t.set_weight("bold")
        else:
            t.set_fontsize(8)


def reserved_key(ax, items, rows=1, fontsize=6.8):
    for t in list(ax.texts): t.set_visible(False)
    for c in list(ax.collections): c.set_visible(False)
    for l in list(ax.lines): l.set_visible(False)
    cols = int(np.ceil(len(items) / rows))
    for i, (label, color, marker, filled) in enumerate(items):
        row, col = divmod(i, cols)
        x, y = .025 + col / cols, 1 - (row + .5) / rows
        ax.scatter([x], [y], s=12, marker=marker, facecolors=color if filled else "white",
                   edgecolors=color, linewidths=.6, clip_on=False)
        ax.text(x + .045, y, label, va="center", fontsize=fontsize)


def small_axes(ax, n=3):
    if ax.get_xscale() == "linear" and not isinstance(ax.xaxis.get_major_locator(), mdates.DateLocator):
        if len(ax.get_xticks()) > n + 2: ax.xaxis.set_major_locator(MaxNLocator(nbins=n, prune="both"))
    if len(ax.get_yticks()) > n + 2: ax.yaxis.set_major_locator(MaxNLocator(nbins=n, prune="both"))


def polish_main(fid, fig):
    a = fig.axes
    if fid == "M1":
        # Four hard-coded connector strokes had no geographic anchors. Use
        # projected viewport outlines with matching inset letters instead.
        for artist in fig.artists:
            if isinstance(artist, Line2D): artist.set_visible(False)
        geo = module("geography_for_locators", BASELINE / "render_figure_family.py")
        for letter, extent in [("b", (-97.5,-82,40,49.5)), ("c", (-87.7,-79.8,24,31.1))]:
            lo,hi,bot,top=extent
            lon=np.r_[np.linspace(lo,hi,41),np.full(41,hi),np.linspace(hi,lo,41),np.full(41,lo)]
            lat=np.r_[np.full(41,bot),np.linspace(bot,top,41),np.full(41,top),np.linspace(top,bot,41)]
            xx,yy=geo.project_5070(lon,lat)
            a[0].add_patch(Polygon(np.c_[xx,yy],closed=True,fill=False,edgecolor=GRAY,
                                  lw=.65,linestyle=(0,(3,2)),zorder=8))
            lx,ly=geo.project_5070(lo,top)
            a[0].annotate(letter,(lx,ly),xytext=(-4,4),textcoords="offset points",
                          fontsize=8,weight="bold",ha="right",va="bottom",zorder=9,
                          bbox={"facecolor":"white","edgecolor":"none","pad":.4})
        # The inset maps retain their geography. One shared categorical key occupies the central gap.
        pos(a[5], 11.6, 6.0, 6, 1.9)
        reserved_key(a[5], [("Neither selected", "#A7ADB4", "o", False),
                           ("Selected by both", "#343A40", "D", True),
                           ("Multi-horizon only", Q, "o", True),
                           ("Direct 60-day only", D, "s", False),
                           ("Event lake", "#8E3B76", "o", False)], rows=3)
        a[5].text(.02, 1.08, "Selection status across 130 prediction dates",
                  transform=a[5].transAxes, ha="left", va="bottom", fontsize=6.8, color=INK)
        a[7].set_visible(False)
        pos(a[4], 12.1, 2.1, 4.8, 3.5)
        pos(a[6], 12.1, 9.0, 4.8, 3.9)
        for t in fig.texts:
            if t.get_text() in {"c", "Florida"}: t.set_y(1 - 8.3 / 14)
        for t in a[1].texts:
            t.set_text(t.get_text().replace("Lakes with field monitoring · 1,679", "Monitored lakes: 1,679")
                       .replace("Event lakes · 355", "Event lakes: 355")
                       .replace("Candidate lakes · 2,844", "Candidate lakes: 2,844"))
        for c in a[1].collections: c.set_edgecolor([GRAY, "#8E3B76"])
        a[0].collections[0].set_edgecolor(GRAY)
        for t in a[2].texts:
            if "50-km cell" in t.get_text():
                t.set_text("Candidate lakes\nper 50-km cell"); t.set_position((.05, -.12))
        pos(a[2], 6, 10, 2.1, 2.6)
        for t in a[3].texts:
            if t.get_text() == "EPSG:5070": t.set_y(-.15)
        # The inherited four-vertex line made a bow-tie scale bar.
        a[3].lines[1].set_visible(False); a[3].lines[2].set_visible(False)
        fig.canvas.draw()
        # Use the overview's EPSG:5070 map units to calibrate the scale length.
        map_length_px=500000*a[0].bbox.width/np.diff(a[0].get_xlim())[0]
        x0=.12; x1=x0+map_length_px/a[3].bbox.width
        a[3].plot([x0,x1],[.32,.32],c=INK,lw=.7)
        for x in [x0,x1]: a[3].plot([x,x],[.28,.36],c=INK,lw=.5)
        for t in a[3].texts:
            if t.get_text()=="500 km": t.set_x((x0+x1)/2)
    elif fid == "M2":
        for artist in fig.artists:
            if isinstance(artist,FancyArrowPatch): artist.set_visible(False)
        for x0,x1 in [(3.45,4.1),(6.45,7.1),(9.95,10.6),(13.45,14.1)]:
            fig.add_artist(FancyArrowPatch((x0/18,1-4.5/14),(x1/18,1-4.5/14),
                transform=fig.transFigure,arrowstyle="->",mutation_scale=7,
                shrinkA=0,shrinkB=0,lw=.7,color=GRAY))
        for ax in a[:5]:
            for t in ax.texts:
                t.set_text(t.get_text().replace("Top 10% selection", "Top 10%\nselection")
                           .replace("Lake-level bootstrap", "Lake-level\nbootstrap")
                           .replace("130 prediction dates", "130 prediction\ndates")
                           .replace("Risk scoring", "Scores fixed\nbefore evaluation"))
                if t.get_text()=="130 prediction\ndates":
                    t.set_va("top")
        pos(a[5], 8.2, .65, 8.8, .6)
        reserved_key(a[5], [("Multi-horizon", Q, "o", True), ("Direct 60-day", D, "s", False),
                           ("Event", "#8E3B76", "o", False)])
        # The old overlay had a second, misaligned 696 reference axis.
        a[9].set_visible(False)
        for t in a[9].texts: t.set_visible(False)
        a[8].text(.98, .08, "Original n = 696", transform=a[8].transAxes, ha="right", fontsize=6.8)
        a[8].set_xlabel("Event copies per\nBootstrap replicate")
        for ax in a[6:8]:
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        for t in fig.texts:
            if t.get_text() == "Bootstrap distribution": t.set_text("Bootstrap distribution")
        # Capacity is common to both strategies and receives a neutral encoding.
        for line in a[7].lines: line.set_color(GRAY)
        for c in a[7].collections: c.set_color(GRAY)
        pos(a[6],1.5,10,3.6,2.5)
        pos(a[7],6.9,10,3.6,2.5)
        a[7].set_ylabel("Selection\ncapacity")
    elif fid == "M4":
        pos(a[1], 5.2, 1.3, 3.7, .65)
        for t in a[1].texts:
            if t.get_text() == "Record density": t.set_position((.5, 1.35))
        # Quadrant labels are outside the dense scatter; leaders point to unchanged quadrants.
        for t in a[2].texts: t.set_visible(False)
        pos(a[2], 11.4, 2.8, 5.6, 4.6)
        # Keep the four definitions inside the zoom frame so they cannot collide
        # with the x-axis title at single-column size.
        labels = [("Multi-horizon\nonly", .052, .148, "left", "top"),
                  ("Neither selected", .148, .148, "right", "top"),
                  ("Selected by both", .052, .052, "left", "bottom"),
                  ("Direct 60-day\nonly", .148, .052, "right", "bottom")]
        for label, x0, y0, ha, va in labels:
            a[2].text(x0, y0, label, ha=ha, va=va, fontsize=6.2,
                      bbox={"facecolor":"white", "edgecolor":"none", "alpha":.78, "pad":.7}, zorder=50)
        a[2].set_xlabel("Multi-horizon normalized rank", labelpad=3)
        a[2].set_ylabel("Direct 60-day\nnormalized rank")
        a[2].set_xticks([.05,.10,.15]); a[2].set_yticks([.05,.10,.15])
        pos(a[3], 11.4, 1.85, 5.6, .45)
        a[3].set_visible(True); a[3].set_axis_off(); a[3].set_xlim(0,1); a[3].set_ylim(0,1)
        for obj in list(a[3].texts) + list(a[3].lines): obj.set_visible(False)
        a[3].text(1, .45, "0.10 selection reference", ha="right", va="center",
                  fontsize=6.8, color=GRAY)
        pos(a[4], 11.3, 10.8, 2.6, 1.7); pos(a[5], 14.8, 10.8, 2.2, 1.7)
        a[4].set_ylabel("Selected\nby both")
        a[5].set_xlabel("Shared selections /\nselection capacity")
        small_axes(a[4], 2); small_axes(a[5], 2)
    elif fid == "M5":
        fig.set_size_inches(18/2.54, 17/2.54)
        pos(a[2], 8.4, .7, 8.5, 1.05)
        pos(a[0], 1.5, 2.2, 6.9, 2.3); pos(a[1], 9.5, 2.2, 7.4, 2.3)
        reserved_key(a[2], [("Neither covered", "#A7ADB4", "o", False),
                           ("Covered by both", "#343A40", "D", True),
                           ("Multi-horizon only", Q, "o", True),
                           ("Direct 60-day only", D, "s", False)], rows=2)
        # Preserve both endpoint compositions and ranges; use equal-width rows so
        # the secondary endpoint remains readable after manuscript reduction.
        for ax in (a[4], a[5], a[7], a[8]):
            ax.set_xlabel("Event coverage\ndifference (pp)")
            small_axes(ax, 3)
        for ax in (a[3], a[6]):
            ax._left_title.set_text("Event coverage")
            ax.set_ylabel("Event coverage (%)")
        for ax in (a[4], a[7]): ax._left_title.set_text("Difference\nand 95% CI")
        for ax in (a[5], a[8]): ax._left_title.set_text("Bootstrap\nECDF")
        for ax, rect in zip(a[3:6], ([1.5,7.6,3.0,2.5],[6.0,7.6,3.4,2.5],[10.9,7.6,5.6,2.5])): pos(ax,*rect)
        for ax, rect in zip(a[6:9], ([1.5,12.5,3.0,2.5],[6.0,12.5,3.4,2.5],[10.9,12.5,5.6,2.5])): pos(ax,*rect)
        panel_titles={"a":(1.0,.8),"Event coverage status":(1.7,.8),
                      "b":(1.0,6.35),"Primary effect":(1.7,6.35),
                      "c":(1.0,11.25),"Secondary effect":(1.7,11.25)}
        for t in fig.texts:
            if t.get_text() in panel_titles:
                x,y=panel_titles[t.get_text()]; t.set_position((x/18,1-y/17))
        # Shade does not carry uncertainty; remove it and label the observed line explicitly.
        for c in a[5].collections: c.set_visible(False)
        a[5].lines[1].set_color(INK); a[5].lines[1].set_linestyle(":"); a[5].lines[1].set_linewidth(.6)
        a[5].text(.98,.08,"Observed",ha="right",transform=a[5].transAxes,fontsize=6.8)
        a[4].text(.98,.86,"+8.05 pp",ha="right",transform=a[4].transAxes,fontsize=7,weight="bold")
        xlim, ylim = a[8].get_xlim(), a[8].get_ylim()
        a[8].axvline(21.982758620689652, color=INK, lw=.6, ls=":")
        a[8].set_xlim(xlim); a[8].set_ylim(ylim)
        a[8].text(.98,.08,"Observed",ha="right",transform=a[8].transAxes,fontsize=6.8)
        a[7].text(.98,.86,"+21.98 pp",ha="right",transform=a[7].transAxes,fontsize=7,weight="bold")
        if a[8].get_xticklabels():
            a[8].get_xticklabels()[-1].set_ha("right")
        for c in a[6].collections[:1]: c.set_facecolor(Q)
        for c in a[7].collections: c.set_facecolor(Q)


def polish_extended(fid, fig):
    a = fig.axes
    if fid == "ED1":
        compact_steps = [r"Chl-a ≥75 μg L$^{-1}$", "Station linked", "Event date", ">60 d apart"]
        for idx, ax in enumerate(a[:2]):
            pos(ax, 1.2, 2.1 if idx == 0 else 5.0, 6.6, 2.1)
            ax.texts[0].set_text("Baseline" if idx == 0 else "Strict 250-m linkage")
            for t, label in zip(ax.texts[1:5], compact_steps):
                t.set_text(label); t.set_y(.34); t.set_fontsize(6.4)
            ax.texts[5].set_text(ax.texts[5].get_text().replace(" · ", "   "))
            ax.texts[5].set_y(.05); ax.texts[5].set_va("bottom"); ax.texts[5].set_fontsize(6.4)
        for ax in a[2:4]:
            ax.set_xticks([2021,2023,2025])
        pos(a[4], 1.5, 8.9, 6.5, 3.6)
        pos(a[5], 10.5, 8.9, 6.4, 3.6)
    elif fid == "ED2":
        pos(a[2], 3.6, 1.5, 4.4, .5)
        for t in a[2].texts: t.set_text(t.get_text().replace("30-day window", "30-day").replace("90-day window", "90-day"))
        a[0].set_title("30-day window", fontsize=7.5)
        a[1].set_title("90-day window", fontsize=7.5)
        for ax in a[:2]: ax.set_xlabel("Valid-day fraction"); ax.set_xticks([0,.5,1])
        for t in a[3].texts:
            t.set_text(t.get_text().replace("No source record in window", "No source record\nin window")
                       .replace("Valid observation available", "Valid observation\navailable")
                       .replace("Observation not yet available", "Observation not\nyet available"))
            if t.get_text() == "Observation not\nyet available":
                t.set_transform(t.get_transform() + ScaledTranslation(0, 3 / 72, fig.dpi_scale_trans))
        a[3].set_xlabel("Lake-prediction-date records")
        pos(a[6], 10, 1.55, 7, .65)
        # Sources are not strategies: distinguish them with neutral line styles.
        for ax in a[4:6]:
            for line, c in zip(ax.lines, [INK, GRAY, "#91989F"]): line.set_color(c)
        for t in a[6].texts: t.set_visible(False)
        for line in a[6].lines: line.set_visible(False)
        for x,label,c,ls in [(.01,"Combined",INK,"-"),(.34,"USGS",GRAY,"--"),(.57,"Water Quality Portal","#91989F",":")]:
            a[6].plot([x,x+.07],[.5,.5],c=c,ls=ls,lw=.8)
            a[6].text(x+.09,.5,label,va="center",fontsize=6.6)
        for ax in (a[7],a[8],a[9]): small_axes(ax,2)
        a[8].set_xlabel("Eligible prediction\ndates per event")
        pos(a[7], 10, 9, 1.5, 3.5); pos(a[8], 12.6, 9, 1.5, 3.5)
        a[9].set_yticks([0,1,2],["All", "1-60 d", "31-60 d"])
        a[9].set_xlabel("Number\nof events")
        pos(a[9],14.8,9,1.3,3.5)
        a[9].yaxis.tick_right()
        a[9].tick_params(axis="y",labelsize=6.4,pad=3)
        a[9].set_yticks([0,1,2],["All", "1-60 d", "31-60 d"])
        a[9].set_xticks([600,700])
    elif fid == "ED3":
        # The inherited renderer concatenated every policy and selected the first
        # capture column for both endpoints. Redraw from the frozen grouped table.
        yf = pd.read_csv(ROOT / "04_analysis_data/development_evaluation/model_selection/by_year_and_fold.csv")
        group_specs = [
            ("year", "captured_31_60", "Year\n31-60 days"),
            ("year", "captured_1_60", "Year\n1-60 days"),
            ("lake_fold", "captured_31_60", "Outer fold\n31-60 days"),
            ("lake_fold", "captured_1_60", "Outer fold\n1-60 days"),
        ]
        policy_styles = [
            ("multi_horizon", Q, "o", True),
            ("direct_60_day", D, "s", False),
        ]
        for i, (old_ax, (group_type, value_col, panel_title)) in enumerate(zip(a[:4], group_specs)):
            # Keep the inherited axis and artists in the geometry ledger while
            # presenting the corrected grouped view on a new display axis.
            old_ax.set_visible(False)
            ax = fig.add_axes(old_ax.get_position())
            sub = (yf[(yf.group_type == group_type) & yf.policy.isin([p[0] for p in policy_styles])]
                   .pivot(index="group", columns="policy", values=value_col)
                   .sort_index())
            x = np.arange(len(sub))
            for policy, color, marker, filled in policy_styles:
                values = sub[policy].to_numpy(float)
                ax.plot(x, values, color=color, lw=.8, zorder=3)
                ax.scatter(x, values, s=10, marker=marker,
                           facecolors=color if filled else "white", edgecolors=color,
                           linewidths=.6, zorder=4)
            labels = [str(int(v)) for v in sub.index]
            if group_type == "year":
                labels = [v[-2:] for v in labels]
            else:
                labels = [str(int(v) + 1) for v in sub.index]
            ax.set_xticks(x, labels)
            ax.set_xlim(-.35, len(x) - .65)
            ax.set_title(panel_title, fontsize=7.2, pad=2)
            ax.set_xlabel("")
            ax.set_ylabel("Covered events" if i % 2 == 0 else "")
            ax.tick_params(labelsize=7, width=.5, length=2.5, pad=2)
            ax.xaxis.label.set_fontsize(7.5); ax.yaxis.label.set_fontsize(7.5)
            for spine in ax.spines.values(): spine.set_linewidth(.5)
            pos(ax, *([1.5,2.5,2.6,1.65] if i == 0 else
                      [5.0,2.5,2.9,1.65] if i == 1 else
                      [1.5,5.45,2.6,1.65] if i == 2 else [5.0,5.45,2.9,1.65]))
            small_axes(ax, 3)
        a[4].set_visible(True)
        pos(a[4], 1.5, 1.35, 6.4, .55)
        reserved_key(a[4], [("Multi-horizon", Q, "o", True),
                            ("Direct 60-day", D, "s", False)], fontsize=6.4)
        for idx, ax in enumerate(a[5:11]):
            pos(ax, [10,12.5,15][idx%3], 3.8 if idx<3 else 6.15, [1.5,1.5,1.9][idx%3], 1.25)
            ax.set_xticks(range(5),["1","2","3","4","5"])
            ax.title.set_text(ax.title.get_text().replace("Average precision", "Average\nprecision")
                              .replace("Calibration ", "Calibration\n"))
            small_axes(ax,2)
            ax.set_xticks(range(5),["1","2","3","4","5"])
        pos(a[11], 9.5, 1.6, 7.5, 1.45)
        for obj in list(a[11].texts) + list(a[11].collections): obj.set_visible(False)
        model_keys=[("1  Multi-horizon",Q,"o",True),
                    ("2  Direct 60-day",D,"s",False),
                    ("3  60-day environment reference",GRAY,"^",True),
                    ("4  60-day backbone reference",GRAY,"D",False),
                    ("5  30-day reference",INK,"v",False)]
        for i,(label,c,m,filled) in enumerate(model_keys):
            y=.90-i*.19
            a[11].scatter([.025],[y],s=11,marker=m,facecolors=c if filled else "white",edgecolors=c,lw=.6,clip_on=False)
            a[11].text(.075,y,label,fontsize=6.8,va="center")
        for ax in a[12:]: ax.set_xlabel("Difference in\ncovered events")
        for ax,rect in zip(a[12:15],[(1.5,9.5,4.1,3.0),(7.0,9.5,4.1,3.0),(13.8,9.5,3.1,3.0)]):
            pos(ax,*rect)
    elif fid == "ED4":
        date_ticks = pd.to_datetime(["2022-01-01", "2024-01-01", "2026-01-01"])
        for ax in a[:4]:
            ax.set_xticks(date_ticks); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            small_axes(ax,2)
        a[2].set_ylabel("Rounding\nincrement")
        for t in a[3].texts: t.set_visible(False)
        a[3].set_ylabel("Selected records")
        a[5].set_xlabel("Shared selections / selection capacity")
        pos(a[4], 1.5, 5.95, 6.4, .8)
        reserved_key(a[4], [("Multi-horizon only",Q,"o",True),
                           ("Selected by both","#343A40","D",True),
                           ("Direct 60-day only",D,"s",False)], rows=2, fontsize=6.4)
        pos(a[8], 10, 6.15, 7, .65)
        for t in a[8].texts: t.set_visible(False)
        for line in a[8].lines: line.set_visible(False)
        a[8].plot([.01,.07],[.5,.5],c=Q,lw=.8); a[8].text(.1,.5,"Multi-horizon only",va="center",fontsize=6.8)
        a[8].plot([.53,.59],[.5,.5],c=D,lw=.8,ls="--"); a[8].text(.62,.5,"Direct 60-day only",va="center",fontsize=6.8)
        for c in a[1].lines: c.set_color(GRAY)
        for ax in a[6:8]:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    elif fid == "ED5":
        # This is the only taller canvas: 19 discrete rows need an honest 7-pt row pitch.
        fig.set_size_inches(18/2.54,17/2.54)
        for i,ax in enumerate(a):
            pos(ax, 5.3 if i%2==0 else 14.0, 2.0 if i<2 else 9.8, 7.0 if i%2==0 else 2.9, 5.8)
            if i%2:
                for line in ax.lines: line.set_linestyle("none")
            else:
                for t in ax.texts:
                    t.set_text(t.get_text().replace("Event-date uncertainty ≤60 d", "Date uncertainty ≤60 d")
                               .replace("Strict 250-m linkage", "250-m linkage"))
                    t.set_fontsize(7)
                    t.set_transform(ax.get_yaxis_transform())
                    t.set_x(-.59)
        for t in fig.texts: t.set_visible(False)
        title(fig,"a","Primary sensitivity: 31-60 days",1.0,.7)
        title(fig,"b","Secondary sensitivity: 1-60 days",1.0,8.8)
    elif fid == "ED6":
        a[0].set_ylabel("1-60-day event coverage\ndifference (pp)")
        for ax in a[1:]: ax.set_xlabel("Event coverage\ndifference (pp)")
        for ax in a[1:3]: small_axes(ax,3)
        for ax,rect in zip(a[1:4],[(1.5,10.5,4.1,2.0),(7.0,10.5,4.1,2.0),(13.8,10.5,3.1,2.0)]):
            pos(ax,*rect)
    elif fid == "ED7":
        for t in fig.texts:
            if t.get_text() == "Event-date uncertainty": t.set_text("Previous-observation gap")
        a[0].set_xlabel("Previous observation gap\n(days)")
        a[2].set_xlabel("Previous high-value\nobservation gap (days)")
        a[3].set_xlabel("Prior observations below threshold\n(1–60 days)")
        a[5].set_xlabel("Event separation (days)")
        a[4].set_xlabel("Event separation (days)")
        pos(a[5], 1.5, 12, 6.4, 1)
        for ax in (a[1],a[4],a[5]): small_axes(ax,3)
        a[1].set_xlabel("Number\nof events")
        a[4].text(60, .96, "Prespecified 60 d", transform=a[4].get_xaxis_transform(),
                  rotation=90, ha="right", va="top", fontsize=6.4, color=GRAY)


def capture_simple(mod, name, fig=None):
    recorded=[]
    def capture(self,*args,**kwargs):
        if not recorded: recorded.append(self)
    with patch.object(sys, "argv", [name,"--output-root",str(TMP / "simple")]), \
         patch.object(Figure,"savefig",capture), patch.object(plt,"close",lambda *a,**k: None):
        if fig is None:
            mod.main()
        elif name == "capacity":
            with patch.object(plt,"figure",lambda *a,**k: fig): mod.main()
        else:
            ax=fig.add_axes([.25,.63,.69,.25])
            with patch.object(plt,"subplots",lambda *a,**k:(fig,ax)), patch.object(Figure,"tight_layout",lambda *a,**k:None): mod.main()
    return recorded[0]


def polish_capacity(fig, composite=False):
    a=fig.axes[:3]
    for t in a[0].texts:
        if t.get_text().startswith("Primary 10%"):
            t.set_text("10% primary selection"); t.set_position((.06,1.04)); t.set_transform(a[0].transAxes)
        elif t.get_text().startswith("+"):
            # Keep direct values beside their confidence interval and the 10%
            # reference, leaving both scientific line segments uninterrupted.
            rightmost=t.xy[0] == max(x.xy[0] for x in a[0].texts
                                     if x.get_text().startswith("+"))
            t.set_position((-5 if rightmost else 5,8))
            t.set_ha("right" if rightmost else "left"); t.set_fontsize(7)
    for ax in (a[0],a[1]): ax._left_title.set_text("")
    for obj in list(a[2].lines)+list(a[2].collections):
        obj.set_color(GRAY)
        if hasattr(obj,"get_markerfacecolor") and obj.get_markerfacecolor() not in {"white","none"}:
            obj.set_markerfacecolor(GRAY)
        if hasattr(obj,"set_markeredgecolor"): obj.set_markeredgecolor(GRAY)
    a[1].set_ylabel("ECDF")
    if composite:
        pos(a[0],1.6,10.0,5.6,5.4)
        pos(a[1],10.2,10.0,6.7,2.3)
        pos(a[2],10.2,14.0,6.7,1.4)
    else:
        fig.set_size_inches(18/2.54,10/2.54)
        pos(a[0],1.6,1.8,5.8,6.4)
        pos(a[1],10.1,1.8,6.8,2.8)
        pos(a[2],10.1,6.9,6.8,1.3)
        title(fig,"a","Capacity response",1.0,.55)
        title(fig,"b","Lake-level weighting",9.5,.55)


def polish_component(fig, ax=None, composite=False):
    ax=ax or fig.axes[0]
    ax.set_yticks([0],["Multi-horizon\nSame-predictor 60-day\nbase model"])
    ax.set_xlabel("Event coverage difference (pp)")
    ax._left_title.set_text("60-day component comparison")
    ax._left_title.set_fontsize(8)
    for t in ax.texts:
        t.set_fontsize(7)
        if "replicates" in t.get_text():
            t.set_text("5,000 lake-level Bootstrap replicates")
            t.set_transform(ax.transAxes)
            t.set_position((.02,.07));t.set_ha("left");t.set_va("bottom")
            t.set_fontsize(6.4)
    if not composite:
        fig.set_size_inches(18/2.54,8/2.54)
        pos(ax,5.3,1.5,11.5,5.1)
    else:
        ax._left_title.set_text("")
        pos(ax,5.3,1.8,11.5,5.0)


def text_audit(fig):
    fig.canvas.draw(); renderer=fig.canvas.get_renderer(); bounds=fig.bbox
    visible=[]; outside=[]
    tick_owners={}
    drawn_ticks=set()
    for ax in fig.axes:
        for axis in [ax.xaxis,ax.yaxis]:
            for tick in [*axis.get_major_ticks(),*axis.get_minor_ticks()]:
                for label in [tick.label1,tick.label2]: tick_owners[id(label)]=ax
            for tick in axis._update_ticks():
                for label in [tick.label1,tick.label2]:
                    if label.get_visible(): drawn_ticks.add(id(label))

    def text_role(text):
        # Tick-label Text instances do not consistently expose their owning axes.
        if id(text) in tick_owners: return "tick",tick_owners[id(text)]
        for ax in fig.axes:
            if text in [ax.xaxis.label, ax.yaxis.label]:
                return "axis_label", ax
            if text in [ax.title, ax._left_title, ax._right_title]:
                return "title", ax
            if text in ax.texts:
                return "annotation", ax
        return "figure", None

    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip() or (t.axes is not None and not t.axes.get_visible()): continue
        role,owner=text_role(t)
        if role=="tick" and id(t) not in drawn_ticks: continue
        if owner is not None and not owner.get_visible(): continue
        if owner is not None and not owner.axison and role in {"tick", "axis_label", "title"}: continue
        b=t.get_window_extent(renderer)
        if b.width==0 or b.height==0: continue
        if b.x1<0 or b.x0>bounds.x1 or b.y1<0 or b.y0>bounds.y1: continue
        if b.x0 < -.5 or b.y0 < -.5 or b.x1>bounds.x1+.5 or b.y1>bounds.y1+.5:
            outside.append(t.get_text())
        visible.append((t,b,role,owner))
    collisions=[]
    for i,(t,b,trole,towner) in enumerate(visible):
        for u,c,urole,uowner in visible[i+1:]:
            if t.get_text()==u.get_text() and np.allclose(b.extents,c.extents): continue
            overlap=min(b.x1,c.x1)-max(b.x0,c.x0),min(b.y1,c.y1)-max(b.y0,c.y0)
            if min(overlap) <= 1.0: continue
            overlap_area=overlap[0]*overlap[1]
            smaller_area=min(b.width*b.height,c.width*c.height)
            ratio=overlap_area/smaller_area if smaller_area else 0
            threshold=.15
            if overlap_area>8 and ratio>=threshold:
                collisions.append([t.get_text(),u.get_text()])
    intrusions=[]
    for text,box,role,owner in visible:
        for idx,ax in enumerate(fig.axes):
            if ax is owner or not ax.get_visible() or not ax.axison: continue
            frame=ax.get_window_extent(renderer)
            ow=min(box.x1,frame.x1)-max(box.x0,frame.x0)
            oh=min(box.y1,frame.y1)-max(box.y0,frame.y0)
            if ow>1 and oh>1 and ow*oh>6:
                intrusions.append({"text":text.get_text(),"foreign_axis":idx})
    clipped_keys=[]
    for idx,ax in enumerate(fig.axes):
        if not ax.get_visible() or ax.axison or ax.images: continue
        for collection in ax.collections:
            if not collection.get_visible() or not collection.get_clip_on(): continue
            offsets=collection.get_offsets()
            if not len(offsets): continue
            points=collection.get_offset_transform().transform(offsets)
            radius=(np.sqrt(max(collection.get_sizes(),default=0))/2+1)*fig.dpi/72
            frame=ax.get_window_extent(renderer)
            if any(x-radius<frame.x0 or x+radius>frame.x1 or y-radius<frame.y0 or y+radius>frame.y1 for x,y in points):
                clipped_keys.append(idx)
    return {"outside_canvas":outside,"potential_text_collisions":collisions,
            "text_into_foreign_data_axes":intrusions,"clipped_legend_markers":clipped_keys,
            "minimum_visible_font_pt":min(t.get_fontsize() for t,b,role,owner in visible),
            "visible_text":[t.get_text() for t,b,role,owner in visible]}


def align_edge_ticklabels(fig):
    """Pull endpoint tick labels inside the canvas without changing tick values."""
    fig.canvas.draw(); renderer=fig.canvas.get_renderer(); bounds=fig.bbox
    for ax in fig.axes:
        if not ax.get_visible(): continue
        for label in ax.get_xticklabels():
            if not label.get_visible(): continue
            box=label.get_window_extent(renderer)
            if box.x0 < bounds.x0: label.set_ha("left")
            if box.x1 > bounds.x1: label.set_ha("right")
        for label in ax.get_yticklabels():
            if not label.get_visible(): continue
            box=label.get_window_extent(renderer)
            if box.y0 < bounds.y0: label.set_va("bottom")
            if box.y1 > bounds.y1: label.set_va("top")
    fig.canvas.draw()


def make_contact_sheets():
    requested_order = [*(f"Figure_{i}" for i in range(1, 6)), *(f"Figure_S{i}" for i in range(1, 9))]
    order = [name for name in requested_order if (OUT / "QA" / f"{name}_89mm.png").exists()]
    cols, rows = 4, max(1, int(np.ceil(len(order) / 4)))
    cell_w, cell_h = 560, 440
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=20)
    for i, name in enumerate(order):
        image = Image.open(OUT / "QA" / f"{name}_89mm.png").convert("RGB")
        image.thumbnail((cell_w - 28, cell_h - 56), Image.Resampling.LANCZOS)
        x, y = (i % cols) * cell_w, (i // cols) * cell_h
        draw.text((x + 14, y + 10), name, fill=INK, font=font)
        sheet.paste(image, (x + (cell_w - image.width) // 2,
                            y + 42 + (cell_h - 50 - image.height) // 2))

    def cvd(matrix):
        arr = np.asarray(sheet, dtype=np.float32) / 255
        out = np.stack([sum(arr[..., j] * row[j] for j in range(3)) for row in matrix], axis=-1)
        return Image.fromarray(np.round(np.clip(out, 0, 1) * 255).astype(np.uint8), "RGB")

    outputs = {
        "contact_sheet_polished.png": sheet,
        "contact_sheet_grayscale.png": ImageOps.grayscale(sheet).convert("RGB"),
        "contact_sheet_protanopia.png": cvd(np.array([[.56667,.43333,0],[.55833,.44167,0],[0,.24167,.75833]])),
        "contact_sheet_deuteranopia.png": cvd(np.array([[.625,.375,0],[.7,.3,0],[0,.3,.7]])),
        "contact_sheet_tritanopia.png": cvd(np.array([[.95,.05,0],[0,.43333,.56667],[0,.475,.525]])),
    }
    for name, image in outputs.items():
        image.save(OUT / "QA" / name, dpi=(150, 150), optimize=True)
    dump(OUT / "QA" / "contact_sheet_hashes.json",
         {name: digest(OUT / "QA" / name) for name in outputs})


def save(fig, name, before, audits):
    # Draw once before asserting limits: any tick/autoscale side effect must be caught.
    align_edge_ticklabels(fig)
    check=text_audit(fig)
    preserved=verify_geometry(before)
    fig.savefig(OUT/f"{name}.pdf",dpi=600,facecolor="white",metadata={"Creator":"Lake monitoring strategy evaluation"})
    fig.savefig(OUT/f"{name}.png",dpi=600,facecolor="white")
    fig.savefig(OUT/f"{name}.svg",dpi=600,facecolor="white")
    fig.savefig(OUT/"QA"/f"{name}_180mm.png",dpi=150,facecolor="white")
    fig.savefig(OUT/"QA"/f"{name}_89mm.png",dpi=150*89/180,facecolor="white")
    audits[name]={**preserved,**check,"canvas_mm":(fig.get_size_inches()*25.4).round(4).tolist(),
                  "sha256_pdf":digest(OUT/f"{name}.pdf"),"sha256_png":digest(OUT/f"{name}.png")}
    dump(OUT/"QA"/"render_audit.json",audits)
    plt.close(fig)
    print(name, "geometry PASS; text flags",len(check["outside_canvas"]),len(check["potential_text_collisions"]),flush=True)


def main():
    global OUT, TMP
    parser=argparse.ArgumentParser();parser.add_argument("--figures",nargs="*")
    parser.add_argument("--output-root",type=Path)
    args=parser.parse_args()
    if args.output_root: OUT=args.output_root.resolve()
    TMP=OUT/'work'
    OUT.mkdir(parents=True,exist_ok=True);(OUT/"QA").mkdir(exist_ok=True);(OUT/"scripts").mkdir(exist_ok=True);TMP.mkdir(parents=True,exist_ok=True)
    base=module("baseline_family",BASELINE/"render_figure_family.py")
    cap=module("baseline_capacity",BASELINE/"render_capacity_and_weighting.py")
    component=module("baseline_component",BASELINE/"render_same_feature_control.py")
    matplotlib.rcParams.update({"font.family":"Arial","pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none"})
    layout=base.load_layout()
    # Audit every source member before/after; no estimator or bootstrap is rerun.
    sources={str(p.relative_to(ROOT)):digest(p) for root in [ROOT/"04_analysis_data/primary_external_evaluation",cap.BASE,
             ROOT/"04_analysis_data/development_evaluation"] for p in root.rglob("*") if p.is_file()}
    dump(OUT/"source_hashes.json",sources)
    audits=json.loads((OUT/"QA"/"render_audit.json").read_text()) if (OUT/"QA"/"render_audit.json").exists() else {}
    mapping={"M5":"Figure_3","M4":"Figure_4","M1":"Figure_1","M2":"Figure_2",**{f"ED{i}":f"Figure_S{i}" for i in range(1,8)}}
    selected=set(args.figures or [*mapping.values(),"Figure_S11","Figure_S8"])
    if any(x in selected for x in mapping.values()):
        data=base.load_common()
        for spec in layout["figures"]:
            fid=spec["figure_id"];name=mapping.get(fid)
            if name not in selected: continue
            rs=copy.deepcopy(spec)
            for p in rs["panels"]:
                for r in p["internal_layout"]["regions"].values():
                    r["rect_cm"]=r.get("axis_slots",{}).get("plot_frame_rect_cm",r["rect_cm"])
            fig=plt.figure(figsize=(18/2.54,14/2.54),dpi=150,facecolor="white")
            base.RENDERERS[fid](fig,rs,data);fig.canvas.draw();before=geometry(fig)
            polish_text(fig)
            if fid.startswith("M"): polish_main(fid,fig)
            else: polish_extended(fid,fig)
            save(fig,name,before,audits)
    if "Figure_S11" in selected:
        fig=capture_simple(cap,"capacity");fig.canvas.draw();before=geometry(fig)
        polish_text(fig);polish_capacity(fig);save(fig,"Figure_S11",before,audits)
    if "Figure_S8" in selected:
        fig=capture_simple(component,"component");fig.canvas.draw();before=geometry(fig)
        polish_text(fig);polish_component(fig);save(fig,"Figure_S8",before,audits)
    for rel,h in sources.items(): assert digest(ROOT/rel)==h, f"SOURCE MODIFIED: {rel}"
    dump(OUT/"QA"/"source_integrity.json",{"status":"PASS","files_verified":len(sources),"data_changed":"NO"})
    make_contact_sheets()
    shutil.copy2(Path(__file__),OUT/"scripts"/Path(__file__).name)


if __name__=="__main__": main()
