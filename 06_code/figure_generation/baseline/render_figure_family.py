from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import hashlib
import io
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, Polygon, Rectangle
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
DISPLAY_ROOT = Path(__file__).resolve().parent
SHARED_GEOGRAPHY = DISPLAY_ROOT.parent / "resources/geography"
NE2_BASE = SHARED_GEOGRAPHY / "natural_earth2_clean_basemap" / "processed"
NE2_TIF = SHARED_GEOGRAPHY / "natural_earth2_clean_basemap" / "NE2_LR_LC_SR_W_DR.tif"
sys.path.insert(0, str(DISPLAY_ROOT))
from display_labels import install_label_adapter, missing_reason_label, public_label

FAMILY = ROOT / ".build/figure_packages"
CONTROL = DISPLAY_ROOT.parent / "resources"
LAYOUT_PATH = CONTROL / "figure_layout.json"
PACKAGES = ["main_1", "main_2", "main_3", "main_4", "main_5", "ed_1", "ed_2", "ed_3", "ed_4", "ed_5", "ed_6", "ed_7"]
FIGURE_TO_PACKAGE = {f"M{i}": f"main_{i}" for i in range(1, 6)} | {f"ED{i}": f"ed_{i}" for i in range(1, 8)}
PACKAGE_TO_FIGURE = {v: k for k, v in FIGURE_TO_PACKAGE.items()}
CANVAS_CM = (18.0, 14.0)
MM_PER_INCH = 25.4
CM_PER_INCH = 2.54
PDF_DATE = dt.datetime(2026, 8, 15, 0, 0, tzinfo=dt.timezone.utc)

PALETTE = {
    "paper": "#FFFFFF",
    "text_axes_references": "#20242A",
    "neutral_context_line": "#68717C",
    "neutral_context_fill": "#B8C0C8",
    "multi_horizon_policy": "#007C91",
    "direct_60_day_policy": "#D55E00",
    "both_selected": "#343A40",
    "neither_selected": "#A7ADB4",
    "formal_event_outline": "#8E3B76",
    "uncertainty_support": "#D0D5DA",
    "background_water_context": "#D6E0E5",
    "map_count_1": "#9DB7D5",
    "map_count_2_3": "#759FC8",
    "map_count_4_7": "#4E87B7",
    "map_count_8_15": "#2D6FA5",
    "map_count_16_47": "#174E7E",
    "map_count_48_233": "#0A3158",
    "rank_density_count_1": "#B9C6CC",
    "rank_density_count_2_3": "#94A8B2",
    "rank_density_count_4_7": "#708A96",
    "rank_density_count_8_15": "#4E6D79",
    "rank_density_count_16_31": "#365461",
    "rank_density_count_32_63": "#243F4B",
    "rank_density_count_64_274": "#132B35",
    "missing_or_outside_domain": "#FFFFFF",
}
Q = PALETTE["multi_horizon_policy"]
D = PALETTE["direct_60_day_policy"]
TEXT = PALETTE["text_axes_references"]
NEUTRAL = PALETTE["neutral_context_line"]
LIGHT = PALETTE["neutral_context_fill"]
EVENT = PALETTE["formal_event_outline"]

matplotlib.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 5.5,
    "axes.labelsize": 6.5,
    "axes.titlesize": 7.0,
    "xtick.labelsize": 5.5,
    "ytick.labelsize": 5.5,
    "legend.fontsize": 5.5,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.835,
    "ytick.major.size": 2.835,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.transparent": False,
})
install_label_adapter()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def rect_norm(rect: list[float]) -> list[float]:
    x, y, w, h = map(float, rect)
    return [x / CANVAS_CM[0], 1.0 - (y + h) / CANVAS_CM[1], w / CANVAS_CM[0], h / CANVAS_CM[1]]


def axis_for(fig: plt.Figure, rect: list[float], *, frame: bool = True):
    ax = fig.add_axes(rect_norm(rect))
    ax.set_facecolor("white")
    ax.tick_params(direction="out", length=2.835, width=0.5, pad=1.5, colors=TEXT)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.5)
        ax.spines[side].set_color(TEXT)
    if not frame:
        ax.set_axis_off()
    return ax


def add_title(fig: plt.Figure, panel: dict) -> None:
    reserve = next((v for k, v in panel["internal_layout"]["reserves"].items() if "title" in k), None)
    if reserve is None:
        x, y, _, _ = panel["outer_rect_cm"]
        rect = [x + 0.5, y + 0.5, 4.0, 0.5]
    else:
        rect = reserve["rect_cm"]
    x, y, w, _ = rect
    label = panel["panel_id"].replace("ED", "")[-1]
    fig.text((x + 0.1) / 18, 1 - (y + 0.10) / 14, label, ha="left", va="top", fontsize=8, fontweight="bold", color=TEXT)
    fig.text((x + 0.7) / 18, 1 - (y + 0.10) / 14, panel["title_text"], ha="left", va="top", fontsize=7, color=TEXT)


def style_axis(ax, xlabel: str = "", ylabel: str = "") -> None:
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=1.5)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=1.5)
    ax.grid(False)


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, float))
    return x, np.arange(1, len(x) + 1) / len(x)


def draw_interval(ax, y, point, lo, hi, color, marker, filled=True, lw=1.0):
    ax.plot([lo, hi], [y, y], color=color, lw=0.5 if lw <= 0.7 else 1.0, solid_capstyle="butt")
    ax.plot([lo, lo], [y - 0.04, y + 0.04], color=color, lw=0.5)
    ax.plot([hi, hi], [y - 0.04, y + 0.04], color=color, lw=0.5)
    ax.scatter([point], [y], s=6.428 if filled else 5.625, marker=marker, facecolors=color if filled else "white", edgecolors=color, linewidths=0.6, zorder=5)


def project_5070(lon, lat):
    # NAD83 / Conus Albers, EPSG:5070, implemented deterministically to avoid a runtime GIS dependency.
    lon = np.deg2rad(np.asarray(lon, float)); lat = np.deg2rad(np.asarray(lat, float))
    a = 6378137.0; invf = 298.257222101; f = 1 / invf; e2 = 2 * f - f * f; e = math.sqrt(e2)
    p1, p2, p0, l0 = map(math.radians, (29.5, 45.5, 23.0, -96.0))
    def m(p): return math.cos(p) / math.sqrt(1 - e2 * math.sin(p) ** 2)
    def qv(p):
        s = np.sin(p)
        return (1 - e2) * (s / (1 - e2 * s * s) - np.log((1 - e * s) / (1 + e * s)) / (2 * e))
    q1, q2, q0 = qv(p1), qv(p2), qv(p0)
    n = (m(p1) ** 2 - m(p2) ** 2) / (q2 - q1); C = m(p1) ** 2 + n * q1
    rho = a * np.sqrt(C - n * qv(lat)) / n; rho0 = a * math.sqrt(C - n * q0) / n
    theta = n * (lon - l0)
    return rho * np.sin(theta), rho0 - rho * np.cos(theta)


def iter_geo_lines(geometry):
    coords = geometry["coordinates"]
    if geometry["type"] == "Polygon":
        for ring in coords: yield np.asarray(ring, float)
    elif geometry["type"] == "MultiPolygon":
        for poly in coords:
            for ring in poly: yield np.asarray(ring, float)


def _inverse_5070(xs, ys):
    """Inverse NAD83 / Conus Albers used by the cached Natural Earth raster."""
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    a = 6378137.0; invf = 298.257222101; f = 1 / invf; e2 = 2 * f - f * f; e = math.sqrt(e2)
    p1, p2, p0, l0 = map(math.radians, (29.5, 45.5, 23.0, -96.0))
    def m(p): return math.cos(p) / math.sqrt(1 - e2 * math.sin(p) ** 2)
    def qv(p):
        s = np.sin(p)
        return (1 - e2) * (s / (1 - e2 * s * s) - np.log((1 - e * s) / (1 + e * s)) / (2 * e))
    q1, q2, q0 = qv(p1), qv(p2), qv(p0)
    n = (m(p1) ** 2 - m(p2) ** 2) / (q2 - q1); C = m(p1) ** 2 + n * q1
    rho0 = a * math.sqrt(C - n * q0) / n
    rho = np.sqrt(xs * xs + (rho0 - ys) * (rho0 - ys)); theta = np.arctan2(xs, rho0 - ys)
    q = (C - (rho * n / a) ** 2) / n
    qp = float(qv(math.pi / 2 - 1e-10)); phi = np.arcsin(np.clip(q / qp, -0.999999, 0.999999))
    for _ in range(8):
        eps = 1e-6; qp0 = qv(phi); dqp = (qv(phi + eps) - qv(phi - eps)) / (2 * eps)
        phi = np.clip(phi - (qp0 - q) / np.where(np.abs(dqp) < 1e-9, 1e-9, dqp), -math.pi / 2 + 1e-8, math.pi / 2 - 1e-8)
    return np.degrees(l0 + theta / n), np.degrees(phi)


def projected_extent_bbox(extent, samples=241):
    """Return the full EPSG:5070 envelope of a geographic rectangular extent."""
    lon0, lon1, lat0, lat1 = extent
    lons = np.linspace(lon0, lon1, samples)
    lats = np.linspace(lat0, lat1, samples)
    lon = np.concatenate([lons, lons, np.full(samples, lon0), np.full(samples, lon1)])
    lat = np.concatenate([np.full(samples, lat0), np.full(samples, lat1), lats, lats])
    x, y = project_5070(lon, lat)
    return float(x.min()), float(x.max()), float(y.min()), float(y.max())


def _reproject_ne2(bbox, width=900):
    """Reproject the cached Natural Earth II raster into the study EPSG:5070 frame.

    The source is a 16,200 x 8,100 geographic raster (0.022222 degrees per
    pixel). Deriving the pixel spacing from the raster dimensions keeps the
    basemap and the frozen state/lake coordinates on the same geographic
    transform.
    """
    xmin, xmax, ymin, ymax = bbox
    aspect = (ymax - ymin) / max(xmax - xmin, 1.0)
    height = max(300, int(round(width * aspect)))
    rounded = "_".join(str(int(round(v / 1000))) for v in bbox)
    out = NE2_BASE / f"ne2_epsg5070_{rounded}_{width}px.png"
    if not out.exists():
        Image.MAX_IMAGE_PIXELS = None
        src = Image.open(NE2_TIF).convert("RGB")
        arr = np.asarray(src)
        xres = 360.0 / arr.shape[1]
        yres = 180.0 / arr.shape[0]
        xx, yy = np.meshgrid(np.linspace(xmin, xmax, width), np.linspace(ymax, ymin, height))
        lon, lat = _inverse_5070(xx, yy)
        cols = np.rint((lon + 180.0 - xres / 2.0) / xres).astype(int)
        rows = np.rint((90.0 - lat - yres / 2.0) / yres).astype(int)
        valid = (rows >= 0) & (rows < arr.shape[0]) & (cols >= 0) & (cols < arr.shape[1])
        out_arr = np.ones((height, width, 3), dtype=np.uint8) * 248
        out_arr[valid] = arr[rows[valid], cols[valid]]
        f = out_arr.astype(float) / 255.0
        f = 1.0 - (1.0 - f) * 0.64
        f = f * 0.95 + np.array([0.965, 0.985, 0.990]) * 0.05
        Image.fromarray(np.clip(f * 255, 0, 255).astype(np.uint8)).save(out)
    return out


def plot_physical_base(ax, extent):
    """Draw an authentic terrain/hydrography raster beneath study overlays."""
    bbox = projected_extent_bbox(extent)
    width = 1800 if (extent[1] - extent[0]) > 40 else 1000
    path = _reproject_ne2(bbox, width)
    arr = np.asarray(Image.open(path).convert("RGB")).astype(float) / 255.0
    arr = 1.0 - (1.0 - arr) * 0.74
    ax.imshow(arr, extent=bbox, origin="upper", interpolation="bilinear", zorder=0)


def plot_basemap(ax, geo, extent=None):
    if extent is not None:
        plot_physical_base(ax, extent)
    for feature in geo["features"]:
        for ring in iter_geo_lines(feature["geometry"]):
            x, y = project_5070(ring[:, 0], ring[:, 1])
            ax.plot(x, y, color="#65707A", lw=0.26, alpha=0.65, zorder=2)
    if extent is not None:
        xmin, xmax, ymin, ymax = projected_extent_bbox(extent)
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box"); ax.set_axis_off()


def source_paths(root: Path):
    ext = root / "04_analysis_data/primary_external_evaluation"
    formal = ext / "evaluation"
    dev = root / "04_analysis_data/development_evaluation"
    shared = SHARED_GEOGRAPHY
    return {
        "formal": formal, "ext": ext, "dev": dev, "shared": shared,
        "events": ext / "event_outcomes/external_incident_events.csv",
        "years": ext / "event_outcomes/external_incident_events_by_year.csv",
        "washout": ext / "event_outcomes/washout_sensitivity.csv",
        "qscore": ext / "multi_horizon_scores/external_outcome_blind_scores.parquet",
        "dscore": ext / "direct_60_day_scores/external_direct_60_day_outcome_blind_scores.parquet",
        "features": ext / "candidate_features/external_field_chla_features.parquet",
        "dev_eval": dev / "model_selection",
        "row_metrics": dev / "fixed_capacity_efficiency/row_probability_metrics.csv",
    }


def load_layout() -> dict:
    return json.loads(LAYOUT_PATH.read_text(encoding="utf-8"))


def load_common():
    p = source_paths(ROOT)
    lakes = pd.read_csv(p["shared"] / "multi_horizon_lakes.csv")
    pop = pd.read_csv(p["shared"] / "multi_horizon_population_summary.csv")
    gcounts = pd.read_csv(p["shared"] / "multi_horizon_geographic_counts.csv")
    geo = json.loads((p["shared"] / "conus_states_2023_20m.geojson").read_text())
    unit = pd.read_csv(p["formal"] / "unit_queue_audit.csv", parse_dates=["origin_date"])
    endpoints = pd.read_csv(p["formal"] / "endpoint_summary.csv")
    points = pd.read_csv(p["formal"] / "point_estimates.csv")
    draws = pd.read_parquet(p["formal"] / "bootstrap_draws.parquet")
    events = pd.read_csv(p["events"], parse_dates=["event_date", "previous_observation_date"])
    years = pd.read_csv(p["years"])
    washout = pd.read_csv(p["washout"])
    q = pd.read_parquet(p["qscore"], columns=["candidate_row_key","lake_id","origin_date","eligible_external_candidate","queue_rank__multi_horizon","queue_capacity__multi_horizon"])
    d = pd.read_parquet(p["dscore"], columns=["candidate_row_key","lake_id","origin_date","eligible_external_candidate","queue_rank__direct_60_day","queue_capacity__direct_60_day"])
    q["origin_date"] = pd.to_datetime(q["origin_date"]); d["origin_date"] = pd.to_datetime(d["origin_date"])
    joined = q.merge(d, on=["candidate_row_key","lake_id","origin_date"], suffixes=("_q","_d"), validate="one_to_one")
    assert len(joined) == 369720
    assert (joined.eligible_external_candidate_q == joined.eligible_external_candidate_d).all()
    rank = joined.loc[joined.eligible_external_candidate_q].copy()
    rank["eligible_count"] = rank.groupby("origin_date")["candidate_row_key"].transform("size")
    rank["q_norm"] = rank.queue_rank__multi_horizon / rank.eligible_count
    rank["d_norm"] = rank.queue_rank__direct_60_day / rank.eligible_count
    rank["q_selected"] = rank.queue_rank__multi_horizon <= rank.queue_capacity__multi_horizon
    rank["d_selected"] = rank.queue_rank__direct_60_day <= rank.queue_capacity__direct_60_day
    assert len(rank) == 365726 and rank.origin_date.nunique() == 130
    qqueue=pd.read_parquet(p["ext"] / "multi_horizon_scores/frozen_external_queues.parquet").query("policy == 'multi_horizon'").copy()
    dqueue=pd.read_parquet(p["ext"] / "direct_60_day_scores/frozen_external_direct_60_day_queues.parquet").query("policy == 'direct_60_day'").copy()
    # Physical queue files contain exactly the selected rows. In the multi_horizon file,
    # selected is the origin capacity count, not a Boolean and is never filtered.
    assert len(qqueue)==len(dqueue)==36631 and qqueue.origin_date.nunique()==dqueue.origin_date.nunique()==130
    def capture(queue,label):
        qx=queue[["lake_id","origin_date"]].copy();qx["lake_id"]=qx.lake_id.astype(str);qx["origin_date"]=pd.to_datetime(qx.origin_date)
        ex=events[["episode_id","lake_id","event_date"]].copy();ex["lake_id"]=ex.lake_id.astype(str);ex["event_date"]=pd.to_datetime(ex.event_date)
        j=ex.merge(qx,on="lake_id",how="left");j["lead_days"]=(j.event_date-j.origin_date).dt.days
        out=ex[["episode_id"]].copy()
        for name,lo,hi in [("primary",31,60),("secondary",1,60)]:
            hit=j[j.lead_days.between(lo,hi)].groupby("episode_id").size().gt(0)
            out[f"{label}_{name}"]=out.episode_id.map(hit).fillna(False).astype(bool)
        return out
    roles=capture(qqueue,"multi_horizon").merge(capture(dqueue,"direct_60_day"),on="episode_id",validate="one_to_one")
    assert [int(roles.multi_horizon_primary.sum()),int(roles.direct_60_day_primary.sum()),int(roles.multi_horizon_secondary.sum()),int(roles.direct_60_day_secondary.sum())]==[263,207,443,290]
    return dict(paths=p,lakes=lakes,pop=pop,gcounts=gcounts,geo=geo,unit=unit,endpoints=endpoints,points=points,draws=draws,events=events,years=years,washout=washout,rank=rank,event_roles=roles)


def render_m1(fig, spec, data):
    lakes, geo = data["lakes"], data["geo"]
    for panel in spec["panels"]: add_title(fig, panel)
    ax = axis_for(fig, spec["panels"][0]["internal_layout"]["regions"]["conus_support_axis"]["rect_cm"], frame=False)
    plot_basemap(ax, geo, (-125, -66, 23, 50))
    cand = lakes[lakes.candidate_universe.astype(bool)].copy(); cand["cx"] = np.floor(cand.x_epsg5070_m / 50000) * 50000; cand["cy"] = np.floor(cand.y_epsg5070_m / 50000) * 50000
    cells = cand.groupby(["cx","cy"], as_index=False).size()
    bins = [0,1,3,7,15,47,np.inf]; colors=[PALETTE[f"map_count_{x}"] for x in ["1","2_3","4_7","8_15","16_47","48_233"]]
    idx=np.digitize(cells["size"],bins[1:],right=True)
    for row, color in zip(cells.itertuples(), np.asarray(colors)[idx]):
        ax.add_patch(Rectangle((row.cx,row.cy),50000,50000,facecolor=color,edgecolor="none",zorder=2))
    mon=lakes[lakes.external_monitoring_lake_domain.astype(bool)]; evt=lakes[lakes.formal_event_lake.astype(bool)]
    ax.scatter(mon.x_epsg5070_m,mon.y_epsg5070_m,s=1.6,facecolors="none",edgecolors=Q,linewidths=.25,zorder=4)
    ax.scatter(evt.x_epsg5070_m,evt.y_epsg5070_m,s=3.0,facecolors="none",edgecolors=EVENT,linewidths=.35,zorder=5)
    # Population legend, ordered count scale, and map furniture occupy frozen reserves.
    r=spec["panels"][0]["internal_layout"]["reserves"]
    la=axis_for(fig,r["population_legend"]["rect_cm"],frame=False); la.set_xlim(0,1);la.set_ylim(0,1)
    la.scatter([.08,.08],[.78,.50],s=[6.4,6.4],facecolors=["none","none"],edgecolors=[Q,EVENT],marker="o",linewidths=.5)
    la.text(.18,.78,"Lakes with field monitoring · 1,679",va="center");la.text(.18,.50,"Event lakes · 355",va="center")
    la.text(.08,.18,"Candidate lakes · 2,844",va="center",color=TEXT)
    ca=axis_for(fig,r["count_colorbar"]["rect_cm"],frame=False);ca.set_xlim(0,1);ca.set_ylim(0,1)
    for i,(c,l) in enumerate(zip(colors,["1 lake","2-3","4-7","8-15","16-47","48-233"])):
        yy=.88-i*.14;ca.add_patch(Rectangle((.05,yy-.04),.18,.08,facecolor=c,edgecolor="none"));ca.text(.30,yy,l,va="center",fontsize=5)
    ca.text(.05,.04,"Candidate lakes / 50-km cell",fontsize=5)
    ma=axis_for(fig,r["map_furniture"]["rect_cm"],frame=False);ma.set_xlim(0,1);ma.set_ylim(0,1)
    ma.annotate("N",(.25,.82),ha="center",va="bottom",fontsize=6.5);ma.plot([.25,.25],[.48,.78],color=TEXT,lw=.7);ma.scatter([.25],[.78],marker="^",s=10,color=TEXT)
    ma.plot([.12,.70],[.32,.32],color=TEXT,lw=1);ma.plot([.12,.12,.70,.70],[.28,.36,.28,.36],color=TEXT,lw=.5);ma.text(.41,.16,"500 km",ha="center",fontsize=5)
    ma.text(.05,.02,"EPSG:5070",fontsize=5)
    inset_axes=[]
    def inset(panel,extent):
        reg=next(iter(panel["internal_layout"]["regions"].values()))["rect_cm"]; ax=axis_for(fig,reg,frame=False);plot_basemap(ax,geo,extent)
        sub=lakes.query(f"longitude >= {extent[0]} and longitude <= {extent[1]} and latitude >= {extent[2]} and latitude <= {extent[3]}").copy()
        order=[("neither",PALETTE["neither_selected"],"o",False),("both",PALETTE["both_selected"],"D",True),("multi_horizon_only",Q,"o",True),("direct_60_day_only",D,"s",False)]
        for role,c,m,fill in order:
            z=sub[sub.ever_selected_role==role]; ax.scatter(z.x_epsg5070_m,z.y_epsg5070_m,s=3.1,marker=m,facecolors=c if fill else "white",edgecolors=c,linewidths=.35,zorder=3)
        ev=sub[sub.formal_event_lake.astype(bool)];ax.scatter(ev.x_epsg5070_m,ev.y_epsg5070_m,s=7.0,facecolors="none",edgecolors=EVENT,linewidths=.4,zorder=5)
        rr=panel["internal_layout"]["reserves"]["role_legend"]["rect_cm"];la=axis_for(fig,rr,frame=False);la.set_xlim(0,1);la.set_ylim(0,1)
        for i,(role,c,m,fill) in enumerate(order):
            y=.9-i*.18;la.scatter([.17],[y],s=6.4,marker=m,facecolors=c if fill else "white",edgecolors=c,linewidths=.5);la.text(.34,y,role.replace("direct_60_day","direct"),va="center",fontsize=5)
        la.scatter([.17],[.17],s=9,facecolors="none",edgecolors=EVENT,linewidths=.5);la.text(.34,.17,"Event lake",va="center",fontsize=5)
        inset_axes.append(ax)
    inset(spec["panels"][1],(-97.5,-82,40,49.5)); inset(spec["panels"][2],(-87.7,-79.8,24,31.1))
    # Frozen map link corridors: two hairlines per inset, no arrowheads.
    for y0,y1 in [(4.0,3.0),(5.8,5.2)]:
        fig.add_artist(matplotlib.lines.Line2D([10.5/18,12/18],[1-y0/14,1-y1/14],transform=fig.transFigure,color=NEUTRAL,lw=.5))
    for y0,y1 in [(8.0,9.2),(9.7,11.0)]:
        fig.add_artist(matplotlib.lines.Line2D([10.5/18,12/18],[1-y0/14,1-y1/14],transform=fig.transFigure,color=NEUTRAL,lw=.5))


def render_m2(fig,spec,data):
    for p in spec["panels"]: add_title(fig,p)
    unit=data["unit"]; draws=data["draws"]
    regs=spec["panels"][0]["internal_layout"]["regions"]
    labels=["Prediction dates","Risk scoring","Top 10% selection","Future events","Whole-lake bootstrap"]
    for i,(rid,r) in enumerate(regs.items()):
        ax=axis_for(fig,r["rect_cm"],frame=False);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.text(.5,.92,labels[i],ha="center",va="top",fontsize=6.5)
        if i==0:
            ax.plot([.5,.5],[.16,.78],color=TEXT,lw=.7);ax.scatter([.5],[.34],s=6.4,color=TEXT);ax.text(.5,.10,"130 prediction dates",ha="center")
        elif i==1:
            y=np.linspace(.2,.72,7);ax.scatter(.25+.45*np.linspace(0,1,7),y,s=6.4,c=Q,marker="o");ax.scatter(.68-.40*np.linspace(0,1,7),y,s=5.6,facecolors="white",edgecolors=D,marker="s")
        elif i==2:
            for yy,c,m,fill in [(.62,Q,"o",True),(.38,D,"s",False)]:
                ax.plot([.15,.82],[yy,yy],color=c,lw=.7,ls="-" if fill else "--");ax.scatter(np.linspace(.18,.78,8),np.full(8,yy),s=4,marker=m,facecolors=c if fill else "white",edgecolors=c,linewidths=.4)
            ax.plot([.78,.78],[.27,.73],color=TEXT,lw=.5);ax.text(.78,.14,"same capacity",ha="center")
        elif i==3:
            xx=np.tile(np.arange(8),5);yy=np.repeat(np.arange(5),8);ax.scatter(.18+xx*.09,.25+yy*.10,s=4,facecolors="white",edgecolors=EVENT,linewidths=.35);ax.text(.5,.10,"696 events",ha="center")
        else:
            x,y=ecdf(draws.total_event_copies);ax.plot((x-x.min())/(x.max()-x.min())*.65+.18,y*.58+.18,color=NEUTRAL,lw=.8);ax.text(.5,.10,"5,000 resamples",ha="center")
    # Computation/observation order connectors in fixed gaps.
    for x in [3.65,6.65,10.15,13.65]:
        fig.add_artist(FancyArrowPatch((x/18,1-4.5/14),((x+.25)/18,1-4.5/14),transform=fig.transFigure,arrowstyle="-|>",mutation_scale=5,lw=.5,color=NEUTRAL))
    lr=spec["panels"][0]["internal_layout"]["reserves"]["policy_legend"]["rect_cm"];la=axis_for(fig,lr,frame=False);la.set_xlim(0,1);la.set_ylim(0,1)
    la.scatter([.03],[.5],s=6.4,c=Q);la.text(.09,.5,"Multi-horizon",va="center");la.scatter([.42],[.5],s=5.6,facecolors="white",edgecolors=D,marker="s");la.text(.49,.5,"Direct 60-day",va="center");la.scatter([.83],[.5],s=7,facecolors="none",edgecolors=EVENT);la.text(.89,.5,"Event",va="center",fontsize=5)
    p=spec["panels"][1]; regs=p["internal_layout"]["regions"]
    piv=unit.pivot(index="origin_date",columns="policy",values=["eligible_candidates","capacity"])
    ax=axis_for(fig,regs["eligible_lakes_axis"]["rect_cm"]);ax.step(piv.index,piv["eligible_candidates"].iloc[:,0],where="mid",color=NEUTRAL,lw=.8);style_axis(ax,"Prediction date","Eligible lakes");ax.xaxis.set_major_locator(mdates.YearLocator());ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax=axis_for(fig,regs["capacity_axis"]["rect_cm"]);ax.step(piv.index,piv["capacity"].iloc[:,0],where="mid",color=Q,lw=1);ax.scatter(piv.index[::7],piv["capacity"].iloc[::7,0],s=5,facecolors=Q,edgecolors=Q);style_axis(ax,"Prediction date","Number selected");ax.xaxis.set_major_locator(mdates.YearLocator());ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"));ax.set_ylim(268,287)
    p=spec["panels"][2]; ax=axis_for(fig,p["internal_layout"]["regions"]["total_event_copies_ecdf"]["rect_cm"]);x,y=ecdf(draws.total_event_copies);ax.plot(x,y,color=NEUTRAL,lw=.8);ax.axvline(696,color=TEXT,lw=.5);style_axis(ax,"Event copies per bootstrap replicate","ECDF")
    ax=axis_for(fig,p["internal_layout"]["regions"]["denominator_reference"]["rect_cm"],frame=False);ax.set_xlim(500,900);ax.set_ylim(0,1);ax.plot([696,696],[.08,.92],color=TEXT,lw=.5);ax.text(696,.94,"696 events",ha="center",va="bottom",fontsize=5)


def render_m3(fig,spec,data):
    for p in spec["panels"]: add_title(fig,p)
    p=data["paths"]; pm=pd.read_csv(p["dev_eval"] / "point_metrics.csv"); comp=pd.read_csv(p["dev_eval"] / "strict_bootstrap_comparisons.csv"); sd=pd.read_parquet(p["dev_eval"] / "strict_bootstrap_draws.parquet")
    regs=spec["panels"][0]["internal_layout"]["regions"]
    ax=axis_for(fig,regs["development_interval"]["rect_cm"],frame=False);ax.set_xlim(2016,2021);ax.set_ylim(0,1);ax.plot([2016,2020.8],[.5,.5],color=NEUTRAL,lw=1);ax.scatter([2016,2020],[.5,.5],s=6,color=NEUTRAL);ax.text(2018.4,.67,"Development evidence",ha="center")
    ax=axis_for(fig,regs["freeze_milestone"]["rect_cm"],frame=False);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.plot([.5,.5],[.2,.8],color=TEXT,lw=1);ax.scatter([.5],[.5],s=12,marker="D",color=TEXT);ax.text(.5,.1,"Frozen before outcomes",ha="center")
    ax=axis_for(fig,regs["external_interval"]["rect_cm"],frame=False);ax.set_xlim(2021,2026);ax.set_ylim(0,1);ax.plot([2021.1,2025.9],[.5,.5],color=EVENT,lw=1);ax.scatter(np.arange(2021,2026),np.full(5,.5),s=5,facecolors="white",edgecolors=EVENT);origins=pd.to_datetime(data["unit"].origin_date.drop_duplicates());ox=origins.dt.year+(origins.dt.dayofyear-1)/365.25;ax.vlines(ox,.42,.50,color=EVENT,lw=.25);ax.text(2023.5,.70,"External 2021-2025",ha="center");ax.text(2023.5,.24,"130 origins",ha="center",fontsize=5,color=EVENT)
    rr=spec["panels"][0]["internal_layout"]["reserves"]["title_and_tier_reserve"]["element_slots"]["evidence_tier_key"]["rect_cm"];la=axis_for(fig,rr,frame=False);la.set_xlim(0,1);la.set_ylim(0,1)
    for i,(lab,c,m) in enumerate([("development",NEUTRAL,"o"),("freeze",TEXT,"D"),("external",EVENT,"o")]):
        x=(i+.5)/3;la.scatter([x],[.68],s=5.5,facecolors="white" if i!=1 else c,edgecolors=c,marker=m);la.text(x,.20,lab,ha="center",va="center",fontsize=5)
    p2=spec["panels"][1]; regs=p2["internal_layout"]["regions"]
    policies=pm.policy.astype(str).tolist()[:8]; ypos=np.arange(len(policies))
    for rid,ep in [("development_primary_interval","31-60 d"),("development_secondary_interval","1-60 d")]:
        ax=axis_for(fig,regs[rid]["rect_cm"]); vals=pm.filter(regex="captur|rate").select_dtypes("number").iloc[:len(ypos),0].to_numpy(float); vals=(vals-vals.min())/(np.ptp(vals)+1e-9)
        ax.scatter(vals,ypos,s=6,facecolors=Q if "primary" in rid else "white",edgecolors=Q,linewidths=.5);ax.set_yticks(ypos[::2],labels=[x.replace("lifecycle_","")[:10] for x in policies[::2]]);style_axis(ax,ep,"Policy");ax.set_xlim(-.05,1.05)
    ax=axis_for(fig,regs["bootstrap_support"]["rect_cm"]);comparators=["early_30_day_reference","direct_60_day","environment_60_day_reference","basic_60_day_reference"]
    for i,c in enumerate(comparators):
        diff=sd["captured_31_60__multi_horizon"]-sd[f"captured_31_60__{c}"];x,y=ecdf(diff);ax.plot(x,y,color=[Q,D,NEUTRAL,EVENT][i],lw=1 if i==0 else .7,ls=["-","--",":","-."][i])
    ax.axvline(0,color=TEXT,lw=.5);style_axis(ax,"Captured-count difference","ECDF")
    rr=p2["internal_layout"]["reserves"]["hierarchy_legend"]["rect_cm"];la=axis_for(fig,rr,frame=False);la.set_xlim(0,1);la.set_ylim(0,1)
    for i,(lab,c,ls) in enumerate([("W0",Q,"-"),("bio",D,"--"),("env",NEUTRAL,":"),("back",EVENT,"-.")]):
        x=(i+.5)/4;la.plot([x-.045,x+.045],[.68,.68],color=c,lw=.8,ls=ls);la.text(x,.20,lab,ha="center",va="center",fontsize=5)
    p3=spec["panels"][2]; rm=pd.read_csv(p["row_metrics"]); metrics=[("auprc_axis","auprc"),("auroc_axis","auroc"),("brier_axis","brier"),("logloss_axis","log_loss"),("calibration_axis","calibration_slope")]
    for rid,col in metrics:
        ax=axis_for(fig,p3["internal_layout"]["regions"][rid]["rect_cm"]); sub=rm.dropna(subset=[col]);x=np.arange(len(sub));ax.scatter(x,sub[col],s=5,c=[Q if "multi_horizon" in str(v) else D if "direct" in str(v) else NEUTRAL for v in sub.policy],marker="o",linewidths=.2);ax.set_title(col.replace("_"," "),fontsize=5.5,pad=1);ticks=sorted(set([0,max(0,len(sub)//2),max(0,len(sub)-1)]));ax.set_xticks(ticks,[str(t+1) for t in ticks]);ax.set_xlabel("policy index",fontsize=5,labelpad=1)
    rr=p3["internal_layout"]["reserves"]["metric_legend"]["rect_cm"];la=axis_for(fig,rr,frame=False);la.set_xlim(0,1);la.set_ylim(0,1)
    for i,(lab,c,m,fill) in enumerate([("multi_horizon",Q,"o",True),("direct",D,"s",False),("other",NEUTRAL,"D",True)]):
        x=(i+.5)/3;la.scatter([x],[.68],s=5.5,facecolors=c if fill else "white",edgecolors=c,marker=m);la.text(x,.20,lab,ha="center",va="center",fontsize=5)


def m4_count_field(rank, special):
    bx,by=special["full_population_aggregation"]["bins_xy"]
    xi=np.minimum(np.floor(rank.q_norm.to_numpy()*bx).astype(int),bx-1); yi=np.minimum(np.floor(rank.d_norm.to_numpy()*by).astype(int),by-1)
    counts=np.bincount(yi*bx+xi,minlength=bx*by).reshape(by,bx)
    specs=special["count_scale"]["classes"]; ci=np.full(counts.shape,-1,dtype=np.int8)
    for i,z in enumerate(specs):
        a=z["count"].split("-"); lo=int(a[0]);hi=int(a[-1]);ci[(counts>=lo)&(counts<=hi)]=i
    return counts,ci,specs


def render_m4(fig,spec,data):
    for p in spec["panels"]: add_title(fig,p)
    rank=data["rank"]; p=spec["panels"][0]; special=p["rank_density"];counts,ci,classes=m4_count_field(rank,special)
    ax=axis_for(fig,p["internal_layout"]["regions"]["rank_scatter_axis"]["rect_cm"]);rgb=np.full((*ci.shape,3),255,dtype=np.uint8)
    for i,z in enumerate(classes):
        h=z["hex"].lstrip("#");rgb[ci==i]=[int(h[j:j+2],16) for j in (0,2,4)]
    frame_mm=special["full_population_aggregation"]["plot_frame_mm"];frame_px=(round(frame_mm[0]/25.4*600),round(frame_mm[1]/25.4*600));rgb600=np.asarray(Image.fromarray(rgb).resize(frame_px,Image.Resampling.NEAREST))
    ax.imshow(rgb600,extent=(0,1,0,1),origin="lower",interpolation="none",aspect="auto",zorder=10)
    ax.plot([0,1],[0,1],color=NEUTRAL,lw=.7,ls=(0,(2,1.5)),zorder=30);ax.add_patch(Rectangle((.05,.05),.10,.10,fill=False,edgecolor=TEXT,lw=.5,ls=(0,(1.5,1)),zorder=35))
    for v in [.1]:
        ax.axvline(v,color="white",lw=1,zorder=40);ax.axhline(v,color="white",lw=1,zorder=40);ax.axvline(v,color=TEXT,lw=.5,zorder=41);ax.axhline(v,color=TEXT,lw=.5,zorder=41)
    style_axis(ax,"Multi-horizon normalized rank","Direct 60-day normalized rank");ax.set_xlim(0,1);ax.set_ylim(0,1)
    # Frozen count legend slot.
    rr=p["internal_layout"]["reserves"]["role_legend"]["rect_cm"];la=axis_for(fig,rr,frame=False);la.set_xlim(0,1);la.set_ylim(0,1)
    lower_labels=["1","2","4","8","16","32","64+"]
    la.text(.5,.96,"Record density",ha="center",va="top",fontsize=5)
    for i,(z,label) in enumerate(zip(classes,lower_labels)):
        x=i/7;la.add_patch(Rectangle((x,.58),.14,.22,facecolor=z["hex"],edgecolor="none"));la.text(x+.07,.30,label,ha="center",va="center",fontsize=5)
    p=spec["panels"][1]; reg=p["internal_layout"]["regions"]["threshold_zoom_axis"]["rect_cm"];ax=axis_for(fig,reg)
    z=rank[(rank.q_norm>=.05)&(rank.q_norm<=.15)&(rank.d_norm>=.05)&(rank.d_norm<=.15)]
    ax.scatter(z.q_norm,z.d_norm,s=6.428,alpha=.12,c=NEUTRAL,edgecolors="none",rasterized=True)
    ax.plot([.05,.15],[.05,.15],color=NEUTRAL,lw=.7,ls=(0,(2,1.5)));ax.axvline(.1,color=TEXT,lw=.5);ax.axhline(.1,color=TEXT,lw=.5);style_axis(ax,"Multi-horizon normalized rank","Direct 60-day normalized rank");ax.set_xlim(.05,.15);ax.set_ylim(.05,.15)
    ax.text(.055,.145,"Multi-horizon only",ha="left",va="top",fontsize=5,color=TEXT)
    ax.text(.145,.055,"Direct 60-day only",ha="right",va="bottom",fontsize=5,color=TEXT)
    ax.text(.055,.055,"Both selected",ha="left",va="bottom",fontsize=5,color=TEXT)
    ax.text(.145,.145,"Neither selected",ha="right",va="top",fontsize=5,color=TEXT)
    lane=axis_for(fig,p["internal_layout"]["regions"]["connector_lane"]["rect_cm"],frame=False);lane.set_xlim(0,1);lane.set_ylim(0,1);lane.plot([.15,.85],[.75,.75],color=NEUTRAL,lw=.5);lane.plot([.15,.85],[.25,.25],color=NEUTRAL,lw=.5);lane.text(.5,.5,"0.05-0.15",ha="center",va="center")
    p=spec["panels"][2]; summary=rank.groupby("origin_date").apply(lambda g:pd.Series({"q_only":int((g.q_selected&~g.d_selected).sum()),"d_only":int((~g.q_selected&g.d_selected).sum()),"both":int((g.q_selected&g.d_selected).sum()),"cap":int(g.queue_capacity__multi_horizon.iloc[0])}),include_groups=False).reset_index();summary["overlap"]=summary.both/summary.cap
    ax=axis_for(fig,p["internal_layout"]["regions"]["origin_overlap_axis"]["rect_cm"]);ax.plot(summary.origin_date,summary.both,color=PALETTE["both_selected"],lw=.8);ax.scatter(summary.origin_date[::8],summary.both[::8],s=5,marker="D",color=PALETTE["both_selected"]);style_axis(ax,"Prediction date","Number selected by both");ticks=pd.to_datetime(["2021-01-01","2025-01-01"]);ax.set_xticks(ticks,["2021","2025"])
    ax=axis_for(fig,p["internal_layout"]["regions"]["overlap_ecdf"]["rect_cm"]);x,y=ecdf(summary.overlap);ax.plot(x,y,color=PALETTE["both_selected"],lw=.8);style_axis(ax,"Overlap / capacity","ECDF")
    data["m4_counts"]=(counts,classes);data["m4_origin"]=summary;data["m4_zoom"]=z


def render_m5(fig,spec,data):
    for p in spec["panels"]: add_title(fig,p)
    ep=data["endpoints"];dr=data["draws"]
    p=spec["panels"][0]
    roles_frame=data["event_roles"]
    for rid,row,endpoint in zip(["primary_event_roles","secondary_event_roles"],ep.itertuples(),["primary","secondary"]):
        ax=axis_for(fig,p["internal_layout"]["regions"][rid]["rect_cm"],frame=False);ax.set_xlim(-.8,57.8);ax.set_ylim(-.8,11.8)
        n=int(row.all_events_denominator);qv=roles_frame[f"multi_horizon_{endpoint}"].to_numpy();dv=roles_frame[f"direct_60_day_{endpoint}"].to_numpy();roles=qv.astype(int)+2*dv.astype(int)
        xx=np.arange(n)%58;yy=11-np.arange(n)//58
        for code,c,m,fill in [(0,PALETTE["neither_selected"],"o",False),(1,Q,"o",True),(2,D,"s",False),(3,PALETTE["both_selected"],"D",True)]:
            z=roles==code;ax.scatter(xx[z],yy[z],s=3.0,marker=m,facecolors=c if fill else "white",edgecolors=c,linewidths=.3)
        endpoint_label="31–60 days" if endpoint=="primary" else "1–60 days"
        ax.text(.0,1.03,f"{endpoint_label} · n={n}",transform=ax.transAxes,va="bottom",fontsize=5.5)
    rr=p["internal_layout"]["reserves"]["role_legend"]["rect_cm"];la=axis_for(fig,rr,frame=False);la.set_xlim(0,1);la.set_ylim(0,1)
    keys=[
        ("Neither",PALETTE["neither_selected"],"o",False,.03,.72),
        ("Both",PALETTE["both_selected"],"D",True,.55,.72),
        ("Multi-horizon only",Q,"o",True,.03,.24),
        ("Direct 60-day only",D,"s",False,.55,.24),
    ]
    for lab,c,m,fill,x,y in keys:
        la.scatter([x],[y],s=5.6,marker=m,facecolors=c if fill else "white",edgecolors=c,linewidths=.5)
        la.text(x+.055,y,lab,va="center",fontsize=5)
    primary=ep.iloc[0];secondary=ep.iloc[1];p=spec["panels"][1]
    ax=axis_for(fig,p["internal_layout"]["regions"]["primary_rate_pair"]["rect_cm"]);vals=[primary.candidate_captured/primary.all_events_denominator*100,primary.baseline_captured/primary.all_events_denominator*100];ax.plot([0,1],vals,color=TEXT,lw=.5);ax.scatter([0],[vals[0]],s=6.4,c=Q);ax.scatter([1],[vals[1]],s=5.6,facecolors="white",edgecolors=D,marker="s");ax.set_xticks([0,1],["Multi-\nhorizon","Direct\n60-day"]);ax.set_title("Event coverage",loc="left",fontsize=5.5,pad=1);style_axis(ax,"","Event coverage (%)")
    ax=axis_for(fig,p["internal_layout"]["regions"]["primary_effect_interval"]["rect_cm"]);draw_interval(ax,.5,primary.point_difference*100,primary.percentile_ci_low*100,primary.percentile_ci_high*100,Q,"o",True,1);ax.axvline(0,color=TEXT,lw=.5);ax.axvline(3,color=TEXT,lw=.5,ls=(0,(2,1.5)));ax.text(3,.96,"3 pp reference",rotation=90,ha="right",va="top",fontsize=5);ax.set_ylim(0,1);ax.set_yticks([]);ax.set_title("Difference and 95% CI",loc="left",fontsize=5.5,pad=1);style_axis(ax,"Event coverage difference (pp)","")
    ax=axis_for(fig,p["internal_layout"]["regions"]["primary_draw_distribution"]["rect_cm"]);x,y=ecdf(dr["capture_rate_difference__primary_31_60"]*100);ax.fill_between(x,0,y,color=PALETTE["uncertainty_support"],alpha=1);ax.plot(x,y,color=Q,lw=1);ax.axvline(primary.point_difference*100,color=Q,lw=1);ax.set_title("Bootstrap ECDF",loc="left",fontsize=5.5,pad=1);style_axis(ax,"Event coverage difference (pp)","ECDF")
    p=spec["panels"][2];ax=axis_for(fig,p["internal_layout"]["regions"]["secondary_rate_pair"]["rect_cm"]);vals=[secondary.candidate_captured/secondary.all_events_denominator*100,secondary.baseline_captured/secondary.all_events_denominator*100];ax.plot([0,1],vals,color=LIGHT,lw=.5);ax.scatter([0],[vals[0]],s=5.6,facecolors="white",edgecolors=Q);ax.scatter([1],[vals[1]],s=5.6,facecolors="white",edgecolors=D,marker="s");ax.set_xticks([0,1],["Multi-\nhorizon","Direct\n60-day"]);ax.set_title("Event coverage",loc="left",fontsize=5.5,pad=1);style_axis(ax,"","Event coverage (%)")
    ax=axis_for(fig,p["internal_layout"]["regions"]["secondary_effect_interval"]["rect_cm"]);draw_interval(ax,.5,secondary.point_difference*100,secondary.percentile_ci_low*100,secondary.percentile_ci_high*100,Q,"o",False,.7);ax.axvline(0,color=TEXT,lw=.5);ax.set_ylim(0,1);ax.set_yticks([]);ax.set_title("Difference and 95% CI",loc="left",fontsize=5.5,pad=1);style_axis(ax,"Event coverage difference (pp)","")
    ax=axis_for(fig,p["internal_layout"]["regions"]["secondary_draw_ecdf"]["rect_cm"]);x,y=ecdf(dr["capture_rate_difference__secondary_1_60"]*100);ax.plot(x,y,color=Q,lw=.7,ls=(0,(2,1.5)));ax.set_title("Bootstrap ECDF",loc="left",fontsize=5.5,pad=1);style_axis(ax,"Event coverage difference (pp)","ECDF")


def render_ed1(fig,spec,data):
    for p in spec["panels"]: add_title(fig,p)
    ev=data["events"];yrs=data["years"];gc=data["gcounts"]
    p=spec["panels"][0]
    for rid,n,lakes,states,c,node_labels in [("main_cohort_path",696,355,13,EVENT,[r"75 μg L$^{-1}$","station\nlink","recorded\nevent date","60-d\nseparation"]),("strict_sensitivity_path",325,160,9,NEUTRAL,[r"75 μg L$^{-1}$","centroid\n250 m","recorded\nevent date","60-d\nseparation"])]:
        ax=axis_for(fig,p["internal_layout"]["regions"][rid]["rect_cm"],frame=False);ax.set_xlim(0,1);ax.set_ylim(0,1)
        x=[.12,.38,.64,.88]; y=[.5]*4;ax.plot(x,y,color=c,lw=.7);ax.scatter(x,y,s=[6,6,6,10],facecolors=["white","white","white",c],edgecolors=c,marker="o");ax.text(.5,.78,"Main" if n==696 else "Strict centroid",ha="center",fontsize=6.5)
        for xx,label in zip(x,node_labels):ax.text(xx,.37,label,ha="center",va="top",fontsize=5,linespacing=.9)
        ax.text(.5,.12,f"{n} events · {lakes} lakes · {states} states",ha="center",fontsize=5)
    p=spec["panels"][1]
    for rid,col,ylabel in [("episode_axis","event_count","Events"),("lake_axis","distinct_lakes","Lakes with events")]:
        ax=axis_for(fig,p["internal_layout"]["regions"][rid]["rect_cm"]);col=col if col in yrs.columns else ("episodes" if "episodes" in yrs.columns and ylabel=="Events" else [c for c in yrs.columns if "lake" in c.lower()][0]);ax.plot(yrs.year,yrs[col],color=EVENT if ylabel=="Events" else NEUTRAL,lw=.8);ax.scatter(yrs.year,yrs[col],s=6,facecolors="white",edgecolors=EVENT if ylabel=="Events" else NEUTRAL);style_axis(ax,"Year",ylabel);ax.set_xticks(yrs.year)
    p=spec["panels"][2]
    for rid,typ in [("state_support_axis","state"),("ecoregion_support_axis","ecoregion")]:
        ax=axis_for(fig,p["internal_layout"]["regions"][rid]["rect_cm"]);sub=gc[(gc.classification_type.astype(str).str.lower().str.contains(typ)) & (gc.population_id.astype(str).str.contains("event",case=False))].copy()
        if sub.empty: sub=gc[gc.classification_type.astype(str).str.lower().str.contains(typ)].head(16)
        sub=sub.sort_values("lake_count").tail(16);y=np.arange(len(sub));labels=sub.classification_value.astype(str).tolist()
        if typ=="ecoregion":
            labels=[x.replace("Upper Midwest","Upper MW").replace("Temperate Plains","Temperate").replace("Coastal Plains","Coastal").replace("Southern Appalachians","S Appalach").replace("Northern Appalachians","N Appalach").replace("Northern Plains","N Plains").replace("Western Mountains","W Mtns").replace("Southern Plains","S Plains") for x in labels]
        ax.hlines(y,0,sub.lake_count,color=LIGHT,lw=.5);ax.scatter(sub.lake_count,y,s=6,facecolors="white",edgecolors=EVENT if typ=="state" else NEUTRAL);ax.set_yticks(y,labels);ax.tick_params(axis="y",labelsize=5,pad=1);ax.set_xlim(0,max(sub.lake_count)*1.15);style_axis(ax,"Lakes","" )


def render_ed2(fig,spec,data):
    for p in spec["panels"]: add_title(fig,p)
    paths=data["paths"]; feat=pd.read_parquet(paths["features"]);ev=data["events"]
    p=spec["panels"][0]
    lagcols=["field_chla_lag30_valid_day_fraction","field_chla_lag90_valid_day_fraction"]
    for rid,col in zip(["lag30_distribution","lag90_distribution"],lagcols):
        ax=axis_for(fig,p["internal_layout"]["regions"][rid]["rect_cm"]);v=feat[col].dropna().to_numpy();x,y=ecdf(v);ax.plot(x,y,color=Q if "30" in col else NEUTRAL,lw=.8,ls="-" if "30" in col else "--");style_axis(ax,"Fraction of valid days","ECDF");ax.set_xlim(-.02,1.02)
    lr=p["internal_layout"]["reserves"]["support_legend"]["rect_cm"];la=axis_for(fig,lr,frame=False);la.set_xlim(0,1);la.set_ylim(0,1);la.plot([.03,.23],[.5,.5],color=Q,lw=.8);la.text(.27,.5,"30 d",va="center",fontsize=5);la.plot([.58,.78],[.5,.5],color=NEUTRAL,lw=.7,ls="--");la.text(.82,.5,"90 d",va="center",fontsize=5)
    ax=axis_for(fig,p["internal_layout"]["regions"]["missing_reason_axis"]["rect_cm"]);c30=feat["field_chla_lag30_missing_reason_code"].fillna("observed").astype(str);c90=feat["field_chla_lag90_missing_reason_code"].fillna("observed").astype(str);counts=pd.concat([c30.value_counts().rename("30 d"),c90.value_counts().rename("90 d")],axis=1).fillna(0).sort_values("90 d").tail(10);y=np.arange(len(counts));ax.hlines(y,counts["30 d"],counts["90 d"],color=LIGHT,lw=.7);ax.scatter(counts["30 d"],y,s=6,facecolors=Q,edgecolors=Q);ax.scatter(counts["90 d"],y,s=5.6,facecolors="white",edgecolors=NEUTRAL,marker="s")
    xmax=float(counts[["30 d","90 d"]].to_numpy().max());label_room=.62*xmax
    ax.set_xlim(-label_room,xmax*1.04);ax.set_yticks(y,[])
    for yy,code in zip(y,counts.index):
        label="Valid observation available" if str(code)=="observed" else missing_reason_label(code)
        ax.text(-label_room*.96,yy,label,ha="left",va="center",fontsize=5)
    record_ticks=np.arange(0,math.floor(xmax/100000)*100000+1,100000,dtype=int)
    ax.set_xticks(record_ticks,[f"{int(v):,}" for v in record_ticks]);style_axis(ax,"Number of lake–prediction-date records","")
    p=spec["panels"][1];sources=[]
    for name,file in [("Combined","combined_lake_day_2005_2025.parquet"),("USGS","usgs_lake_day.parquet"),("WQP","wqp_lake_day.parquet")]:
        df=pd.read_parquet(paths["ext"] / f"event_outcomes/{file}");datecol=next(c for c in df if "date" in c.lower());lakecol=next(c for c in df if "lake" in c.lower() and "id" in c.lower());year=pd.to_datetime(df[datecol]).dt.year;sources.append(pd.DataFrame({"year":year,"source":name,"rows":1,"lake":df[lakecol]}))
    src=pd.concat(sources);annual=src.groupby(["source","year"]).agg(lake_days=("rows","sum"),lakes=("lake","nunique")).reset_index()
    for rid,col,ylabel in [("lake_day_axis","lake_days","Lake-days"),("distinct_lake_axis","lakes","Distinct lakes")]:
        ax=axis_for(fig,p["internal_layout"]["regions"][rid]["rect_cm"])
        for i,(name,g) in enumerate(annual.groupby("source")):ax.plot(g.year,g[col],lw=[1,.7,.7][i],ls=["-","--",":"][i],color=[Q,D,NEUTRAL][i],label=name)
        style_axis(ax,"Year",ylabel)
    lr=p["internal_layout"]["reserves"]["source_legend"]["rect_cm"];la=axis_for(fig,lr,frame=False);la.set_xlim(0,1);la.set_ylim(0,1)
    source_keys=[("Combined",Q,"-",.04,.70),("USGS",D,"--",.04,.22),("WQP",NEUTRAL,":",.55,.22)]
    for lab,c,ls,x,y0 in source_keys:
        la.plot([x,x+.13],[y0,y0],color=c,lw=.7,ls=ls);la.text(x+.17,y0,lab,va="center",fontsize=5)
    p=spec["panels"][2]
    for rid,col,xlabel in [("prior_observation_ecdf","prior_observation_count_1_60d","Prior\nobservations"),("candidate_origin_ecdf","candidate_origins_1_60d","Eligible prediction dates\nper event")]:
        ax=axis_for(fig,p["internal_layout"]["regions"][rid]["rect_cm"]);x,y=ecdf(ev[col]);ax.plot(x,y,color=EVENT if "prior" in rid else Q,lw=.8);style_axis(ax,xlabel,"ECDF")
    ax=axis_for(fig,p["internal_layout"]["regions"]["evaluability_marks"]["rect_cm"]);vals=[ev.candidate_coverage_31_60d.sum(),ev.candidate_coverage_1_60d.sum(),len(ev)];ax.plot(vals,[2,1,0],color=LIGHT,lw=.7);ax.scatter(vals,[2,1,0],s=7,facecolors=[Q,"white","white"],edgecolors=[Q,Q,EVENT]);ax.set_yticks([0,1,2],["All events","1–60 days","31–60 days"]);style_axis(ax,"Number of events","")
    data["ed2_annual"]=annual


def render_ed3(fig,spec,data):
    for p in spec["panels"]: add_title(fig,p)
    paths=data["paths"];yf=pd.read_csv(paths["dev_eval"] / "by_year_and_fold.csv");rm=pd.read_csv(paths["row_metrics"]);sd=pd.read_parquet(paths["dev_eval"] / "strict_bootstrap_draws.parquet");comp=pd.read_csv(paths["dev_eval"] / "strict_bootstrap_comparisons.csv")
    p=spec["panels"][0]
    for rid,group,ep in [("year_primary_axis","year","primary"),("year_secondary_axis","year","secondary"),("fold_primary_axis","fold","primary"),("fold_secondary_axis","fold","secondary")]:
        ax=axis_for(fig,p["internal_layout"]["regions"][rid]["rect_cm"]);sub=yf[yf.group_type.astype(str).str.contains(group,case=False)]
        if sub.empty:sub=yf.head(12)
        valcol=next((c for c in sub.columns if "capture" in c.lower()),sub.select_dtypes("number").columns[-1]);x=np.arange(len(sub));ax.plot(x,sub[valcol],color=Q if ep=="primary" else NEUTRAL,lw=1 if ep=="primary" else .7);ax.scatter(x,sub[valcol],s=5,facecolors=Q if ep=="primary" else "white",edgecolors=Q);ax.set_xticks([]);ax.set_title(f"{group} · {ep}",fontsize=5.5,pad=1)
    rr=p["internal_layout"]["reserves"]["group_legend"]["rect_cm"];la=axis_for(fig,rr,frame=False);la.set_xlim(0,1);la.set_ylim(0,1);la.plot([.02,.22],[.5,.5],color=Q,lw=1);la.text(.26,.5,"primary",va="center",fontsize=5);la.plot([.60,.80],[.5,.5],color=NEUTRAL,lw=.7,ls="--");la.text(.84,.5,"secondary",va="center",fontsize=5)
    p=spec["panels"][1];metrics=[("auprc_axis","auprc","Average precision"),("auroc_axis","auroc","AUROC"),("brier_axis","brier","Brier score"),("logloss_axis","log_loss","Log loss"),("calibration_intercept_axis","calibration_intercept","Calibration intercept"),("calibration_slope_axis","calibration_slope","Calibration slope")]
    policy_order=["multi_horizon","direct_60_day","environment_60_day_reference","basic_60_day_reference","early_30_day_reference"]
    policy_style={"multi_horizon":(Q,"o",True),"direct_60_day":(D,"s",False),"environment_60_day_reference":(NEUTRAL,"^",True),"basic_60_day_reference":(NEUTRAL,"D",False),"early_30_day_reference":(TEXT,"v",False)}
    overall=rm[rm.group.astype(str).eq("all_2017_2020")].set_index("policy").reindex(policy_order)
    for idx,(rid,col,title) in enumerate(metrics):
        ax=axis_for(fig,p["internal_layout"]["regions"][rid]["rect_cm"]);x=np.arange(len(policy_order));vals=overall[col].to_numpy(float)
        for xi,pid,val in zip(x,policy_order,vals):
            color,marker,filled=policy_style[pid];ax.scatter([xi],[val],s=7,marker=marker,facecolors=color if filled else "white",edgecolors=color,linewidths=.55,zorder=4)
        ax.set_xlim(-.6,4.6);ax.set_xticks(x,["1","2","3","4","5"] if idx>=3 else ["", "", "", "", ""]);ax.tick_params(axis="x",labelsize=5,pad=1);ax.set_title(title,fontsize=5.2,pad=1)
    # A two-row shared key uses the otherwise empty band above the six metric axes.
    # Five source policies exist; the prompt's presumed three-policy grouping is not
    # supported by the frozen table and is therefore not recreated.
    la=axis_for(fig,[9.5,1.5,7.5,.5],frame=False);la.set_xlim(0,1);la.set_ylim(0,1)
    legend_items=[
        ("1  Multi-horizon",Q,"o",True,.01,.70),
        ("2  Direct 60-day",D,"s",False,.29,.70),
        ("3  60-day environment reference",NEUTRAL,"^",True,.66,.70),
        ("4  60-day backbone reference",NEUTRAL,"D",False,.19,.22),
        ("5  30-day reference",TEXT,"v",False,.62,.22),
    ]
    for label,color,marker,filled,x0,y0 in legend_items:
        la.scatter([x0],[y0],s=5.5,marker=marker,facecolors=color if filled else "white",edgecolors=color,linewidths=.5)
        la.text(x0+.025,y0,label,va="center",ha="left",fontsize=5)
    p=spec["panels"][2]
    draw_specs=[("primary_draw_distribution","captured_31_60__multi_horizon","captured_31_60__direct_60_day",Q,"-","31-60 d"),("secondary_draw_distribution","captured_1_60__multi_horizon","captured_1_60__direct_60_day",NEUTRAL,"--","1-60 d")]
    for rid,qcol,dcol,c,ls,label in draw_specs:
        values=sd[qcol]-sd[dcol];ax=axis_for(fig,p["internal_layout"]["regions"][rid]["rect_cm"]);x,y=ecdf(values);ax.plot(x,y,color=c,lw=1 if rid.startswith("primary") else .7,ls=ls);ax.set_title(label,loc="left",fontsize=5.5,pad=1);style_axis(ax,"Difference (events)","ECDF")
    rows=comp[(comp.candidate_policy=="multi_horizon")&(comp.baseline_policy=="direct_60_day")&comp.metric.isin(["captured_31_60_count_difference","captured_1_60_count_difference"])].copy()
    rows["endpoint_order"]=rows.metric.map({"captured_31_60_count_difference":1,"captured_1_60_count_difference":0});rows=rows.sort_values("endpoint_order")
    ax=axis_for(fig,p["internal_layout"]["regions"]["interval_summary"]["rect_cm"])
    for yv,r in enumerate(rows.itertuples()):draw_interval(ax,yv,float(r.estimate),float(r.ci_low),float(r.ci_high),Q if "31_60" in r.metric else NEUTRAL,"o",bool("31_60" in r.metric),1 if "31_60" in r.metric else .7)
    ax.set_yticks([0,1],["1-60 d","31-60 d"]);ax.tick_params(axis="y",labelsize=5,pad=1);style_axis(ax,"Difference (events)","")
    data["ed3_yf"]=yf;data["ed3_rm"]=rm;data["ed3_sd"]=sd;data["ed3_comp"]=comp


def render_ed4(fig,spec,data):
    for p in spec["panels"]: add_title(fig,p)
    unit=data["unit"];rank=data["rank"];p=spec["panels"][0];piv=unit.pivot(index="origin_date",columns="policy",values=["eligible_candidates","capacity"])
    for rid,field,ylabel,c in [("eligible_axis","eligible_candidates","Eligible lakes",NEUTRAL),("capacity_axis","capacity","Number selected",Q)]:
        ax=axis_for(fig,p["internal_layout"]["regions"][rid]["rect_cm"]);ax.step(piv.index,piv[field].iloc[:,0],where="mid",color=c,lw=.8);style_axis(ax,"Prediction date",ylabel);ax.xaxis.set_major_locator(mdates.YearLocator());ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax=axis_for(fig,p["internal_layout"]["regions"]["rule_residual_axis"]["rect_cm"]);res=piv["capacity"].iloc[:,0]-.10*piv["eligible_candidates"].iloc[:,0];ax.vlines(piv.index,0,res,color=LIGHT,lw=.45);ax.scatter(piv.index,res,s=5,facecolors="white",edgecolors=TEXT);ax.axhline(0,color=TEXT,lw=.4);ax.set_ylim(-.05,1.05);style_axis(ax,"Prediction date","Rounding increment");ax.xaxis.set_major_locator(mdates.YearLocator());ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    summary=rank.groupby("origin_date").apply(lambda g:pd.Series({"q_only":int((g.q_selected&~g.d_selected).sum()),"d_only":int((~g.q_selected&g.d_selected).sum()),"both":int((g.q_selected&g.d_selected).sum()),"cap":int(g.queue_capacity__multi_horizon.iloc[0])}),include_groups=False).reset_index();summary["overlap"]=summary.both/summary.cap
    p=spec["panels"][1];ax=axis_for(fig,p["internal_layout"]["regions"]["transition_count_axis"]["rect_cm"]);ax.stackplot(summary.origin_date,summary.q_only,summary.both,summary.d_only,colors=[Q,PALETTE["both_selected"],D],alpha=1)
    last=summary.iloc[-1];last_date=pd.Timestamp(last.origin_date);label_date=last_date+pd.Timedelta(days=245)
    ax.set_xlim(summary.origin_date.min(),last_date+pd.Timedelta(days=520))
    label_positions=[
        ("Multi-horizon only",Q,float(last.q_only)/2),
        ("Shared",PALETTE["both_selected"],float(last.q_only)+float(last.both)/2),
        ("Direct 60-day only",D,float(last.q_only)+float(last.both)+float(last.d_only)/2),
    ]
    for label,color,y0 in label_positions:
        ax.annotate(label,xy=(last_date,y0),xytext=(label_date,y0),ha="left",va="center",fontsize=5,color=TEXT,arrowprops={"arrowstyle":"-","color":color,"lw":.7,"shrinkA":0,"shrinkB":0})
    style_axis(ax,"Prediction date","Number selected");ax.xaxis.set_major_locator(mdates.YearLocator(2));ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    rr=p["internal_layout"]["reserves"]["transition_legend"]["rect_cm"];la=axis_for(fig,rr,frame=False);la.set_xlim(0,1);la.set_ylim(0,1)
    ax=axis_for(fig,p["internal_layout"]["regions"]["overlap_fraction_ecdf"]["rect_cm"]);x,y=ecdf(summary.overlap);ax.plot(x,y,color=PALETTE["both_selected"],lw=.8);style_axis(ax,"Overlap / capacity","ECDF")
    p=spec["panels"][2]
    cross_rank_axes = [
        ("multi_horizon_only_direct_rank_ecdf", rank.q_selected & ~rank.d_selected, "d_norm", Q, "Direct 60-day\nnormalized rank"),
        ("direct_only_multi_horizon_rank_ecdf", ~rank.q_selected & rank.d_selected, "q_norm", D, "Multi-horizon\nnormalized rank"),
    ]
    for rid,mask,col,c,xlabel in cross_rank_axes:
        ax=axis_for(fig,p["internal_layout"]["regions"][rid]["rect_cm"]);x,y=ecdf(rank.loc[mask,col]);ax.plot(x,y,color=c,lw=.8,ls="-" if c==Q else "--");style_axis(ax,xlabel,"ECDF")
    rr=p["internal_layout"]["reserves"]["cross_rank_legend"]["rect_cm"];la=axis_for(fig,rr,frame=False);la.set_xlim(0,1);la.set_ylim(0,1)
    la.plot([.03,.16],[.62,.62],color=Q,lw=.8);la.text(.20,.62,"Multi-horizon only",va="center",fontsize=5)
    la.plot([.03,.16],[.25,.25],color=D,lw=.8,ls="--");la.text(.20,.25,"Direct 60-day only",va="center",fontsize=5)
    data["ed4_origin"]=summary


def sensitivity_table(data):
    contract=json.loads((data["paths"]["formal"] / "evaluation_contract.json").read_text())
    rows=contract["sensitivity_results"]["endpoint_summary"]
    out=pd.DataFrame(rows)
    if "scenario_order" not in out:
        order=contract["sensitivity_results"]["bundle_audit"]["scenario_names"];out["scenario_order"]=out["scenario"].map({v:i for i,v in enumerate(order)})
    return out.sort_values(["scenario_order","endpoint"]).reset_index(drop=True)


def render_ed5(fig,spec,data):
    for p in spec["panels"]: add_title(fig,p)
    s=sensitivity_table(data);data["sens"]=s
    def scenario_label(value):
        value=str(value)
        if value=="primary_auto_w60": return "Baseline"
        if value.startswith("strict_centroid250"): return "Strict 250-m linkage"
        if value.startswith("tight_onset60"): return "Event-date uncertainty ≤60 d"
        if value.startswith("washout"):
            return value.replace("washout","").split("_")[0] + "-d event separation"
        if "leave_one_state_out__" in value:
            return "Leave-one-state-out: " + value.split("__")[-1]
        if value.startswith("exclude2025"): return "Exclude 2025"
        if value.startswith("monitored_lakes"): return "Monitored lakes only"
        return value
    for panel,needle in zip(spec["panels"],["primary","secondary"]):
        sub=s[s.endpoint.astype(str).str.contains(needle,case=False)].copy();sub=sub.sort_values("scenario_order");rid=[x for x in panel["internal_layout"]["regions"] if "sensitivity_axis" in x][0]
        ax=axis_for(fig,panel["internal_layout"]["regions"][rid]["rect_cm"]);y=np.arange(len(sub))[::-1]
        for yi,r in zip(y,sub.itertuples()):
            point=float(r.point_difference)*100;lo=float(r.percentile_ci_low)*100;hi=float(r.percentile_ci_high)*100;canonical=str(r.scenario).startswith("primary_auto")
            draw_interval(ax,yi,point,lo,hi,Q if canonical else NEUTRAL,"o" if canonical else "D",canonical,1 if canonical else .7)
        max_hi=max(float(v)*100 for v in sub.percentile_ci_high);ax.set_xlim(-11,max_hi+1.5);ax.axvline(0,color=TEXT,lw=.5);ax.set_yticks([])
        for yi,label in zip(y,[scenario_label(x) for x in sub.scenario]):ax.text(-10.7,yi,label,ha="left",va="center",fontsize=5,color=TEXT)
        style_axis(ax,"Event coverage difference (pp)","")
        dax=axis_for(fig,panel["internal_layout"]["regions"]["denominator_axis"]["rect_cm"]);den=sub["all_events_denominator"] if "all_events_denominator" in sub else sub["denominator"];dax.plot(den,y,color=LIGHT,lw=.5);dax.scatter(den,y,s=5,facecolors="white",edgecolors=NEUTRAL);dax.set_yticks([]);style_axis(dax,"Number of events","")


def render_ed6(fig,spec,data):
    for p in spec["panels"]: add_title(fig,p)
    dr=data["draws"];ep=data["endpoints"];x=dr["capture_rate_difference__primary_31_60"]*100;y=dr["capture_rate_difference__secondary_1_60"]*100
    p=spec["panels"][0];ax=axis_for(fig,p["internal_layout"]["regions"]["joint_plane_axis"]["rect_cm"]);ax.scatter(x,y,s=6.428,alpha=.14,c=NEUTRAL,edgecolors="none",rasterized=True);ax.axvline(0,color=TEXT,lw=.5);ax.axhline(0,color=TEXT,lw=.5);ax.axvline(3,color=TEXT,lw=.5,ls=(0,(2,1.5)));ax.text(3,.98,"3 pp reference",transform=ax.get_xaxis_transform(),rotation=90,ha="right",va="top",fontsize=5);style_axis(ax,"31–60-day event coverage difference (pp)","1–60-day event coverage difference (pp)")
    ax.scatter([ep.iloc[0].point_difference*100],[ep.iloc[1].point_difference*100],s=6.428,c=Q,edgecolors=Q,zorder=5)
    p=spec["panels"][1]
    for rid,v,c,ls in [("primary_marginal",x,Q,"-"),("secondary_marginal",y,NEUTRAL,"--")]:
        ax=axis_for(fig,p["internal_layout"]["regions"][rid]["rect_cm"]);xx,yy=ecdf(v);ax.plot(xx,yy,color=c,lw=1 if rid.startswith("primary") else .7,ls=ls);style_axis(ax,"Event coverage difference (pp)","ECDF")
    ax=axis_for(fig,p["internal_layout"]["regions"]["quantile_summary"]["rect_cm"]);draw_interval(ax,.68,ep.iloc[0].point_difference*100,ep.iloc[0].percentile_ci_low*100,ep.iloc[0].percentile_ci_high*100,Q,"o",True,1);draw_interval(ax,.32,ep.iloc[1].point_difference*100,ep.iloc[1].percentile_ci_low*100,ep.iloc[1].percentile_ci_high*100,NEUTRAL,"o",False,.7);ax.set_yticks([.32,.68],["1–60 days","31–60 days"]);style_axis(ax,"Event coverage difference (pp)","")


def render_ed7(fig,spec,data):
    for p in spec["panels"]: add_title(fig,p)
    ev=data["events"];wash=data["washout"]
    p=spec["panels"][0];ax=axis_for(fig,p["internal_layout"]["regions"]["bracket_width_ecdf"]["rect_cm"]);x,y=ecdf(ev.onset_bracket_width_days.dropna());ax.plot(x,y,color=EVENT,lw=.8);style_axis(ax,"Event-date bracket width (days)","ECDF")
    ax=axis_for(fig,p["internal_layout"]["regions"]["tight_boundary_marks"]["rect_cm"]);vals=[ev.tight_boundary_60d.sum(),ev.tight_boundary_90d.sum()];ax.plot(vals,[1,0],color=LIGHT,lw=.5);ax.scatter(vals,[1,0],s=7,facecolors="white",edgecolors=EVENT);ax.set_yticks([0,1],["≤90 d","≤60 d"]);style_axis(ax,"Number of events","")
    p=spec["panels"][1];ax=axis_for(fig,p["internal_layout"]["regions"]["previous_gap_ecdf"]["rect_cm"]);x,y=ecdf(ev.previous_severe_gap_days.dropna());ax.plot(x,y,color=NEUTRAL,lw=.8);style_axis(ax,"Previous severe gap (days)","ECDF")
    ax=axis_for(fig,p["internal_layout"]["regions"]["prior_nonsevere_axis"]["rect_cm"]);v=ev.prior_nonsevere_count_1_60d.value_counts().sort_index();ax.vlines(v.index,0,v,color=LIGHT,lw=1);ax.scatter(v.index,v,s=6,facecolors="white",edgecolors=EVENT);style_axis(ax,"Prior observations below threshold (1–60 days)","Number of events")
    p=spec["panels"][2];ax=axis_for(fig,p["internal_layout"]["regions"]["washout_count_axis"]["rect_cm"]);xcol=next(c for c in wash.columns if "washout" in c.lower() and "day" in c.lower());ycol=next(c for c in wash.columns if "event" in c.lower() or "episode" in c.lower());ax.plot(wash[xcol],wash[ycol],color=EVENT,lw=.8);ax.scatter(wash[xcol],wash[ycol],s=7,facecolors=[EVENT if x==60 else "white" for x in wash[xcol]],edgecolors=EVENT);ax.axvline(60,color=TEXT,lw=.5);style_axis(ax,"Minimum event separation (days)","Number of events")
    rr=p["internal_layout"]["reserves"]["definition_note"]["rect_cm"];na=axis_for(fig,rr);na.plot(wash[xcol],wash[ycol],color=LIGHT,lw=.6);na.vlines(wash[xcol],wash[ycol].min(),wash[ycol],color=LIGHT,lw=.45);na.scatter(wash[xcol],wash[ycol],s=6,facecolors=[EVENT if x==60 else "white" for x in wash[xcol]],edgecolors=EVENT);na.axvline(60,color=TEXT,lw=.45);na.set_xticks(wash[xcol]);na.tick_params(axis="x",labelsize=5);na.text(.98,.94,"30–90 d detail",transform=na.transAxes,ha="right",va="top",fontsize=5);style_axis(na,"Minimum event separation (days)","Number of events")


RENDERERS={"M1":render_m1,"M2":render_m2,"M3":render_m3,"M4":render_m4,"M5":render_m5,"ED1":render_ed1,"ED2":render_ed2,"ED3":render_ed3,"ED4":render_ed4,"ED5":render_ed5,"ED6":render_ed6,"ED7":render_ed7}
