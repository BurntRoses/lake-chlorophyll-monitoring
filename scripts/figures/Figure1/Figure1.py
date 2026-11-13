"""Draw study lakes, monitoring workload and event-level evaluation."""
from pathlib import Path
from datetime import datetime
import sys
sys.dont_write_bytecode = True
import csv
import hashlib
import json
from collections import Counter

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from matplotlib.ticker import AutoMinorLocator
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, PathPatch, FancyArrowPatch, ConnectionPatch
from matplotlib.path import Path as MPath
from matplotlib.transforms import ScaledTranslation
import rasterio
import geopandas as gpd
from pyproj import Transformer, Geod
from scipy.optimize import brentq
from scipy.ndimage import gaussian_filter
from shapely.geometry import shape
from shapely.ops import transform as shape_transform

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT.parents[2] / 'figures'
DATA = ROOT / "data"
BASE = ROOT / "basemap"
from figure_style import STYLE, FS, COLORS, LINES, apply_style
W, H = float(STYLE['canvas']['main_width_mm']), float(STYLE['canvas']['figure1_height_mm'])
INK = COLORS['ink']
BLUE = COLORS['blue']
RED = COLORS['vermilion']
GREEN = COLORS['green']
GREY = COLORS['grey']
WATER = COLORS['water']
LAND = COLORS['surrounding_land']
MAP_EVENT = STYLE['geographic_lake_categories']['primary_event_bearing']
MAP_OTHER = STYLE['geographic_lake_categories']['other_study_lake']
PROJ = Transformer.from_crs(4326, 5070, always_xy=True)
UNPROJ = Transformer.from_crs(5070, 4326, always_xy=True)
GEOD = Geod(ellps="GRS80")

FONT_SELECTED = apply_style()
from unified_style import apply as apply_unified, export_figure
from connector_geometry import component
apply_unified()

def csv_rows(name):
    with (DATA / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

LAKES = csv_rows("lake_coordinates.csv")
WORK = csv_rows("monitoring_workload.csv")
CALENDAR = csv_rows("evaluation_calendar.csv")
assert len(LAKES) == len({r["lake_id"] for r in LAKES}) == 2844
assert len(WORK) == 78 and len(CALENDAR) == 9
dictionary = json.loads((ROOT / "data_dictionary.json").read_text(encoding="utf-8-sig"))
for item in dictionary["data_files"]:
    assert hashlib.sha256((ROOT / item["file"]).read_bytes()).hexdigest() == item["sha256"]
lon = np.array([float(r["longitude_deg"]) for r in LAKES])
lat = np.array([float(r["latitude_deg"]) for r in LAKES])
PX, PY = PROJ.transform(lon, lat)

# All methods share the primary event cohort.
event_dictionary = json.loads((ROOT / 'primary_event_dictionary.json').read_text(encoding='utf-8-sig'))
event_entry = next(x for x in event_dictionary['data_files'] if x['file'].endswith('primary_event_fingerprint.csv'))
assert hashlib.sha256((DATA / 'primary_event_fingerprint.csv').read_bytes()).hexdigest() == event_entry['sha256']
EVENT_ROWS = csv_rows('primary_event_fingerprint.csv')
method_ids = sorted({r['method_id'] for r in EVENT_ROWS})
event_sets = [{(r['episode_id'], r['lake_id'], r['event_date']) for r in EVENT_ROWS if r['method_id'] == method} for method in method_ids]
assert len(method_ids) == 4 and all(s == event_sets[0] and len(s) == 593 for s in event_sets)
EVENT_COUNTS = Counter(x[1] for x in event_sets[0])
assert set(EVENT_COUNTS) <= {r['lake_id'] for r in LAKES}
HAS_PRIMARY_EVENT = np.array([r['lake_id'] in EVENT_COUNTS for r in LAKES])
assert int(HAS_PRIMARY_EVENT.sum()) == 354
TOTAL_SAMPLED = sum(int(r["sampled_lakes"]) for r in WORK)
TOTAL_ELIGIBLE = sum(int(r["eligible_lakes"]) for r in WORK)
POOLED = 100 * TOTAL_SAMPLED / TOTAL_ELIGIBLE
assert (TOTAL_SAMPLED, TOTAL_ELIGIBLE) == (20027, 218838)
DATES = [datetime.fromisoformat(r["decision_date"]) for r in WORK]
FRACTIONS = np.array([float(r["sampled_fraction_pct"]) for r in WORK])
assert all((b - a).days == 14 for a, b in zip(DATES, DATES[1:]))

def project_geom(geom):
    return shape_transform(PROJ.transform, geom)

outline_json = json.loads((DATA / "us_outline.geojson").read_text(encoding="utf-8"))
state_json = json.loads((DATA / "us_states.geojson").read_text(encoding="utf-8"))
US = project_geom(shape(outline_json["features"][0]["geometry"]))
STATES = [project_geom(shape(f["geometry"])) for f in state_json["features"]]
lands = gpd.read_file(BASE / "ne_50m_land.zip", bbox=(-131, 20, -60, 56)).to_crs(5070)
waters = gpd.read_file(BASE / "ne_50m_lakes.zip", bbox=(-131, 20, -60, 56)).to_crs(5070)

def geom_path(geom):
    verts, codes = [], []
    geoms = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    for polygon in geoms:
        if polygon.geom_type != "Polygon":
            continue
        for ring in [polygon.exterior, *polygon.interiors]:
            xy = np.asarray(ring.coords)
            verts.extend(xy)
            codes.extend([MPath.MOVETO] + [MPath.LINETO] * (len(xy) - 2) + [MPath.CLOSEPOLY])
    return MPath(np.array(verts), codes)

US_PATH = geom_path(US)
LAND_PATHS = [geom_path(g) for g in lands.geometry if not g.is_empty]
WATER_PATHS = [geom_path(g) for g in waters.geometry if not g.is_empty]
STATE_PATHS = [geom_path(g) for g in STATES]

# A physical height ramp, shared by all maps. Negative heights are shown as zero.
ELEV_CMAP = mcolors.LinearSegmentedColormap.from_list("hypsometric", [
    (0.000, "#9FC27A"), (0.055, "#C3D690"), (0.180, "#E5D993"),
    (0.350, "#C8AE67"), (0.550, "#AF884D"), (0.750, "#927252"),
    (0.900, "#C8B59D"), (1.000, "#F3EEE4")])
ELEV_NORM = mcolors.Normalize(0, 4000, clip=True)

def relief(name):
    with rasterio.open(BASE / f"{name}_dem_5070.tif") as src:
        zz = src.read(1, masked=True)
        valid = ~np.ma.getmaskarray(zz)
        zz = np.asarray(zz.filled(0), dtype=float)
        zz[~np.isfinite(zz)] = 0
        z_smooth = gaussian_filter(zz, 0.45)
        shade = mcolors.LightSource(azdeg=315, altdeg=42).hillshade(
            z_smooth, vert_exag=3.0, dx=src.res[0], dy=src.res[1])
        rgb = ELEV_CMAP(ELEV_NORM(zz))[..., :3]
        # Shade the elevation surface.
        multiplier = 0.73 + 0.37 * shade
        rgba = np.dstack([np.clip(rgb * multiplier[..., None], 0, 1), valid.astype(float)])
        b = src.bounds
        return rgba, (b.left, b.right, b.bottom, b.top), list(src.res)

fig = plt.figure(figsize=(W / 25.4, H / 25.4))
page = fig.add_axes([0, 0, 1, 1], frameon=False, zorder=30)
page.set(xlim=(0, W), ylim=(H, 0))
page.axis("off")

def ax_at(x, y, w, h, **kwargs):
    return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H], **kwargs)

def text(x, y, s, fs=FS['body'], ha="left", va="center", color=INK, **kwargs):
    return page.text(x, y, s, ha=ha, va=va, fontsize=fs, color=color, **kwargs)

def p_line(pts, colour=INK, lw=0.65, **kwargs):
    return page.plot([p[0] for p in pts], [p[1] for p in pts], color=colour, lw=lw, **kwargs)[0]

def p_arrow(a, b, colour=INK, lw=0.8, **kwargs):
    return component(fig,[a,b],key='event-time-axis',scope='timeline',color=colour,lw=lw,radius=0,zorder=30)

def panel(letter, title, x, y):
    text(x, y, "("+letter+")", fs=FS['panel_letter'], fontweight="bold")

def map_background(ax, name, bounds=None, point_size=5.3, inset=False):
    rgba, extent, res = relief(name)
    ax.set_facecolor(WATER)
    for path in LAND_PATHS:
        ax.add_patch(PathPatch(path, facecolor=LAND, edgecolor="#918F83", lw=0.35, zorder=1))
    image = ax.imshow(rgba, extent=extent, origin="upper", interpolation="bilinear", zorder=2)
    clip = PathPatch(US_PATH, transform=ax.transData)
    image.set_clip_path(clip)
    for path in WATER_PATHS:
        ax.add_patch(PathPatch(path, facecolor=WATER, edgecolor="#7FA0B0", lw=0.35, zorder=4))
    for path in STATE_PATHS:
        ax.add_patch(PathPatch(path, facecolor="none", edgecolor="#676E64", lw=LINES['state_boundary'], zorder=5, alpha=0.8))
    ax.add_patch(PathPatch(US_PATH, facecolor="none", edgecolor="#565C55", lw=LINES['country_boundary'], zorder=6))
    if bounds is None:
        l, b, r, t = US.bounds
        bounds = (l - 85000, r + 85000, b - 85000, t + 85000)
    xmin, xmax, ymin, ymax = bounds
    keep = (PX >= xmin) & (PX <= xmax) & (PY >= ymin) & (PY <= ymax)
    other = keep & ~HAS_PRIMARY_EVENT
    event = keep & HAS_PRIMARY_EVENT
    p_other = ax.scatter(PX[other], PY[other], s=point_size, marker=MAP_OTHER['marker'], facecolors='none',
                         edgecolors=MAP_OTHER['colour'], linewidths=0.48, alpha=0.82, zorder=9)
    p_event = ax.scatter(PX[event], PY[event], s=point_size * 1.75, marker=MAP_EVENT['marker'], facecolors=MAP_EVENT['colour'],
                         edgecolors='white', linewidths=0.18, alpha=0.97, zorder=10)
    pts = (p_other, p_event)
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set(lw=LINES['axis'], color="#4C5356")
    return bounds, int(keep.sum()), pts, res

def scale_bar(ax, length_km, anchor=(0.065, 0.07), fs=FS['cartographic_detail'], compact=False, below=False):
    """Straight projected bar calibrated with ellipsoidal geodesic distance."""
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x0 = xmin + anchor[0] * (xmax - xmin)
    y0 = ymin + anchor[1] * (ymax - ymin)
    lon0, lat0 = UNPROJ.transform(x0, y0)
    def distance(dx):
        lon1, lat1 = UNPROJ.transform(x0 + dx, y0)
        return GEOD.inv(lon0, lat0, lon1, lat1)[2]
    ends = [0.0] + [brentq(lambda dx: distance(dx) - target * 1000,
                           target * 700, target * 1400)
                    for target in [length_km / 2, length_km]]
    height = (ymax - ymin) * 0.012
    width = ends[-1]
    draw_transform = ax.transData
    if below:
        # Translate the calibrated bar onto white space without changing its length.
        draw_transform = draw_transform + ScaledTranslation(0, -4.8/25.4, fig.dpi_scale_trans)
    for i in range(2):
        ax.add_patch(Rectangle((x0 + ends[i], y0), ends[i + 1] - ends[i], height,
                               facecolor=INK if i == 0 else "white", edgecolor=INK,
                               lw=0.55, zorder=19, transform=draw_transform, clip_on=not below))
    label_items = [(ends[-1] / 2, f"{length_km:,.0f} km")] if compact else [(0, "0"), (ends[-1], f"{length_km:,.0f} km")]
    for dx, label in label_items:
        ax.text(x0 + dx, y0 + height * (-1.8 if below else 1.8), label, fontsize=fs,
                ha="center", va="top" if below else "bottom", zorder=20, color=INK,
                transform=draw_transform, clip_on=False)
    assert xmin <= x0 < x0 + ends[-1] <= xmax
    independent_geod = Geod(ellps='WGS84')
    independent_distances = []
    for endpoint, target_km in zip(ends[1:], [length_km / 2, length_km]):
        lon1, lat1 = UNPROJ.transform(x0 + endpoint, y0)
        measured = independent_geod.inv(lon0, lat0, lon1, lat1)[2]
        assert abs(measured - target_km * 1000) < 0.01
        independent_distances.append(measured)
    return {"length_km": length_km, "calibration_lonlat": [lon0, lat0],
            "geodesic_distance_m": distance(ends[-1]),
            "independent_wgs84_half_and_full_m": independent_distances,
            "display_placement": "below map, on white space" if below else "inside map",
            "projected_endpoints": [[x0, y0], [x0 + ends[-1], y0]],
            "axis": ax}

def north_arrow(ax, lon0=-122, lat0=42.3):
    xx, yy = PROJ.transform(lon0, lat0)
    lon1, lat1, _ = GEOD.fwd(lon0, lat0, 0, 350000)
    x1, y1 = PROJ.transform(lon1, lat1)
    ax.annotate("", xy=(x1, y1), xytext=(xx, yy), zorder=20,
                arrowprops={"arrowstyle": "-|>", "mutation_scale": 12,
                            "lw": 1.0, "color": INK})
    ax.annotate("N", xy=(x1, y1), xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=FS['body'], fontweight="bold", zorder=20)

def geo_label(ax, longitude, latitude, label, fs=FS['geographic_name'], rotation=0):
    x, y = PROJ.transform(longitude, latitude)
    ax.text(x, y, label, fontsize=fs, fontfamily=FONT_SELECTED, fontstyle="italic",
            color="#24281F", rotation=rotation, ha="center", va="center", zorder=10)

# Study-area map and regional insets.
panel("a", "US study lakes", 5, 6.0)
main = ax_at(5, 11.0, 104.0, 61.0)
MAIN_BOUNDS, MAIN_COUNT, main_points, main_res = map_background(main, "conus", point_size=5.0)
assert MAIN_COUNT == 2844
scale_checks = [scale_bar(main, 1000)]
north_arrow(main)
geo_label(main, -111.0, 39.0, "Rocky Mountains", rotation=66)
geo_label(main, -100.0, 36.1, "Great Plains")

# Sparse geographic coordinates are drawn on the same projection as the map.
for longitude in [-120, -100, -80]:
    ll = np.linspace(20, 55, 220)
    gx, gy = PROJ.transform(np.full_like(ll, longitude), ll)
    main.plot(gx, gy, color="#526D7D", lw=0.38, ls=(0, (2, 3)), alpha=0.52, zorder=7)
    visible = (gy >= MAIN_BOUNDS[2]) & (gy <= MAIN_BOUNDS[3]) & (gx >= MAIN_BOUNDS[0]) & (gx <= MAIN_BOUNDS[1])
    i = np.flatnonzero(visible)[0]
    if longitude == -120:
        main.annotate('120°W', xy=(gx[i], gy[i]), xytext=(-5, 5),
                      textcoords='offset points', ha='center', va='center', rotation=90,
                      fontsize=FS['cartographic_detail'], annotation_clip=False)
    else:
        main.annotate(f"{abs(longitude)}°W", xy=(gx[i], gy[i]), xytext=(0, -6),
                      textcoords="offset points", ha="center", va="top", fontsize=FS['cartographic_detail'],
                      annotation_clip=False)

# Lake categories and geographic legend.


inset_info = []
for inset_id, name, title, rect, scale, anchor in [
    ("A", "midwest", "Upper Midwest", (133.0, 11.0, 40.85, 29.5), 200, (0.69, 0.07)),
    ("B", "florida", "Florida", (113.5, 45.0, 23.6, 30.0), 200, (0.23, 0.07)),
]:
    rgba, extent, resolution = relief(name)
    axis = ax_at(*rect)
    bounds, count, points, resolution = map_background(axis, name, bounds=extent,
                                                      point_size=6.0 if name == "midwest" else 5.5,
                                                      inset=True)
    text(rect[0], rect[1] - 3.3, inset_id, fs=FS['inset_title'])
    # Count all points within each inset extent.
    scale_checks.append(scale_bar(axis, scale, anchor=anchor, compact=True, below=name == 'midwest'))
    x0, x1, y0, y1 = bounds
    main.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="none",
                             edgecolor=RED, lw=LINES['inset_box'],
                             linestyle=STYLE['canvas']['inset_locator_box_style'], zorder=15))
    if inset_id == 'A':
        origin, destination = (x1, y1), (0, 1)
    else:
        origin, destination = (x1, (y0 + y1) / 2), (0, 0.5)
    connector = ConnectionPatch(xyA=origin, coordsA=main.transData,
                               xyB=destination, coordsB=axis.transAxes,
                               axesA=main, axesB=axis, color=RED, lw=LINES['guide'],
                               linestyle=STYLE['canvas']['inset_connector_style'], clip_on=False, zorder=16)
    fig.add_artist(connector)
    main.annotate(inset_id, xy=(x0, y1), xytext=(2, -2), textcoords='offset points',
                  ha='left', va='top', fontsize=FS['inset_title'], fontweight='bold', color=RED,
                  bbox={'facecolor':'white','edgecolor':'none','alpha':0.93,'pad':0.8}, zorder=25)
    in_view = (PX >= x0) & (PX <= x1) & (PY >= y0) & (PY <= y1)
    inset_info.append({"region": title, "view_bounds_epsg5070": list(bounds),
                       "displayed_lakes": count, "event_bearing_lakes": int((in_view & HAS_PRIMARY_EVENT).sum()),
                       "pixel_size_metres": resolution, 'locator_id': inset_id})

# Geographic legend beside the Florida inset.

page.plot(146.0, 55.0, marker=MAP_EVENT['marker'], ms=5.2, color=MAP_EVENT['colour'], markeredgecolor='white', markeredgewidth=0.3)
text(151.0, 55.0, 'Event-bearing', fs=FS['legend'])
page.plot(146.0, 62.0, marker=MAP_OTHER['marker'], ms=4.5, markerfacecolor='none', markeredgecolor=MAP_OTHER['colour'], markeredgewidth=0.8)
text(151.0, 62.0, 'Other lakes', fs=FS['legend'])

# Elevation scale.
cax = ax_at(6, 80.5, 36, 1.8)
cb = mpl.colorbar.ColorbarBase(cax, cmap=ELEV_CMAP, norm=ELEV_NORM, orientation="horizontal")
cb.set_ticks([0, 2000, 4000])
cb.set_ticklabels(["0", "2,000", "4,000"])
cb.ax.tick_params(labelsize=FS['cartographic_detail'], length=1.8, pad=1.0, width=0.55)
cb.outline.set_linewidth(0.5)
text(48, 81.3, "Elevation (m)", fs=FS['legend'])

# b: measured workload at every round, and the separate evaluation quota.
panel("b", "Observed monitoring workload", 5, 90.0)
work_ax = ax_at(17, 99.0, 69.5, 29)
work_ax.plot(DATES, FRACTIONS, color=BLUE, lw=LINES['curve'], marker="o", ms=2.6,
             markeredgecolor="white", markeredgewidth=0.25, zorder=4)
work_ax.axhline(POOLED, color=INK, lw=.7, ls=(0, (1, 2.1)), zorder=2)
work_ax.axhline(10, color=RED, lw=.8, ls=(0, (5, 2.5)), zorder=3)
work_ax.set(xlim=(datetime(2016, 1, 1), datetime(2019, 1, 1)), ylim=(0, 32))
work_ax.set_xticks([datetime(y, 1, 1) for y in [2016, 2017, 2018]])
work_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
work_ax.get_xticklabels()[0].set_horizontalalignment('left')
work_ax.set_yticks([0, 10, 20, 30])
work_ax.yaxis.set_minor_locator(AutoMinorLocator(2))
work_ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=6))
work_ax.set_ylabel("Sampled lakes (%)", fontsize=FS['axis_label'], labelpad=2.5)
work_ax.tick_params(labelsize=FS['tick'], length=2.5, pad=2, width=LINES['axis'])
work_ax.spines[["top", "right"]].set_visible(True)
work_ax.tick_params(which="both", top=True, right=True, direction="in")
work_ax.legend(handles=[
    Line2D([], [], color=INK, lw=.7, ls=(0, (1, 2.1)), label="Pooled 9.15%"),
    Line2D([], [], color=RED, lw=.8, ls=(0, (5, 2.5)), label="Quota 10%"),
], loc="center", bbox_to_anchor=(.46, 1.155), ncol=2, frameon=False,
    fontsize=FS['legend'], handlelength=1.6, handletextpad=0.4,
    columnspacing=.9, borderaxespad=0)
peak = int(FRACTIONS.argmax())
work_ax.annotate(f"{FRACTIONS[peak]:.1f}%", xy=(DATES[peak], FRACTIONS[peak]),
                 xytext=(DATES[peak], 31), ha="center", va="top", fontsize=FS['annotation'],
                 color=BLUE, arrowprops={"arrowstyle": "-", "color": BLUE, "lw": 0.65},
                 bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.3})

# c: selection, explicit symbol definitions and one event credited once.
panel("c", "Event-level evaluation", 96, 90.0)
page.plot(100.0, 94.5, marker='o', ms=4.3, markerfacecolor=BLUE,
          markeredgecolor=BLUE, markeredgewidth=0.7, linestyle='none')
text(104.5, 94.5, 'Selected', fs=FS['legend'])
page.plot(132.0, 94.5, marker='o', ms=4.3, markerfacecolor='white',
          markeredgecolor=BLUE, markeredgewidth=0.9, linestyle='none')
text(137.0, 94.5, 'Not selected', fs=FS['legend'])

TLEFT, TRIGHT, TY = 100.5, 175, 116.0
def lead_x(lead):
    return TRIGHT - (TRIGHT - TLEFT) * lead / 60
page.add_patch(Rectangle((lead_x(60), 111.0), lead_x(31) - lead_x(60), 7.5,
                         facecolor="#E3ECF8", edgecolor="none"))
text((lead_x(60) + lead_x(31)) / 2, 112.9, "31–60-day window", fs=FS['annotation'], ha="center", color=BLUE)
p_arrow((TLEFT - 1, TY), (178, TY), colour=GREY, lw=.8)
for lead, selected in [(59, True), (45, True), (31, False)]:
    px = lead_x(lead)
    page.plot(px, TY, marker="o", ms=6.2, markerfacecolor=BLUE if selected else "white",
              markeredgecolor=BLUE, markeredgewidth=.8, linestyle="none", zorder=40)
    p_line([(px, TY + 1.6), (px, 119.2)], colour="#7E96B9", lw=0.75)
p_line([(lead_x(59), 110.0), (lead_x(59), 108.4), (lead_x(45), 108.4), (lead_x(45), 110.0)],
       colour=BLUE, lw=.75)
for lead in [59, 45]:
    p_line([(lead_x(lead), 110.0), (lead_x(lead), TY - 2.1)], colour=BLUE, lw=0.65,
           ls=(0, (2, 2)))
text(110.8, 104.7, "1 event", fs=FS['body'], ha="center", color=BLUE)
p_line([(TRIGHT, 110.8), (TRIGHT, 119.2)], colour=RED, lw=.8)
page.plot(TRIGHT, TY, marker="D", ms=6.5, markerfacecolor=RED,
          markeredgecolor="white", markeredgewidth=0.6, linestyle="none", zorder=40)
text(176.0, 103.3, "First high", fs=FS['annotation'], ha="right")
text(176.0, 107.7, "Chl-a ≥75 µg/L", fs=FS['annotation'], ha="right", color=RED)
for lead in [59, 45, 31, 0]:
    px = lead_x(lead)
    p_line([(px, 119.3), (px, 120.3)], colour=GREY, lw=0.75)
    text(px, 122.6, str(lead), fs=FS['tick'], ha="center")
text((TLEFT + TRIGHT) / 2, 128.5, "Lead time (d)", fs=FS['axis_label'], ha="center")

fig.canvas.draw()
from matplotlib.text import Text
visible_text = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
assert all(STYLE['font']['minimum_pt'] <= t.get_fontsize() <= STYLE['font']['maximum_pt'] for t in visible_text)
for check in scale_checks:
    axis = check.pop('axis')
    xy = np.array(check['projected_endpoints'])
    displayed = axis.transData.transform(xy)
    check['rendered_length_mm'] = float(np.linalg.norm(displayed[1] - displayed[0]) / fig.dpi * 25.4)
    check['rendered_length_px_at_600dpi'] = check['rendered_length_mm'] / 25.4 * 600
    check['background_fill'] = False
EXPORT = STYLE['exports']
export_figure(fig, ROOT, 1, (OUTPUT/"Figure1_caption.txt").read_text(encoding="utf-8"),
    {"all_2844_lakes_and_354_event_lakes_verified": True, "all_78_workload_rounds_verified": True,
     "all_3_scale_bars_independently_geodesic_calibrated": True, "scale_bars": scale_checks})
plt.close(fig)
print(json.dumps({"main_lakes": MAIN_COUNT, "insets": inset_info, "scale_bars": scale_checks}, indent=2))
