"""Shared typography and export settings for scientific figures."""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import re
import sys
import xml.etree.ElementTree as ET
import matplotlib as mpl
from matplotlib.text import Text
from matplotlib.legend import Legend
from matplotlib import font_manager
from matplotlib.transforms import Bbox
from PIL import Image

FONT = 'Times New Roman'
NAVY = '#1B3B8B'
TEAL = '#006D2C'
GOLD = '#A6761D'
COPPER = '#B2182B'
MINIMUM_PT = 10.5
AXIS_LABEL_PT = 11.0
PANEL_PT = 13.0


def apply():
    mpl.rcParams.update({'font.family': FONT, 'font.serif': [FONT, 'DejaVu Serif'],
        'font.size': 11, 'axes.labelsize': AXIS_LABEL_PT, 'xtick.labelsize': MINIMUM_PT,
        'ytick.labelsize': MINIMUM_PT, 'legend.fontsize': MINIMUM_PT, 'legend.frameon': False,
        'axes.linewidth': .65, 'axes.edgecolor': 'black', 'axes.labelcolor': 'black',
        'text.color': 'black', 'xtick.color': 'black', 'ytick.color': 'black',
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.major.width': .6, 'ytick.major.width': .6,
        'xtick.minor.width': .45, 'ytick.minor.width': .45,
        'xtick.major.size': 3.0, 'ytick.major.size': 3.0,
        'xtick.minor.size': 1.6, 'ytick.minor.size': 1.6,
        'svg.fonttype': 'none', 'pdf.fonttype': 42, 'mathtext.fontset': 'stix',
        'savefig.facecolor': 'white', 'lines.solid_capstyle': 'butt'})


def normalize(fig):
    axis_labels = {a for ax in fig.axes for a in (ax.xaxis.label, ax.yaxis.label)}
    for t in fig.findobj(Text):
        if not t.get_text():
            continue
        t.set_fontfamily(FONT)
        if re.fullmatch(r'\([a-f]\)', t.get_text()):
            t.set_fontsize(PANEL_PT)
        elif t in axis_labels:
            t.set_fontsize(AXIS_LABEL_PT)
        else:
            t.set_fontsize(max(MINIMUM_PT, t.get_fontsize()))
        t.set_fontweight('bold' if re.fullmatch(r'\([a-f]\)', t.get_text()) else 'normal')
        # Keep labels outside the axes clipping path.
        t.set_clip_on(False)
    for legend in fig.findobj(Legend):
        legend.set_frame_on(False)
    for ax in fig.axes:
        for spine in ax.spines.values():
            if spine.get_visible():
                spine.set_edgecolor('black')
                spine.set_linewidth(.5 if hasattr(ax, '_colorbar') else .65)
        for axis in (ax.xaxis, ax.yaxis):
            for minor, ticks in [(False, axis.get_major_ticks()), (True, axis.get_minor_ticks())]:
                for tick in ticks:
                    for mark in (tick.tick1line, tick.tick2line):
                        mark.set_markeredgewidth(.45 if minor else .6)
                        mark.set_markersize(min(mark.get_markersize(), 1.6 if minor else 3.0))


def audit_text(fig, renderer):
    """Measure glyphs at export resolution; annotations exclude leader extents."""
    scale = 25.4 / fig.dpi
    rows, artists = [], []
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        b = Text.get_window_extent(t, renderer)
        if not b.width or not b.height:
            continue
        margins = [b.x0*scale, (fig.bbox.height-b.y1)*scale,
                   (fig.bbox.width-b.x1)*scale, b.y0*scale]
        rows.append({'text': t.get_text(), 'font_pt': t.get_fontsize(),
            'bbox_mm': [round(b.x0*scale, 4), round((fig.bbox.height-b.y1)*scale, 4),
                        round(b.width*scale, 4), round(b.height*scale, 4)],
            'canvas_clearance_mm': round(min(margins), 4),
            'axes_index': fig.axes.index(t.axes) if t.axes in fig.axes else None,
            'clip_on': t.get_clip_on()})
        artists.append((t,b))
    overlaps = []
    for (i,(ta,ba)), (j,(tb,bb)) in itertools.combinations(enumerate(artists),2):
        cut = Bbox.intersection(ba,bb)
        if cut is not None and cut.width*scale > .25 and cut.height*scale > .25:
            overlaps.append({'texts':[ta.get_text(),tb.get_text()],
                'overlap_mm':[round(cut.width*scale,3),round(cut.height*scale,3)]})
    occlusion = []
    for t,b in artists:
        if t.axes not in fig.axes:
            continue
        for ax in fig.axes:
            if (ax is t.axes or not ax.axison or not ax.get_frame_on()
                    or not ax.patch.get_visible() or ax.patch.get_alpha()==0
                    or ax.patch.get_facecolor()[3]==0):
                continue
            later = ax.get_zorder()>t.axes.get_zorder() or (ax.get_zorder()==t.axes.get_zorder() and fig.axes.index(ax)>fig.axes.index(t.axes))
            cut = Bbox.intersection(b,ax.bbox)
            if later and cut is not None and cut.width*scale>.25 and cut.height*scale>.25:
                occlusion.append({'text':t.get_text(),'occluding_axes':fig.axes.index(ax)})
    return {'measurement_dpi':fig.dpi,'texts':rows,'text_overlap_candidates':overlaps,
            'later_axes_occlusion_candidates':occlusion,
            'minimum_canvas_clearance_mm':min(r['canvas_clearance_mm'] for r in rows)}


def export_figure(fig, root, number, caption, checks, hashes=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--qa-dir', type=Path)
    args = parser.parse_args()
    # Measure text at the PNG export resolution.
    fig.set_dpi(600)
    fig.canvas.draw()
    normalize(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    audit = audit_text(fig, renderer)
    diagnostics = Path(args.qa_dir) if args.qa_dir else root.parent/'diagnostics'
    diagnostics.mkdir(parents=True, exist_ok=True)
    outside, small = [], []
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text():
            continue
        b = Text.get_window_extent(t, renderer)
        if not b.width or not b.height:
            continue
        if b.x0 < -.5 or b.y0 < -.5 or b.x1 > fig.bbox.width+.5 or b.y1 > fig.bbox.height+.5:
            outside.append(t.get_text())
        if t.get_fontsize() < MINIMUM_PT:
            small.append(t.get_text())
    assert not outside, outside
    assert not small, small
    assert all(not legend.get_frame_on() for legend in fig.findobj(Legend))
    (diagnostics/'text_layout_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    assert audit['minimum_canvas_clearance_mm'] >= 3.0, audit['minimum_canvas_clearance_mm']
    from connector_geometry import audit_connectors
    connector_report = audit_connectors(fig,diagnostics)
    out = root.parent
    stem = f'Figure{number}'
    fig.savefig(out / (stem + '.png'), dpi=600)
    fig.savefig(out / (stem + '.svg'), dpi=600, metadata={'Date': None, 'Creator': 'Python/matplotlib'})
    fig.savefig(out / (stem + '.pdf'), metadata={'Title':stem,'Creator':'Python Matplotlib'})
    with Image.open(out / (stem + '.png')) as original_png:
        original_png.convert('RGB').save(out / (stem + '.tif'),compression='tiff_lzw',dpi=(600,600))
    caption_path=out / (stem + '_caption.txt')
    if not caption_path.exists():caption_path.write_text(caption.strip()+'\n',encoding='utf-8')
    if args.qa_dir:
        args.qa_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.qa_dir / 'preview.png', dpi=210)
        import resvg_py
        (args.qa_dir / 'svg_preview.png').write_bytes(resvg_py.svg_to_bytes(
            svg_path=str(out / (stem + '.svg')), width=1700, dpi=96))
    svg = ET.parse(out / (stem + '.svg')).getroot()
    ntext = len(svg.findall('.//{http://www.w3.org/2000/svg}text'))
    clipped = []
    def inspect_svg(node, inherited_clip=False):
        clipped_here = inherited_clip or 'clip-path' in node.attrib
        if node.tag.endswith('}text') and clipped_here:
            clipped.append(''.join(node.itertext()))
        for child in node:
            inspect_svg(child, clipped_here)
    inspect_svg(svg)
    assert not clipped, clipped
    with Image.open(out / (stem + '.png')) as im:
        pixels = list(im.size)
        dpi = list(im.info['dpi'])
    report = {'figure': number, 'status': 'Scientific figure export',
        'backend': 'Python/matplotlib', 'canvas_mm': [round(float(x)*25.4, 4) for x in fig.get_size_inches()],
        'font': font_manager.FontProperties(fname=font_manager.findfont(FONT, fallback_to_default=False)).get_name(),
        'minimum_pt': MINIMUM_PT, 'axis_label_pt': AXIS_LABEL_PT, 'panel_pt': PANEL_PT,
        'all_legends_unframed': True,
        'text_outside_canvas': outside, 'text_below_minimum_pt': small,
        'minimum_text_canvas_clearance_mm':audit['minimum_canvas_clearance_mm'],
        'text_measurement_dpi':600,'svg_text_with_clip_paths':clipped,
        'text_overlap_candidates':audit['text_overlap_candidates'],
        'later_axes_occlusion_candidates':audit['later_axes_occlusion_candidates'],
        'png_pixels': pixels, 'png_dpi': dpi, 'svg_editable_text_elements': ntext,
        'data_validation': checks, 'source_sha256': hashes or {}, 'connector_count':connector_report['count'],
        'axes_positions_mm': [[round(float(x)*183, 3), round((1-float(y)-float(h))*fig.get_size_inches()[1]*25.4, 3),
            round(float(w)*183, 3), round(float(h)*fig.get_size_inches()[1]*25.4, 3)]
            for ax in fig.axes for x, y, w, h in [ax.get_position().bounds]],
        'software': {'python': sys.version.split()[0], 'matplotlib': mpl.__version__}}
    (diagnostics / 'validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'figure': number, 'pixels': pixels, 'editable_text': ntext,
        'font': report['font'], 'unframed_legends': True, 'checks': checks}, ensure_ascii=False))
