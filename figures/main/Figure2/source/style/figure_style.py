"""Shared publication style. All point sizes refer to the final physical canvas."""
from pathlib import Path
import json
import matplotlib as mpl
from matplotlib import font_manager

STYLE = json.loads(Path(__file__).with_suffix('.json').read_text(encoding='utf-8'))
FS = STYLE['font']['sizes_pt']
COLORS = STYLE['palette']
LINES = STYLE['lines_pt']

def apply_style():
    preferred = STYLE['font']['family']
    try:
        font_manager.findfont(preferred, fallback_to_default=False)
        family = preferred
    except ValueError:
        family = STYLE['font']['fallback']
    mpl.rcParams.update({
        'font.family': family,
        'font.size': FS['body'],
        'text.color': COLORS['ink'],
        'axes.labelcolor': COLORS['ink'],
        'axes.edgecolor': COLORS['ink'],
        'axes.linewidth': LINES['axis'],
        'axes.labelsize': FS['axis_label'],
        'xtick.color': COLORS['ink'],
        'ytick.color': COLORS['ink'],
        'xtick.labelsize': FS['tick'],
        'ytick.labelsize': FS['tick'],
        'legend.fontsize': FS['legend'],
        'legend.frameon': False,
        'svg.fonttype': 'none',
        'pdf.fonttype': 42,
        'savefig.facecolor': 'white',
        'lines.solid_capstyle': 'round'
    })
    return family
