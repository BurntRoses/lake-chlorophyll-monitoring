"""Configuration-by-window effects, paired intervals and definition robustness."""
from pathlib import Path
from publication import export_publication
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.transforms import Bbox

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'figures'
DATA = ROOT / 'data/figures/FigureS7'
WIDTH, HEIGHT = 183, 155
BLUE, GREEN, RED, GOLD, GREY = '#1B3B8B', '#006D2C', '#B2182B', '#A6761D', '#606060'
CONFIGS = ['single', 'three_long_primary_only', 'three_time_primary_only',
           'three_long_joint', 'three_time_joint']
NAMES = ['Single [3, 0]', 'Same · primary [3, 0]', 'Multi · primary [3, 0]',
         'Same · joint [4, .5]', 'Multi · joint [3, .1]']
ENDPOINTS = ['1_14', '15_30', '1_30', '31_60', '1_60']
METHODS = ['xgboost', 'elastic_net', 'single_window', 'multi_window']
DEFINITIONS = ['absolute25', 'absolute50', 'absolute75', 'absolute100',
               'absolute75_gap30', 'absolute75_gap90']


def axis(fig, x, y, w, h):
    ax = fig.add_axes([x / WIDTH, 1 - (y + h) / HEIGHT, w / WIDTH, h / HEIGHT])
    ax.tick_params(which='both', top=True, right=True, pad=2)
    return ax


def text(fig, x, y, value, **kwargs):
    return fig.text(x / WIDTH, 1 - y / HEIGHT, value, va='center', **kwargs)


def export(fig, checks):
    export_publication(fig, OUT, 'FigureS7')


def main():
    font_manager.findfont('Times New Roman', fallback_to_default=False)
    plt.rcParams.update({
        'font.family': 'Times New Roman', 'font.size': 10.5,
        'axes.labelsize': 11, 'xtick.labelsize': 10.5, 'ytick.labelsize': 10.5,
        'legend.fontsize': 10.5, 'legend.frameon': False,
        'axes.linewidth': .65, 'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.major.width': .6, 'ytick.major.width': .6,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'svg.fonttype': 'none', 'pdf.fonttype': 42, 'mathtext.fontset': 'stix',
        'savefig.facecolor': 'white', 'lines.solid_capstyle': 'butt'})
    fig = plt.figure(figsize=(WIDTH / 25.4, HEIGHT / 25.4), dpi=160, facecolor='white')
    captures = pd.read_csv(DATA / 'configuration_effects.csv')
    effects = pd.read_csv(DATA / 'paired_effects.csv')
    robust = pd.read_csv(DATA / 'event_definition_robustness.csv')
    checks = json.loads((ROOT / 'docs/figure_statistics.json').read_text(encoding='utf-8'))['FigureS7']
    assert len(captures) == 25 and len(effects) == 6 and len(robust) == 24

    # a: Cell numbers are absolute coverage; a common colour scale encodes effects.
    text(fig, 5, 7, '(a)', fontsize=13, fontweight='bold')
    text(fig, 120, 6, 'Evaluation window (d)', ha='center', fontsize=11)
    text(fig, 62, 20, r'Configuration [depth, $\alpha$]', ha='right')
    a = axis(fig, 65, 25, 110, 30)
    raw = captures.pivot(index='method', columns='endpoint', values='capture_percent').loc[CONFIGS, ENDPOINTS]
    delta = captures.pivot(index='method', columns='endpoint', values='difference_pp').loc[CONFIGS, ENDPOINTS]
    assert delta.to_numpy().min() >= -4 and delta.to_numpy().max() <= 4
    signed = LinearSegmentedColormap.from_list('configuration_effect', [RED, 'white', GREEN])
    aim = a.imshow(delta, cmap=signed, norm=Normalize(-4, 4), aspect='auto',
                   interpolation='nearest', extent=(-.5, 4.5, 4.5, -.5))
    for i in range(5):
        for j in range(5):
            a.text(j, i, f'{raw.iloc[i, j]:.1f}', ha='center', va='center',
                   color='white' if abs(delta.iloc[i, j]) > 2.3 else 'black')
    a.set(xticks=range(5), xticklabels=['1–14', '15–30', '1–30', '31–60', '1–60'],
          yticks=range(5), yticklabels=NAMES)
    a.xaxis.tick_top()
    a.tick_params(axis='both', length=0, pad=5)
    for item, color in zip(a.get_yticklabels(), [BLUE, 'black', 'black', GREY, GREEN]):
        item.set_color(color)
    a.add_patch(Rectangle((2.51, -.49), .98, 4.98, fill=False, ec=GOLD, lw=1.1))
    for line in np.arange(.5, 4.5, 1):
        a.axhline(line, color='#D9D9D9', lw=.4)
        a.axvline(line, color='#D9D9D9', lw=.4)
    for j, endpoint in enumerate(ENDPOINTS):
        n = int(captures.query('endpoint == @endpoint').events.iloc[0])
        text(fig, 65 + (j + .5) * 22, 15, f'n = {n}', ha='center', color=GREY)
    acb = fig.colorbar(aim, cax=axis(fig, 85, 64, 70, 2.5), orientation='horizontal', ticks=[-4, -2, 0, 2, 4])
    acb.ax.tick_params(length=2, pad=2)
    acb.set_label('Change from Single (pp)', labelpad=3)

    # b: Observed effects plus exactly reconstructed paired bootstrap intervals.
    text(fig, 5, 81, '(b)', fontsize=13, fontweight='bold')
    b = axis(fig, 39, 94, 53, 25)
    controls = ['single', 'three_long_primary_only', 'three_time_primary_only']
    for endpoint, color, marker, offset in [('31_60', 'black', 'o', .13), ('1_60', GREEN, '^', -.13)]:
        for i, control in enumerate(controls):
            row = effects.query('endpoint == @endpoint and comparator == @control').iloc[0]
            b.errorbar(row.difference_pp, 2 - i + offset,
                       xerr=[[row.difference_pp - row.lower95], [row.upper95 - row.difference_pp]],
                       fmt=marker, color=color, mfc='white' if marker == '^' else color,
                       ms=4.5, lw=.9, capsize=2)
    b.axvline(0, color=GREY, lw=.65, ls=(0, (3, 2)))
    b.set(xlim=(-3.5, 3.5), ylim=(-.5, 2.5), xticks=[-3, 0, 3], yticks=[2, 1, 0],
          yticklabels=['Single', 'Same · primary', 'Multi · primary'], xlabel='Multi-joint gain (pp)')
    b.tick_params(axis='y', length=0)
    handles = [Line2D([], [], color=color, marker=mark, ms=4, lw=.8,
                      mfc='white' if mark == '^' else color, label=label)
               for color, mark, label in [('black', 'o', '31–60 d'), (GREEN, '^', '1–60 d')]]
    fig.legend(handles=handles, loc='center', bbox_to_anchor=(.35, 1 - 86 / HEIGHT),
               ncol=2, handlelength=1, columnspacing=.9, handletextpad=.4)

    # c: Six unique reconstructed definitions, absolute percentages and denominators.
    text(fig, 103, 81, '(c)', fontsize=13, fontweight='bold')
    text(fig, 124, 86, 'Chl-a / gap', ha='right')
    text(fig, 175, 86, 'n', ha='center')
    c = axis(fig, 125, 94, 43, 25)
    absolute = robust.pivot(index='threshold_id', columns='method_id', values='coverage_pct').loc[DEFINITIONS, METHODS]
    assert absolute.to_numpy().min() >= 20 and absolute.to_numpy().max() <= 90
    count_cmap = LinearSegmentedColormap.from_list('coverage', ['#F7F9FC', '#BDC9E0', '#6A84BA', BLUE, '#071C4D'])
    cim = c.imshow(absolute, cmap=count_cmap, vmin=20, vmax=90, aspect='auto',
                   interpolation='nearest', extent=(-.5, 3.5, 5.5, -.5))
    for i in range(6):
        for j in range(4):
            value = absolute.iloc[i, j]
            c.text(j, i, f'{value:.1f}', ha='center', va='center', color='white' if value > 65 else 'black')
    c.add_patch(Rectangle((-.49, 1.51), 3.98, .98, fill=False, ec=GOLD, lw=1.1))
    c.set(xticks=range(4), xticklabels=['XGB', 'EN', 'Single', 'Multi'], yticks=range(6),
          yticklabels=['25 / 60', '50 / 60', '75 / 60', '100 / 60', '75 / 30', '75 / 90'])
    c.tick_params(axis='both', length=0)
    c.tick_params(axis='x', labelrotation=45)
    for i, definition in enumerate(DEFINITIONS):
        n = int(robust.query('threshold_id == @definition').evaluable_events.iloc[0])
        text(fig, 175, 94 + (i + .5) * 25 / 6, f'{n:,}', ha='center')
    ccb = fig.colorbar(cim, cax=axis(fig, 125, 133, 43, 2.5), orientation='horizontal', ticks=[20, 50, 90])
    ccb.ax.tick_params(length=2, pad=2)
    ccb.set_label('Coverage (%)', labelpad=3)
    export(fig, checks)
    plt.close(fig)


if __name__ == '__main__':
    main()
