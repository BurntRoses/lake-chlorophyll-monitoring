"""Draw a stage-replay figure from complete, paired event-level diagnostics."""
from pathlib import Path
from publication import export_publication
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.transforms import Bbox

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'figures'
DATA = ROOT / 'data/figures/FigureS3'
BLUE, GREEN, GOLD, GREY = '#1B3B8B', '#006D2C', '#A6761D', '#606060'
METHODS = ['single_window', 'multi_window']
COLORS = dict(single_window=BLUE, multi_window=GREEN)
LABELS = dict(single_window='Single-window', multi_window='Multi-window')
WIDTH, HEIGHT = 183, 243


def axis(fig, x, y, w, h):
    ax = fig.add_axes([x / WIDTH, 1 - (y + h) / HEIGHT, w / WIDTH, h / HEIGHT])
    ax.tick_params(which='both', top=True, right=True, pad=2)
    return ax


def text(fig, x, y, value, **kwargs):
    return fig.text(x / WIDTH, 1 - y / HEIGHT, value, va='center', **kwargs)


def marker(method):
    return dict(marker='o' if method == 'single_window' else '^',
                color=COLORS[method], ms=4,
                mfc=COLORS[method] if method == 'single_window' else 'white', mew=.8)


def export(fig, data_checks, color_limit):
    export_publication(fig, OUT, 'FigureS3')


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
    coverage = pd.read_csv(DATA / 'stage_coverage.csv')
    turnover = pd.read_csv(DATA / 'coverage_turnover.csv')
    priorities = np.load(DATA / 'event_priorities.npz')
    data_checks = json.loads((ROOT / 'docs/figure_statistics.json').read_text(encoding='utf-8'))['FigureS3']
    assert len(coverage) == len(turnover) == 244
    legend_handles = [Line2D([], [], lw=1.1, label=LABELS[m], **marker(m)) for m in METHODS]
    fig.legend(handles=legend_handles, loc='center', bbox_to_anchor=(.52, 1 - 7 / HEIGHT),
               ncol=2, handlelength=1.7, columnspacing=1.8, handletextpad=.5)

    # a: Complete stage-level coverage, with the saved 60-tree endpoint retained.
    text(fig, 5, 17, '(a)', fontsize=13, fontweight='bold')
    a = axis(fig, 24, 23, 150, 34)
    for method in METHODS:
        for capacity in [5, 10]:
            rows = coverage.query('method == @method and capacity_pct == @capacity').sort_values('stage')
            a.plot(rows.stage, rows.coverage_pct, color=COLORS[method],
                   ls='-' if capacity == 10 else (0, (4, 2)), lw=1.1)
            a.plot(60, rows.coverage_pct.iloc[-1], ls='none', **marker(method))
    a.set(xlim=(0, 62), ylim=(35, 84), xticks=[0, 10, 20, 30, 40, 50, 60],
          yticks=[40, 60, 80], xlabel='Saved tree prefix', ylabel='31–60 d coverage (%)')
    a.legend(handles=[Line2D([], [], color=GREY, lw=1, label='10% capacity'),
                      Line2D([], [], color=GREY, lw=1, ls=(0, (4, 2)), label='5% capacity')],
             loc='lower right', handlelength=1.7, handletextpad=.5, labelspacing=.2)

    # b: All event rows, matched order, matched complete colour range.
    text(fig, 5, 75, '(b)', fontsize=13, fontweight='bold')
    text(fig, 99, 70, '10% capacity · 593 events', ha='center')
    cmap = LinearSegmentedColormap.from_list('priority_change', [GREEN, '#FFFFFF', GOLD])
    all_changes = [np.log2(priorities[f'{m}_10'] / priorities[f'{m}_10'][:, :1]) for m in METHODS]
    assert all(np.allclose(v[:, 0], 0) for v in all_changes)
    limit = float(np.ceil(max(np.abs(v).max() for v in all_changes)))
    norm = Normalize(vmin=-limit, vmax=limit)
    binary = ListedColormap(['white', '#252525'])
    initial_count = int((priorities['single_window_10'][:, 0] <= 1).sum())
    assert initial_count == 329
    for method, x, change in zip(METHODS, [24, 107], all_changes):
        b = axis(fig, x, 86, 57, 63)
        im = b.imshow(change, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest',
                      extent=(-.5, 60.5, 593.5, .5), rasterized=True)
        b.axhline(initial_count + .5, color='black', ls=(0, (2, 2)), lw=.65)
        b.set(xticks=[0, 20, 40, 60], yticks=[1, 329, 593], xlabel='Saved tree prefix')
        if method == 'single_window':
            b.set_ylabel('Event order')
        else:
            b.set_yticklabels([])
        text(fig, x + 28.5, 77, LABELS[method], ha='center', color=COLORS[method])
        for stage, offset in [(0, 61), (60, 66)]:
            status = (priorities[f'{method}_10'][:, stage] <= 1).astype(float)
            strip = axis(fig, x + offset, 86, 2, 63)
            strip.imshow(status[:, None], cmap=binary, vmin=0, vmax=1, aspect='auto',
                         interpolation='nearest', extent=(0, 1, 593.5, .5), rasterized=True)
            strip.axhline(initial_count + .5, color=GREY, ls=(0, (2, 2)), lw=.6)
            strip.set(xticks=[], yticks=[])
            text(fig, x + offset + 1, 82, str(stage), ha='center')
        text(fig, x + 64.5, 77, 'Covered', ha='center')
    colorbar = fig.colorbar(im, cax=axis(fig, 48, 169, 102, 2.5),
                           orientation='horizontal', ticks=[-8, -4, 0, 4, 8])
    colorbar.ax.tick_params(length=2, pad=2)
    colorbar.set_label(r'$\Delta\,\log_2$(event priority)', labelpad=3)
    text(fig, 31, 170.25, 'Improved', ha='center', color=GREEN)
    text(fig, 165, 170.25, 'Worsened', ha='center', color=GOLD)

    # c: Opposite changes are separately counted relative to the initial coverage set.
    text(fig, 5, 190, '(c)', fontsize=13, fontweight='bold')
    fig.legend(handles=[Line2D([], [], color=GREY, lw=1, label='Newly covered (+)'),
                        Line2D([], [], color=GREY, lw=1, ls=(0, (4, 2)), label='Lost (−)')],
               loc='center', bbox_to_anchor=(.56, 1 - 190 / HEIGHT), ncol=2,
               handlelength=1.7, columnspacing=1.8, handletextpad=.5)
    for capacity, x in [(5, 24), (10, 107)]:
        c = axis(fig, x, 199, 67, 28)
        for method in METHODS:
            rows = turnover.query('method == @method and capacity_pct == @capacity').sort_values('stage')
            c.plot(rows.stage, rows.gained, color=COLORS[method], lw=1.1)
            c.plot(rows.stage, -rows.lost, color=COLORS[method], ls=(0, (4, 2)), lw=1.1)
            c.plot(60, rows.gained.iloc[-1], ls='none', **marker(method))
            c.plot(60, -rows.lost.iloc[-1], ls='none', **marker(method))
        c.axhline(0, color=GREY, lw=.6)
        c.set(xlim=(0, 62), ylim=(-42, 200), xticks=[0, 20, 40, 60],
              yticks=[0, 100, 200], xlabel='Saved tree prefix')
        if capacity == 5:
            c.set_ylabel('Events gained / lost')
        else:
            c.set_yticklabels([])
        c.text(.03, .94, f'{capacity}% capacity', transform=c.transAxes, ha='left', va='top')
    export(fig, data_checks, limit)
    plt.close(fig)


if __name__ == '__main__':
    main()
