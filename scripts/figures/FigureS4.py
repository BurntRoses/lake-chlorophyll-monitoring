"""Paired allocation, cumulative reach, full interval CDFs and concentration."""
from pathlib import Path
from publication import export_publication
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.transforms import Bbox

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'figures'
DATA = ROOT / 'data/figures/FigureS4'
WIDTH, HEIGHT = 183, 198
BLUE, GREEN, GOLD, GREY = '#1B3B8B', '#006D2C', '#A6761D', '#606060'
METHODS = ['xgboost', 'elastic_net', 'single_window', 'multi_window']
NAMES = dict(xgboost='XGBoost', elastic_net='Elastic Net', single_window='Single-window', multi_window='Multi-window')
COLORS = dict(xgboost='#111111', elastic_net=GREY, single_window=BLUE, multi_window=GREEN)
MARKERS = dict(xgboost='s', elastic_net='D', single_window='o', multi_window='^')
STYLES = dict(xgboost=(0, (5, 2.2)), elastic_net=(0, (1, 1.6)), single_window='-', multi_window='-')
COUNTS = LinearSegmentedColormap.from_list('counts', ['#F7F9FC', '#BDC9E0', '#6A84BA', BLUE, '#071C4D'])


def axis(fig, x, y, w, h):
    ax = fig.add_axes([x / WIDTH, 1 - (y + h) / HEIGHT, w / WIDTH, h / HEIGHT])
    ax.tick_params(which='both', top=True, right=True, pad=2)
    return ax


def text(fig, x, y, value, **kwargs):
    return fig.text(x / WIDTH, 1 - y / HEIGHT, value, va='center', **kwargs)


def symbol(method):
    return dict(marker=MARKERS[method], color=COLORS[method], ms=3.4,
                mfc='white' if method in ['elastic_net', 'multi_window'] else COLORS[method], mew=.7)


def export(fig, checks):
    export_publication(fig, OUT, 'FigureS4')


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
    checks = json.loads((ROOT / 'docs/figure_statistics.json').read_text(encoding='utf-8'))['FigureS4']
    fig.legend(handles=[Line2D([], [], lw=0, ls='none', label=NAMES[m], **symbol(m)) for m in METHODS],
               loc='center', bbox_to_anchor=(.51, 1 - 7 / HEIGHT), ncol=4,
               handlelength=1.45, columnspacing=.95, handletextpad=.4)

    # a: Preserve both 5x5 matrices; a central colourbar reserves vertical space.
    text(fig, 5, 18, '(a)', fontsize=13, fontweight='bold')
    transitions = pd.read_csv(DATA / 'paired_visit_transitions.csv')
    bands = ['0', '1–13', '14–26', '27–65', '66–130']
    for method, x in [('single_window', 28), ('multi_window', 117)]:
        a = axis(fig, x, 29, 57, 34)
        rows = transitions.query('method == @method')
        mat = rows.pivot(index='history_bin', columns='original_bin', values='lakes').reindex(index=range(5), columns=range(5)).to_numpy()
        assert mat.sum() == 2844
        a.imshow(np.ma.masked_where(mat == 0, mat), origin='lower', cmap=COUNTS,
                 norm=LogNorm(1, 1600), aspect='auto', interpolation='nearest')
        for i in range(5):
            for j in range(5):
                if mat[i, j]:
                    a.text(j, i, f'{mat[i, j]:,}', ha='center', va='center',
                           color='white' if mat[i, j] > 65 else 'black')
        a.add_patch(Rectangle((-.48, .52), .96, 3.96, fill=False, ec=GOLD, lw=.8))
        a.set(xticks=range(5), yticks=range(5), xticklabels=bands,
              yticklabels=bands if method == 'single_window' else [], xlabel='Score-only selections',
              ylabel='History-aware selections' if method == 'single_window' else '')
        a.tick_params(length=0)
        a.tick_params(axis='x', labelrotation=45)
        text(fig, x + 28.5, 23, NAMES[method], ha='center', color=COLORS[method])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=LogNorm(1, 1600), cmap=COUNTS),
                      cax=axis(fig, 96, 29, 2, 34), ticks=[1, 10, 100, 1000])
    cb.ax.set_yticklabels(['1', '10', '100', '1000'])
    cb.ax.tick_params(length=2, pad=2)
    cb.set_label('Lakes', labelpad=3)

    # b: Eight complete trajectories share axes and a common schedule key.
    text(fig, 5, 85, '(b)', fontsize=13, fontweight='bold')
    reach = pd.read_csv(DATA / 'cumulative_lake_reach.csv')
    b = axis(fig, 24, 94, 151, 28)
    for method in METHODS:
        for schedule in ['original', 'rotation']:
            rows = reach.query('method == @method and schedule == @schedule').sort_values('round_index')
            b.plot(rows.round_index, rows.cumulative_selected_lakes, color=COLORS[method],
                   ls='-' if schedule == 'rotation' else (0, (3, 2)), lw=1.05)
            marks = rows[rows.round_index.isin([1, 26, 52, 78, 104, 130])]
            b.plot(marks.round_index, marks.cumulative_selected_lakes, ls='none', **symbol(method))
    b.set(xlim=(1, 132), ylim=(0, 1530), xticks=[1, 26, 52, 78, 104, 130],
          yticks=[0, 500, 1000, 1500], xlabel='Decision round', ylabel='Distinct lakes selected')
    schedule_handles = [Line2D([], [], color=GREY, lw=1, ls=(0, (3, 2)), label='Score-only'),
                        Line2D([], [], color=GREY, lw=1, label='History-aware')]
    fig.legend(handles=schedule_handles, loc='center', bbox_to_anchor=(.61, 1 - 85 / HEIGHT),
               ncol=2, handlelength=2, columnspacing=1.5, handletextpad=.5)

    # c: Same metric and full support permit comparison of both schedules in one axis.
    text(fig, 5, 144, '(c)', fontsize=13, fontweight='bold')
    intervals = pd.read_csv(DATA / 'selection_intervals.csv')
    c = axis(fig, 24, 154, 62, 27)
    for schedule in ['original', 'rotation']:
        for method in METHODS:
            rows = intervals.query('method == @method and schedule == @schedule').sort_values('interval_days')
            assert rows.interval_days.min() >= 14 and rows.interval_days.max() <= 1610
            assert np.isclose(rows.ecdf_pct.iloc[-1], 100)
            c.step(np.r_[10, rows.interval_days, 1700], np.r_[0, rows.ecdf_pct, 100], where='post',
                   color=COLORS[method], ls='-' if schedule == 'rotation' else (0, (3, 2)), lw=1.05)
            marks = rows[rows.interval_days.isin([14, 28, 84]) | rows.interval_days.eq(rows.interval_days.max())]
            c.plot(marks.interval_days, marks.ecdf_pct, ls='none', **symbol(method))
    c.axvline(30, color=GOLD, lw=.7, ls=':')
    c.set(xscale='log', xlim=(10, 1700), ylim=(0, 103), xticks=[14, 84, 365, 1610],
          yticks=[0, 50, 100], xlabel='Reselection interval (d)', ylabel='Cumulative intervals (%)')
    c.set_xticklabels(['14', '84', '365', '1610'])
    c.text(34, 12, '30 d', color=GOLD)

    # d: All eight concentration curves share one percentage coordinate system.
    text(fig, 99, 144, '(d)', fontsize=13, fontweight='bold')
    concentration = pd.read_csv(DATA / 'slot_concentration_curves.csv')
    d = axis(fig, 115, 154, 60, 27)
    d.plot([0, 100], [0, 100], color='#B7B7B7', lw=.6)
    for schedule in ['original', 'rotation']:
        for method in METHODS:
            rows = concentration.query('method == @method and schedule == @schedule').sort_values('lake_rank')
            assert len(rows) == 2845
            xx, yy = 100 * rows.lake_fraction, 100 * rows.slot_fraction
            d.plot(xx, yy, color=COLORS[method], ls='-' if schedule == 'rotation' else (0, (3, 2)), lw=1.05)
            indices = [int(np.argmin(np.abs(xx.to_numpy() - value))) for value in [70, 85, 95]]
            d.plot(xx.iloc[indices], yy.iloc[indices], ls='none', **symbol(method))
    d.set(xlim=(0, 100), ylim=(0, 100), xticks=[0, 50, 100], yticks=[0, 50, 100],
          xlabel='Candidate lakes (%)', ylabel='Monitoring slots (%)')
    # Verify the actual plotted artists against all stored method/schedule paths.
    # This guards against dropping a trajectory while merging coordinate systems.
    geometry = []
    for panel, ax, curves in [('b', b, b.lines[::2]),
                               ('c', c, c.lines[:-1:2]),
                               ('d', d, d.lines[1::2])]:
        assert len(curves) == 8, panel
        order = ([(m, s) for m in METHODS for s in ['original', 'rotation']]
                 if panel == 'b' else [(m, s) for s in ['original', 'rotation'] for m in METHODS])
        for line, (method, schedule) in zip(curves, order):
            if panel == 'b':
                rows = reach.query('method == @method and schedule == @schedule').sort_values('round_index')
                xx, yy = rows.round_index, rows.cumulative_selected_lakes
            elif panel == 'c':
                rows = intervals.query('method == @method and schedule == @schedule').sort_values('interval_days')
                xx, yy = np.r_[10, rows.interval_days, 1700], np.r_[0, rows.ecdf_pct, 100]
            else:
                rows = concentration.query('method == @method and schedule == @schedule').sort_values('lake_rank')
                xx, yy = 100 * rows.lake_fraction, 100 * rows.slot_fraction
            assert np.array_equal(line.get_xdata(), xx) and np.array_equal(line.get_ydata(), yy)
            assert line.get_color() == COLORS[method]
            assert line.get_linestyle() == ('-' if schedule == 'rotation' else '--')
            geometry.append(dict(panel=panel, method=method, schedule=schedule, points=len(xx)))
    export(fig, checks)
    plt.close(fig)


if __name__ == '__main__':
    main()
