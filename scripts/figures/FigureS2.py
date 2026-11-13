"""Complete event timing with explicit matrix scales and exact paired changes."""
from pathlib import Path
from publication import export_publication
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.text import Text
from matplotlib.ticker import NullLocator
from matplotlib.transforms import Bbox

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'figures'
DATA = ROOT / 'data/figures/FigureS2'
WIDTH, HEIGHT = 183, 164
BLUE, GREEN, GOLD, GREY = '#1B3B8B', '#006D2C', '#A6761D', '#606060'
METHODS = ['xgboost', 'elastic_net', 'single_window', 'multi_window']
NAMES = {'xgboost': 'XGBoost', 'elastic_net': 'Elastic Net',
         'single_window': 'Single-window', 'multi_window': 'Multi-window'}
COLORS = {'xgboost': '#111111', 'elastic_net': GREY, 'single_window': BLUE, 'multi_window': GREEN}
MARKERS = {'xgboost': 's', 'elastic_net': 'D', 'single_window': 'o', 'multi_window': '^'}
STYLES = {'xgboost': (0, (5, 2.2)), 'elastic_net': (0, (1, 1.6)),
          'single_window': '-', 'multi_window': '-'}
COUNTS = LinearSegmentedColormap.from_list('counts', ['#F7F9FC', '#BDC9E0', '#6A84BA', BLUE, '#071C4D'])


def axis(fig, x, y, w, h):
    ax = fig.add_axes([x / WIDTH, 1 - (y + h) / HEIGHT, w / WIDTH, h / HEIGHT])
    ax.tick_params(which='both', top=True, right=True, pad=2)
    return ax


def text(fig, x, y, value, **kwargs):
    return fig.text(x / WIDTH, 1 - y / HEIGHT, value, va='center', **kwargs)


def colourbar(fig, x, norm, ticks, label):
    ax = axis(fig, 126, 75, 49, 2.2)
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=COUNTS),
                       cax=ax, orientation='horizontal', ticks=ticks)
    bar.ax.xaxis.set_minor_locator(NullLocator())
    bar.ax.tick_params(top=False, length=1.5, pad=1.5)
    bar.ax.set_xticklabels([str(value) for value in ticks])
    bar.set_label(label, labelpad=2, fontsize=11)
    return bar


def export(fig, checks):
    export_publication(fig, OUT, 'FigureS2')


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
    checks = json.loads((ROOT / 'docs/figure_statistics.json').read_text(encoding='utf-8'))['FigureS2']
    handles = [Line2D([], [], color=COLORS[method], ls=STYLES[method], marker=MARKERS[method],
                      ms=3.5, mfc='white' if method in ['elastic_net', 'multi_window'] else COLORS[method],
                      mew=.8, lw=1, label=NAMES[method]) for method in METHODS]
    fig.legend(handles=handles, loc='center', bbox_to_anchor=(.5, 1 - 8 / HEIGHT), ncol=4,
               handlelength=1.5, columnspacing=1, handletextpad=.4)

    # a: Keep the fixed broad-window cohort and every daily cumulative estimate.
    text(fig, 5, 17, '(a)', fontsize=13, fontweight='bold')
    a = axis(fig, 24, 26, 68, 35)
    cumulative = pd.read_csv(DATA / 'coverage_accumulation.csv')
    for method in METHODS:
        rows = cumulative[cumulative.method_id.eq(method)].sort_values('lead_days', ascending=False)
        assert len(rows) == 60 and rows.cohort_events.eq(681).all()
        a.plot(rows.lead_days, rows.cumulative_coverage_pct, drawstyle='steps-post',
               color=COLORS[method], ls=STYLES[method], lw=1.05,
               marker=MARKERS[method], markevery=[2, 15, 30, 45, 57], ms=3.5,
               mfc='white' if method in ['elastic_net', 'multi_window'] else COLORS[method], mew=.8)
    a.axvline(30, color=GOLD, ls='--', lw=.65)
    a.set(xlim=(60, 1), ylim=(0, 100), xticks=[60, 45, 30, 15, 1], yticks=[0, 25, 50, 75, 100],
          xlabel='Lead time (d)', ylabel='Cumulative coverage (%)')
    a.tick_params(axis='y', pad=5)
    a.text(58, 94, 'n = 681', va='top')

    # b: Retain annual primary-window support with an explicit percentage scale.
    text(fig, 99, 17, '(b)', fontsize=13, fontweight='bold')
    text(fig, 150.5, 14, '31–60 d', ha='center')
    b = axis(fig, 126, 26, 49, 35)
    annual = pd.read_csv(DATA / 'annual_coverage.csv')
    matrix = annual.pivot(index='method_id', columns='event_year', values='coverage_pct').loc[METHODS]
    b.imshow(matrix, cmap=COUNTS, vmin=40, vmax=90, aspect='auto', interpolation='nearest')
    for i, method in enumerate(METHODS):
        for j, year in enumerate(matrix.columns):
            value = matrix.loc[method, year]
            b.text(j, i, f'{value:.1f}', ha='center', va='center', color='white' if value > 68 else 'black')
    b.set(xticks=np.arange(5), xticklabels=[str(year) for year in matrix.columns],
          yticks=np.arange(4), yticklabels=[NAMES[method] for method in METHODS])
    b.tick_params(length=0, axis='both')
    b.tick_params(axis='x', labelrotation=45)
    denominators = annual.groupby('event_year').evaluable_events.first()
    for j, n in enumerate(denominators):
        b.text(j, -.83, str(n), ha='center', va='center', clip_on=False)
    text(fig, 122, 22.5, 'n', ha='right')
    colourbar(fig, 33, Normalize(40, 90), [40, 65, 90], 'Annual coverage (%)')

    # c: Keep the full paired matrix, including coverage gains and losses.
    text(fig, 5, 94, '(c)', fontsize=13, fontweight='bold')
    fig.legend(handles=[Patch(fc='none', ec=GREEN, lw=.8, label='Earlier / gained'),
                        Patch(fc='none', ec=GOLD, lw=.8, label='Later / lost')],
               loc='center', bbox_to_anchor=(61 / WIDTH, 1 - 95 / HEIGHT),
               handlelength=1.1, handletextpad=.5, labelspacing=.1)
    c = axis(fig, 29, 102, 57, 36)
    transitions = pd.read_csv(DATA / 'earliest_selection_transition.csv')
    categories = ['31–60 d', '15–30 d', '1–14 d', 'Uncovered']
    counts = transitions.pivot(index='multi_category', columns='single_category', values='count').loc[categories, categories]
    assert counts.to_numpy().sum() == 681 and counts.size == 16
    c.imshow(counts, cmap=COUNTS, norm=LogNorm(1, 500), aspect='auto', interpolation='nearest')
    for i in range(4):
        for j in range(4):
            value = int(counts.iloc[i, j])
            c.text(j, i, str(value), ha='center', va='center', color='white' if value >= 70 else 'black')
            if i != j and value:
                c.add_patch(Rectangle((j - .48, i - .48), .96, .96, fill=False,
                                       ec=GREEN if i < j else GOLD, lw=.7))
    c.set(xticks=np.arange(4), yticks=np.arange(4),
          xticklabels=['31–60', '15–30', '1–14', 'None'],
          yticklabels=['31–60', '15–30', '1–14', 'None'],
          xlabel='Single-window lead (d)', ylabel='Multi-window lead (d)')
    c.tick_params(length=0)
    c.tick_params(axis='x', labelrotation=45)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=LogNorm(1, 500), cmap=COUNTS),
                       cax=axis(fig, 95, 102, 2.2, 36), ticks=[1, 10, 100, 500])
    cb.ax.yaxis.set_minor_locator(NullLocator())
    cb.ax.set_yticklabels(['1', '10', '100', '500'])
    cb.ax.tick_params(length=1.5, pad=1.5)
    cb.set_label('Events (log)', labelpad=2, fontsize=11)

    # d: State zero-day pairs separately; retain every exact nonzero lead bin.
    text(fig, 117, 94, '(d)', fontsize=13, fontweight='bold')
    text(fig, 150, 100, '582 paired; 514 at 0 d', ha='center')
    d = axis(fig, 125, 107, 50, 31)
    differences = pd.read_csv(DATA / 'paired_lead_difference_counts.csv')
    zero = differences[differences.lead_difference_days.eq(0)]
    assert zero.events.item() == 514 and differences.events.sum() == 582
    changes = differences[differences.lead_difference_days.ne(0)]
    assert len(changes) == 6 and changes.events.sum() == 68
    d.bar(changes.lead_difference_days, changes.events, width=2,
          color=[GREEN if value > 0 else GOLD for value in changes.lead_difference_days], lw=0)
    for row in changes.itertuples(index=False):
        d.text(row.lead_difference_days, row.events + .8, str(row.events), ha='center', va='bottom')
    d.axvline(0, color=GREY, lw=.65, ls=':')
    d.set(xlim=(-60, 60), ylim=(0, 45), xticks=[-42, 0, 42], yticks=[0, 10, 20, 30, 40],
          xlabel='Earliest-lead difference (d)\nMulti-window − Single-window', ylabel='Events')
    d.tick_params(axis='y', pad=5)
    d.get_xticklabels()[0].set_ha('left')
    d.text(-55, 42, 'Later', color=GOLD, ha='left', va='top')
    d.text(55, 42, 'Earlier', color=GREEN, ha='right', va='top')

    export(fig, checks)
    plt.close(fig)


if __name__ == '__main__':
    main()
