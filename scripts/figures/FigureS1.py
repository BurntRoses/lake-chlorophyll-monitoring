"""Information timing, nested time/lake validation and sparse supervision."""
from pathlib import Path
from publication import export_publication
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.transforms import Bbox

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'figures'
DATA = ROOT / 'data/figures/FigureS1'
WIDTH, HEIGHT = 183, 253
BLUE, GREEN, RED, GOLD, GREY = '#1B3B8B', '#006D2C', '#B2182B', '#A6761D', '#606060'
WINDOWS = ['1_14', '15_30', '31_60']
PALETTE = {'Positive': BLUE, 'Negative': '#A8ADB6', 'Unknown': '#F2F2F2'}


def axis(fig, x, y, w, h):
    ax = fig.add_axes([x / WIDTH, 1 - (y + h) / HEIGHT, w / WIDTH, h / HEIGHT])
    ax.tick_params(which='both', top=True, right=True, pad=2)
    return ax


def text(fig, x, y, value, **kwargs):
    return fig.text(x / WIDTH, 1 - y / HEIGHT, value, va='center', **kwargs)


def export(fig, checks):
    export_publication(fig, OUT, 'FigureS1')


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
    checks = json.loads((ROOT / 'docs/figure_statistics.json').read_text(encoding='utf-8'))['FigureS1']
    timing = pd.read_csv(DATA / 'information_timing_examples.csv').set_index('example_id')

    # a: Preserve all five observation/outcome roles and the original seven-day lag.
    text(fig, 5, 8, '(a)', fontsize=13, fontweight='bold')
    a = axis(fig, 30, 20, 85, 38)
    a.axvspan(31, 60, color='#E5EBF5', zorder=0)
    a.axvline(0, color='black', lw=.8)
    a.axvline(7, color=GREY, lw=.45, ls=':')
    roles = [('Positive', 4, 'available'), ('Negative', 3, 'available'),
             ('Unknown', 2, 'available'), ('Pending', 1, 'pending'), ('Active', 0, 'known_active')]
    for label, y, example in roles:
        record = timing.loc[example]
        color = GREY if label == 'Pending' else BLUE
        a.plot([record.sample_day, record.available_day], [y, y], color=color, lw=.85)
        if label != 'Active':
            a.plot(record.sample_day, y, 'o', mfc='white', mec=color, ms=4.5)
        a.plot(record.available_day, y, 's', color=color, ms=4.5)
        if label != 'Pending':
            a.plot(0, y, 'x' if label == 'Active' else '|',
                   color=GREY if label == 'Active' else GREEN, ms=6, mew=1)
    a.plot(42, 4, '^', color=RED, ms=5.5)
    a.plot(47, 3, 'o', mfc='white', mec=BLUE, ms=5)
    a.text(45, 2, '?', ha='center', va='center', color=GREY)
    a.plot(-20, 0, '^', color=RED, ms=5.5)
    a.set(xlim=(-24, 63), ylim=(-.7, 4.8), xticks=[-20, 0, 7, 31, 60],
          yticks=range(4, -1, -1), yticklabels=[r[0] for r in roles],
          xlabel='Days relative to decision')
    a.tick_params(axis='y', length=0)
    a.text(45.5, 4.9, '31–60 d', ha='center', va='bottom', color=BLUE)
    handles = [Line2D([], [], marker='o', mfc='white', color=BLUE, ls='none', label='Sample'),
               Line2D([], [], marker='s', color=BLUE, ls='none', label='Available'),
               Line2D([], [], marker='^', color=RED, ls='none', label='High Chl-a')]
    fig.legend(handles=handles, loc='center', bbox_to_anchor=(.37, 1 - 8 / HEIGHT),
               ncol=3, handletextpad=.3, columnspacing=.7, handlelength=1)

    # b: Schematic event dates, grouping and first-record markers.
    text(fig, 126, 8, '(b)', fontsize=13, fontweight='bold')
    b = axis(fig, 131, 20, 44, 38)
    events = pd.read_csv(DATA / 'event_grouping_examples.csv')
    b.plot([10, 50], [0, 0], color=RED, lw=1)
    for row in events.itertuples(index=False):
        b.plot(row.record_day, 0, marker='^', ms=6,
               mfc=RED if row.first_record_of_event else 'white', mec=RED, mew=.85)
    b.annotate('', (120, .38), (50, .38),
               arrowprops=dict(arrowstyle='<->', color=GREY, lw=.7))
    b.text(85, .57, '70 d', ha='center', va='center')
    b.text(30, 1, 'Event 1', ha='center', color=RED)
    b.text(117, 1, 'Event 2', ha='center', color=RED)
    b.set(xlim=(-2, 137), ylim=(-.55, 1.3), xticks=[0, 60, 120], yticks=[], xlabel='High-record day')

    # c: Full forward-year design and explicit nested fold exclusions.
    text(fig, 5, 77, '(c)', fontsize=13, fontweight='bold')
    text(fig, 59, 77, 'Time-forward fitting', ha='center')
    text(fig, 148, 77, 'Nested base scores', ha='center')
    text(fig, 148, 84, '2020 / F0; rows: 2019', ha='center', color=GREY)
    c = axis(fig, 23, 91, 72, 39)
    timeline = pd.read_csv(DATA / 'forward_timeline.csv', dtype={'row': str})
    row_order = ['2016', '2017', '2018', '2019', '2020', 'Final']
    role_values = {'fit': 0, 'validation': 1, 'retrospective': 2, 'unused': np.nan}
    values = timeline.pivot(index='row', columns='record_year', values='role').loc[row_order, range(2011, 2026)]
    numeric = values.apply(lambda col: col.map(role_values)).to_numpy(dtype=float)
    c.imshow(np.ma.masked_invalid(numeric), aspect='auto', interpolation='nearest',
             cmap=ListedColormap(['#D6D6D6', GOLD, GREEN]), vmin=0, vmax=2)
    c.set(xticks=[0, 4, 9, 14], xticklabels=['2011', '2015', '2020', '2025'],
          yticks=np.arange(6), yticklabels=row_order, xlabel='Record year')
    c.tick_params(which='both', length=0)
    for row in range(1, 6):
        c.axhline(row - .5, color='white', lw=.7)
    c.axvline(9.5, color='black', lw=.75)

    nested = pd.read_csv(DATA / 'nested_example_matrix.csv')
    admissible = nested.pivot(index='row_fold', columns='base_fit_fold', values='admissible').loc[range(1, 5), range(5)]
    assert admissible.to_numpy().sum() == 12
    cn = axis(fig, 122, 91, 53, 39)
    cn.imshow(admissible.to_numpy(dtype=int), cmap=ListedColormap(['white', '#D6D6D6']),
              vmin=0, vmax=1, interpolation='nearest', aspect='auto', extent=(-.5, 4.5, 3.5, -.5))
    for row in range(4):
        for col in range(5):
            if not admissible.iloc[row, col]:
                cn.plot(col, row, marker='x', color='black', ms=5, mew=.85, ls='none')
    for value in np.arange(.5, 4.5):
        cn.axvline(value, color='white', lw=.7)
    for value in np.arange(.5, 3.5):
        cn.axhline(value, color='white', lw=.7)
    cn.set(xticks=range(5), xticklabels=[f'F{i}' for i in range(5)],
           yticks=range(4), yticklabels=[f'F{i}' for i in range(1, 5)],
           xlabel='Base fitting fold', ylabel='Scored row fold')
    cn.tick_params(which='both', length=0)
    cn.get_xticklabels()[0].set_color(GOLD)
    validation_legend = [Patch(fc='#D6D6D6', label='Earlier-year fit'),
                         Patch(fc=GOLD, label='Lake-fold validation'),
                         Patch(fc=GREEN, label='Retrospective'),
                         Line2D([], [], color='black', marker='x', ls='none', ms=5, label='Excluded')]
    fig.legend(handles=validation_legend, loc='center', bbox_to_anchor=(.52, 1 - 153 / HEIGHT),
               ncol=4, handlelength=1, handletextpad=.4, columnspacing=.7)

    # d: Complete proportions and aligned positive counts; both axes start at zero.
    text(fig, 5, 167, '(d)', fontsize=13, fontweight='bold')
    d = axis(fig, 24, 183, 151, 26)
    count = axis(fig, 24, 220, 151, 17)
    support = pd.read_csv(DATA / 'label_support.csv')
    positive = pd.read_csv(DATA / 'positive_records.csv')
    assert len(support) == 72 and len(positive) == 24
    for j, window in enumerate(WINDOWS):
        distribution = support[support.window.eq(window)].pivot(index='year', columns='label', values='percent').sort_index()
        xx = np.arange(8) + (j - 1) * .25
        bottom = np.zeros(8)
        for label in ['Positive', 'Negative', 'Unknown']:
            yy = distribution[label].to_numpy()
            d.bar(xx, yy, bottom=bottom, width=.23, color=PALETTE[label], edgecolor='#B7B7B7', lw=.25)
            bottom += yy
        assert np.allclose(bottom, 100)
        pp = positive[positive.window.eq(window)].sort_values('year')
        count.bar(xx, pp.records, width=.23, color=BLUE, edgecolor='white', lw=.25)
    d.set(xlim=(-.6, 7.6), ylim=(0, 100), xticks=np.arange(8), xticklabels=[],
          yticks=[0, 50, 100], ylabel='Records (%)')
    count.set(xlim=(-.6, 7.6), ylim=(0, 650), xticks=np.arange(8), xticklabels=np.arange(2013, 2021),
              yticks=[0, 300, 600], ylabel='Positive n', xlabel='Training-context year')
    count.tick_params(axis='x', top=False)
    fig.legend(handles=[Patch(fc=PALETTE[label], ec='#B7B7B7', lw=.3, label=label) for label in PALETTE],
               loc='center', bbox_to_anchor=(.52, 1 - 167 / HEIGHT), ncol=3,
               handlelength=1, handletextpad=.4, columnspacing=1)
    text(fig, 175, 175, '1–14 / 15–30 / 31–60 d', ha='right')
    export(fig, checks)
    plt.close(fig)


if __name__ == '__main__':
    main()
