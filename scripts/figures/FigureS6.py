"""Reference-defining background, support, cumulative and paired event outcomes."""
from pathlib import Path
from publication import export_publication
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.transforms import Bbox

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'figures'
DATA = ROOT / 'data/figures/FigureS6'
WIDTH, HEIGHT = 183, 244
GREEN, RED, GOLD, GREY = '#006D2C', '#B2182B', '#A6761D', '#606060'
REGIONS = ['china_2020_2022', 'china_2023', 'oceania', 'new_zealand']


def axis(fig, x, y, w, h):
    ax = fig.add_axes([x / WIDTH, 1 - (y + h) / HEIGHT, w / WIDTH, h / HEIGHT])
    ax.tick_params(which='both', top=True, right=True, pad=2)
    return ax


def text(fig, x, y, value, **kwargs):
    return fig.text(x / WIDTH, 1 - y / HEIGHT, value, va='center', **kwargs)


def export(fig, checks):
    export_publication(fig, OUT, 'FigureS6')


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
    checks = json.loads((ROOT / 'docs/figure_statistics.json').read_text(encoding='utf-8'))['FigureS6']

    # a: Give the reference-defining lake-weighted distribution the principal width.
    text(fig, 5, 8, '(a)', fontsize=13, fontweight='bold')
    fig.legend(handles=[Line2D([], [], color=GREY, ls='--', marker='s', ms=4, label='Source'),
                        Line2D([], [], color=GREEN, marker='o', ms=4, label='China')],
               loc='center', bbox_to_anchor=(.55, 1 - 8 / HEIGHT), ncol=2,
               handlelength=1.8, columnspacing=1.5, handletextpad=.5)
    text(fig, 67, 17, 'Lake-weighted references', ha='center')
    text(fig, 157.5, 17, 'Observation-weighted', ha='center')
    a = axis(fig, 24, 25, 86, 36)
    references = pd.read_csv(DATA / 'lake_reference_distributions.csv')
    for region, color, ls in [('source', GREY, '--'), ('china_2020_2022', GREEN, '-')]:
        values = references[references.region.eq(region) & references.reference_supported].historical_q90.sort_values().to_numpy()
        assert len(values) == checks['supported_lake_cdf_counts'][region]
        assert values.min() > .08 and values.max() < 650
        xx = np.r_[.08, values, 650]
        yy = np.r_[0, 100 * np.arange(1, len(values) + 1) / len(values), 100]
        a.step(xx, yy, color=color, ls=ls, lw=1.1, where='post')
    for value in [25, 75]:
        a.axvline(value, color=GOLD, lw=.55, ls=':')
    a.set(xscale='log', xlim=(.08, 650), ylim=(0, 102), xticks=[.1, 1, 5, 25, 75, 500],
          yticks=[0, 50, 100], xlabel='Pre-2011 lake Chl-a q90 (µg/L)', ylabel='Lakes (%)')
    a.set_xticklabels(['0.1', '1', '5', '25', '75', '500'])
    quantiles = pd.read_csv(DATA / 'pre2020_concentration_quantiles.csv')
    aq = axis(fig, 140, 25, 35, 36)
    for region, color, marker, ls in [('source', GREY, 's', '--'), ('china', GREEN, 'o', '-')]:
        rows = quantiles[quantiles.region_id.eq(region)].sort_values('percentile')
        assert len(rows) == 6
        aq.plot(rows.percentile, rows.chl_a_ug_L, color=color, marker=marker, ms=3.5, lw=1, ls=ls)
    aq.set(yscale='log', xlim=(20, 103), ylim=(1, 250), xticks=[25, 50, 75, 99], yticks=[1, 10, 100],
           xlabel='Percentile', ylabel='Pre-2020 Chl-a (µg/L)')
    aq.set_yticklabels(['1', '10', '100'])

    # b: Preserve all support denominators and unsupported candidate populations.
    text(fig, 5, 83, '(b)', fontsize=13, fontweight='bold')
    b = axis(fig, 32, 96, 54, 40)
    support = pd.read_csv(DATA / 'historical_reference_support.csv')
    for i, region in enumerate(REGIONS):
        row = support[support.region_id.eq(region)].iloc[0]
        b.barh(3-i, row.candidate_lakes, height=.55, color='#D9D9D9', lw=0)
        b.barh(3-i, row.supported_reference_lakes, height=.55, color=GREEN, lw=0)
        b.text(row.candidate_lakes + 8, 3-i, f'{int(row.supported_reference_lakes)}/{int(row.candidate_lakes)}',
               ha='left', va='center')
    b.set(xlim=(0, 570), xticks=[0, 200, 400], yticks=[3, 2, 1, 0],
          yticklabels=['China 20–22', 'China 2023', 'Oceania', 'NZ'], xlabel='Candidate lakes')
    b.tick_params(axis='y', length=0)
    fig.legend(handles=[Patch(fc=GREEN, label='Supported'), Patch(fc='#D9D9D9', label='Unsupported')],
               loc='center', bbox_to_anchor=(.32, 1 - 86 / HEIGHT), ncol=2,
               handlelength=1, columnspacing=.6, handletextpad=.4)

    # c: Accumulate exactly the same paired cohort used in panel e.
    text(fig, 99, 83, '(c)', fontsize=13, fontweight='bold')
    text(fig, 145, 87, 'China 2020–22', ha='center')
    c = axis(fig, 115, 96, 60, 40)
    accumulation = pd.read_csv(DATA / 'china_background_accumulation.csv')
    for column, color, ls, label in [('cumulative_evaluable_events', GREY, ':', 'Evaluable'),
                                   ('cumulative_absolute_covered', GOLD, '--', 'Absolute-trained'),
                                   ('cumulative_relative_covered', GREEN, '-', 'Relative-trained')]:
        c.step(np.r_[.25, accumulation.historical_q90_ug_L], np.r_[0, accumulation[column]],
               color=color, ls=ls, lw=1.1, where='post', label=label)
    c.set(xscale='log', xlim=(.25, 200), ylim=(0, 750), xticks=[1, 5, 25, 75], yticks=[0, 250, 500, 750],
          xlabel='Historical Chl-a q90 (µg/L)', ylabel='Cumulative events')
    c.set_xticklabels(['1', '5', '25', '75'])
    for value in [25, 75]:
        c.axvline(value, color=GOLD, lw=.55, ls=':')
    c.legend(loc='upper left', handlelength=1.4, handletextpad=.3, labelspacing=.1)

    # d: Keep both windows visible when points coincide (NZ).
    text(fig, 5, 160, '(d)', fontsize=13, fontweight='bold')
    d = axis(fig, 24, 173, 62, 49)
    reach = pd.read_csv(DATA / 'regional_relative_reach.csv')
    positions = {'china_2020_2022': (52, 92, 'China 20–22', 'right'),
                 'china_2023': (61, 58, 'China 2023', 'left'),
                 'oceania': (73, 80, 'Oceania', 'left'), 'new_zealand': (96, 105, 'NZ', 'right')}
    for region in REGIONS:
        rows = reach[reach.region_id.eq(region)].set_index('endpoint_id')
        primary, broad = rows.loc['31_60'], rows.loc['1_60']
        if not np.allclose([primary.coverage_pct, primary.lake_coverage_pct], [broad.coverage_pct, broad.lake_coverage_pct]):
            d.annotate('', (broad.coverage_pct, broad.lake_coverage_pct), (primary.coverage_pct, primary.lake_coverage_pct),
                       arrowprops=dict(arrowstyle='-|>', lw=.8, color=GREY, shrinkA=3, shrinkB=4))
        d.plot(primary.coverage_pct, primary.lake_coverage_pct, 'o', ms=6, mfc='white', mec=GREEN, mew=.9)
        d.plot(broad.coverage_pct, broad.lake_coverage_pct, 'o', ms=3.8, color=GREEN)
        x, y, label, ha = positions[region]
        d.annotate(label, (broad.coverage_pct, broad.lake_coverage_pct), xytext=(x, y), ha=ha, va='center',
                   arrowprops=dict(arrowstyle='-', lw=.5, color=GREY, shrinkB=4))
    d.set(xlim=(25, 110), ylim=(35, 110), xticks=[25, 50, 75, 100], yticks=[50, 75, 100],
          xlabel='Events covered (%)', ylabel='Event lakes covered (%)')
    fig.legend(handles=[Line2D([], [], color=GREEN, marker='o', ms=6, mfc='white', ls='none', label='31–60 d'),
                        Line2D([], [], color=GREEN, marker='o', ms=3.8, ls='none', label='1–60 d')],
               loc='center', bbox_to_anchor=(55 / WIDTH, 1 - 235 / HEIGHT), ncol=2,
               handlelength=.7, handletextpad=.4, columnspacing=1)

    # e: All three strata retain both positive and negative paired changes.
    text(fig, 99, 160, '(e)', fontsize=13, fontweight='bold')
    e = axis(fig, 115, 173, 60, 49)
    strata = pd.read_csv(DATA / 'background_paired_strata.csv')
    for i, row in enumerate(strata.itertuples(index=False)):
        e.barh(2-i, row.gained_pct, height=.52, color=GREEN)
        e.barh(2-i, -row.lost_pct, height=.52, color=RED)
        e.plot(row.net_change_pp, 2-i, 'D', ms=4, mfc='white', mec='black', mew=.85)
        e.text(2, 2-i+.36, f'n = {row.events}', ha='left', va='bottom')
    e.axvline(0, color='black', lw=.65)
    e.set(xlim=(-52, 65), ylim=(-.6, 2.8), xticks=[-50, 0, 50], yticks=[2, 1, 0],
          yticklabels=['<25', '25–<75', '≥75'], xlabel='Events gained / lost (%)', ylabel='Historical q90 (µg/L)')
    e.tick_params(axis='y', length=0)
    fig.legend(handles=[Patch(fc=RED, label='Lost'),
                        Line2D([], [], color='black', marker='D', mfc='white', ls='none', ms=4, label='Net'),
                        Patch(fc=GREEN, label='Gained')], loc='center',
               bbox_to_anchor=(145 / WIDTH, 1 - 235 / HEIGHT), ncol=3,
               handlelength=.8, handletextpad=.3, columnspacing=.7)
    export(fig, checks)
    plt.close(fig)


if __name__ == '__main__':
    main()
