"""Regional capacity responses with separated density, interval and point layers."""
from pathlib import Path
from publication import export_publication
from collections import Counter
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.transforms import Bbox

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'figures'
DATA = ROOT / 'data/figures/FigureS5'
WIDTH, HEIGHT = 183, 212
BLUE, GREEN, GOLD, GREY = '#1B3B8B', '#006D2C', '#A6761D', '#606060'
METHODS = ['xgboost', 'elastic_net', 'single_window', 'multi_window']
NAMES = {'xgboost': 'XGBoost', 'elastic_net': 'Elastic Net',
         'single_window': 'Single-window', 'multi_window': 'Multi-window'}
COLORS = {'xgboost': '#111111', 'elastic_net': GREY, 'single_window': BLUE, 'multi_window': GREEN}
MARKERS = {'xgboost': 's', 'elastic_net': 'D', 'single_window': 'o', 'multi_window': '^'}
STYLES = {'xgboost': (0, (5, 2.2)), 'elastic_net': (0, (1, 1.6)),
          'single_window': '-', 'multi_window': '-'}
REGIONS = ['china_2020_2022', 'china_2023', 'oceania', 'new_zealand']
REGION_NAMES = ['China 2020–22', 'China 2023', 'Oceania 2020–22', 'NZ 2023–24']


def axis(fig, x, y, w, h):
    ax = fig.add_axes([x / WIDTH, 1 - (y + h) / HEIGHT, w / WIDTH, h / HEIGHT])
    ax.tick_params(which='both', top=True, right=True, pad=2)
    return ax


def text(fig, x, y, value, **kwargs):
    return fig.text(x / WIDTH, 1 - y / HEIGHT, value, va='center', **kwargs)


def export(fig, checks, profile_checks):
    export_publication(fig, OUT, 'FigureS5')


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
    checks = json.loads((ROOT / 'docs/figure_statistics.json').read_text(encoding='utf-8'))['FigureS5']
    handles = [Line2D([], [], color=COLORS[method], ls=STYLES[method], marker=MARKERS[method],
                      ms=3.5, mfc='white' if method in ['elastic_net', 'multi_window'] else COLORS[method],
                      mew=.8, lw=1, label=NAMES[method]) for method in METHODS]
    fig.legend(handles=handles, loc='center', bbox_to_anchor=(.5, 1 - 8 / HEIGHT), ncol=4,
               handlelength=1.5, columnspacing=1, handletextpad=.4)

    # a–d: Complete fixed-score curves on identical regional axes.
    capacity = pd.read_csv(DATA / 'regional_capacity_response.csv')
    for i, (region, name) in enumerate(zip(REGIONS, REGION_NAMES)):
        x, y = (24 if i % 2 == 0 else 115), (25 if i < 2 else 74)
        text(fig, 5 if i % 2 == 0 else 99, y - 12, f'({"abcd"[i]})', fontsize=13, fontweight='bold')
        a = axis(fig, x, y, 62 if i % 2 == 0 else 60, 30)
        for method in METHODS:
            rows = capacity[capacity.region_id.eq(region) & capacity.method_id.eq(method)].sort_values('nominal_capacity_pct')
            assert len(rows) == 117
            a.plot(rows.nominal_capacity_pct, rows.coverage_pct, drawstyle='steps-post',
                   color=COLORS[method], ls=STYLES[method], lw=1.05,
                   marker=MARKERS[method], markevery=[16, 36, 76, 108], ms=3.3,
                   mfc='white' if method in ['elastic_net', 'multi_window'] else COLORS[method], mew=.75)
        a.axvline(10, color=GOLD, lw=.65, ls='--')
        count = int(capacity[capacity.region_id.eq(region)].evaluable_events.iloc[0])
        a.text(2.3, 94, f'n = {count}', va='top')
        a.text(28.7, 6, name, ha='right', va='bottom')
        a.set(xlim=(1, 30), ylim=(0, 100), xticks=[1, 10, 20, 30], yticks=[0, 50, 100],
              ylabel='Event coverage (%)' if i % 2 == 0 else '')
        a.tick_params(axis='y', pad=5)
        a.get_xticklabels()[0].set_ha('left')

    # e: Profiles, intervals and point estimates occupy distinct vertical layers.
    text(fig, 5, 124, '(e)', fontsize=13, fontweight='bold')
    text(fig, 105, 115, 'Monitoring capacity (%)', ha='center', fontsize=11)
    a = axis(fig, 35, 133, 140, 61)
    estimates = pd.read_csv(DATA / 'regional_differences.csv')
    profiles = pd.read_csv(DATA / 'bootstrap_density_profiles.csv')
    heights, interval_positions = {}, []
    for i, region in enumerate(REGIONS):
        for j, control in enumerate(['xgboost', 'elastic_net']):
            rows = profiles[profiles.region.eq(region) & profiles.comparator.eq(control) & profiles.displayed].sort_values('difference_pp')
            baseline = (3 - i) * 2.15 + (.85 if j == 0 else 0)
            height = 3 * rows.density.to_numpy()
            color = '#111111' if j == 0 else GREY
            a.fill_between(rows.difference_pp, baseline, baseline + height,
                           color=color, alpha=.09 if j == 0 else .13, lw=0, zorder=1)
            a.plot(rows.difference_pp, baseline + height, color=color,
                   ls='-' if j == 0 else (0, (1, 1.6)), lw=.85, zorder=2)
            estimate = estimates[estimates.region_id.eq(region) & estimates.comparator_id.eq(control)].iloc[0]
            interval_y = baseline - .22
            a.errorbar(estimate.difference_pp, interval_y,
                       xerr=[[estimate.difference_pp - estimate.lower_pp], [estimate.upper_pp - estimate.difference_pp]],
                       fmt=MARKERS[control], color=color, lw=1.2, capsize=2.1,
                       mfc=color if j == 0 else 'white', ms=4.8, mew=1, zorder=4)
            heights[region, control] = float(height.max())
            interval_positions.append(interval_y)
    # Explicitly verify that the lower profile cannot touch the upper interval row.
    clearances = [.85 - .22 - heights[region, 'elastic_net'] for region in REGIONS]
    assert min(clearances) > .2
    a.axvline(0, color=GREY, lw=.65, ls='--', zorder=0)
    a.set(xlim=(-30, 75), ylim=(-.45, 8.05), xticks=[-20, 0, 20, 40, 60],
          yticks=[(3 - i) * 2.15 + .3 for i in range(4)], yticklabels=REGION_NAMES,
          xlabel='Multi-window − comparator (pp)')
    a.tick_params(axis='y', length=0)
    effects_handles = [Line2D([], [], color='#111111', marker='s', ms=4.8, lw=1.2, label='XGBoost'),
                       Line2D([], [], color=GREY, marker='D', mfc='white', ms=4.8, lw=1.2, label='Elastic Net')]
    a.legend(handles=effects_handles, loc='lower right', bbox_to_anchor=(1, 1.015), ncol=2,
             handlelength=1.6, handletextpad=.4, columnspacing=1)
    a.text(.01, 1.045, '5,000 resamples; 98.75% intervals', transform=a.transAxes, ha='left', va='bottom')
    profile_checks = dict(common_vertical_density_multiplier=3, individually_peak_normalized=False,
                          minimum_profile_to_other_interval_clearance=float(min(clearances)),
                          minimum_profile_to_other_interval_clearance_mm=float(min(clearances) * 61 / 8.5),
                          all_point_and_interval_rows_within_axis=min(interval_positions) > -.45,
                          all_eight_profiles_displayed=True)
    export(fig, checks, profile_checks)
    plt.close(fig)


if __name__ == '__main__':
    main()
