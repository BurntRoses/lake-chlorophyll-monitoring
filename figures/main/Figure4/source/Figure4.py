"""Figure 4. Regional coverage and target refitting."""
from results_common import *
from evidence_validation import validate as validate_previous_evidence
from matplotlib.ticker import AutoMinorLocator, FixedLocator, NullLocator, FuncFormatter, LogLocator

ROOT = Path(__file__).resolve().parent
PAIR_METHODS = ['single_window', 'multi_window']


def draw(f):
    p = Page(181)
    for x, y, letter in [(5, 11, 'a'), (97, 11, 'b'), (5, 102, 'c'), (97, 102, 'd')]:
        p.text(x, y, '(' + letter + ')', 12, weight='bold')
    handles = [Line2D([], [], ls='none', marker=ENC[m]['marker'], ms=4.8,
        mfc=ENC[m]['color'] if ENC[m]['filled'] else 'white',
        mec=ENC[m]['color'], mew=.8, label=name)
        for m, name in zip(['elastic_net'] + PAIR_METHODS,
            ['Elastic Net', 'Single-window', 'Multi-window'])]
    legend = p.fig.legend(handles=handles, loc='center', bbox_to_anchor=(98/183, 1-5.5/181),
        ncol=3, frameon=False, fancybox=False, edgecolor='black', facecolor='white',
        framealpha=1, fontsize=10.5, handlelength=1, handletextpad=.45,
        columnspacing=1.4, borderpad=.35, borderaxespad=0)
    legend.get_frame().set_linewidth(.6)

    a = p.axes(18, 15, 68, 68)
    a.set(xlim=(-1.2, 24), ylim=(-1.2, 24), xticks=[0, 10, 20], yticks=[0, 10, 20])
    a.set_aspect('equal', adjustable='box')
    a.xaxis.set_minor_locator(AutoMinorLocator(2))
    a.yaxis.set_minor_locator(AutoMinorLocator(2))
    a.plot([0, 23], [0, 23], ls=(0, (4, 3)), color='#404040', lw=.65, zorder=0)
    a.set_xlabel('Events lost (%)', labelpad=2)
    a.set_ylabel('Events gained (%)', labelpad=2)
    placements = {
        ('elastic_net', 'china_2020_2022'): (15.1, 6.5, 'China 20–22', 'left'),
        ('elastic_net', 'china_2023'): (13.5, 17.1, 'China 2023', 'left'),
        ('elastic_net', 'oceania'): (1.5, .9, 'Oceania', 'left'),
        ('elastic_net', 'new_zealand'): (7.8, 6.0, 'NZ', 'right'),
        ('single_window', 'china_2020_2022'): (14.6, 9.2, 'China 20–22', 'left'),
        ('single_window', 'china_2023'): (14.6, 12.1, 'China 2023', 'left'),
        ('single_window', 'oceania'): (1.2, 17.5, 'Oceania', 'left'),
        ('single_window', 'new_zealand'): (13.6, 4.8, 'NZ', 'left'),
        ('multi_window', 'china_2020_2022'): (.3, 11.7, 'China 20–22', 'left'),
        ('multi_window', 'china_2023'): (.3, 3.7, 'China 2023', 'left'),
        ('multi_window', 'oceania'): (5.7, 21.8, 'Oceania', 'left'),
        ('multi_window', 'new_zealand'): (13.5, 2.0, 'NZ', 'left')}
    for row in f['absolute_event_exchanges'].itertuples():
        e = ENC[row.method_id]
        a.plot(row.lost_pct, row.gained_pct, marker=e['marker'], ms=5, ls='none',
            mfc=e['color'] if e['filled'] else 'white', mec=e['color'], mew=.85, zorder=5)
        x, y, label, align = placements[row.method_id, row.region_id]
        a.annotate(label, (row.lost_pct, row.gained_pct), xytext=(x, y),
            fontsize=9.5, ha=align, va='center', color=e['color'], linespacing=1.0,
            bbox=dict(fc='white', ec='none', pad=.06, alpha=.97),
            arrowprops=dict(arrowstyle='-', color=e['color'], lw=.6, shrinkA=2, shrinkB=3.4, connectionstyle='arc3,rad=0'), zorder=8)

    b = p.axes(110, 15, 68, 68)
    b.set(xlim=(-.5, 3.5), ylim=(0, 108), xticks=range(4), yticks=[0, 25, 50, 75, 100],
        xticklabels=['China\n2020–22', 'China\n2023', 'Oceania\n2020–22', 'NZ\n2023–24'])
    b.tick_params(axis='x', length=3, pad=3)
    b.yaxis.set_minor_locator(AutoMinorLocator(2))
    b.set_ylabel('Relative − absolute (pp)', labelpad=2)
    for m in PAIR_METHODS:
        e = ENC[m]
        for row in f['relative_training_effects'].query('method_id==@m').itertuples():
            x = REGIONS.index(row.region_id) + (-.12 if m == 'single_window' else .12)
            b.errorbar(x, row.difference_pp,
                yerr=[[row.difference_pp-row.lower_pp], [row.upper_pp-row.difference_pp]],
                fmt=e['marker'], ms=4.8, mfc=e['color'] if e['filled'] else 'white',
                mec=e['color'], mew=.75, ecolor=e['color'], elinewidth=.75, capsize=2, zorder=4)
    joint = p.axes(18, 106, 68, 62)
    top = p.axes(18, 98, 68, 6)
    side = p.axes(88, 106, 4, 62)
    lim = [.009, 25]
    joint.set_xscale('log'); joint.set_yscale('log')
    joint.set(xlim=lim, ylim=lim)
    for axis in [joint.xaxis, joint.yaxis]:
        axis.set_major_locator(FixedLocator([.1, 1, 10]))
        axis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:g}'))
        axis.set_minor_locator(LogLocator(base=10, subs=[2, 5], numticks=20))
    joint.set_xlabel('Min. rank/quota (absolute)', fontsize=9.5, labelpad=3)
    joint.set_ylabel('Min. rank/quota (relative)', fontsize=9.5, labelpad=2)
    joint.axhline(1, color='black', ls=(0, (4, 2)), lw=.6)
    joint.axvline(1, color='black', ls=(0, (4, 2)), lw=.6)
    pairs = f['event_priority_pairs']
    for state, color, marker, area in [
        ('neither', '#8C8C8C', 'o', 9), ('retained', GOLD, 's', 13),
        ('gained', TEAL, 'o', 12), ('lost', COPPER, 'x', 19)]:
        d = pairs[pairs.status.eq(state)]
        kw = dict(s=area, marker=marker, alpha=.73, linewidths=.45, zorder=3)
        if state == 'lost':
            joint.scatter(d.absolute_best_rank_over_quota, d.relative_best_rank_over_quota, color=color, **kw)
        else:
            joint.scatter(d.absolute_best_rank_over_quota, d.relative_best_rank_over_quota,
                facecolors='white' if state == 'retained' else color, edgecolors=color, **kw)
    for x, y, label, color, ha, va in [
        (.05, .96, 'Lost (27)', COPPER, 'left', 'top'),
        (.96, .96, 'Neither (342)', MID, 'right', 'top'),
        (.05, .04, 'Retained (35)', GOLD, 'left', 'bottom'),
        (.96, .04, 'Gained (314)', TEAL, 'right', 'bottom')]:
        joint.text(x, y, label, fontsize=9.5, ha=ha, va=va, color=color, transform=joint.transAxes)
    bins = np.geomspace(lim[0], lim[1], 25)
    top.set_xscale('log'); top.set_xlim(lim)
    side.set_yscale('log'); side.set_ylim(lim)
    top.hist(pairs.absolute_best_rank_over_quota, bins=bins, color='#D6D6D6',
        edgecolor='black', lw=.45, histtype='stepfilled')
    side.hist(pairs.relative_best_rank_over_quota, bins=bins, orientation='horizontal',
        color='#D6D6D6', edgecolor='black', lw=.45, histtype='stepfilled')
    for margin in [top, side]:
        margin.set_xticks([]); margin.set_yticks([])
        margin.xaxis.set_minor_locator(NullLocator()); margin.yaxis.set_minor_locator(NullLocator())
        margin.tick_params(which='both', bottom=False, left=False, top=False, right=False)
        for spine in margin.spines.values(): spine.set_visible(False)
    top.spines['bottom'].set_visible(True)
    side.spines['left'].set_visible(True)

    lake_ax = p.axes(110, 106, 68, 62)
    lake_ax.set_xscale('log')
    lake_ax.set(xlim=(.2, 200), ylim=(-115, 118), yticks=[-100, -50, 0, 50, 100])
    lake_ax.xaxis.set_major_locator(FixedLocator([1, 5, 25, 75, 150]))
    lake_ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:g}'))
    lake_ax.xaxis.set_minor_locator(NullLocator())
    lake_ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    lake_ax.axhline(0, color='black', lw=.6)
    for q in [25, 75]:
        lake_ax.axvline(q, color=GOLD, lw=.6, ls=(0, (4, 3)), zorder=0)
    lake_ax.set_xlabel('Historical Chl-a q90 (µg/L)', fontsize=9.5, labelpad=3)
    lake_ax.set_ylabel('Lake coverage change (pp)', fontsize=9.5, labelpad=2)
    lake = f['china_lake_changes'].sort_values('evaluable_events', ascending=False)
    for sign, color, marker in [(0, '#8C8C8C', 'o'), (1, TEAL, 'o'), (-1, COPPER, 'v')]:
        d = lake[np.sign(lake.coverage_change_pp) == sign]
        # Unit circle and triangle have areas pi/4 and 1/2. Correct the
        # triangle size so all outcome shapes share the circle area legend.
        area_correction = np.pi/2 if marker == 'v' else 1
        lake_ax.scatter(d.historical_q90_ug_L, d.coverage_change_pp,
            s=d.evaluable_events*11*area_correction, marker=marker, facecolors=color,
            edgecolors='white', linewidths=.35, alpha=.82, zorder=3 if sign else 2)
    lake_ax.text(.38, -30, 'Events / lake', fontsize=9.5)
    for n, x in zip([1, 3, 6], [.64, 1.25, 2.5]):
        lake_ax.scatter([x], [-62], s=n*11, facecolors='none', edgecolors='black', lw=.55, zorder=5)
        lake_ax.annotate(str(n), (x, -62), xytext=(0, -7), textcoords='offset points',
            ha='center', va='top', fontsize=9.5)
    return p


CAPTION = 'Figure 4. Cross-region coverage gains and losses under absolute and lake-relative targets.\n(a) Gained and lost absolute events relative to XGBoost, expressed as percentages of each regional event cohort. Points above the diagonal have a positive net coverage difference. Grey diamonds, blue circles and open green triangles denote Elastic Net, Single-window and Multi-window. Denominators are 337, 86, 29 and 77 events for China 2020–2022, China 2023, Oceania 2020–2022 and New Zealand 2023–2024, respectively. Absolute events use Chl-a ≥75 μg/L and within-lake high-record gaps ≤60 days.\n(b) Relative-target minus absolute-target training effects on coverage of relative events. Bars show 98.75% paired whole-lake bootstrap intervals from 5,000 resamples. Denominators are 718, 168, 128 and 24 relative events. Panels a and b evaluate different event cohorts; NZ denotes New Zealand.\n(c) Multi-window priorities for the same 718 Chinese relative events in 258 lakes. Each coordinate is the minimum rank/quota across the event’s eligible 31–60-day opportunities under the corresponding training target. Values ≤1 indicate coverage. Quadrants contain 314 gained, 27 lost, 35 retained and 342 uncovered events. Both axes are logarithmic; marginal histograms show the corresponding priority distributions. The two minima can occur on different decision dates.\n(d) Change in the fraction of evaluable events covered within each lake, plotted against its pre-2011 Chl-a q90 (90th percentile). Symbol area is proportional to the number of evaluable events per lake (1–6). Green, grey and red indicate increased, unchanged and decreased coverage. Vertical guides mark q90 values of 25 and 75 μg/L.\nAll panels use 10% capacity and the 31–60-day window. Models are fitted outside China and all Oceania, then held fixed for target-region evaluation. China and Oceania use satellite records; New Zealand uses field records. Relative references require ≥50 pre-2011 observations spanning ≥365 days. Relative-target fitting rebuilds labels and threshold-dependent inputs before retraining; candidates lacking a reference remain in the ranking. Figures S5 and S6 report capacity responses and historical-reference availability.'


def validate(f):
    checks = validate_previous_evidence(f)
    queue = f['paired_candidate_ranks'].copy()
    links = f['event_opportunity_links']
    points = f['event_priority_pairs'].set_index('episode_id')
    assert len(queue) == 21148 and len(links) == 1409 and len(points) == 718
    queue['origin_date'] = pd.to_datetime(queue.origin_date)
    assert not queue.candidate_row_key.duplicated().any()
    assert queue.equals(queue.sort_values(['origin_date', 'lake_id']).reset_index(drop=True))
    sizes = queue.groupby('origin_date').lake_id.transform('size')
    assert (queue.queue_size == sizes).all()
    assert (queue.quota == np.ceil(.1 * sizes)).all()
    for prefix in ['absolute', 'relative']:
        ranks = queue.groupby('origin_date')[prefix+'_score'].rank(method='first', ascending=False)
        assert np.array_equal(ranks, queue[prefix+'_rank'])
        assert np.array_equal(ranks <= queue.quota, queue[prefix+'_selected'].astype(bool))
    joined = links.merge(queue, on='candidate_row_key', validate='many_to_one')
    event_dates = f['china_paired_relative_events'].set_index('episode_id').event_date
    leads = (pd.to_datetime(joined.episode_id.map(event_dates)) - joined.origin_date).dt.days
    assert (leads == joined.lead_days).all() and leads.between(31, 60).all()
    for prefix in ['absolute', 'relative']:
        ratios = joined[prefix+'_rank'] / joined.quota
        best = ratios.groupby(joined.episode_id).min().reindex(points.index)
        assert np.allclose(best, points[prefix+'_best_rank_over_quota'], rtol=1e-14, atol=0)
        assert np.array_equal(best <= 1, points[prefix+'_covered'].astype(bool))
    assert points.status.value_counts().to_dict() == {'neither': 342, 'gained': 314, 'retained': 35, 'lost': 27}
    assert np.array_equal(links.groupby('episode_id').size().reindex(points.index), points.opportunity_count)
    original = f['china_paired_relative_events'].set_index('episode_id').loc[points.index]
    for key in ['lake_id', 'absolute_covered', 'relative_covered']:
        assert np.array_equal(points[key], original[key])
    checks.update(all_21148_saved_candidate_ranks_rebuilt=True,
        all_1409_event_opportunities_reconciled=True,
        all_718_best_rank_pairs_and_quadrants_reconciled=True,
        saved_queue_tie_break_and_ceiling_quota_preserved=True,
        no_new_embedding_or_density_estimate=True)
    return checks


if __name__ == '__main__':
    frames, hashes = read_data(ROOT)
    export(draw(frames), ROOT, 4, CAPTION, validate(frames), hashes)
