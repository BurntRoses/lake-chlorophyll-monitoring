"""Lead-time structure, temporal alignment and allocation of US monitoring policies."""
import json
import joblib
import numpy as np
import pandas as pd
from .paths import DATA, MODELS
from .evaluation import selected_mask


def warning_counts(mask, ri, ei, count):
    result = np.zeros(count, np.int64)
    np.add.at(result, ei, mask[ri])
    return result


def repeat_adjusted_selection(frame, scores, capacity, power, interval):
    """Use previous allocations and current risk ranks only, with unchanged quotas."""
    ranks = frame.groupby('origin_date')[scores].rank(method='average', pct=True).to_numpy()
    lakes = pd.factorize(frame.lake_id, sort=True)[0]
    dates = frame.origin_date.to_numpy(dtype='datetime64[D]').astype(np.int64)
    last = np.full(lakes.max()+1, -1000000, np.int64)
    chosen = np.zeros(len(frame), bool)
    for _, g in frame.groupby('origin_date', sort=False):
        rows = g.index.to_numpy()
        elapsed = np.minimum(dates[rows]-last[lakes[rows]], interval)/interval
        adjusted = ranks[rows]*elapsed**power
        order = np.lexsort((g.lake_id.to_numpy(), -adjusted))
        take = rows[order[:int(np.ceil(capacity*len(rows)))]]
        chosen[take] = True
        last[lakes[take]] = dates[take]
    return chosen


def run_structural_analysis(frame, spec, ops, masks):
    from .us_analysis import OUT, event_hits
    methods = list(spec['methods'])
    definitions = spec['endpoints']
    ledger, ri, ei = ops[('absolute75', '1_60')]
    common = ledger[['episode_id','lake_id','event_date']].copy()
    counts, timing, common_rows, lead_rows, yield_rows = [], [], [], [], []
    for endpoint in definitions:
        subset, r, e = ops[('absolute75', endpoint)]
        common['available_'+endpoint] = common.episode_id.isin(subset.episode_id)
        for method in methods:
            hits = subset.set_index('episode_id')[method]
            common[method+'__'+endpoint] = common.episode_id.map(hits).eq(True)
            n = warning_counts(masks[method], r, e, len(subset))
            denominator = np.bincount(e, minlength=len(subset))
            for possible in sorted(np.unique(denominator)):
                use = denominator == possible
                yield_rows.append(dict(endpoint=endpoint, method=method, opportunities=int(possible),
                    events=int(use.sum()), captured=int((n[use]>0).sum()), selected_opportunities=int(n[use].sum())))
        print('US_LEAD_TIME', endpoint, len(subset), flush=True)
    common_mask = common.available_1_30 & common.available_31_60
    for method in methods:
        if not np.array_equal(common[method+'__1_60'], common[method+'__1_30'] | common[method+'__31_60']):
            raise ValueError('Near and long event captures do not partition the full warning interval')
        a = common.loc[common_mask, method+'__1_30'].to_numpy()
        b = common.loc[common_mask, method+'__31_60'].to_numpy()
        common_rows.append(dict(method=method, events=int(common_mask.sum()), near_only=int((a & ~b).sum()),
            long_only=int((b & ~a).sum()), both=int((a & b).sum()), neither=int((~a & ~b).sum()),
            near_captured=int(a.sum()), long_captured=int(b.sum()), full_captured=int((a | b).sum())))
        n = warning_counts(masks[method], ri, ei, len(ledger))
        opportunities = np.bincount(ei, minlength=len(ledger))
        first = np.full(len(ledger), -1, np.int64)
        leads = (ledger.event_date.to_numpy()[ei]-frame.origin_date.to_numpy()[ri])/np.timedelta64(1,'D')
        use = masks[method][ri]
        np.maximum.at(first, ei[use], leads[use].astype(np.int64))
        first = np.where(first >= 0, first.astype(float), np.nan)
        common[method+'__first_lead_days'] = first
        counts.extend(dict(episode_id=k, endpoint='1_60', method=method, possible_visits=int(o),
            selected_visits=int(nc), first_lead_days=float(l))
            for k,o,nc,l in zip(ledger.episode_id, opportunities, n, first))
        bins = {'31_60':np.isfinite(first)&(first>=31), '15_30':(first>=15)&(first<=30),
                '1_14':(first>=1)&(first<=14), 'uncovered':~np.isfinite(first)}
        for group, mask in bins.items():
            timing.append(dict(method=method, first_warning_interval=group, events=int(mask.sum()),
                all_events=len(ledger), percent=100*mask.mean()))
    for proposed, control in spec['comparisons']:
        a, b = common[proposed+'__first_lead_days'], common[control+'__first_lead_days']
        use = a.notna() & b.notna()
        delta = a[use]-b[use]
        lead_rows.append(dict(proposed=proposed, comparator=control, jointly_captured=int(use.sum()),
            median_lead_difference_days=float(delta.median()), mean_lead_difference_days=float(delta.mean()),
            earlier=int(delta.gt(0).sum()), same=int(delta.eq(0).sum()), later=int(delta.lt(0).sum())))
    pd.DataFrame(common_rows).to_csv(OUT/'common_population_captures.csv', index=False)
    pd.DataFrame(timing).to_csv(OUT/'first_warning_summary.csv', index=False)
    pd.DataFrame(lead_rows).to_csv(OUT/'paired_warning_timing.csv', index=False)
    pd.DataFrame(counts).to_parquet(OUT/'event_warning_visits.parquet', index=False)
    pd.DataFrame(yield_rows).to_csv(OUT/'opportunity_yield.csv', index=False)
    subgroup_analysis(frame, spec, ops, common, ri, ei)
    common.to_parquet(OUT/'event_timing.parquet', index=False)
    schedules, schedule_allocation = [], []
    for method in methods:
        for alpha in [0., *spec['diagnostics']['repeat_penalty_powers']]:
            mask = masks[method] if alpha == 0 else repeat_adjusted_selection(
                frame, method, .1, alpha, spec['diagnostics']['repeat_interval_days'])
            if mask.sum() != masks[method].sum():
                raise ValueError('Scheduling changed the monitoring budget')
            visits = frame.loc[mask].groupby('lake_id').size()
            schedule_allocation.append(dict(method=method, penalty_power=alpha, slots=int(mask.sum()),
                lakes=len(visits), top10percent_lake_slot_share=100*visits.nlargest(int(np.ceil(.1*frame.lake_id.nunique()))).sum()/mask.sum()))
            for endpoint in definitions:
                subset, r, e = ops[('absolute75', endpoint)]
                hits = event_hits(mask, r, e, len(subset))
                for year in ['ALL', *sorted(subset.event_date.dt.year.unique())]:
                    use = np.ones(len(subset), bool) if year == 'ALL' else subset.event_date.dt.year.eq(year).to_numpy()
                    schedules.append(dict(method=method, penalty_power=alpha, endpoint=endpoint, year=str(year),
                        captured=int(hits[use].sum()), events=int(use.sum()), capture_percent=100*hits[use].mean()))
    pd.DataFrame(schedules).to_csv(OUT/'repeat_scheduling.csv', index=False)
    pd.DataFrame(schedule_allocation).to_csv(OUT/'repeat_scheduling_allocation.csv', index=False)
    temporal_alignment(frame, spec, ops)
    tree_feature_use(spec)
    component_analysis(frame, spec, ops)


def subgroup_analysis(frame, spec, ops, common, ri, ei):
    from .us_analysis import OUT, reference_observations
    # Earliest 1-60-day opportunity is a common, method-independent reference date.
    reference = np.full(len(common), np.iinfo(np.int64).max, np.int64)
    days = frame.origin_date.to_numpy(dtype='datetime64[D]').astype(np.int64)
    np.minimum.at(reference, ei, days[ri])
    common['reference_date'] = pd.to_datetime(reference, unit='D')
    history = reference_observations()
    history['available_date'] = history.date+pd.Timedelta(days=7)
    history = {lake:g for lake,g in history.groupby('lake_id', sort=False)}
    n, age, severe = [], [], []
    for event in common.itertuples():
        h = history.get(event.lake_id)
        prior = None if h is None else h[h.available_date.lt(event.reference_date)]
        n.append(0 if prior is None else len(prior))
        age.append(np.nan if prior is None or prior.empty else (event.reference_date-prior.date.max()).days)
        severe.append(0 if prior is None else int(prior.chla.ge(75).sum()))
    common['history_observations'] = n
    common['latest_observation_age_days'] = age
    common['prior_high_observations'] = severe
    develop = pd.read_parquet(DATA/'processed/us_training/common_development_table.parquet',
        columns=['candidate_row_key','lake_id','ecoregion','common_trainable','label_mature_date','origin_date'])
    train = develop.common_trainable & develop.label_mature_date.lt('2021-01-01') & develop.origin_date.lt('2021-01-01')
    if int(train.sum()) != 136122:
        raise ValueError('Final training population differs')
    from .paths import RESULTS
    keys = pd.read_parquet(RESULTS/'us/main/final_common_training_keys.parquet').candidate_row_key
    if set(keys) != set(develop.loc[train,'candidate_row_key']):
        raise ValueError('Stored final training identities differ')
    training_lakes = set(develop.loc[train,'lake_id'].astype(str))
    common['training_membership'] = np.where(common.lake_id.isin(training_lakes),'training_lake','other_lake')
    eco = develop.drop_duplicates('lake_id').assign(lake_id=lambda x:x.lake_id.astype(str)).set_index('lake_id').ecoregion
    common['ecoregion'] = common.lake_id.map(eco).fillna('unknown').astype(str)
    common['history_count_group'] = pd.cut(common.history_observations, [-1,0,9,49,np.inf], labels=['0','1-9','10-49','50+']).astype(str)
    common['observation_age_group'] = pd.cut(common.latest_observation_age_days, [-1,30,90,365,np.inf],
        labels=['0-30','31-90','91-365','366+']).astype('object').fillna('no_history')
    common['prior_high_group'] = np.where(common.prior_high_observations.gt(0),'prior_recorded_high','no_prior_recorded_high')
    rows = []
    groups = ['training_membership','ecoregion','history_count_group','observation_age_group','prior_high_group']
    for endpoint in spec['endpoints']:
        subset, _, _ = ops[('absolute75',endpoint)]
        values = subset.merge(common[['episode_id',*groups]],on='episode_id',validate='one_to_one')
        for group in groups:
            for name, g in values.groupby(group, observed=True, sort=True):
                for method in spec['methods']:
                    rows.append(dict(endpoint=endpoint, grouping=group, stratum=str(name), method=method,
                        events=len(g), captured=int(g[method].sum()), capture_percent=100*g[method].mean()))
    pd.DataFrame(rows).to_csv(OUT/'event_strata.csv',index=False)


def temporal_alignment(frame, spec, ops):
    from .us_analysis import OUT, event_hits
    methods = list(spec['methods'])
    ranks = frame.groupby('origin_date')[methods].rank(method='average',pct=True)
    groups = frame.groupby([frame.lake_id,frame.origin_date.dt.year],sort=False).indices
    means = ranks.groupby([frame.lake_id,frame.origin_date.dt.year]).transform('mean')
    rng = np.random.default_rng(spec['diagnostics']['phase_seed'])
    rows = []
    for rep in range(-2,spec['diagnostics']['phase_replicates']):
        if rep == -2:
            values, control = ranks.to_numpy(), 'aligned'
        elif rep == -1:
            values, control = means.to_numpy(), 'lake_year_mean'
        else:
            permutation = np.arange(len(frame))
            for indices in groups.values():
                if len(indices)>1:
                    permutation[indices] = np.roll(indices,int(rng.integers(1,len(indices))))
            values, control = ranks.to_numpy()[permutation], 'phase_displaced'
        for j, method in enumerate(methods):
            mask = selected_mask(frame,values[:,j])
            for endpoint in spec['endpoints']:
                ledger,ri,ei = ops[('absolute75',endpoint)]
                hits = event_hits(mask,ri,ei,len(ledger))
                rows.append(dict(control=control,replicate=max(rep,0),method=method,endpoint=endpoint,
                    events=len(ledger),captured=int(hits.sum()),capture_percent=100*hits.mean()))
    pd.DataFrame(rows).to_csv(OUT/'temporal_alignment.csv',index=False)


def tree_feature_use(spec):
    from .us_analysis import OUT
    from .coverage import FirstOrderCoverageBooster
    from .io import read
    from .paths import CONFIG
    rows = []
    for correction in read(CONFIG/'monitoring_models.json')['corrections']:
        name = correction['policy_name']
        model = FirstOrderCoverageBooster.load(MODELS/'us/coverage'/(name+'.joblib'))
        expression = joblib.load(MODELS/'us/experts'/correction['variant']/'common_expression.joblib')
        names = list(expression.names)+['expert_logit::'+x for x in correction['expert_keys']]+['window_head']
        if len(names)!=model.feature_count_:
            raise ValueError('Coverage tree feature names differ from saved model dimensions')
        def collect(node, counts):
            if 'split' not in node:
                return
            index = int(node['split'].removeprefix('f'))
            counts[index] = counts.get(index,0)+1
            for child in node['children']:
                collect(child,counts)
        for j, tree in enumerate(model.trees_):
            counts = {}
            collect(json.loads(tree.get_booster().get_dump(dump_format='json')[0]),counts)
            for index, count in counts.items():
                feature = names[index]
                rows.append(dict(method=correction['policy_name'],tree=j,feature=feature,split_count=count,
                    group='expert_prediction' if feature.startswith('expert_logit::') else 'head' if feature=='window_head' else 'environment_and_history'))
    pd.DataFrame(rows).to_csv(OUT/'tree_feature_usage.csv',index=False)


def component_analysis(frame, spec, ops):
    """Compare primary-only and joint training."""
    from .us_analysis import OUT, event_hits
    from .paths import RESULTS
    methods = {'single':'single_window','three_long_primary_only':'ablation_same_window',
        'three_time_primary_only':'ablation_multi_window',
        'three_long_joint':'same_window_control','three_time_joint':'multi_window'}
    points, intervals = [], []
    original = pd.read_parquet(RESULTS/'us/ablations/evaluation/bootstrap_draws.parquet')
    joint = pd.read_parquet(RESULTS/'us/main/evaluation/bootstrap_draws.parquet')
    supplementary = pd.read_parquet(OUT/'bootstrap_draws.parquet')
    for endpoint in spec['endpoints']:
        ledger,ri,ei = ops[('absolute75',endpoint)]
        for method,column in methods.items():
            mask = selected_mask(frame,frame[column])
            hits = event_hits(mask,ri,ei,len(ledger))
            points.append(dict(endpoint=endpoint,method=method,captured=int(hits.sum()),events=len(ledger),
                capture_percent=100*hits.mean()))
        if endpoint not in ['31_60','1_60']:
            continue
        for name in ['single_window','multi_window']:
            current = supplementary[f'absolute75__{endpoint}__1000__{name}'].to_numpy()
            if not np.array_equal(current,joint[f'{endpoint}__{name}'].to_numpy()):
                raise ValueError('Component analysis requires identical paired resamples')
        if not np.array_equal(original[f'{endpoint}__single_window'],joint[f'{endpoint}__single_window']):
            raise ValueError('Primary-window and joint resamples are not paired')
        for control,table,column in [('single',joint,'single_window'),
            ('three_long_primary_only',original,'same_window_control'),
            ('three_time_primary_only',original,'multi_window')]:
            delta = 100*(joint[f'{endpoint}__multi_window']-table[f'{endpoint}__{column}'])
            low,high = np.quantile(delta,[.025,.975])
            intervals.append(dict(endpoint=endpoint,proposed='three_time_joint',comparator=control,
                lower95=low,upper95=high,bootstrap_reps=len(delta)))
    pd.DataFrame(points).to_csv(OUT/'component_captures.csv',index=False)
    pd.DataFrame(intervals).to_csv(OUT/'component_intervals.csv',index=False)
