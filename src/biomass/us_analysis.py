"""Sensitivity and practical comparisons for the frozen US joint models."""
import time
import joblib
import numpy as np
import pandas as pd
from numba import njit
from .paths import CONFIG, DATA, MODELS, RESULTS
from .io import read, write
from .evaluation import selected_mask, opportunity_index
OUT = RESULTS / 'us/analysis'

def monitoring_workload():
    """Describe observed development-period sampling in each 14-day decision round."""
    f = pd.read_parquet(DATA / 'derived/development_candidates.parquet', columns=['lake_id', 'origin_date', 'eligible_common_development_candidate', 'future_chla_observation_days_1_14'])
    f = f[f.eligible_common_development_candidate & f.origin_date.ge('2016-01-01') & f.origin_date.add(pd.Timedelta(days=14)).lt('2019-01-01')].copy()
    ledger = pd.read_parquet(DATA / 'derived/development_chla_observation_ledger.parquet', columns=['lake_id', 'sample_date'])
    groups = {str(lake): np.sort(g.sample_date.to_numpy(dtype='datetime64[D]').astype('int64')) for lake, g in ledger.groupby('lake_id', sort=False)}
    counts = np.zeros(len(f), dtype=int)
    f = f.reset_index(drop=True)
    for lake, g in f.groupby('lake_id', sort=False):
        days = groups.get(str(lake), np.array([], dtype='int64'))
        origins = g.origin_date.to_numpy(dtype='datetime64[D]').astype('int64')
        counts[g.index] = np.searchsorted(days, origins + 14, side='right') - np.searchsorted(days, origins, side='right')
    if not np.array_equal(counts, f.future_chla_observation_days_1_14.to_numpy()):
        raise ValueError('Development-period sampling counts differ from the observation ledger')
    f['sample_days'] = counts
    f['sampled'] = counts > 0
    rounds = f.groupby('origin_date').agg(eligible_lakes=('lake_id', 'size'), sampled_lakes=('sampled', 'sum'), lake_sample_days=('sample_days', 'sum')).reset_index()
    rounds['sampled_lake_fraction_percent'] = 100 * rounds.sampled_lakes / rounds.eligible_lakes
    OUT.mkdir(parents=True, exist_ok=True)
    rounds.to_csv(OUT / 'development_monitoring_workload.csv', index=False)
    write(OUT / 'monitoring_workload.json', dict(period='2016-2018', decision_interval_days=14, decision_rounds=len(rounds), eligible_lake_periods=len(f), sampled_lake_periods=int(f.sampled.sum()), lake_sample_days=int(counts.sum()), sampled_lake_period_percent=float(100 * f.sampled.mean()), lake_sample_days_per_candidate_percent=float(100 * counts.sum() / len(f)), median_round_sampled_percent=float(rounds.sampled_lake_fraction_percent.median()), interpretation='Historical multi-program workload; contextual support for a 10% comparison scenario'))
    return rounds

def observations():
    f = pd.read_parquet(DATA / 'derived/retrospective_observations.parquet', columns=['lake_id', 'date', 'chla_ug_l']).rename(columns={'chla_ug_l': 'chla'})
    f['lake_id'] = f.lake_id.astype(str)
    f = f[np.isfinite(f.chla) & f.chla.ge(0)].copy()
    return f.sort_values(['lake_id', 'date'], kind='stable').reset_index(drop=True)

def historical_references(obs, spec):
    f = obs[obs.date.add(pd.Timedelta(days=7)).lt(spec['reference_cutoff'])]
    g = f.groupby('lake_id', sort=True)
    ref = g.agg(history_observations=('chla', 'count'), first_date=('date', 'min'), last_date=('date', 'max'))
    ref['span_days'] = (ref.last_date - ref.first_date).dt.days
    ref['reference_supported'] = ref.history_observations.ge(spec['relative_min_observations']) & ref.span_days.ge(spec['relative_min_span_days'])
    for q in spec['relative_quantiles']:
        name = 'historical_q' + str(round(100 * q))
        ref[name] = g.chla.quantile(q)
        ref.loc[~ref.reference_supported, name] = np.nan
    return ref.reset_index()

def reference_observations():
    """Full predictor-history ledger, including the frozen early observations."""
    f = pd.read_parquet(DATA / 'derived/retrospective_chla_history.parquet', columns=['lake_id', 'sample_date', 'value_median', 'field_available_date'])
    if not f.field_available_date.eq(f.sample_date + pd.Timedelta(days=7)).all():
        raise ValueError('Historical reference availability differs')
    f = f.rename(columns={'sample_date': 'date', 'value_median': 'chla'})
    f['lake_id'] = f.lake_id.astype(str)
    return f[np.isfinite(f.chla) & f.chla.ge(0)].sort_values(['lake_id', 'date']).reset_index(drop=True)

def reconstruct_events(obs, threshold, gap=60):
    high = obs.loc[obs.chla.ge(threshold) & pd.notna(threshold), ['lake_id', 'date']].copy()
    elapsed = high.groupby('lake_id', sort=False).date.diff().dt.days
    events = high[elapsed.isna() | elapsed.gt(gap)].rename(columns={'date': 'event_date'})
    events['episode_id'] = events.lake_id + '|' + events.event_date.dt.strftime('%Y-%m-%d')
    return events.reset_index(drop=True)

def queue_frame(spec):
    """Use the saved frozen scores and independently rebuild practical scores."""
    f = pd.read_parquet(RESULTS / 'us/main/predictions.parquet')
    f['lake_id'] = f.lake_id.astype(str)
    f = f.merge(pd.read_parquet(RESULTS / 'us/ablations/predictions.parquet'), on='candidate_row_key', validate='one_to_one')
    h = pd.read_parquet(DATA / 'derived/development_chla_observation_ledger.parquet')
    h['lake_id'] = h.lake_id.astype(str)
    h = h[h.assumed_result_available_date.lt(spec['reference_cutoff'])]
    risk = h.groupby('lake_id').primary_severe.mean()
    training = pd.read_parquet(DATA / 'processed/us_training/common_development_table.parquet', columns=['common_trainable', 'label_mature_date', 'fixed_history_risk'])
    fallback = float(training.loc[training.common_trainable & training.label_mature_date.lt(spec['reference_cutoff']), 'fixed_history_risk'].median())
    f['fixed_history'] = f.lake_id.map(risk).fillna(fallback)
    extra_path = OUT / 'base_predictions.parquet'
    extra_keys = ['xgb4__y_31_60', 'xgb4__y_1_14', 'xgb4__y_15_30', 'xgb4__long_seed1', 'xgb4__long_seed2']
    import pyarrow.parquet as pq
    required_extra = {'matched_xgboost', *extra_keys[1:]}
    if not extra_path.exists() or not required_extra.issubset(pq.read_schema(extra_path).names):
        from .prediction import evaluation_inputs, predict_experts
        pieces = []
        for year in range(2021, 2026):
            frame = evaluation_inputs(year)
            _, values = predict_experts(frame, MODELS / 'us/experts/seasonal', extra_keys)
            piece = frame[['candidate_row_key', 'seasonal_31_60_expected_log_chla']].copy()
            for key in extra_keys:
                piece['matched_xgboost' if key == 'xgb4__y_31_60' else key] = values[key]
            piece = piece.rename(columns={'seasonal_31_60_expected_log_chla': 'seasonal_reference'})
            pieces.append(piece)
            print('US_ANALYSIS_ADDITIONAL_SCORES', year, len(piece), flush=True)
        pd.concat(pieces, ignore_index=True).to_parquet(extra_path, index=False)
    extra = pd.read_parquet(extra_path)
    f = f.merge(extra, on='candidate_row_key', how='left', validate='one_to_one')
    expr = joblib.load(MODELS / 'us/experts/seasonal/common_expression.joblib')
    name = 'seasonal_31_60_expected_log_chla'
    seasonal_fallback = float(expr.median[expr.numeric.index(name)])
    f['seasonal_reference_supported'] = f.seasonal_reference.notna()
    f['seasonal_reference'] = f.seasonal_reference.fillna(seasonal_fallback)
    f[list(spec['methods'])] = f[list(spec['methods'])].astype(float)
    if not np.isfinite(f[list(spec['methods'])].to_numpy(dtype=float)).all():
        raise ValueError('Incomplete US analysis scores')
    return f.sort_values(['origin_date', 'lake_id'], kind='stable').reset_index(drop=True)

def event_hits(mask, ri, ei, size):
    hit = np.zeros(size, bool)
    np.logical_or.at(hit, ei, mask[ri])
    return hit

def run():
    started = time.perf_counter()
    from .prediction import verify_models
    verify_models()
    spec = read(CONFIG / 'analysis.json')
    OUT.mkdir(parents=True, exist_ok=True)
    monitoring_workload()
    f = queue_frame(spec)
    methods = list(spec['methods'])
    obs = observations()
    ref = historical_references(reference_observations(), spec)
    ref.to_csv(OUT / 'historical_thresholds.csv', index=False)
    cases = {}
    for value in spec['absolute_thresholds_ugL']:
        cases['absolute' + str(value)] = reconstruct_events(obs, value, spec['event_gap_days'])
    for q in spec['relative_quantiles']:
        name = 'historical_q' + str(round(100 * q))
        cases[name] = reconstruct_events(obs, obs.lake_id.map(ref.set_index('lake_id')[name]), spec['event_gap_days'])
    supported = ref.loc[ref.reference_supported, 'lake_id']
    cases['absolute75_supported'] = cases['absolute75'][cases['absolute75'].lake_id.isin(supported)].copy()
    cases['absolute75_gap30'] = reconstruct_events(obs, 75, 30)
    cases['absolute75_gap90'] = reconstruct_events(obs, 75, 90)
    event_support = observation_support(obs, cases['absolute75'])
    event_support.to_csv(OUT / 'event_observation_support.csv', index=False)
    cases['absolute75_confirmed2'] = cases['absolute75'][event_support.severe_days_0_13.ge(2)].copy()
    cases['absolute75_recent_normal'] = cases['absolute75'][event_support.normal_days_before30.ge(1)].copy()
    original = pd.read_parquet(DATA / 'processed/us_evaluation/retrospective_episodes.parquet')
    original['lake_id'] = original.lake_id.astype(str)
    keys = ['lake_id', 'event_date']
    rebuilt = cases['absolute75']
    rebuilt = rebuilt[rebuilt.event_date.ge('2021-01-01')]
    if set(map(tuple, rebuilt[keys].to_numpy())) != set(map(tuple, original[keys].to_numpy())):
        raise ValueError('Reconstructed 75 event dates differ from the main study')
    masks = {m: selected_mask(f, f[m], 0.1) for m in methods}
    points, annual, exchanges, ledgers, ops = ([], [], [], [], {})
    for threshold, episodes in cases.items():
        episodes = episodes[episodes.event_date.between('2021-01-01', '2025-12-31')]
        for endpoint, (lo, hi) in spec['endpoints'].items():
            ledger, ri, ei = opportunity_index(f, episodes, lo, hi)
            ledger = ledger.copy()
            for method in methods:
                ledger[method] = event_hits(masks[method], ri, ei, len(ledger))
                points.append(dict(threshold=threshold, endpoint=endpoint, method=method, captured=int(ledger[method].sum()), events=len(ledger), capture_percent=100 * ledger[method].mean()))
                for year, g in ledger.groupby(ledger.event_date.dt.year):
                    annual.append(dict(threshold=threshold, endpoint=endpoint, year=int(year), method=method, captured=int(g[method].sum()), events=len(g), capture_percent=100 * g[method].mean()))
            for proposed, control in spec['comparisons']:
                a, b = (ledger[proposed].to_numpy(), ledger[control].to_numpy())
                exchanges.append(dict(threshold=threshold, endpoint=endpoint, proposed=proposed, comparator=control, events=len(ledger), both=int((a & b).sum()), multi_only=int((a & ~b).sum()), control_only=int((~a & b).sum()), neither=int((~a & ~b).sum()), gain_pp=100 * (a.mean() - b.mean())))
            ops[threshold, endpoint] = (ledger, ri, ei)
            ledgers.append(ledger.assign(threshold=threshold, endpoint=endpoint))
    points = pd.DataFrame(points)
    saved = pd.read_csv(RESULTS / 'us/main/evaluation/event_coverage.csv').set_index(['endpoint', 'method'])
    for row in points[points.threshold.eq('absolute75') & points.endpoint.isin(['31_60', '1_60']) & points.method.isin(['xgboost', 'elastic_net', 'single_window', 'same_window_control', 'multi_window'])].itertuples():
        prior = saved.loc[row.endpoint, row.method]
        if row.captured != prior.captured or row.events != prior.events:
            raise ValueError('Main US point result changed')
    points.to_csv(OUT / 'threshold_sensitivity.csv', index=False)
    pd.DataFrame(annual).to_csv(OUT / 'annual_event_coverage.csv', index=False)
    pd.DataFrame(exchanges).to_csv(OUT / 'event_exchanges.csv', index=False)
    pd.concat(ledgers, ignore_index=True).to_parquet(OUT / 'event_captures.parquet', index=False)
    grid = spec['capacity_basis_points']
    capacities = range(grid['first'], grid['last'] + 1, grid['step'])
    curve = []
    sizes = f.groupby('origin_date').lake_id.transform('size').to_numpy()
    ranks = {method: f.groupby('origin_date')[method].rank(method='first', ascending=False).to_numpy() for method in methods}
    for bp in capacities:
        for method in methods:
            mask = ranks[method] <= np.ceil(sizes * (bp / 10000))
            for endpoint in spec['endpoints']:
                ledger, ri, ei = ops['absolute75', endpoint]
                hits = event_hits(mask, ri, ei, len(ledger))
                curve.append(dict(endpoint=endpoint, method=method, capacity_basis_points=bp, nominal_capacity_percent=bp / 100, selected_slots=int(mask.sum()), candidate_rows=len(f), realized_slot_fraction_percent=100 * mask.mean(), captured=int(hits.sum()), events=len(ledger), capture_percent=100 * hits.mean()))
    curve = pd.DataFrame(curve)
    curve.to_csv(OUT / 'capacity_sensitivity.csv', index=False)
    allocation(f, methods, masks, ops, curve)
    print('US_ANALYSIS_POINTS_COMPLETE', len(curve), len(points), flush=True)
    bootstrap(f, spec, ops, points, curve)
    from .us_structure import run_structural_analysis
    run_structural_analysis(f, spec, ops, masks)
    from .monitoring_schedule import run_schedule
    run_schedule((f, ops))

def observation_support(obs, events):
    groups = {lake: group for lake, group in obs.groupby('lake_id', sort=False)}
    rows = []
    for e in events.itertuples():
        group = groups[e.lake_id]
        delta = (group.date - e.event_date).dt.days
        rows.append(dict(episode_id=e.episode_id, lake_id=e.lake_id, event_date=e.event_date, severe_days_0_13=int((delta.between(0, 13) & group.chla.ge(75)).sum()), normal_days_before30=int((delta.between(-30, -1) & group.chla.lt(75)).sum())))
    return pd.DataFrame(rows)

def allocation(f, methods, masks, ops, curve):
    rows, equivalence = ([], [])
    for method in methods:
        selected = f.loc[masks[method], ['lake_id', 'origin_date']]
        by_lake = selected.groupby('lake_id').size()
        previous = selected.groupby('lake_id').origin_date.diff().dt.days
        slots = len(selected)
        rows.append(dict(method=method, selected_slots=slots, unique_selected_lakes=len(by_lake), candidate_lakes=f.lake_id.nunique(), repeat_slots_within30days=int(previous.le(30).sum()), repeat_slot_percent=100 * previous.le(30).mean(), top10percent_lake_slot_share=100 * by_lake.nlargest(int(np.ceil(0.1 * f.lake_id.nunique()))).sum() / slots, effective_allocated_lakes=slots ** 2 / np.square(by_lake).sum()))
    for threshold, endpoint in ops:
        if threshold != 'absolute75':
            continue
        ledger, _, _ = ops['absolute75', endpoint]
        target = int(ledger.multi_window.sum())
        for method in methods:
            possible = curve[curve.endpoint.eq(endpoint) & curve.method.eq(method) & curve.captured.ge(target)]
            first = possible.sort_values('capacity_basis_points').head(1)
            equivalence.append(dict(endpoint=endpoint, target_method='multi_window', target_capacity_percent=10, target_events=target, comparator=method, minimum_capacity_percent=float(first.nominal_capacity_percent.iloc[0]) if len(first) else np.nan, resolution_percentage_points=0.25))
    pd.DataFrame(rows).to_csv(OUT / 'allocation.csv', index=False)
    pd.DataFrame(equivalence).to_csv(OUT / 'capacity_equivalence.csv', index=False)

def bootstrap(f, spec, ops, points, curve):
    methods = list(spec['methods'])
    lakes = np.sort(f.lake_id.unique())
    row_lake = pd.Index(lakes).get_indexer(f.lake_id).astype(np.int32)
    dates = pd.factorize(f.origin_date, sort=True)[0]
    starts = np.r_[0, np.flatnonzero(dates[1:] != dates[:-1]) + 1].astype(np.int64)
    ends = np.r_[starts[1:], len(f)].astype(np.int64)
    orders = np.array([np.lexsort((f.lake_id.to_numpy(), -f[m].to_numpy(), dates)) for m in methods], np.int64)
    capacities = np.array(spec['interval_capacity_basis_points']) / 10000
    rows, ids, el, slices = ([], [], [], {})
    offset = 0
    for key, (ledger, ri, ei) in ops.items():
        rows.append(ri)
        ids.append(ei + offset)
        el.extend(pd.Index(lakes).get_indexer(ledger.lake_id))
        slices[key] = (offset, offset + len(ledger))
        offset += len(ledger)
    ri = np.concatenate(rows)
    ei = np.concatenate(ids)
    event_lake = np.asarray(el, np.int32)
    bounds = np.array(list(slices.values()), np.int64)
    rng = np.random.default_rng(spec['bootstrap_seed'])
    draws = []
    for rep in range(spec['bootstrap_reps']):
        mult = np.bincount(rng.integers(0, len(lakes), len(lakes)), minlength=len(lakes)).astype(np.int64)
        num, den = bootstrap_endpoint_counts(mult, row_lake, starts, ends, orders, capacities, ri, ei, event_lake, bounds)
        record = {'replicate': rep}
        for k, (threshold, endpoint) in enumerate(slices):
            for c, bp in enumerate(spec['interval_capacity_basis_points']):
                if threshold != 'absolute75' and bp != 1000:
                    continue
                for j, method in enumerate(methods):
                    record[f'{threshold}__{endpoint}__{bp}__{method}'] = num[c, j, k] / den[k] if den[k] else np.nan
        draws.append(record)
        if (rep + 1) % 500 == 0:
            print('US_ANALYSIS_BOOTSTRAP', rep + 1, '/', spec['bootstrap_reps'], flush=True)
    d = pd.DataFrame(draws)
    d.to_parquet(OUT / 'bootstrap_draws.parquet', index=False)
    intervals = []
    for threshold, endpoint in slices:
        budgets = spec['interval_capacity_basis_points'] if threshold == 'absolute75' else [1000]
        for bp in budgets:
            prefix = f'{threshold}__{endpoint}__{bp}__'
            for method in methods:
                values = 100 * d[prefix + method].dropna().to_numpy()
                lo, hi = np.quantile(values, [0.025, 0.975]) if len(values) else [np.nan, np.nan]
                intervals.append(dict(threshold=threshold, endpoint=endpoint, capacity_basis_points=bp, proposed=method, comparator='', lower95=lo, upper95=hi))
            for proposed, method in spec['comparisons']:
                values = 100 * (d[prefix + proposed] - d[prefix + method]).dropna().to_numpy()
                lo, hi = np.quantile(values, [0.025, 0.975]) if len(values) else [np.nan, np.nan]
                intervals.append(dict(threshold=threshold, endpoint=endpoint, capacity_basis_points=bp, proposed=proposed, comparator=method, lower95=lo, upper95=hi))
    pd.DataFrame(intervals).to_csv(OUT / 'supplement_intervals.csv', index=False)

@njit
def bootstrap_endpoint_counts(mult, row_lake, starts, ends, orders, capacities, ri, ei, event_lake, bounds):
    before = np.empty((len(orders), len(row_lake)), np.int64)
    total = np.empty(len(row_lake), np.int64)
    for start, end in zip(starts, ends):
        n = 0
        for row in range(start, end):
            n += mult[row_lake[row]]
        for row in range(start, end):
            total[row] = n
        for method in range(len(orders)):
            used = 0
            for pos in range(start, end):
                row = orders[method, pos]
                before[method, row] = used
                used += mult[row_lake[row]]
    numerator = np.zeros((len(capacities), len(orders), len(bounds)), np.int64)
    denominator = np.zeros(len(bounds), np.int64)
    for k in range(len(bounds)):
        for e in range(bounds[k, 0], bounds[k, 1]):
            denominator[k] += mult[event_lake[e]]
    for c in range(len(capacities)):
        hits = np.zeros((len(orders), len(event_lake)), np.int64)
        for j in range(len(ri)):
            row, event = (ri[j], ei[j])
            quota = int(np.ceil(capacities[c] * total[row]))
            copies = mult[row_lake[row]]
            for method in range(len(orders)):
                selected = min(copies, max(0, quota - before[method, row]))
                hits[method, event] = max(hits[method, event], selected)
        for method in range(len(orders)):
            for k in range(len(bounds)):
                numerator[c, method, k] = hits[method, bounds[k, 0]:bounds[k, 1]].sum()
    return (numerator, denominator)
if __name__ == '__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        run()
