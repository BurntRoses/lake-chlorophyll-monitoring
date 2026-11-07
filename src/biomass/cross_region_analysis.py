"""Common-framework regional sensitivity and observation-support analysis."""
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from .paths import DATA, CONFIG
from .io import read, write, digest
from .cross_region_data import DEST, specification
from .cross_region_training import RESULT, MODEL, CROSSFIT
from .evaluation import selected_mask, opportunity_index
from .us_analysis import historical_references, reconstruct_events, observation_support, event_hits, bootstrap_endpoint_counts

def panel_observations(name):
    """Return the same observation history used to construct each regional input."""
    spec = specification()
    if name == 'new_zealand':
        return pd.read_parquet(DATA / 'derived/cross_region_field_observations.parquet')
    obs = pd.read_parquet(DATA / 'derived/cross_region_satellite_observations.parquet')
    panel = spec['panels'][name]
    obs = obs[obs.country.eq(panel['country'])] if 'country' in panel else obs[obs.continent.eq(panel['continent'])]
    if name == 'china_2023':
        cohort = set(obs.loc[obs.date.add(pd.Timedelta(days=spec['result_delay_days'])).lt(spec['training_cutoff']), 'lake_id'])
        parts = [obs[['lake_id', 'date', 'chla']]]
        for path in sorted((DATA / 'raw/cross_region/cci_v3/daily_2023').glob('*.parquet')):
            daily = pd.read_parquet(path)
            parts.append(daily.loc[daily.usable_chla & daily.lake_id.isin(cohort), ['lake_id', 'date', 'chla']])
        obs = pd.concat(parts, ignore_index=True)
    obs = obs[['lake_id', 'date', 'chla']].sort_values(['lake_id', 'date']).reset_index(drop=True)
    if obs.duplicated(['lake_id', 'date']).any() or not np.isfinite(obs.chla).all():
        raise ValueError('Invalid regional lake-day observations')
    return obs

def analyze_panel(name, frame):
    """Apply the US fixed-score capacity and event-definition analyses unchanged."""
    from .cross_region_evaluation import METHODS, ENDPOINTS
    spec = specification()
    analysis = read(CONFIG / 'analysis.json')
    output = RESULT / name
    frame = frame.sort_values(['origin_date', 'lake_id']).reset_index(drop=True)
    obs = panel_observations(name)
    obs = obs[obs.lake_id.isin(frame.lake_id.unique())].reset_index(drop=True)
    reference_spec = dict(analysis, reference_cutoff=spec['training_cutoff'])
    refs = historical_references(obs, reference_spec)
    refs.to_csv(output / 'historical_thresholds.csv', index=False)
    cases = {f'absolute{v}': reconstruct_events(obs, v, analysis['event_gap_days']) for v in analysis['absolute_thresholds_ugL']}
    for q in analysis['relative_quantiles']:
        key = 'historical_q' + str(round(100 * q))
        cases[key] = reconstruct_events(obs, obs.lake_id.map(refs.set_index('lake_id')[key]), analysis['event_gap_days'])
    supported = refs.loc[refs.reference_supported, 'lake_id']
    cases['absolute75_supported'] = cases['absolute75'][cases['absolute75'].lake_id.isin(supported)].copy()
    for gap in [30, 90]:
        cases[f'absolute75_gap{gap}'] = reconstruct_events(obs, 75, gap)
    event_support = observation_support(obs, cases['absolute75'])
    event_support.to_csv(output / 'event_observation_support.csv', index=False)
    cases['absolute75_confirmed2'] = cases['absolute75'][event_support.severe_days_0_13.ge(2)].copy()
    cases['absolute75_recent_normal'] = cases['absolute75'][event_support.normal_days_before30.ge(1)].copy()
    panel = spec['panels'][name]
    cases = {k: v[v.event_date.between(panel['event_start'], panel['event_end'])].reset_index(drop=True) for k, v in cases.items()}
    saved = pd.read_parquet(DEST / name / 'episodes.parquet')
    if set(cases['absolute75'].episode_id) != set(saved.episode_id):
        raise ValueError(f'Reconstructed primary events changed in {name}')
    masks = {m: selected_mask(frame, frame[m], spec['capacity']) for m in METHODS}
    points, ops, ledgers = ([], {}, [])
    for threshold, events in cases.items():
        for endpoint, (lo, hi) in ENDPOINTS.items():
            ledger, ri, ei = opportunity_index(frame, events, lo, hi)
            for m in METHODS:
                hits = event_hits(masks[m], ri, ei, len(ledger))
                ledger[m] = hits
                points.append(dict(threshold=threshold, endpoint=endpoint, method=m, captured=int(hits.sum()), events=len(ledger), capture_percent=float(100 * hits.mean()) if len(hits) else np.nan))
            ops[threshold, endpoint] = (ledger, ri, ei)
            ledgers.append(ledger.assign(threshold=threshold, endpoint=endpoint))
    pd.DataFrame(points).to_csv(output / 'threshold_sensitivity.csv', index=False)
    pd.concat(ledgers, ignore_index=True).to_parquet(output / 'sensitivity_event_captures.parquet', index=False)
    grid = analysis['capacity_basis_points']
    curve = []
    sizes = frame.groupby('origin_date').lake_id.transform('size').to_numpy()
    ranks = {m: frame.groupby('origin_date')[m].rank(method='first', ascending=False).to_numpy() for m in METHODS}
    for bp in range(grid['first'], grid['last'] + 1, grid['step']):
        for m in METHODS:
            mask = ranks[m] <= np.ceil(sizes * (bp / 10000))
            for endpoint in ENDPOINTS:
                ledger, ri, ei = ops['absolute75', endpoint]
                hits = event_hits(mask, ri, ei, len(ledger))
                curve.append(dict(endpoint=endpoint, method=m, capacity_basis_points=bp, nominal_capacity_percent=bp / 100, selected_slots=int(mask.sum()), candidate_rows=len(frame), realized_slot_fraction_percent=100 * mask.mean(), captured=int(hits.sum()), events=len(ledger), capture_percent=float(100 * hits.mean()) if len(hits) else np.nan))
    pd.DataFrame(curve).to_csv(output / 'capacity_sensitivity.csv', index=False)
    _sensitivity_bootstrap(frame, ops, output, analysis, METHODS)

def _sensitivity_bootstrap(frame, ops, output, analysis, methods):
    spec = specification()
    lakes = np.sort(frame.lake_id.unique())
    row_lake = pd.Index(lakes).get_indexer(frame.lake_id).astype(np.int32)
    dates = pd.factorize(frame.origin_date, sort=True)[0]
    starts = np.r_[0, np.flatnonzero(dates[1:] != dates[:-1]) + 1].astype(np.int64)
    ends = np.r_[starts[1:], len(frame)].astype(np.int64)
    orders = np.asarray([np.lexsort((frame.lake_id.to_numpy(), -frame[m].to_numpy(), dates)) for m in methods], np.int64)
    rows, ids, event_lakes, bounds = ([], [], [], [])
    offset = 0
    for ledger, ri, ei in ops.values():
        rows.append(ri)
        ids.append(ei + offset)
        event_lakes.extend(pd.Index(lakes).get_indexer(ledger.lake_id))
        bounds.append((offset, offset + len(ledger)))
        offset += len(ledger)
    ri, ei = (np.concatenate(rows), np.concatenate(ids))
    event_lakes, bounds = (np.asarray(event_lakes, np.int32), np.asarray(bounds, np.int64))
    bps = analysis['interval_capacity_basis_points']
    budgets = np.asarray(bps) / 10000
    keys = [(k, bp, m) for k in ops for bp in (bps if k[0] == 'absolute75' else [1000]) for m in methods]
    draws = np.empty((spec['bootstrap_reps'], len(keys)), float)
    rng = np.random.default_rng(spec['bootstrap_seed'])
    for rep in range(len(draws)):
        mult = np.bincount(rng.integers(0, len(lakes), len(lakes)), minlength=len(lakes)).astype(np.int64)
        num, den = bootstrap_endpoint_counts(mult, row_lake, starts, ends, orders, budgets, ri, ei, event_lakes, bounds)
        pos = 0
        for k, (threshold, endpoint) in enumerate(ops):
            for c, bp in enumerate(bps):
                if threshold != 'absolute75' and bp != 1000:
                    continue
                for j, m in enumerate(methods):
                    draws[rep, pos] = num[c, j, k] / den[k] if den[k] else np.nan
                    pos += 1
        if (rep + 1) % 1000 == 0:
            print('REGIONAL_SENSITIVITY', output.name, rep + 1, '/', len(draws), flush=True)
    names = [f'{t}__{e}__{bp}__{m}' for (t, e), bp, m in keys]
    d = pd.DataFrame(draws, columns=names)
    d.to_parquet(output / 'sensitivity_draws.parquet', index=False)
    primary = pd.read_parquet(output / 'bootstrap_draws.parquet')
    for column in primary:
        if not np.array_equal(primary[column], d['absolute75__' + column.replace('__', '__1000__', 1)], equal_nan=True):
            raise ValueError('Primary and sensitivity bootstrap calculations differ')
    comparisons = [p for p in analysis['comparisons'] if p[0] in methods and p[1] in methods]
    intervals = []
    for threshold, endpoint in ops:
        for bp in bps if threshold == 'absolute75' else [1000]:
            prefix = f'{threshold}__{endpoint}__{bp}__'
            for proposed, comparator in [(m, '') for m in methods] + [tuple(p) for p in comparisons]:
                values = d[prefix + proposed] - (d[prefix + comparator] if comparator else 0)
                values = 100 * values.dropna().to_numpy()
                lower, upper = np.quantile(values, [0.025, 0.975]) if len(values) else [np.nan, np.nan]
                intervals.append(dict(threshold=threshold, endpoint=endpoint, capacity_basis_points=bp, proposed=proposed, comparator=comparator, lower95=lower, upper95=upper, valid_replicates=len(values)))
    pd.DataFrame(intervals).to_csv(output / 'supplement_intervals.csv', index=False)
