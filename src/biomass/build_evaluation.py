"""Build retrospective predictors from strictly available source histories."""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from .paths import ROOT, CONFIG, DATA
from .datasets import input_paths
from . import biomass_features, meteorology_features

def source_fields(frame, history, source, names, reference, availability, observation, value, quality, censor=None, sites=None, threshold=0):
    added = {name: np.full(len(frame), np.nan, dtype='float32') for name in names if name not in frame}
    history['lake_id'] = history.lake_id.astype(str)
    history_groups = dict(tuple(history.groupby('lake_id', sort=False)))
    empty = history.iloc[:0].copy()
    checks = 0
    for i, (lake, group) in enumerate(frame.groupby('lake_id', sort=False)):
        h = history_groups.get(lake, empty)
        computed = reference.compute_source_features(source=source, origins=group.origin_date.to_numpy(), history=h, availability_col=availability, observation_col=observation, value_col=value, quality_col=quality, censor_col=censor, site_count_col=sites, positive_threshold=threshold, windows=(14, 30, 60, 90))
        ix = group.index.to_numpy()
        for name in names:
            if name not in computed:
                raise RuntimeError(f'Feature reconstruction missing {name}')
            if name in added:
                added[name][ix] = computed[name]
            elif len(h):
                actual = group[name].to_numpy(dtype=float, na_value=np.nan)
                expected = np.asarray(computed[name], dtype=float)
                if not np.allclose(actual, expected, rtol=2e-05, atol=2e-05, equal_nan=True):
                    bad = int((~np.isclose(actual, expected, rtol=2e-05, atol=2e-05, equal_nan=True)).sum())
                    raise RuntimeError(f'Material source mismatch in {source}/{lake}/{name}: {bad} rows')
                checks += len(group)
        for name, values in computed.items():
            if name.endswith('_availability_date') or name.endswith('_observation_date'):
                dates = pd.to_datetime(values)
                if (dates >= group.origin_date.to_numpy()).any():
                    raise RuntimeError(f'After-origin reconstructed source date: {source}/{name}')
        if (i + 1) % 500 == 0:
            print('RECONSTRUCTED_SOURCE_LAKES', source, i + 1, flush=True)
    if added:
        frame = pd.concat([frame, pd.DataFrame(added, index=frame.index)], axis=1)
    return (frame, {'source': source, 'added_features': len(added), 'overlapping_row_values_checked': checks})

def meteorology_fields(frame, paths, names, reference):
    inventory_path = Path(paths['retrospective_meteorology_inventory'])
    folder = inventory_path.parent
    inventory = json.loads(inventory_path.read_text(encoding='utf-8'))
    parts = []
    sources = []
    start = (frame.origin_date.min() - pd.Timedelta(days=90)).strftime('%Y-%m-%d')
    end = (frame.origin_date.max() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    entries = [entry for entry in inventory['days'] if start <= entry['date'] <= end]
    for i, entry in enumerate(entries):
        p = folder / 'daily' / f"{entry['date']}.parquet"
        if not p.is_file():
            raise RuntimeError(f"Local daily meteorology source absent: {entry['date']}")
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest != entry['parquet_sha256']:
            raise RuntimeError(f'Daily meteorology hash differs: {p}')
        parts.append(pd.read_parquet(p))
        sources.append({'date': entry['date'], 'path': p.relative_to(ROOT).as_posix(), 'sha256': digest})
        if (i + 1) % 250 == 0:
            print('LOCAL_METEOROLOGY_CACHE_READ', i + 1, '/', len(entries), flush=True)
    daily = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    if daily.duplicated(['era5_grid_id', 'date']).any():
        raise RuntimeError('Duplicate daily meteorology keys')
    precip = daily.precipitation_mm.to_numpy(dtype=float)
    if (precip < -1e-10).any():
        raise RuntimeError('Substantive negative precipitation in cached source')
    daily['precipitation_mm'] = np.where(np.abs(precip) <= 1e-10, 0, precip)
    source_groups = dict(tuple(daily.groupby('era5_grid_id', sort=False)))
    added = {name: np.full(len(frame), np.nan, dtype='float32') for name in names if name not in frame}
    checks = 0
    for i, (grid, group) in enumerate(frame.groupby('era5_grid_id', sort=False)):
        if grid not in source_groups:
            raise RuntimeError(f'Missing meteorology grid: {grid}')
        unique = group.origin_date.drop_duplicates().sort_values()
        computed = reference.compute_group_features(unique.to_numpy(), source_groups[grid])
        positions = pd.Index(unique).get_indexer(group.origin_date)
        ix = group.index.to_numpy()
        for name in names:
            if name not in computed:
                raise RuntimeError(f'Missing met reconstruction: {name}')
            values = np.asarray(computed[name])[positions]
            if name in added:
                added[name][ix] = values
            else:
                actual = group[name].to_numpy(dtype=float, na_value=np.nan)
                if not np.allclose(actual, values, rtol=2e-05, atol=2e-05, equal_nan=True):
                    raise RuntimeError(f'Meteorology source mismatch: {grid}/{name}')
                checks += len(group)
        if (i + 1) % 200 == 0:
            print('RECONSTRUCTED_METEOROLOGY_GRIDS', i + 1, flush=True)
    frame = pd.concat([frame, pd.DataFrame(added, index=frame.index)], axis=1)
    return (frame, {'source': 'meteorology', 'added_features': len(added), 'overlapping_row_values_checked': checks, 'source_days_verified': len(sources)}, sources)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, choices=range(2021, 2026))
    parser.add_argument('--output-root', type=Path, default=ROOT)
    args = parser.parse_args()
    output = args.output_root.resolve()
    paths = input_paths()
    spec = pd.read_csv(CONFIG / 'features.csv')
    frame = pd.read_parquet(paths['retrospective_candidate_ledger'])
    frame['lake_id'] = frame.lake_id.astype(str)
    if frame.candidate_row_key.duplicated().any() or frame.duplicated(['lake_id', 'origin_date']).any():
        raise RuntimeError('Retrospective candidate keys duplicated')
    if args.year:
        frame = frame[frame.origin_year.eq(args.year)].reset_index(drop=True)
    expected_rows = len(frame)
    suffix = f'_{args.year}' if args.year else ''
    names = spec.feature_name.tolist()
    for asset in ['retrospective_field_features', 'retrospective_wq_features', 'retrospective_meteorology_features', 'retrospective_satellite_features']:
        schema = pq.ParquetFile(paths[asset]).schema_arrow.names
        needed = [name for name in names if name in schema and name not in frame]
        if asset == 'retrospective_meteorology_features':
            needed += ['era5_grid_id']
        data = pd.read_parquet(paths[asset], columns=['lake_id', 'origin_date'] + needed)
        data['lake_id'] = data.lake_id.astype(str)
        frame = frame.merge(data, on=['lake_id', 'origin_date'], how='left', validate='one_to_one')
        if len(frame) != expected_rows:
            raise RuntimeError('Retrospective merge changed row count')
    frame = frame.reset_index(drop=True)
    history = pd.read_parquet(paths['development_chla_observation_ledger'])
    history['lake_id'] = history.lake_id.astype(str)
    prior = history[history.assumed_result_available_date.lt('2021-01-01')]
    h = prior.groupby('lake_id').primary_severe.agg(['size', 'sum'])
    frame['fixed_history_risk'] = frame.lake_id.map(h['sum'] / h['size']).astype('float32')
    frame['fixed_history_observation_count'] = frame.lake_id.map(h['size']).fillna(0).astype('float32')
    frame['fixed_history_severe_count'] = frame.lake_id.map(h['sum']).fillna(0).astype('float32')
    del history, prior
    gc.collect()
    window = biomass_features
    met = meteorology_features
    diagnostics = []
    wq = pd.read_parquet(paths['retrospective_wq_history'])
    wq['available_date'] = wq.sample_date + pd.Timedelta(days=7)
    for variable in ['tn', 'tp', 'do', 'ph', 'conductivity', 'turbidity', 'secchi_depth']:
        prefix = f'wq_{variable}'
        needed = [name for name in names if name.startswith(prefix + '_')]
        frame, record = source_fields(frame, wq[wq.variable.eq(variable)].copy(), prefix, needed, window, 'available_date', 'sample_date', 'value_median', 'record_quality_fraction', 'censored_fraction_among_numeric_valid', 'site_count')
        diagnostics.append(record)
    del wq
    gc.collect()
    weekly = pd.read_parquet(paths['retrospective_cyan_weekly_history'])
    weekly['availability_date'] = weekly.week_end + pd.Timedelta(days=2)
    weekly['bridge_quality'] = 1.0
    needed = [name for name in names if name.startswith('satellite_cyan_')]
    frame, record = source_fields(frame, weekly, 'satellite_cyan', needed, window, 'availability_date', 'week_end', 'cell_dn_raw', 'bridge_quality')
    diagnostics.append(record)
    del weekly
    gc.collect()
    needed = spec.loc[spec.source.eq('development_meteorology_features'), 'feature_name'].tolist()
    frame, record, daily_sources = meteorology_fields(frame, paths, needed, met)
    diagnostics.append(record)
    missing = set(names) - set(frame)
    if missing:
        raise RuntimeError(f'Retrospective features incomplete: {sorted(missing)}')
    for name in names:
        if name != 'ecoregion':
            frame[name] = pd.to_numeric(frame[name], errors='raise').astype('float32')
    events = pd.read_csv(paths['retrospective_episode_ledger'], dtype={'lake_id': str}, parse_dates=['event_date'])
    if events.episode_id.duplicated().any():
        raise RuntimeError('Retrospective episode IDs duplicated')
    if not events.event_chla_ug_l.ge(75).all():
        raise RuntimeError('Retrospective severe threshold differs')
    metadata = ['candidate_row_key', 'lake_id', 'origin_date', 'origin_year', 'outer_lake_fold', 'eligible_external_candidate', 'lake_centroidstate', 'era5_grid_id']
    dest = output / 'data/processed/us_evaluation'
    dest.mkdir(parents=True, exist_ok=True)
    frame[list(dict.fromkeys(metadata + names))].to_parquet(dest / f'common_retrospective_table{suffix}.parquet', index=False)
    events.to_parquet(dest / 'retrospective_episodes.parquet', index=False)
    out = output / 'data/processed/us_evaluation'
    out.mkdir(parents=True, exist_ok=True)
    coverage = pd.DataFrame({'feature_name': names, 'nonnull': frame[names].notna().sum().to_numpy()})
    coverage.to_csv(out / f'retrospective_feature_coverage{suffix}.csv', index=False)
    result = {'status': 'passed', 'rows': len(frame), 'eligible_rows': int(frame.eligible_external_candidate.sum()), 'lakes': frame.lake_id.nunique(), 'origins': frame.origin_date.nunique(), 'feature_pool': len(names), 'history_cutoff': 'available before 2021-01-01, fixed throughout evaluation', 'event_rows': len(events), 'structural_missing_water_quality_variables': ['tn', 'turbidity'], 'satellite_contract': 'inherited operational zero for absent API output; not measured zero concentration'}
    print(json.dumps(result, indent=2), flush=True)
if __name__ == '__main__':
    main()
