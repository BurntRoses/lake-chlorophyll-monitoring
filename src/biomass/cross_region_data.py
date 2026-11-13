"""Common regional inputs with complete, historically eligible capacity queues."""
import hashlib
import time
from pathlib import Path
import numpy as np
import pandas as pd
from .paths import DATA, CONFIG
from .io import read, write, digest
from . import seasonal

DEST = DATA / 'processed/external_absolute'
LABELS = ['y_1_14', 'y_15_30', 'y_31_60']
BINS = [(1, 14), (15, 30), (31, 60)]


def specification():
    return read(CONFIG / 'external.json')


def features():
    spec = specification()
    return spec['features'] + (seasonal.FEATURES if spec['seasonal_features'] else [])


def calendar(first, last):
    spec = specification()
    return pd.date_range(spec['calendar_anchor'], last, freq=f"{spec['spacing_days']}D").to_numpy(dtype='datetime64[D]')[
        pd.date_range(spec['calendar_anchor'], last, freq=f"{spec['spacing_days']}D") >= pd.Timestamp(first)]


def episode_days(days, values, threshold=None):
    high = days[np.asarray(values) >= (specification()['threshold_ugL'] if threshold is None else threshold)]
    return high[np.r_[True, np.diff(high) > specification()['washout_days']]] if len(high) else high


def labels_for(origins, days, event_days):
    j = np.searchsorted(event_days, origins, side='right')
    delta = np.full(len(origins), np.inf)
    good = j < len(event_days)
    delta[good] = event_days[j[good]] - origins[good]
    labels = np.full((len(origins), 3), -1, np.int8)
    for h, (lo, hi) in enumerate(BINS):
        support = np.searchsorted(days, origins + hi, side='right') - np.searchsorted(days, origins + lo)
        labels[(support > 0) | ((delta >= 1) & (delta <= 60)), h] = 0
        labels[(delta >= lo) & (delta <= hi), h] = 1
    return labels, j


def build_lake(lake, obs, origins, morphology, threshold=None, exclude_known_active=True):
    spec = specification()
    threshold = spec['threshold_ugL'] if threshold is None else threshold
    obs = obs.sort_values('date')
    dates = obs.date.to_numpy(dtype='datetime64[D]')
    days = dates.astype(np.int64)
    values = obs.chla.to_numpy(float)
    event = episode_days(days, values, threshold)
    available = np.searchsorted(days + spec['result_delay_days'], origins.astype(np.int64), side='left')
    origins = origins[available > 0]
    if not len(origins):
        return None, None
    years = pd.DatetimeIndex(origins).year.to_numpy()
    history = seasonal.histories(origins, years, dates, values, threshold)
    frame = pd.DataFrame(history, columns=seasonal.HISTORY)
    # The same known-active exclusion is used by the US main analysis.
    eligible = ~(frame.known_severe_age.notna() & frame.known_severe_age.le(spec['washout_days'])) if exclude_known_active else np.ones(len(frame),bool)
    frame = frame.loc[eligible].reset_index(drop=True)
    origins = origins[eligible]
    if not len(origins):
        return None, None
    oi = origins.astype(np.int64)
    end = np.searchsorted(days + spec['result_delay_days'], oi, side='left')
    for variable in ['temperature', 'turbidity']:
        a = obs[variable].to_numpy(float) if variable in obs else np.full(len(obs), np.nan)
        valid = np.isfinite(a)
        sums = np.r_[0., np.cumsum(np.where(valid, a, 0.))]
        counts = np.r_[0, np.cumsum(valid)]
        for window in [14, 30, 90]:
            low = np.minimum(np.searchsorted(days, oi - window), end)
            n = counts[end] - counts[low]
            frame[f'{variable}_mean{window}'] = np.divide(sums[end] - sums[low], n,
                out=np.full(len(oi), np.nan), where=n > 0)
        frame[variable + '_latest'] = a[end - 1]
    for window in [14, 30]:
        frame[f'temperature_shift{window}_90'] = frame[f'temperature_mean{window}'] - frame.temperature_mean90
        frame[f'turbidity_relative{window}_90'] = np.log1p(frame[f'turbidity_mean{window}']) - np.log1p(frame.turbidity_mean90)
    for source, target in [('Lake_area', 'log_area'), ('Depth_avg', 'log_depth'),
                           ('Res_time', 'log_residence_days'), ('Vol_total', 'log_volume')]:
        value = morphology.get(source, np.nan)
        frame[target] = np.log1p(value) if np.isfinite(value) and value >= 0 else np.nan
    frame['lake_type'] = morphology.get('Lake_type', np.nan)
    latitude = float(obs.lat.median())
    frame['abs_lat'] = abs(latitude)
    phase = 2 * np.pi * pd.DatetimeIndex(origins).dayofyear.to_numpy() / 365.25
    sign = -1 if latitude < 0 else 1
    frame['season_sin'] = sign * np.sin(phase)
    frame['season_cos'] = sign * np.cos(phase)
    if spec['seasonal_features']:
        hist = seasonal.history_features(origins, dates, values, frame.latest.to_numpy(), frame.mean90.to_numpy(), threshold)
        frame = pd.concat([frame, pd.DataFrame(hist, columns=seasonal.FEATURES)], axis=1)
    lab, j = labels_for(oi, days, event)
    ids = np.array([str(lake) + '|' + str(np.datetime64(int(day), 'D')) for day in event], object)
    following = np.full(len(frame), None, object)
    has = j < len(ids)
    following[has] = ids[j[has]]
    for h, label in enumerate(LABELS):
        frame[label] = lab[:, h]
    frame['lake_id'] = str(lake)
    frame['origin_date'] = pd.to_datetime(origins)
    frame['origin_year'] = frame.origin_date.dt.year.astype(np.int16)
    frame['candidate_row_key'] = str(lake) + '|' + frame.origin_date.dt.strftime('%Y-%m-%d')
    frame['next_episode_id'] = following
    frame['common_trainable'] = frame.y_31_60.ge(0)
    frame['label_mature_date'] = frame.origin_date + pd.Timedelta(days=spec['label_maturity_days'])
    frame['eligible_common_development_candidate'] = True
    frame['outer_lake_fold'] = int(hashlib.sha256((spec['fold_hash_salt'] + '|' + str(lake)).encode()).hexdigest()[:12], 16) % spec['lake_folds']
    frame['country'] = str(obs.country.iloc[0])
    frame['continent'] = str(obs.continent.iloc[0])
    frame['ecoregion'] = '__not_available__'
    for name in features():
        a = frame[name].to_numpy(float)
        a[~np.isfinite(a)] = np.nan
        frame[name] = a.astype(np.float32)
    events = pd.DataFrame({'lake_id': str(lake), 'event_date': pd.to_datetime(event, unit='D'), 'episode_id': ids})
    return frame, events


def build_table(obs, morphology, first, last, cohort_cutoff, folder, event_start=None, event_end=None):
    folder.mkdir(parents=True, exist_ok=True)
    origins = calendar(first, last)
    parts, events = [], []
    spec = specification()
    cohort = obs.loc[obs.date.add(pd.Timedelta(days=spec['result_delay_days'])).lt(pd.Timestamp(cohort_cutoff)), 'lake_id'].unique()
    attributes = morphology.set_index('lake_id')
    if not attributes.index.is_unique:
        raise ValueError('Repeated physical lake in morphology')
    started = time.perf_counter()
    for i, (lake, group) in enumerate(obs[obs.lake_id.isin(cohort)].groupby('lake_id', sort=True)):
        m = attributes.loc[lake].to_dict() if lake in attributes.index else {}
        f, e = build_lake(lake, group, origins, m)
        if f is not None:
            parts.append(f)
            events.append(e)
        if (i + 1) % 500 == 0:
            print('INPUT_LAKES', folder.name, i + 1, round(time.perf_counter() - started, 1), flush=True)
    frame = pd.concat(parts, ignore_index=True).sort_values(['origin_date', 'lake_id'], kind='stable').reset_index(drop=True)
    ledger = pd.concat(events, ignore_index=True)
    if event_start:
        ledger = ledger[ledger.event_date.ge(event_start) & ledger.event_date.le(event_end)].reset_index(drop=True)
    if not frame.candidate_row_key.is_unique or not ledger.episode_id.is_unique:
        raise ValueError('Repeated candidate or episode identity')
    frame.to_parquet(folder / 'candidates.parquet', index=False)
    ledger.to_parquet(folder / 'episodes.parquet', index=False)
    row = dict(panel=folder.name, candidate_rows=len(frame), lakes=int(frame.lake_id.nunique()),
               origin_dates=int(frame.origin_date.nunique()), first_origin=str(frame.origin_date.min()),
               last_origin=str(frame.origin_date.max()), events=len(ledger),
               known_long_rows=int(frame.common_trainable.sum()), unknown_long_rows=int((~frame.common_trainable).sum()),
               seconds=time.perf_counter() - started)
    write(folder / 'cohort_statistics.json', row)
    print('INPUT_COMPLETE', row, flush=True)
    return row


def observation_support():
    rows = []
    for name in ['source'] + list(specification()['panels']):
        f = pd.read_parquet(DEST / name / 'candidates.parquet')
        row = dict(panel=name, lakes=int(f.lake_id.nunique()), countries=int(f.country.nunique()),
            rows=len(f), known_long=int(f.common_trainable.sum()),
            median_observation_age_days=float(f.age.median()))
        for label, column in [('temperature', 'temperature_mean30'), ('turbidity', 'turbidity_mean30'),
                              ('area', 'log_area'), ('long_seasonal', 'seasonal_31_60_expected_log_chla')]:
            row['fraction_' + label + '_available'] = float(f[column].notna().mean())
        rows.append(row)
    pd.DataFrame(rows).to_csv(DEST / 'observation_support.csv', index=False)


def build_inputs():
    spec = specification()
    DEST.mkdir(parents=True, exist_ok=True)
    signature = digest(CONFIG / 'external.json')
    state = DEST / 'cohort_statistics.json'
    if state.exists():
        if read(state)['config_sha256'] != signature:
            raise ValueError('Input configuration changed; use an explicit rebuild')
        print('Using completed common regional inputs', flush=True)
        return
    satellite = pd.read_parquet(DATA / 'derived/cross_region_satellite_observations.parquet')
    morphology = pd.read_csv(DATA / 'raw/cross_region/cci_v2/lake_morphology.csv')
    cutoff = pd.Timestamp(spec['training_cutoff'])
    source = satellite[~satellite.country.isin(spec['excluded_source_countries']) &
        ~satellite.continent.isin(spec['excluded_source_continents']) &
        satellite.date.add(pd.Timedelta(days=spec['result_delay_days'])).lt(cutoff)]
    summaries = [build_table(source, morphology, spec['source_first_origin'], spec['source_last_origin'],
                             spec['training_cutoff'], DEST / 'source', spec['source_first_origin'], spec['source_last_origin'])]
    for name in ['china_2020_2022', 'oceania']:
        panel = spec['panels'][name]
        obs = satellite[satellite.country.eq(panel['country'])] if 'country' in panel else satellite[satellite.continent.eq(panel['continent'])]
        summaries.append(build_table(obs, morphology, panel['first_origin'], panel['last_origin'],
            spec['training_cutoff'], DEST / name, panel['event_start'], panel['event_end']))
    old = satellite[satellite.country.eq('China')].copy()
    cohort = set(old.loc[old.date.add(pd.Timedelta(days=7)).lt(cutoff), 'lake_id'])
    pieces = []
    for path in sorted((DATA / 'raw/cross_region/cci_v3/daily_2023').glob('*.parquet')):
        daily = pd.read_parquet(path)
        pieces.append(daily[daily.usable_chla & daily.lake_id.isin(cohort)][['lake_id', 'date', 'chla', 'temperature', 'turbidity']])
    combined = pd.concat([old, *pieces], ignore_index=True).sort_values(['lake_id', 'date'])
    metadata = old.groupby('lake_id').first()
    for col in ['country', 'continent', 'lat', 'lon']:
        combined[col] = combined.lake_id.map(metadata[col])
    panel = spec['panels']['china_2023']
    summaries.append(build_table(combined, morphology, panel['first_origin'], panel['last_origin'],
        spec['training_cutoff'], DEST / 'china_2023', panel['event_start'], panel['event_end']))
    field = pd.read_parquet(DATA / 'derived/cross_region_field_observations.parquet')
    field['country'] = 'New Zealand'
    field['continent'] = 'Oceania'
    field['temperature'] = field['turbidity'] = np.nan
    field_morph = pd.read_csv(DATA / 'raw/cross_region/lawa_nz/lake_morphology.csv')
    panel = spec['panels']['new_zealand']
    summaries.append(build_table(field, field_morph, panel['first_origin'], panel['last_origin'],
        '2023-01-01', DEST / 'new_zealand', panel['event_start'], panel['event_end']))
    pd.DataFrame(summaries).to_csv(DEST / 'cohort_statistics.csv', index=False)
    observation_support()
    write(state, dict(config_sha256=signature, summaries=summaries))
