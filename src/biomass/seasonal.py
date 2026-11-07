"""Strictly available history and prior-year seasonal features."""
import numpy as np
import pandas as pd

def known_state(origins, years, days, concentrations, threshold):
    """Apply the seven-day observation-availability delay."""
    risks = np.full(len(origins), np.nan, np.float32)
    ages = np.full(len(origins), np.nan, np.float32)
    counts = np.zeros(len(origins), np.float32)
    boundaries = np.asarray([np.datetime64(f'{year}-01-01', 'D').astype(np.int64)
                             for year in years])
    n = np.searchsorted(days + 7, boundaries, side='left')
    counts[:] = n
    if not np.isfinite(threshold):
        return risks, ages, counts
    cumulative = np.r_[0, np.cumsum(concentrations >= threshold)]
    np.divide(cumulative[n], n, out=risks, where=n > 0)
    severe = days[concentrations >= threshold]
    if len(severe):
        previous = np.searchsorted(severe + 7, origins, side='left') - 1
        known = previous >= 0
        ages[known] = origins[known] - severe[previous[known]]
    return risks, ages, counts

HISTORY = ['latest', 'mean14', 'mean30', 'mean90', 'slope30', 'slope90', 'age', 'sample_count90',
    'history_risk', 'history_n', 'known_severe_age', 'latest_relative90', 'mean14_relative90',
    'mean30_relative90', 'slope30_relative90', 'latest_minus_log_threshold',
    'mean30_minus_log_threshold', 'mean90_minus_log_threshold']

def histories(origins, years, dates, values, threshold=75.):
    origins = np.asarray(origins, dtype='datetime64[D]').astype(np.int64)
    days = np.asarray(dates, dtype='datetime64[D]').astype(np.int64)
    values = np.asarray(values, float)
    matrix = np.full((len(origins), len(HISTORY)), np.nan, np.float32)
    col = {name: j for j, name in enumerate(HISTORY)}
    ix = np.searchsorted(days + 7, origins, side='left') - 1
    observed = ix >= 0
    matrix[observed, col['latest']] = values[ix[observed]]
    matrix[observed, col['age']] = origins[observed] - days[ix[observed]]
    end = np.maximum(ix + 1, 0)
    prefix = np.r_[0., np.cumsum(values)]
    for window in [14, 30, 90]:
        low = np.minimum(np.searchsorted(days, origins - window), end)
        count = end - low
        matrix[:, col[f'mean{window}']] = np.divide(prefix[end] - prefix[low], count,
            out=np.full(len(origins), np.nan), where=count > 0)
        if window == 90:
            matrix[:, col['sample_count90']] = count
        if window in [30, 90]:
            valid = ix > low
            slope = np.full(len(origins), np.nan)
            slope[valid] = (values[ix[valid]] - values[low[valid]]) / (days[ix[valid]] - days[low[valid]])
            matrix[:, col[f'slope{window}']] = slope
    risk, age, number = known_state(origins, np.asarray(years), days, values, threshold)
    for name, value in [('history_risk', risk), ('known_severe_age', age), ('history_n', number)]:
        matrix[:, col[name]] = value
    mean = matrix[:, col['mean90']]
    for source, name in [('latest', 'latest_relative90'), ('mean14', 'mean14_relative90'), ('mean30', 'mean30_relative90')]:
        matrix[:, col[name]] = np.log1p(matrix[:, col[source]]) - np.log1p(mean)
    matrix[:, col['slope30_relative90']] = matrix[:, col['slope30']] / np.maximum(mean, 1.)
    for source, name in [('latest', 'latest_minus_log_threshold'), ('mean30', 'mean30_minus_log_threshold'),
                         ('mean90', 'mean90_minus_log_threshold')]:
        matrix[:, col[name]] = np.log1p(matrix[:, col[source]]) - np.log1p(threshold)
    matrix[~np.isfinite(matrix)] = np.nan
    return matrix

WINDOWS = {'now': (0., 21), '1_14': (7.5, 21), '15_30': (22.5, 22), '31_60': (45.5, 30)}

STATS = ['expected_log_chla', 'annual_log_sd', 'high75_fraction', 'support_years', 'support_samples']

FEATURES = [f'seasonal_{window}_{stat}' for window in WINDOWS for stat in STATS] + [
    f'seasonal_{window}_minus_now' for window in list(WINDOWS)[1:]] + [
    'latest_minus_seasonal_now', 'mean90_minus_seasonal_now']

MIN_YEARS = 3

MIN_SAMPLES = 6

def phase(dates):
    index = pd.DatetimeIndex(dates)
    offsets = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
    result = offsets[index.month.to_numpy()-1] + index.day.to_numpy()-1.
    result[(index.month == 2) & (index.day == 29)] = 58.5
    return result

def circular_sum(values, radius):
    extended = np.r_[values[-radius:], values, values[:radius]]
    cumulative = np.r_[0., np.cumsum(extended)]
    width = 2*radius+1
    return cumulative[width:] - cumulative[:-width]

def annual_curves(dates, values, threshold=75.):
    dates = pd.DatetimeIndex(dates)
    years = dates.year.to_numpy(); bins = np.floor(phase(dates)).astype(int)
    values = np.asarray(values, float)
    curves = {}
    for year in np.unique(years):
        year_rows = years == year
        boundary = pd.Timestamp(int(year)+1, 1, 1)
        for limited in [False, True]:
            use = year_rows & ((dates + pd.Timedelta(days=7) < boundary) if limited else True)
            counts = np.bincount(bins[use], minlength=365).astype(float)
            logs = np.bincount(bins[use], weights=np.log1p(values[use]), minlength=365)
            highs = np.bincount(bins[use], weights=(values[use] >= threshold).astype(float), minlength=365)
            for radius in sorted({radius for _, radius in WINDOWS.values()}):
                number = circular_sum(counts, radius)
                mean = np.divide(circular_sum(logs, radius), number,
                    out=np.zeros(365), where=number > 0)
                risk = np.divide(circular_sum(highs, radius), number,
                    out=np.zeros(365), where=number > 0)
                curves[(int(year), limited, radius)] = (mean, risk, number)
    return curves, np.unique(years)

def history_features(origins, dates, values, latest, mean90, threshold=75.):
    origins = pd.DatetimeIndex(origins)
    if len(origins) != len(latest) or len(origins) != len(mean90):
        raise RuntimeError('Seasonal feature input rows differ')
    curves, historical_years = annual_curves(dates, values, threshold)
    output = np.full((len(origins), len(FEATURES)), np.nan, np.float32)
    positions = {name: j for j, name in enumerate(FEATURES)}
    for year in np.unique(origins.year):
        rows = np.flatnonzero(origins.year == year)
        past_years = historical_years[historical_years < year]
        if not len(past_years):
            for window in WINDOWS:
                for stat in ['support_years', 'support_samples']:
                    output[rows, positions[f'seasonal_{window}_{stat}']] = 0.
            continue
        calendar = phase(origins[rows])
        for window, (center, radius) in WINDOWS.items():
            ix = np.floor((calendar + center) % 365).astype(int)
            a = [curves[(int(previous), previous == year-1, radius)] for previous in past_years]
            annual_means = np.stack([item[0][ix] for item in a])
            annual_risk = np.stack([item[1][ix] for item in a])
            number = np.stack([item[2][ix] for item in a])
            observed = number > 0
            support_years = observed.sum(axis=0)
            samples = number.sum(axis=0)
            denominator = np.maximum(support_years, 1)
            expectation = np.sum(annual_means*observed, axis=0)/denominator
            risk = np.sum(annual_risk*observed, axis=0)/denominator
            variance = np.sum((annual_means-expectation)**2*observed, axis=0)/denominator
            valid = (support_years >= MIN_YEARS) & (samples >= MIN_SAMPLES)
            for stat, values_for_rows in [('expected_log_chla', expectation), ('annual_log_sd', np.sqrt(variance)),
                ('high75_fraction', risk)]:
                output[rows, positions[f'seasonal_{window}_{stat}']] = np.where(valid, values_for_rows, np.nan)
            output[rows, positions[f'seasonal_{window}_support_years']] = support_years
            output[rows, positions[f'seasonal_{window}_support_samples']] = samples
    current = output[:, positions['seasonal_now_expected_log_chla']]
    for window in list(WINDOWS)[1:]:
        output[:, positions[f'seasonal_{window}_minus_now']] = output[:, positions[f'seasonal_{window}_expected_log_chla']] - current
    output[:, positions['latest_minus_seasonal_now']] = np.log1p(np.asarray(latest)) - current
    output[:, positions['mean90_minus_seasonal_now']] = np.log1p(np.asarray(mean90)) - current
    output[~np.isfinite(output)] = np.nan
    return output

def observations(path):
    frame = pd.read_parquet(path, columns=['lake_id', 'sample_date', 'value_median',
        'assumed_result_available_date', 'canonical_unit'], filters=[('sample_date', '<', pd.Timestamp('2021-01-01'))])
    frame['lake_id'] = frame.lake_id.astype(str)
    if not frame.assumed_result_available_date.eq(frame.sample_date + pd.Timedelta(days=7)).all():
        raise RuntimeError('The US observed-result delay differs from the frozen seven-day seasonal reference')
    if not frame.canonical_unit.eq('ug/L').all():
        raise RuntimeError('The US development ledger contains an unspecified Chl-a unit')
    numeric = pd.to_numeric(frame.value_median, errors='coerce')
    frame = frame[np.isfinite(numeric) & numeric.ge(0)].copy()
    frame = frame.rename(columns={'sample_date': 'date', 'value_median': 'chla'})
    frame = frame.sort_values(['lake_id', 'date'], kind='stable').reset_index(drop=True)
    if frame.duplicated(['lake_id', 'date']).any():
        raise RuntimeError('The US lake-day median ledger repeats a physical lake/day')
    return frame

def lake_features(origins, dates, values):
    years = pd.DatetimeIndex(origins).year.to_numpy()
    recent = histories(origins, years, dates, values)
    latest = recent[:, HISTORY.index('latest')]
    mean90 = recent[:, HISTORY.index('mean90')]
    return history_features(origins, dates, values, latest, mean90)
