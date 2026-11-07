"""Independent-event evaluation of the common frozen regional models."""
import time
import numpy as np
import pandas as pd
from .io import read, write, digest
from .paths import CONFIG
from .evaluation import opportunity_index, selected_mask, select_copies, event_counts
from .cross_region_data import DEST, specification
from .cross_region_training import MODEL, RESULT, score_panel

METHODS = ['xgboost', 'elastic_net', 'single_window', 'same_window_control', 'multi_window', 'matched_xgboost']
NAMES = {'xgboost': 'XGBoost', 'elastic_net': 'Elastic Net', 'single_window': '容量约束单窗',
         'same_window_control': '同窗多模型对照', 'multi_window': '容量约束多窗',
         'matched_xgboost': '同配置长窗预测'}
ENDPOINTS = read(CONFIG / 'analysis.json')['endpoints']


def evaluate_frame(frame, events, output, reps=0, seed=20261101):
    """Use every eligible competitor, including those with unknown outcomes."""
    spec = specification()
    output.mkdir(parents=True, exist_ok=True)
    frame = frame.sort_values(['origin_date', 'lake_id'], kind='stable').reset_index(drop=True)
    if not frame.candidate_row_key.is_unique or not np.isfinite(frame[METHODS].to_numpy()).all():
        raise ValueError('Invalid evaluation identities or scores')
    lakes = np.sort(frame.lake_id.unique())
    row_lake = pd.Index(lakes).get_indexer(frame.lake_id).astype(np.int32)
    dates = pd.factorize(frame.origin_date, sort=True)[0]
    starts = np.r_[0, np.flatnonzero(dates[1:] != dates[:-1]) + 1].astype(np.int64)
    ends = np.r_[starts[1:], len(frame)].astype(np.int64)
    orders = np.asarray([np.lexsort((frame.lake_id.to_numpy(), -frame[m].to_numpy(), dates)) for m in METHODS], np.int64)
    ones = np.ones(len(lakes), np.int64)
    selected = select_copies(ones, row_lake, starts, ends, orders, spec['capacity'])
    quotas = np.ceil(spec['capacity'] * (ends - starts)).astype(int)
    if not np.array_equal(np.add.reduceat(selected, starts, axis=1), np.tile(quotas, (len(METHODS), 1))):
        raise ValueError('Monitoring capacity differs from the declared quota')
    selections = frame[['candidate_row_key', 'lake_id', 'origin_date']].copy()
    for i, method in enumerate(METHODS):
        selections[method] = selected[i] > 0
    selections.to_parquet(output / 'selected_candidates.parquet', index=False)
    pd.DataFrame({'origin_date': frame.origin_date.iloc[starts].to_numpy(),
        'eligible_lakes': ends - starts, 'quota': quotas}).to_csv(output / 'queues.csv', index=False)
    points, annual, endpoints, support = [], [], {}, []
    for endpoint, (lo, hi) in ENDPOINTS.items():
        ledger, ri, ei = opportunity_index(frame, events, lo, hi)
        event_lake = pd.Index(lakes).get_indexer(ledger.lake_id).astype(np.int32)
        endpoints[endpoint] = (ri, ei, event_lake)
        numerator, denominator = event_counts(selected, ri, ei, event_lake, ones)
        support.append(dict(endpoint=endpoint, recorded_events=len(events), evaluable_events=int(denominator),
                            events_without_eligible_opportunity=int(len(events)-denominator)))
        captures = ledger.copy()
        for k, method in enumerate(METHODS):
            hits = np.zeros(len(ledger), bool)
            np.logical_or.at(hits, ei, selected[k, ri] > 0)
            captures[method] = hits
            points.append(dict(endpoint=endpoint, method=method, method_name=NAMES[method],
                captured=int(numerator[k]), events=int(denominator),
                capture_percent=float(100*numerator[k]/denominator) if denominator else np.nan))
            for year in sorted(ledger.event_date.dt.year.unique()):
                take = ledger.event_date.dt.year.eq(year).to_numpy()
                annual.append(dict(endpoint=endpoint, method=method, year=int(year),
                    captured=int(hits[take].sum()), events=int(take.sum()), capture_percent=float(100*hits[take].mean())))
        captures.to_csv(output / (endpoint + '_event_coverage.csv'), index=False)
    pd.DataFrame(points).to_csv(output / 'event_coverage.csv', index=False)
    pd.DataFrame(annual).to_csv(output / 'annual_event_coverage.csv', index=False)
    pd.DataFrame(support).to_csv(output / 'event_support.csv', index=False)
    if reps:
        rng = np.random.default_rng(seed)
        draws = np.empty((reps, len(endpoints), len(METHODS)), float)
        start = time.perf_counter()
        for rep in range(reps):
            multiplicity = np.bincount(rng.integers(0, len(lakes), len(lakes)), minlength=len(lakes)).astype(np.int64)
            chosen = select_copies(multiplicity, row_lake, starts, ends, orders, spec['capacity'])
            for e, (ri, ei, event_lake) in enumerate(endpoints.values()):
                numerator, denominator = event_counts(chosen, ri, ei, event_lake, multiplicity)
                draws[rep, e] = numerator/denominator if denominator else np.nan
            if (rep + 1) % 1000 == 0:
                print('REGIONAL_BOOTSTRAP', output.name, rep+1, '/', reps,
                      'seconds', round(time.perf_counter()-start, 1), flush=True)
        columns = [e+'__'+m for e in endpoints for m in METHODS]
        pd.DataFrame(draws.reshape(reps, -1), columns=columns).to_parquet(output / 'bootstrap_draws.parquet', index=False)
        contrasts, intervals = [], []
        for e, endpoint in enumerate(endpoints):
            point = {r['method']: r['capture_percent'] for r in points if r['endpoint'] == endpoint}
            for k, method in enumerate(METHODS):
                sample = 100*draws[:, e, k]
                valid = sample[np.isfinite(sample)]
                lower, upper = np.quantile(valid, [.025, .975]) if len(valid) else [np.nan, np.nan]
                intervals.append(dict(endpoint=endpoint, method=method, capture_percent=point[method],
                    lower95=lower, upper95=upper, valid_replicates=len(valid)))
            for family, pairs in spec['comparison_families'].items():
                # Familywise 95% intervals for the primary endpoint; the secondary
                # endpoint is reported with exploratory, unadjusted 95% intervals.
                alpha = .05/len(pairs) if endpoint == '31_60' else .05
                for method, comparator in pairs:
                    sample = 100*(draws[:, e, METHODS.index(method)] - draws[:, e, METHODS.index(comparator)])
                    valid = sample[np.isfinite(sample)]
                    lower, upper = np.quantile(valid, [alpha/2, 1-alpha/2]) if len(valid) else [np.nan, np.nan]
                    contrasts.append(dict(endpoint=endpoint, family=family, method=method, comparator=comparator,
                        difference_pp=point[method]-point[comparator], lower=lower, upper=upper,
                        interval_level=1-alpha, valid_replicates=len(valid),
                        adjustment='Bonferroni within family' if endpoint == '31_60' else 'exploratory unadjusted'))
        pd.DataFrame(contrasts).to_csv(output / 'method_comparisons.csv', index=False)
        pd.DataFrame(intervals).to_csv(output / 'coverage_intervals.csv', index=False)
    return points


def evaluate():
    spec = specification()
    training = read(RESULT / 'training_statistics.json')
    signature = digest(CONFIG / 'external.json')
    if training['config_sha256'] != signature:
        raise ValueError('The frozen model configuration changed after training')
    all_points = []
    source = pd.read_parquet(RESULT / 'source_validation/predictions.parquet')
    # Use chronological held-out predictions for each source cell.
    if 'matched_xgboost' not in source:
        from .cross_region_training import cell_folder
        parts = [pd.read_parquet(cell_folder(year, (fold,))/'predictions.parquet',
                 columns=['candidate_row_key', 'xgb4__y_31_60'])
                 for year in spec['validation_years'] for fold in range(spec['lake_folds'])]
        base = pd.concat(parts).set_index('candidate_row_key')
        source['matched_xgboost'] = base.loc[source.candidate_row_key, 'xgb4__y_31_60'].to_numpy()
        source.to_parquet(RESULT / 'source_validation/predictions.parquet', index=False)
    events = pd.read_parquet(DEST / 'source/episodes.parquet')
    points = evaluate_frame(source, events, RESULT / 'source_validation/evaluation')
    all_points.extend([dict(panel='source_validation', **row) for row in points])
    fold_points = []
    for (year, fold), group in source.groupby(['origin_year', 'outer_lake_fold']):
        group = group.sort_values(['origin_date', 'lake_id']).reset_index(drop=True)
        selected = {m: selected_mask(group, group[m], spec['capacity']) for m in METHODS}
        for endpoint, (lo, hi) in ENDPOINTS.items():
            ledger, ri, ei = opportunity_index(group, events, lo, hi)
            for method in METHODS:
                hits = np.zeros(len(ledger), bool)
                np.logical_or.at(hits, ei, selected[method][ri])
                fold_points.append(dict(year=int(year), fold=int(fold), endpoint=endpoint,
                    method=method, captured=int(hits.sum()), events=len(ledger),
                    capture_percent=float(100*hits.mean()) if len(hits) else np.nan))
    pd.DataFrame(fold_points).to_csv(RESULT / 'source_validation/fold_captures.csv', index=False)
    from .cross_region_analysis import analyze_panel
    for name in spec['panels']:
        folder = RESULT / name
        folder.mkdir(parents=True, exist_ok=True)
        frame = pd.read_parquet(DEST / name / 'candidates.parquet')
        events = pd.read_parquet(DEST / name / 'episodes.parquet')
        pred = score_panel(frame)
        if not pred.candidate_row_key.equals(frame.candidate_row_key):
            raise ValueError('Model scores are not aligned with the complete target queue')
        pred.to_parquet(folder / 'predictions.parquet', index=False)
        points = evaluate_frame(pred, events, folder, spec['bootstrap_reps'], spec['bootstrap_seed'])
        analyze_panel(name, pred)
        all_points.extend([dict(panel=name, **row) for row in points])
        primary = {r['method']: f"{r['captured']}/{r['events']}" for r in points if r['endpoint'] == '31_60'}
        print('REGIONAL_EVALUATION_COMPLETE', name, primary, flush=True)
    pd.DataFrame(all_points).to_csv(RESULT / 'event_coverage.csv', index=False)
    contrasts = pd.concat([pd.read_csv(RESULT/name/'method_comparisons.csv').assign(panel=name) for name in spec['panels']], ignore_index=True)
    contrasts.to_csv(RESULT / 'method_comparisons.csv', index=False)
    write(RESULT / 'evaluation_statistics.json', dict(config_sha256=signature,
        model_checksums_sha256=digest(MODEL/'checksums.json'), panels=list(spec['panels']),
        bootstrap_reps=spec['bootstrap_reps'], bootstrap_seed=spec['bootstrap_seed'],
        methods=NAMES, endpoints=ENDPOINTS,
        analysis_config_sha256=digest(CONFIG/'analysis.json')))

