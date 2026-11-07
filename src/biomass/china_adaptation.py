"""One fixed historical-lake-panel adaptation of the regional study framework."""
import gc
import hashlib
import time
import joblib
import numpy as np
import pandas as pd
from .paths import DATA, MODELS, RESULTS, CONFIG
from .io import read, write, digest
from .cross_region_data import DEST, specification, build_table
from .cross_region_training import fit_base, correction_predictions, score_panel, IDENTITY, KEYS
from .cross_region_evaluation import METHODS, ENDPOINTS, evaluate_frame
from .experts import transform_batched
INPUT = DATA / 'processed/china_adaptation'
MODEL = MODELS / 'china_adaptation'
RESULT = RESULTS / 'china_adaptation'

def parameter_basis():
    """Describe predeployment concentration and event support, without fitting."""
    from .us_analysis import reconstruct_events
    common = specification()
    obs = pd.read_parquet(DATA / 'derived/cross_region_satellite_observations.parquet', columns=['lake_id', 'date', 'chla'])
    obs = obs[obs.date.add(pd.Timedelta(days=common['result_delay_days'])).lt(common['training_cutoff'])]
    source = pd.read_parquet(DEST / 'source/candidates.parquet', columns=['lake_id']).lake_id.unique()
    china = pd.read_parquet(DEST / 'china_2020_2022/candidates.parquet', columns=['lake_id']).lake_id.unique()
    panel = pd.DataFrame({'lake_id': np.sort(china)})
    local = read(CONFIG / 'china_adaptation.json')
    panel['historical_panel'] = [panel_member(l, local) for l in panel.lake_id]
    groups = [('source', source), ('china', china), ('china_historical_panel', panel.loc[panel.historical_panel, 'lake_id'])]
    distribution, thresholds = ([], [])
    for name, lakes in groups:
        history = obs[obs.lake_id.isin(lakes)].sort_values(['lake_id', 'date']).reset_index(drop=True)
        for period, first in [('available_pre2020', '2002-01-01'), ('2016_2019', '2016-01-01')]:
            f = history[history.date.ge(first)]
            row = dict(region=name, period=period, lakes=int(f.lake_id.nunique()), observations=len(f))
            for q in [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
                row[f'observation_q{round(100 * q)}_ugL'] = float(f.chla.quantile(q))
            row['median_lake_median_ugL'] = float(f.groupby('lake_id').chla.median().median())
            distribution.append(row)
            for threshold in [25, 50, 75, 100]:
                events = reconstruct_events(history, threshold, common['washout_days'])
                events = events[events.event_date.ge(first)]
                high = f.chla.ge(threshold)
                thresholds.append(dict(region=name, period=period, threshold_ugL=threshold, observations=len(f), high_observation_percent=float(100 * high.mean()), lakes_with_high_observation=int(f.loc[high, 'lake_id'].nunique()), recorded_events=len(events), event_lakes=int(events.lake_id.nunique()), event_gap_days=common['washout_days']))
    RESULT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(distribution).to_csv(RESULT / 'predeployment_concentration.csv', index=False)
    pd.DataFrame(thresholds).to_csv(RESULT / 'predeployment_event_support.csv', index=False)
    return (pd.DataFrame(distribution), pd.DataFrame(thresholds))

def panel_member(lake, spec):
    value = hashlib.sha256((spec['panel_hash_salt'] + '|' + str(lake)).encode()).hexdigest()
    return int(value[:12], 16) % spec['panel_hash_modulus'] == spec['panel_hash_remainder']

def fit_rows(frame, year, excluded):
    """Mature past labels, with every lake of the prediction fold excluded."""
    return frame[frame.origin_year.lt(year) & ~frame.outer_lake_fold.isin(excluded) & frame.common_trainable & frame.label_mature_date.lt(pd.Timestamp(year, 1, 1))].reset_index(drop=True)

def inputs():
    spec, common = (read(CONFIG / 'china_adaptation.json'), specification())
    if spec['training_cutoff'] != common['training_cutoff'] or spec['local_label_weight'] != 1:
        raise ValueError('Adaptation must use the same cutoff and pooled observation weighting')
    signature = {p.name: digest(p) for p in [CONFIG / 'china_adaptation.json', CONFIG / 'external.json', CONFIG / 'monitoring_models.json', CONFIG / 'base_models.json']}
    signature['source_candidates'] = digest(DEST / 'source/candidates.parquet')
    signature['satellite_observations'] = digest(DATA / 'derived/cross_region_satellite_observations.parquet')
    signature['morphology'] = digest(DATA / 'raw/cross_region/cci_v2/lake_morphology.csv')
    INPUT.mkdir(parents=True, exist_ok=True)
    state = INPUT / 'configuration.json'
    if state.exists() and read(state) != signature:
        raise ValueError('Existing adaptation fits belong to different inputs or parameters')
    write(state, signature)
    cohort = pd.read_parquet(DEST / 'china_2020_2022/candidates.parquet', columns=['lake_id']).lake_id.unique()
    panel = pd.DataFrame({'lake_id': np.sort(cohort)})
    panel['historical_panel'] = [panel_member(l, spec) for l in panel.lake_id]
    path = INPUT / 'lake_panel.csv'
    if path.exists():
        pd.testing.assert_frame_equal(pd.read_csv(path), panel)
    else:
        panel.to_csv(path, index=False)
    local_lakes = set(panel.loc[panel.historical_panel, 'lake_id'])
    if not (INPUT / 'local_history/candidates.parquet').exists():
        obs = pd.read_parquet(DATA / 'derived/cross_region_satellite_observations.parquet')
        obs = obs[obs.lake_id.isin(local_lakes) & obs.date.add(pd.Timedelta(days=common['result_delay_days'])).lt(spec['training_cutoff'])]
        morphology = pd.read_csv(DATA / 'raw/cross_region/cci_v2/lake_morphology.csv')
        build_table(obs, morphology, common['source_first_origin'], common['source_last_origin'], spec['training_cutoff'], INPUT / 'local_history', common['source_first_origin'], common['source_last_origin'])
    local = pd.read_parquet(INPUT / 'local_history/candidates.parquet')
    local_events = pd.read_parquet(INPUT / 'local_history/episodes.parquet')
    if not set(local.lake_id).issubset(local_lakes):
        raise ValueError('Unselected Chinese lakes entered local fitting')
    source = pd.read_parquet(DEST / 'source/candidates.parquet')
    events = pd.read_parquet(DEST / 'source/episodes.parquet')
    if set(source.lake_id) & set(cohort):
        raise ValueError('Chinese evaluation cohort entered the original source')
    frame = pd.concat([source, local], ignore_index=True).sort_values(['origin_date', 'lake_id']).reset_index(drop=True)
    events = pd.concat([events, local_events], ignore_index=True)
    if not frame.candidate_row_key.is_unique or not events.episode_id.is_unique:
        raise ValueError('Repeated pooled training identity')
    if set(frame.loc[frame.country.eq('China'), 'lake_id']) != local_lakes:
        raise ValueError('Historical adaptation panel identities differ')
    write(INPUT / 'cohort_statistics.json', dict(source_lakes=int(source.lake_id.nunique()), chinese_cohort_lakes=len(cohort), historical_panel_lakes=len(local_lakes), held_out_chinese_lakes=len(cohort) - len(local_lakes), local_candidates=len(local), local_mature_supervised_rows=int((local.common_trainable & local.label_mature_date.lt(spec['training_cutoff'])).sum()), local_mature_positive_events=int(local.loc[local.y_31_60.eq(1) & local.label_mature_date.lt(spec['training_cutoff']), 'next_episode_id'].nunique()), total_candidates=len(frame), training_cutoff=spec['training_cutoff'], panel_rule='SHA256(salt|physical lake ID), first 12 hex digits, modulo 10 equals 0'))
    return (frame, events, panel)

def train(frame, events, input_root=INPUT, model_root=MODEL, result_root=RESULT, feature_names=None, expression_config=None):
    common = specification()
    result_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    splits = []
    for year in range(common['context_first_year'], 2020):
        for fold in range(common['lake_folds']):
            folder = input_root / 'crossfit' / str(year) / str(fold)
            path = folder / 'predictions.parquet'
            val = frame[frame.origin_year.eq(year) & frame.outer_lake_fold.eq(fold)].reset_index(drop=True)
            training = fit_rows(frame, year, [fold])
            if set(training.lake_id) & set(val.lake_id):
                raise ValueError('Adaptation crossfit shares a training and prediction lake')
            if path.exists():
                saved = pd.read_parquet(path, columns=['candidate_row_key'])
                if not saved.candidate_row_key.equals(val.candidate_row_key):
                    raise ValueError('Adaptation crossfit identities changed')
            else:
                start = time.perf_counter()
                prediction = fit_base(training, val, folder, feature_names=feature_names, expression_config=expression_config)
                prediction.to_parquet(path, index=False)
                write(folder / 'split.json', dict(year=year, excluded_folds=[fold], training_rows=len(training), prediction_rows=len(val), last_label_maturity=str(training.label_mature_date.max())))
                print('CHINA_ADAPTATION_CELL', year, fold, 'train', len(training), 'predict', len(val), 'seconds', round(time.perf_counter() - start, 1), flush=True)
            splits.append(read(folder / 'split.json'))
            del training, val
            gc.collect()
    pd.DataFrame(splits).to_csv(result_root / 'crossfit_splits.csv', index=False)
    cutoff = pd.Timestamp(common['training_cutoff'])
    training = fit_rows(frame, 2020, [])
    base_path = input_root / 'final_expert_predictions.parquet'
    if not base_path.exists():
        base = fit_base(training, frame, model_root / 'experts', controls=True, save_models=True, feature_names=feature_names, expression_config=expression_config)
        base.to_parquet(base_path, index=False)
    context = frame[frame.origin_year.ge(common['context_first_year'])].reset_index(drop=True)
    parts = [pd.read_parquet(input_root / 'crossfit' / str(year) / str(fold) / 'predictions.parquet') for year in range(common['context_first_year'], 2020) for fold in range(common['lake_folds'])]
    oof = pd.concat(parts, ignore_index=True).set_index('candidate_row_key')
    if not oof.index.is_unique or set(oof.index) != set(context.candidate_row_key):
        raise ValueError('Incomplete adaptation crossfit coverage')
    oof = oof.loc[context.candidate_row_key].reset_index()
    oof.to_parquet(input_root / 'coverage_training_predictions.parquet', index=False)
    expr = joblib.load(model_root / 'experts/expression.joblib')
    correction_predictions(context, events, transform_batched(expr, context, 8192), oof, cutoff, model_root, 2020, ())
    write(model_root / 'checksums.json', {p.relative_to(model_root).as_posix(): digest(p) for p in model_root.rglob('*') if p.suffix in ['.ubj', '.joblib']})
    write(result_root / 'training_statistics.json', dict(expert_training_rows=len(training), coverage_training_rows=len(context), local_training_rows=int(training.country.eq('China').sum()), crossfit_cells=len(splits), transformed_features=len(expr.names), training_cutoff=common['training_cutoff'], inputs_sha256=digest(input_root / 'configuration.json')))
    print('CHINA_ADAPTATION_TRAINING_COMPLETE', flush=True)

def evaluate(panel):
    spec = read(CONFIG / 'china_adaptation.json')
    selected = set(panel.loc[panel.historical_panel, 'lake_id'])
    points, effects = ([], [])
    for name in spec['evaluation_panels']:
        frame = pd.read_parquet(DEST / name / 'candidates.parquet')
        events = pd.read_parquet(DEST / name / 'episodes.parquet')
        events = events[~events.lake_id.isin(selected)].reset_index(drop=True)
        source = pd.read_parquet(RESULTS / 'external_absolute' / name / 'predictions.parquet')
        adapted = score_panel(frame, MODEL)
        if not source.candidate_row_key.equals(adapted.candidate_row_key):
            raise ValueError('Source and adapted candidate queues differ')
        for regime, pred in [('source', source), ('adapted', adapted)]:
            folder = RESULT / name / regime
            rows = evaluate_frame(pred, events, folder, spec['bootstrap_reps'], spec['bootstrap_seed'])
            pred.to_parquet(folder / 'predictions.parquet', index=False)
            points.extend((dict(panel=name, regime=regime, **r) for r in rows))
        a = pd.read_parquet(RESULT / name / 'source/bootstrap_draws.parquet')
        b = pd.read_parquet(RESULT / name / 'adapted/bootstrap_draws.parquet')
        source_points = pd.read_csv(RESULT / name / 'source/event_coverage.csv').set_index(['endpoint', 'method'])
        adapted_points = pd.read_csv(RESULT / name / 'adapted/event_coverage.csv').set_index(['endpoint', 'method'])
        pd.testing.assert_frame_equal(pd.read_csv(RESULT / name / 'source/queues.csv'), pd.read_csv(RESULT / name / 'adapted/queues.csv'))
        for endpoint in ENDPOINTS:
            for method in METHODS:
                old, new = (source_points.loc[endpoint, method], adapted_points.loc[endpoint, method])
                if old.events != new.events:
                    raise ValueError('Adaptation changed the evaluation event population')
                alpha = 0.05 / len(spec['adaptation_effect_methods']) if endpoint == '31_60' and method in spec['adaptation_effect_methods'] else 0.05
                values = 100 * (b[endpoint + '__' + method] - a[endpoint + '__' + method]).dropna().to_numpy()
                lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2]) if len(values) else [np.nan, np.nan]
                effects.append(dict(panel=name, endpoint=endpoint, method=method, events=int(old.events), source_captured=int(old.captured), adapted_captured=int(new.captured), source_percent=float(old.capture_percent), adapted_percent=float(new.capture_percent), difference_pp=float(new.capture_percent - old.capture_percent), lower=lo, upper=hi, interval_level=1 - alpha, valid_replicates=len(values)))
        print('CHINA_ADAPTATION_EVALUATION_COMPLETE', name, flush=True)
    pd.DataFrame(points).to_csv(RESULT / 'event_coverage.csv', index=False)
    pd.DataFrame(effects).to_csv(RESULT / 'adaptation_effects.csv', index=False)
    write(RESULT / 'evaluation_statistics.json', dict(population=spec['evaluation_population'], evaluation_panels=spec['evaluation_panels'], historical_panel_lakes=len(selected), full_queue_retained=True, paired_lake_bootstrap_reps=spec['bootstrap_reps'], model_checksums_sha256=digest(MODEL / 'checksums.json')))

def run():
    frame,events,panel=inputs()
    train(frame,events)
    evaluate(panel)
