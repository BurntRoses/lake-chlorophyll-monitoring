"""Historical-q90 retraining with reference, labels, inputs and queues aligned."""
import copy
import time
import numpy as np
import pandas as pd
from .paths import DATA, MODELS, RESULTS, CONFIG
from .io import read, write, digest
from .cross_region_data import DEST, build_lake, calendar, features, specification
from .cross_region_training import score_panel
from .cross_region_evaluation import METHODS, ENDPOINTS, evaluate_frame
from .china_adaptation import train as train_fixed
from .us_analysis import historical_references, reconstruct_events, event_hits
from .evaluation import opportunity_index, selected_mask
from .cross_region_analysis import panel_observations
INPUT = DATA / 'processed/external_relative'
MODEL = MODELS / 'external_relative'
RESULT = RESULTS / 'external_relative'

def relative_features():
    return [s.replace('high75_fraction', 'high_reference_fraction') for s in features()]

def relative_expression():
    return copy.deepcopy(specification()['expression'])

def build_panel(name, obs, attrs, cohort, first, last, event_first, event_last, refs):
    folder = INPUT / name
    if (folder / 'cohort_statistics.json').exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    origins = calendar(first, last)
    parts, events, absolute = ([], [], [])
    for lake, g in obs[obs.lake_id.isin(cohort)].groupby('lake_id', sort=True):
        threshold = refs.loc[lake, 'historical_q90'] if lake in refs.index else np.nan
        morphology = attrs.loc[lake].to_dict() if lake in attrs.index else {}
        f, e = build_lake(lake, g, origins, morphology, threshold=threshold)
        if f is None:
            continue
        f = f.rename(columns=lambda s: s.replace('high75_fraction', 'high_reference_fraction'))
        if not np.isfinite(threshold):
            f[['y_1_14', 'y_15_30', 'y_31_60']] = -1
            f['common_trainable'] = False
            for c in relative_features():
                if 'high_reference_fraction' in c:
                    f[c] = np.nan
        f['relative_reference_ugL'] = threshold
        parts.append(f)
        events.append(e)
        if name != 'source':
            a, _ = build_lake(lake, g, origins, morphology, threshold=75.0, exclude_known_active=False)
            a = a.set_index('candidate_row_key').loc[f.candidate_row_key].reset_index()
            absolute.append(a)
    frame = pd.concat(parts, ignore_index=True).sort_values(['origin_date', 'lake_id']).reset_index(drop=True)
    ledger = pd.concat(events, ignore_index=True)
    ledger = ledger[ledger.event_date.between(event_first, event_last)].reset_index(drop=True)
    if not frame.candidate_row_key.is_unique or not ledger.episode_id.is_unique:
        raise ValueError('Relative candidates or events repeat')
    frame.to_parquet(folder / 'candidates.parquet', index=False)
    ledger.to_parquet(folder / 'episodes.parquet', index=False)
    if absolute:
        a = pd.concat(absolute, ignore_index=True).sort_values(['origin_date', 'lake_id']).reset_index(drop=True)
        if not a.candidate_row_key.equals(frame.candidate_row_key):
            raise ValueError('Absolute-input comparison queues differ')
        a.to_parquet(folder / 'absolute_inputs.parquet', index=False)
    write(folder / 'cohort_statistics.json', dict(panel=name, lakes=int(frame.lake_id.nunique()), candidates=len(frame), supported_reference_lakes=int(frame.loc[frame.relative_reference_ugL.notna(), 'lake_id'].nunique()), events=len(ledger), event_lakes=int(ledger.lake_id.nunique()), known_long_rows=int(frame.common_trainable.sum())))
    print('RELATIVE_INPUT_COMPLETE', name, len(frame), 'events', len(ledger), flush=True)

def build():
    spec, common = (read(CONFIG / 'relative.json'), specification())
    INPUT.mkdir(parents=True, exist_ok=True)
    signature = {p.name: digest(p) for p in [CONFIG / 'relative.json', CONFIG / 'external.json', CONFIG / 'china_adaptation.json', CONFIG / 'monitoring_models.json', CONFIG / 'base_models.json']}
    for p in [DATA / 'derived/cross_region_satellite_observations.parquet', DATA / 'derived/cross_region_field_observations.parquet', DATA / 'raw/cross_region/cci_v2/lake_morphology.csv']:
        signature[str(p.relative_to(DATA))] = digest(p)
    state = INPUT / 'configuration.json'
    if state.exists() and read(state) != signature:
        raise ValueError('Relative task inputs or parameters changed')
    write(state, signature)
    satellite = pd.read_parquet(DATA / 'derived/cross_region_satellite_observations.parquet')
    field = pd.read_parquet(DATA / 'derived/cross_region_field_observations.parquet')
    reference_spec = dict(reference_cutoff=spec['reference_cutoff'], relative_min_observations=spec['minimum_observations'], relative_min_span_days=spec['minimum_span_days'], relative_quantiles=[spec['quantile']])
    refs = historical_references(pd.concat([satellite[['lake_id', 'date', 'chla']], field[['lake_id', 'date', 'chla']]]), reference_spec)
    refs.loc[refs.historical_q90.le(0), 'historical_q90'] = np.nan
    refs.to_csv(INPUT / 'historical_references.csv', index=False)
    refs = refs.set_index('lake_id')
    attrs = pd.read_csv(DATA / 'raw/cross_region/cci_v2/lake_morphology.csv').set_index('lake_id')
    source_ids = pd.read_parquet(DEST / 'source/candidates.parquet', columns=['lake_id']).lake_id.unique()
    obs = satellite[satellite.lake_id.isin(source_ids) & satellite.date.add(pd.Timedelta(days=7)).lt(spec['training_cutoff'])]
    build_panel('source', obs, attrs, source_ids, common['source_first_origin'], common['source_last_origin'], common['source_first_origin'], common['source_last_origin'], refs)
    for name in spec['evaluation_panels']:
        panel = common['panels'][name]
        cohort = pd.read_parquet(DEST / name / 'candidates.parquet', columns=['lake_id']).lake_id.unique()
        if name == 'new_zealand':
            obs = field.copy()
            obs['country'], obs['continent'] = ('New Zealand', 'Oceania')
            local_attrs = pd.read_csv(DATA / 'raw/cross_region/lawa_nz/metadata/all_LFENZ_lake_morphology.csv').set_index('lake_id')
        elif name == 'china_2023':
            obs = satellite[satellite.country.eq('China')].copy()
            p = panel_observations(name)
            metadata = obs.groupby('lake_id').first()
            for col in ['country', 'continent', 'lat', 'lon']:
                p[col] = p.lake_id.map(metadata[col])
            v3 = []
            for path in sorted((DATA / 'raw/cross_region/cci_v3/daily_2023').glob('*.parquet')):
                d = pd.read_parquet(path)
                v3.append(d.loc[d.usable_chla & d.lake_id.isin(cohort), ['lake_id', 'date', 'temperature', 'turbidity']])
            measured = pd.concat([obs[['lake_id', 'date', 'temperature', 'turbidity']], *v3], ignore_index=True)
            obs = p.merge(measured, on=['lake_id', 'date'], validate='one_to_one')
            local_attrs = attrs
        else:
            obs = satellite[satellite.lake_id.isin(cohort)]
            local_attrs = attrs
        build_panel(name, obs, local_attrs, cohort, panel['first_origin'], panel['last_origin'], panel['event_start'], panel['event_end'], refs)
    selected = pd.read_csv(DATA / 'processed/china_adaptation/lake_panel.csv')
    selected = set(selected.loc[selected.historical_panel, 'lake_id'])
    obs = satellite[satellite.lake_id.isin(selected) & satellite.date.add(pd.Timedelta(days=7)).lt(spec['training_cutoff'])]
    build_panel('china_history', obs, attrs, selected, common['source_first_origin'], common['source_last_origin'], common['source_first_origin'], common['source_last_origin'], refs)

def evaluate_regime(regime, names, relative_model, absolute_model):
    spec = read(CONFIG / 'relative.json')
    all_points, effects, reach = ([], [], [])
    panel = pd.read_csv(DATA / 'processed/china_adaptation/lake_panel.csv')
    selected = set(panel.loc[panel.historical_panel, 'lake_id'])
    for name in names:
        frame = pd.read_parquet(INPUT / name / 'candidates.parquet')
        absolute = pd.read_parquet(INPUT / name / 'absolute_inputs.parquet')
        events = pd.read_parquet(INPUT / name / 'episodes.parquet')
        if regime == 'china_adapted':
            events = events[~events.lake_id.isin(selected)].reset_index(drop=True)
        predictions = {'absolute75_trained': score_panel(absolute, absolute_model), 'q90_trained': score_panel(frame, relative_model)}
        for strategy, pred in predictions.items():
            folder = RESULT / regime / name / strategy
            rows = evaluate_frame(pred, events, folder, spec['bootstrap_reps'], spec['bootstrap_seed'])
            pred.to_parquet(folder / 'predictions.parquet', index=False)
            all_points.extend((dict(regime=regime, panel=name, strategy=strategy, **r) for r in rows))
            for endpoint in ENDPOINTS:
                ledger = pd.read_csv(folder / (endpoint + '_event_coverage.csv'))
                for m in METHODS:
                    hits = ledger[m]
                    reach.append(dict(regime=regime, panel=name, strategy=strategy, endpoint=endpoint, method=m, evaluable_event_lakes=int(ledger.lake_id.nunique()), captured_event_lakes=int(ledger.loc[hits, 'lake_id'].nunique()), captured_events=int(hits.sum()), events=len(ledger)))
        a = pd.read_parquet(RESULT / regime / name / 'absolute75_trained/bootstrap_draws.parquet')
        b = pd.read_parquet(RESULT / regime / name / 'q90_trained/bootstrap_draws.parquet')
        points = {s: pd.read_csv(RESULT / regime / name / s / 'event_coverage.csv').set_index(['endpoint', 'method']) for s in predictions}
        for endpoint in ENDPOINTS:
            for m in METHODS:
                alpha = 0.05 / 4 if endpoint == '31_60' and m in spec['comparison_methods'] else 0.05
                values = 100 * (b[endpoint + '__' + m] - a[endpoint + '__' + m]).dropna().to_numpy()
                lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2]) if len(values) else [np.nan, np.nan]
                old, new = (points['absolute75_trained'].loc[endpoint, m], points['q90_trained'].loc[endpoint, m])
                effects.append(dict(regime=regime, panel=name, endpoint=endpoint, method=m, events=int(old.events), absolute75_captured=int(old.captured), q90_captured=int(new.captured), absolute75_percent=float(old.capture_percent), q90_percent=float(new.capture_percent), difference_pp=float(new.capture_percent - old.capture_percent), lower=lo, upper=hi, interval_level=1 - alpha))
        print('RELATIVE_EVALUATION_COMPLETE', regime, name, flush=True)
    pd.DataFrame(all_points).to_csv(RESULT / regime / 'event_coverage.csv', index=False)
    pd.DataFrame(effects).to_csv(RESULT / regime / 'retraining_effects.csv', index=False)
    pd.DataFrame(reach).to_csv(RESULT / regime / 'lake_reach.csv', index=False)

def run():
    started = time.perf_counter()
    build()
    source = pd.read_parquet(INPUT / 'source/candidates.parquet')
    source_events = pd.read_parquet(INPUT / 'source/episodes.parquet')
    for regime in read(CONFIG / 'relative.json')['regimes']:
        frame, events = (source, source_events)
        if regime == 'china_adapted':
            frame = pd.concat([source, pd.read_parquet(INPUT / 'china_history/candidates.parquet')], ignore_index=True)
            frame = frame.sort_values(['origin_date', 'lake_id']).reset_index(drop=True)
            events = pd.concat([source_events, pd.read_parquet(INPUT / 'china_history/episodes.parquet')], ignore_index=True)
        root = INPUT / regime / 'training'
        root.mkdir(parents=True, exist_ok=True)
        signature = dict(task_sha256=digest(INPUT / 'configuration.json'), regime=regime)
        if (root / 'configuration.json').exists() and read(root / 'configuration.json') != signature:
            raise ValueError('Relative training regime changed')
        write(root / 'configuration.json', signature)
        train_fixed(frame, events, input_root=root, model_root=MODEL / regime, result_root=RESULT / regime, feature_names=relative_features(), expression_config=relative_expression())
        names = read(CONFIG / 'relative.json')['evaluation_panels'] if regime == 'source' else ['china_2020_2022', 'china_2023']
        evaluate_regime(regime, names, MODEL / regime, MODELS / ('external_absolute' if regime == 'source' else 'china_adaptation'))
    analyze()

def analyze():
    """Measure event-lake breadth, lower-baseline reach and capacity response."""
    spec = read(CONFIG / 'relative.json')
    allocations, strata, curves, contrasts = ([], [], [], [])
    refs = pd.read_csv(INPUT / 'historical_references.csv').set_index('lake_id').historical_q90
    panel = pd.read_csv(DATA / 'processed/china_adaptation/lake_panel.csv')
    history_lakes = set(panel.loc[panel.historical_panel, 'lake_id'])
    for regime in spec['regimes']:
        names = spec['evaluation_panels'] if regime == 'source' else ['china_2020_2022', 'china_2023']
        for name in names:
            events = pd.read_parquet(INPUT / name / 'episodes.parquet')
            if regime == 'china_adapted':
                events = events[~events.lake_id.isin(history_lakes)].reset_index(drop=True)
            for strategy in ['absolute75_trained', 'q90_trained']:
                folder = RESULT / regime / name / strategy
                frame = pd.read_parquet(folder / 'predictions.parquet').sort_values(['origin_date', 'lake_id']).reset_index(drop=True)
                selected = pd.read_parquet(folder / 'selected_candidates.parquet')
                pairs = pd.read_csv(folder / 'method_comparisons.csv')
                contrasts.append(pairs.assign(regime=regime, panel=name, strategy=strategy))
                for m in METHODS:
                    chosen = selected[m].to_numpy(bool)
                    supported = frame.lake_id.map(refs).notna().to_numpy()
                    allocations.append(dict(regime=regime, panel=name, strategy=strategy, method=m, selected_slots=int(chosen.sum()), selected_lakes=int(frame.loc[chosen, 'lake_id'].nunique()), slots_without_reference=int((chosen & ~supported).sum()), supported_reference_lakes=int(frame.loc[supported, 'lake_id'].nunique())))
                ops = {}
                for endpoint, (lo, hi) in ENDPOINTS.items():
                    ledger, ri, ei = opportunity_index(frame, events, lo, hi)
                    ops[endpoint] = (ledger, ri, ei)
                    reference = ledger.lake_id.map(refs)
                    groups = {'below25': reference.lt(25), '25to75': reference.ge(25) & reference.lt(75), '75plus': reference.ge(75)}
                    for label, take in groups.items():
                        for m in METHODS:
                            hits = event_hits(selected[m].to_numpy(bool), ri, ei, len(ledger))
                            strata.append(dict(regime=regime, panel=name, strategy=strategy, endpoint=endpoint, reference_group=label, method=m, events=int(take.sum()), captured=int((hits & take).sum()), event_lakes=int(ledger.loc[take, 'lake_id'].nunique()), captured_event_lakes=int(ledger.loc[hits & take, 'lake_id'].nunique())))
                sizes = frame.groupby('origin_date').lake_id.transform('size').to_numpy()
                for m in METHODS:
                    ranks = frame.groupby('origin_date')[m].rank(method='first', ascending=False).to_numpy()
                    for bp in range(100, 3001, 25):
                        chosen = ranks <= np.ceil(sizes * (bp / 10000))
                        for endpoint, (ledger, ri, ei) in ops.items():
                            hits = event_hits(chosen, ri, ei, len(ledger))
                            curves.append(dict(regime=regime, panel=name, strategy=strategy, method=m, endpoint=endpoint, capacity_basis_points=bp, selected_slots=int(chosen.sum()), captured=int(hits.sum()), events=len(ledger), captured_event_lakes=int(ledger.loc[hits, 'lake_id'].nunique()), capture_percent=float(100 * hits.mean()) if len(hits) else np.nan))
    pd.DataFrame(allocations).to_csv(RESULT / 'lake_allocation.csv', index=False)
    pd.DataFrame(strata).to_csv(RESULT / 'baseline_strata.csv', index=False)
    pd.DataFrame(curves).to_csv(RESULT / 'capacity_sensitivity.csv', index=False)
    pd.concat(contrasts, ignore_index=True).to_csv(RESULT / 'method_comparisons.csv', index=False)
    adaptation_effects = []
    for name in ['china_2020_2022', 'china_2023']:
        frame = pd.read_parquet(INPUT / name / 'candidates.parquet')
        events = pd.read_parquet(INPUT / name / 'episodes.parquet')
        events = events[~events.lake_id.isin(history_lakes)].reset_index(drop=True)
        folder = RESULT / 'china_adapted' / name / 'source_q90_held_out'
        pred = pd.read_parquet(RESULT / 'source' / name / 'q90_trained/predictions.parquet')
        rows = evaluate_frame(pred, events, folder, spec['bootstrap_reps'], spec['bootstrap_seed'])
        before = pd.DataFrame(rows).set_index(['endpoint', 'method'])
        after = pd.read_csv(RESULT / 'china_adapted' / name / 'q90_trained/event_coverage.csv').set_index(['endpoint', 'method'])
        a = pd.read_parquet(folder / 'bootstrap_draws.parquet')
        b = pd.read_parquet(RESULT / 'china_adapted' / name / 'q90_trained/bootstrap_draws.parquet')
        for endpoint in ENDPOINTS:
            for m in METHODS:
                old, new = (before.loc[endpoint, m], after.loc[endpoint, m])
                if old.events != new.events:
                    raise ValueError('Relative adaptation event populations differ')
                alpha = 0.05 / 4 if endpoint == '31_60' and m in spec['comparison_methods'] else 0.05
                values = 100 * (b[endpoint + '__' + m] - a[endpoint + '__' + m]).dropna().to_numpy()
                lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2]) if len(values) else [np.nan, np.nan]
                adaptation_effects.append(dict(panel=name, endpoint=endpoint, method=m, events=int(old.events), source_captured=int(old.captured), adapted_captured=int(new.captured), source_percent=float(old.capture_percent), adapted_percent=float(new.capture_percent), difference_pp=float(new.capture_percent - old.capture_percent), lower=lo, upper=hi, interval_level=1 - alpha))
    pd.DataFrame(adaptation_effects).to_csv(RESULT / 'china_adaptation_effects.csv', index=False)
