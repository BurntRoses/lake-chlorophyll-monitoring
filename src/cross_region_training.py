"""US coverage learning on a common, target-excluded regional source cohort."""
import gc
import itertools
import time
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import logit
from threadpoolctl import threadpool_limits
from .paths import CONFIG, MODELS, RESULTS
from .io import read, write, digest
from .expression import CommonExpression, StandardElasticNet
from .experts import transform_batched, event_balanced_weights, tree_parameters, TREE_GRID, EN_GRID, SEED
from .training import fit_correction
from .coverage import FirstOrderCoverageBooster, head_features, infer_cpu
from .cross_region_data import DEST, features, specification
CROSSFIT = DEST / 'source/crossfit'
MODEL = MODELS / 'external_absolute'
RESULT = RESULTS / 'external_absolute'
KEYS = ['xgb4__y_31_60', 'xgb4__y_1_14', 'xgb4__y_15_30', 'xgb4__long_seed1', 'xgb4__long_seed2']
IDENTITY = ['candidate_row_key', 'lake_id', 'origin_date', 'origin_year', 'outer_lake_fold']

def expression(train, feature_names=None, expression_config=None):
    spec = pd.DataFrame({'feature_name': features() if feature_names is None else feature_names, 'role': 'dynamic'})
    spec.loc[spec.feature_name.isin(['log_area', 'log_depth', 'log_volume', 'log_residence_days', 'lake_type', 'abs_lat']), 'role'] = 'base'
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'All-NaN slice encountered')
        return CommonExpression(spec, specification()['expression'] if expression_config is None else expression_config).fit(train)

def fit_base(train, val, folder, controls=False, save_models=False, feature_names=None, expression_config=None):
    """Fit US-selected binary trees; CUDA changes the backend, not the objective."""
    folder.mkdir(parents=True, exist_ok=True)
    expr = expression(train, feature_names, expression_config)
    X = transform_batched(expr, train, 8192)
    V = transform_batched(expr, val, 8192)
    joblib.dump(expr, folder / 'expression.joblib')
    output = val[IDENTITY].copy()
    cutoff_year = int(train.origin_year.max()) + 1
    excluded = sorted(set(val.outer_lake_fold) - set(train.outer_lake_fold))
    definitions = [(key, 'xgb4', key.split('__')[1] if '__y_' in key else 'y_31_60', SEED + (int(key[-1]) if 'long_seed' in key else 0)) for key in KEYS]
    if controls:
        definitions.append(('xgboost', 'xgb2', 'y_31_60', SEED))
    matrices = {}
    prediction_matrix = xgb.DMatrix(V, nthread=4)
    for key, tree_key, label, seed in definitions:
        if label not in matrices:
            known = train[label].ge(0).to_numpy()
            subset = train.loc[known]
            if subset[label].nunique() != 2:
                raise ValueError(f'Source cell lacks both {label} classes')
            matrices[label] = xgb.DMatrix(X[known], label=subset[label].to_numpy(int), weight=event_balanced_weights(subset, label), nthread=4)
        params = tree_parameters(TREE_GRID[tree_key], seed)
        params.update(device='cuda', nthread=4, objective='binary:logistic', disable_default_eval_metric=1)
        start = time.perf_counter()
        model = xgb.train(params, matrices[label], num_boost_round=TREE_GRID[tree_key]['n_estimators'])
        output[key] = model.predict(prediction_matrix)
        if save_models:
            model.save_model(folder / (key + '.ubj'))
        del model
    if controls:
        start = time.perf_counter()
        cfg = EN_GRID['lr4']
        with threadpool_limits(limits=4):
            en = StandardElasticNet(C=cfg['C'], l1_ratio=cfg['l1_ratio']).fit(np.ascontiguousarray(X, dtype=np.float64), train.y_31_60.to_numpy(int))
        if not en.converged:
            raise RuntimeError(f'Elastic Net KKT={en.kkt_max}; source fit not converged')
        output['elastic_net'] = en.predict_proba(V)[:, 1]
        if save_models:
            joblib.dump(en, folder / 'elastic_net.joblib')
    if not np.isfinite(output.drop(columns=IDENTITY).to_numpy()).all():
        raise ValueError('Nonfinite source prediction')
    del X, V, matrices, prediction_matrix
    gc.collect()
    return output

def cell_folder(year, excluded):
    return CROSSFIT / str(year) / '_'.join(map(str, sorted(excluded)))

def fit_cell(frame, year, excluded):
    spec = specification()
    folder = cell_folder(year, excluded)
    saved = folder / 'predictions.parquet'
    if saved.exists():
        return
    cutoff = pd.Timestamp(year, 1, 1)
    train = frame[frame.origin_year.lt(year) & ~frame.outer_lake_fold.isin(excluded) & frame.common_trainable & frame.label_mature_date.lt(cutoff)].reset_index(drop=True)
    val = frame[frame.origin_year.eq(year) & frame.outer_lake_fold.isin(excluded)].reset_index(drop=True)
    if train.empty or val.empty or set(train.lake_id) & set(val.lake_id):
        raise ValueError('Invalid chronological/lake-excluded source cell')
    controls = len(excluded) == 1 and year in spec['validation_years']
    start = time.perf_counter()
    pred = fit_base(train, val, folder, controls)
    pred.to_parquet(saved, index=False)
    write(folder / 'split.json', dict(year=year, excluded_folds=list(excluded), training_rows=len(train), prediction_rows=len(val), last_label_maturity=str(train.label_mature_date.max())))
    print('SOURCE_CELL', year, list(excluded), 'train', len(train), 'predict', len(val), 'seconds', round(time.perf_counter() - start, 1), flush=True)

def crossfit_predictions(frame, cutoff_year, outer_fold=None):
    parts = []
    for year in range(specification()['context_first_year'], cutoff_year):
        for fold in range(specification()['lake_folds']):
            if fold == outer_fold:
                continue
            excluded = (fold,) if outer_fold is None else tuple(sorted((fold, outer_fold)))
            p = pd.read_parquet(cell_folder(year, excluded) / 'predictions.parquet')
            parts.append(p[p.outer_lake_fold.eq(fold)])
    result = pd.concat(parts, ignore_index=True)
    if not result.candidate_row_key.is_unique:
        raise ValueError('Nested source prediction identities repeat')
    return result.set_index('candidate_row_key')

def correction_predictions(context, events, raw, base, cutoff, folder, year, excluded):
    for spec in read(CONFIG / 'monitoring_models.json')['corrections']:
        name = spec['policy_name']
        path = folder / (name + '.joblib')
        start = time.perf_counter()
        if path.exists():
            model = FirstOrderCoverageBooster.load(path)
        else:
            model = fit_correction(context, events, raw, base, spec, 'joint', cutoff, path, 'cuda')
            print('COVERAGE_FIT', year, list(excluded), name, 'rows', len(context), 'seconds', round(time.perf_counter() - start, 1), flush=True)
        del model
        gc.collect()

def predict_with(raw, base, model_folder):
    result = base[IDENTITY + ['xgboost', 'elastic_net']].copy()
    result['matched_xgboost'] = base['xgb4__y_31_60'].to_numpy()
    for spec in read(CONFIG / 'monitoring_models.json')['corrections']:
        name = spec['policy_name']
        logits = logit(np.clip(base[spec['expert_keys']].to_numpy(float), 1e-06, 1 - 1e-06))
        model = FirstOrderCoverageBooster.load(model_folder / (name + '.joblib'))
        result[name] = infer_cpu(model, head_features(raw, logits), logits[:, 0])
    return result

def train():
    spec = specification()
    CROSSFIT.mkdir(parents=True, exist_ok=True)
    signature = digest(CONFIG / 'external.json')
    signature_path = CROSSFIT / 'configuration.json'
    if signature_path.exists() and read(signature_path)['sha256'] != signature:
        raise ValueError('Crossfit predictions belong to a different configuration')
    write(signature_path, {'sha256': signature})
    frame = pd.read_parquet(DEST / 'source/candidates.parquet')
    events = pd.read_parquet(DEST / 'source/episodes.parquet')
    if frame.country.isin(spec['excluded_source_countries']).any() or frame.continent.isin(spec['excluded_source_continents']).any():
        raise ValueError('Target geography entered source training')
    fit_cell(frame, 2019, (0,))
    cells = {(year, (fold,)) for year in range(spec['context_first_year'], 2020) for fold in range(spec['lake_folds'])}
    cells |= {(year, pair) for year in range(spec['context_first_year'], max(spec['validation_years'])) for pair in itertools.combinations(range(spec['lake_folds']), 2)}
    for year, excluded in sorted(cells):
        fit_cell(frame, year, excluded)
    validation = []
    for year in spec['validation_years']:
        cutoff = pd.Timestamp(year, 1, 1)
        for fold in range(spec['lake_folds']):
            folder = MODEL / 'validation' / str(year) / str(fold)
            saved = RESULT / 'source_validation' / f'predictions_{year}_fold{fold}.parquet'
            if saved.exists():
                validation.append(pd.read_parquet(saved))
                continue
            context = frame[frame.origin_year.between(spec['context_first_year'], year - 1) & frame.outer_lake_fold.ne(fold)].reset_index(drop=True)
            nested = crossfit_predictions(frame, year, fold).loc[context.candidate_row_key].reset_index()
            expr = joblib.load(cell_folder(year, (fold,)) / 'expression.joblib')
            raw = transform_batched(expr, context, 8192)
            correction_predictions(context, events, raw, nested, cutoff, folder, year, (fold,))
            val = frame[frame.origin_year.eq(year) & frame.outer_lake_fold.eq(fold)].reset_index(drop=True)
            base = pd.read_parquet(cell_folder(year, (fold,)) / 'predictions.parquet').set_index('candidate_row_key').loc[val.candidate_row_key].reset_index()
            pred = predict_with(transform_batched(expr, val, 8192), base, folder)
            saved.parent.mkdir(parents=True, exist_ok=True)
            pred.to_parquet(saved, index=False)
            validation.append(pred)
            print('SOURCE_VALIDATION', year, fold, 'complete', flush=True)
            del context, nested, raw, val, base, pred, expr
            gc.collect()
    combined = pd.concat(validation, ignore_index=True).sort_values(['origin_date', 'lake_id']).reset_index(drop=True)
    combined.to_parquet(RESULT / 'source_validation/predictions.parquet', index=False)
    cutoff = pd.Timestamp(spec['training_cutoff'])
    training = frame[frame.common_trainable & frame.label_mature_date.lt(cutoff)].reset_index(drop=True)
    final_base_path = DEST / 'source/final_expert_predictions.parquet'
    if not final_base_path.exists():
        base = fit_base(training, frame, MODEL / 'experts', controls=True, save_models=True)
        base.to_parquet(final_base_path, index=False)
    final_expr = joblib.load(MODEL / 'experts/expression.joblib')
    context = frame[frame.origin_year.ge(spec['context_first_year'])].reset_index(drop=True)
    oof = crossfit_predictions(frame, 2020).loc[context.candidate_row_key].reset_index()
    oof.to_parquet(DEST / 'source/coverage_training_predictions.parquet', index=False)
    correction_predictions(context, events, transform_batched(final_expr, context, 8192), oof, cutoff, MODEL, 2020, ())
    write(MODEL / 'checksums.json', {p.relative_to(MODEL).as_posix(): digest(p) for p in MODEL.rglob('*') if p.suffix in ['.ubj', '.joblib']})
    write(RESULT / 'training_statistics.json', dict(config_sha256=signature, expert_training_rows=len(training), coverage_training_rows=len(context), source_lakes=int(frame.lake_id.nunique()), source_validation_rows=len(combined), base_cells=len(cells), source_validation_cells=len(spec['validation_years']) * spec['lake_folds'], transformed_features=len(final_expr.names), input_features=features(), training_cutoff=spec['training_cutoff']))
    print('CROSS_REGION_TRAINING_COMPLETE', flush=True)

def score_panel(frame, model_root=MODEL):
    expr = joblib.load(model_root / 'experts/expression.joblib')
    raw = transform_batched(expr, frame, 8192)
    data = xgb.DMatrix(raw, nthread=4)
    base = frame[IDENTITY].copy()
    for key in KEYS + ['xgboost']:
        model = xgb.Booster()
        model.load_model(model_root / 'experts' / (key + '.ubj'))
        model.set_param({'device': 'cpu', 'nthread': 4})
        base[key] = model.predict(data)
    en = joblib.load(model_root / 'experts/elastic_net.joblib')
    base['elastic_net'] = en.predict_proba(raw)[:, 1]
    return predict_with(raw, base, model_root)
