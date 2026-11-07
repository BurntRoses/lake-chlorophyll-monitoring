"""Common feature expression and fixed individual prediction tasks."""
from pathlib import Path
import copy
import gc
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from threadpoolctl import threadpool_limits
from .paths import ROOT, CONFIG
from .io import read, write
from .expression import CommonExpression, StandardElasticNet
from . import seasonal
SEARCH = read(CONFIG / 'base_models.json')
TREE_GRID = {row['id']: row for row in SEARCH['XGBoost']}
EN_GRID = {row['id']: row for row in SEARCH['ElasticNet_logistic']}
LABELS = ['y_1_14', 'y_15_30', 'y_31_60']
SEED = 20261015

def event_balanced_weights(frame, label):
    """Keep positive mass but prevent an event with 3 origins dominating."""
    y = frame[label].to_numpy(dtype=int)
    weights = np.ones(len(frame), dtype=np.float64)
    positive = y == 1
    if positive.any():
        ids = frame.loc[positive, 'next_episode_id'].astype(str)
        counts = ids.value_counts()
        raw = ids.map(1.0 / counts).to_numpy(dtype=float)
        raw *= positive.sum() / raw.sum()
        weights[positive] = raw
    return weights

def feature_setup(variant, original_spec=None):
    spec = pd.read_csv(CONFIG / 'features.csv') if original_spec is None else original_spec.copy()
    expression = copy.deepcopy(SEARCH['expression'])
    if variant == 'seasonal':
        extra = pd.DataFrame({'feature_name': seasonal.FEATURES, 'source': 'causal_prior_year_history', 'role': 'dynamic'})
        spec = pd.concat([spec, extra], ignore_index=True)
        expression['spline_features'] += [f'seasonal_{window}_expected_log_chla' for window in ['now', '1_14', '15_30', '31_60']]
        anomaly = next((name for name in seasonal.FEATURES if 'latest' in name and 'minus' in name))
        expression['spline_features'].append(anomaly)
        expression['interactions'].append(['seasonal_31_60_expected_log_chla', anomaly])
    return (spec, expression)

def tasks():
    result = []
    for name in TREE_GRID:
        for label in LABELS:
            result.append((name + '__' + label, 'binary', name, label, SEED))
        for seed in [1, 2]:
            result.append((name + '__long_seed' + str(seed), 'binary', name, 'y_31_60', SEED + seed))
    for name in EN_GRID:
        result.append((name, 'elastic_net', name, None, SEED))
    return result

def transform_batched(expression, frame, batch_size=2048):
    result = np.empty((len(frame), len(expression.names)), dtype=np.float32)
    for start in range(0, len(frame), batch_size):
        result[start:start + batch_size] = expression.transform(frame.iloc[start:start + batch_size])
    return result

def tree_parameters(config, seed):
    return {'tree_method': 'hist', 'nthread': 2, 'max_bin': 127, 'max_depth': config['max_depth'], 'min_child_weight': config['min_child_weight'], 'eta': config['learning_rate'], 'subsample': config['subsample'], 'colsample_bytree': config['colsample_bytree'], 'reg_lambda': config['reg_lambda'], 'reg_alpha': 0.05, 'seed': seed}

def fit_task(train, X, V, task, directory):
    key, kind, cfg_key, target, seed = task
    directory.mkdir(parents=True, exist_ok=True)
    if kind == 'elastic_net':
        path = directory / (key + '.joblib')
        if path.exists():
            model = joblib.load(path)
        else:
            cfg = EN_GRID[cfg_key]
            with threadpool_limits(limits=2):
                model = StandardElasticNet(C=cfg['C'], l1_ratio=cfg['l1_ratio']).fit(np.ascontiguousarray(X, dtype=np.float64), train.y_31_60.to_numpy(int))
            joblib.dump(model, path)
        return model.predict_proba(V)[:, 1] if model.converged else np.full(len(V), np.nan)
    path = directory / (key + '.ubj')
    cfg = TREE_GRID[cfg_key]
    params = tree_parameters(cfg, seed)
    if path.exists():
        model = xgb.Booster()
        model.load_model(path)
    elif kind == 'binary':
        known = train[target].ge(0).to_numpy()
        subset = train.loc[known].reset_index(drop=True)
        if subset[target].nunique() != 2:
            raise RuntimeError('A known binary US training label lacks both classes')
        data = xgb.DMatrix(X[known], label=subset[target].to_numpy(int), weight=event_balanced_weights(subset, target), nthread=2)
        params.update(objective='binary:logistic', eval_metric='logloss')
        model = xgb.train(params, data, num_boost_round=cfg['n_estimators'])
        model.save_model(path)
        del data
    else:
        raise ValueError('Final coverage experts use binary or elastic-net tasks')
    result = model.predict(xgb.DMatrix(V, nthread=2))
    del model
    gc.collect()
    if not np.isfinite(result).all():
        raise RuntimeError('A US tree produced incomplete predictions')
    return result
