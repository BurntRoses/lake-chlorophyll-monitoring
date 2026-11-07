"""Residual-tree learners for primary-window and joint event-coverage objectives."""
from pathlib import Path
import gc
import json
import time
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import expit, logit
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits
from .objectives import coverage_value_gradient, log_coverage_value_gradient
from .io import write
class CoverageContext:
    def __init__(self, group, row_index, event_index, labels, anchor_margin,
                 capacity=.1, temperature=.5, classification_weight=.1,
                 anchor_weight=.1, coverage_weight=1., event_weight=None):
        self.group = np.asarray(group, np.int64).copy()
        self.row_index = np.asarray(row_index, np.int64).copy()
        self.event_index = np.asarray(event_index, np.int64).copy()
        self.labels = np.asarray(labels, float).copy()
        self.anchor_margin = np.asarray(anchor_margin, float).copy()
        if self.labels.shape != self.group.shape or self.anchor_margin.shape != self.group.shape:
            raise ValueError('All queue context arrays must align')
        if not np.isfinite(self.anchor_margin).all():
            raise ValueError('The frozen anchor must be finite')
        self.known = np.isfinite(self.labels)
        self.unknown = ~self.known
        if np.isinf(self.labels).any() or not np.isin(self.labels[self.known], [0., 1.]).all():
            raise ValueError('Unknown labels must remain NaN; supported labels are zero or one')
        if (self.row_index < 0).any() or (self.row_index >= len(self.group)).any():
            raise ValueError('Unmapped event opportunity')
        if not np.all(self.labels[self.row_index] == 1.):
            raise ValueError('Reward opportunities require supported mature positive labels')
        coefficients = np.asarray([classification_weight, anchor_weight, coverage_weight])
        if not np.isfinite(coefficients).all() or (coefficients < 0).any():
            raise ValueError('Loss coefficients must be finite and nonnegative')
        self.capacity, self.temperature = capacity, temperature
        self.classification_weight = classification_weight
        self.anchor_weight, self.coverage_weight = anchor_weight, coverage_weight
        self.event_weight = event_weight

    def value_gradient(self, margin):
        margin = np.asarray(margin, float)
        if margin.shape != self.group.shape or not np.isfinite(margin).all():
            raise ValueError('Expected one finite score per competitor')
        state = coverage_value_gradient(margin, self.group, self.row_index,
            self.event_index, self.capacity, self.temperature, self.event_weight)
        gradient = self.coverage_weight * state['gradient']
        classification, anchor = 0., 0.
        if self.known.any() and self.classification_weight:
            z, y = margin[self.known], self.labels[self.known]
            classification = float(np.mean(np.logaddexp(0., z) - y * z))
            gradient[self.known] += self.classification_weight * (expit(z) - y) / len(z)
        if self.unknown.any() and self.anchor_weight:
            z = margin[self.unknown]
            base = self.anchor_margin[self.unknown]
            prior = expit(base)
            # KL(Bernoulli(frozen prior) || Bernoulli(current risk)); the
            # subtraction makes the exact unchanged-prior penalty equal zero.
            anchor = float(np.mean(np.logaddexp(0., z) - np.logaddexp(0., base) - prior * (z - base)))
            gradient[self.unknown] += self.anchor_weight * (expit(z) - prior) / len(z)
        return {'loss': self.coverage_weight * state['loss']
                + self.classification_weight * classification + self.anchor_weight * anchor,
                'gradient': gradient, 'coverage_loss': state['loss'],
                'supported_label_logloss': classification, 'unknown_prior_kl': anchor,
                'soft_selection': state['soft_selection'], 'quota': state['quota']}

class FirstOrderCoverageBooster:
    """Fit residual trees using first-order gradients and backtracking descent.

    The baseline margins are offsets supplied by an external frozen expert.
    At deployment the same expert must supply offsets for new candidate rows.
    Every tree, clipping rule and accepted step is retained for saved inference.
    """
    def __init__(self, rounds=40, depth=2, min_leaf=5, learning_rate=1.,
                 max_direction=3., max_backtracks=30, armijo=1e-4, seed=56):
        if rounds < 0 or depth < 1 or min_leaf < 1 or learning_rate <= 0:
            raise ValueError('Invalid learner dimensions or step size')
        if max_direction <= 0 or max_backtracks < 1 or not 0 < armijo < 1:
            raise ValueError('Invalid line-search parameters')
        self.rounds, self.depth, self.min_leaf = rounds, depth, min_leaf
        self.learning_rate, self.max_direction = learning_rate, max_direction
        self.max_backtracks, self.armijo, self.seed = max_backtracks, armijo, seed

    def fit(self, features, base_margin, context):
        features = np.asarray(features, np.float32)
        margin = np.asarray(base_margin, float).copy()
        if features.ndim != 2 or len(features) != len(margin) or not np.isfinite(features).all():
            raise ValueError('Expected a finite training-only-transformed feature matrix')
        self.feature_count_ = features.shape[1]
        self.trees_, self.steps_, self.trace_ = [], [], []
        state = context.value_gradient(margin)
        self.trace_.append({'round': 0, 'loss': state['loss'], 'accepted': True})
        self.stop_reason_ = 'round_limit'
        for iteration in range(self.rounds):
            tree = DecisionTreeRegressor(max_depth=self.depth, min_samples_leaf=self.min_leaf,
                                         random_state=self.seed + iteration)
            # Scaling counteracts averaging in the empirical risk. It changes
            # step units only; Armijo uses the true unscaled loss and gradient.
            with threadpool_limits(limits=1):
                tree.fit(features, -state['gradient'] * len(margin))
                direction = np.clip(tree.predict(features), -self.max_direction, self.max_direction)
            slope = float(np.dot(state['gradient'], direction))
            if slope >= -1e-15:
                self.stop_reason_ = 'no_descending_tree_direction'
                break
            step, accepted = self.learning_rate, False
            for backtrack in range(self.max_backtracks):
                candidate = margin + step * direction
                new_state = context.value_gradient(candidate)
                if new_state['loss'] <= state['loss'] + self.armijo * step * slope:
                    accepted = True
                    break
                step *= .5
            if not accepted:
                self.stop_reason_ = 'no_accepted_step'
                break
            self.trees_.append(tree)
            self.steps_.append(step)
            self.trace_.append({'round': iteration + 1, 'loss': new_state['loss'],
                'coverage_loss': new_state['coverage_loss'],
                'supported_label_logloss': new_state['supported_label_logloss'],
                'unknown_prior_kl': new_state['unknown_prior_kl'], 'accepted': True,
                'backtracks': backtrack, 'step': step, 'directional_derivative': slope})
            margin, state = candidate, new_state
        self.training_margin_ = margin
        return self

    def predict_margin(self, features, base_margin):
        features = np.asarray(features, np.float32)
        margin = np.asarray(base_margin, float).copy()
        if features.ndim != 2 or features.shape[1] != self.feature_count_ or len(features) != len(margin):
            raise ValueError('Inference features and frozen expert offsets must align')
        for tree, step in zip(self.trees_, self.steps_):
            margin += step * np.clip(tree.predict(features), -self.max_direction, self.max_direction)
        return margin

    def save(self,path):
        parameters={name:getattr(self,name) for name in ['rounds','depth','min_leaf','learning_rate','max_direction','max_backtracks','armijo','seed']}
        joblib.dump({'schema_version':2,'parameters':parameters,'trees':self.trees_,'steps':self.steps_,'feature_count':self.feature_count_},path)

    @classmethod
    def load(cls,path):
        payload=joblib.load(path)
        if payload['schema_version']!=2:raise ValueError('Unsupported coverage model format')
        model=cls(**payload['parameters'])
        model.trees_,model.steps_=payload['trees'],payload['steps']
        model.feature_count_=payload['feature_count']
        model.trace_,model.stop_reason_=[],'fitted'
        return model

class LogCoverageContext(CoverageContext):
    def __init__(self, *arguments, log_fraction=.1, epsilon=1e-10, **settings):
        super().__init__(*arguments, **settings)
        if not np.isfinite(log_fraction) or not 0 <= log_fraction <= 1:
            raise ValueError('Invalid log coverage fraction')
        if not np.isfinite(epsilon) or not 0 < epsilon < 1:
            raise ValueError('Invalid log coverage stabilizer')
        self.log_fraction, self.epsilon = log_fraction, epsilon

    def value_gradient(self, margin):
        margin = np.asarray(margin, float)
        if margin.shape != self.group.shape or not np.isfinite(margin).all():
            raise ValueError('Expected one finite margin per competitor')
        state = log_coverage_value_gradient(margin, self.group, self.row_index,
            self.event_index, self.capacity, self.temperature, self.event_weight,
            self.log_fraction, self.epsilon)
        gradient = self.coverage_weight * state['gradient']
        classification, anchor = 0., 0.
        if self.known.any() and self.classification_weight:
            z, y = margin[self.known], self.labels[self.known]
            classification = float(np.mean(np.logaddexp(0., z) - y * z))
            gradient[self.known] += self.classification_weight * (expit(z) - y) / len(z)
        if self.unknown.any() and self.anchor_weight:
            z, base = margin[self.unknown], self.anchor_margin[self.unknown]
            prior = expit(base)
            anchor = float(np.mean(np.logaddexp(0., z) - np.logaddexp(0., base) - prior * (z - base)))
            gradient[self.unknown] += self.anchor_weight * (expit(z) - prior) / len(z)
        return {'loss': self.coverage_weight * state['loss']
                + self.classification_weight * classification + self.anchor_weight * anchor,
                'gradient': gradient, 'coverage_loss': state['loss'],
                'supported_label_logloss': classification, 'unknown_prior_kl': anchor,
                'soft_selection': state['soft_selection'], 'quota': state['quota'],
                'mean_soft_event_coverage': float(state['event_coverage'].mean())}

class ResidualMatrix:
    """Reuse one quantization within a fitted model; never serialize the cache."""
    def __init__(self, device='cuda', max_bin=256, threads=2):
        self.device, self.max_bin, self.threads = device, max_bin, threads
        self.features, self.matrix = None, None

    def update(self, features, residual):
        if self.matrix is None:
            self.features = features
            self.matrix = xgb.QuantileDMatrix(features, label=residual,
                max_bin=self.max_bin, nthread=self.threads)
        else:
            if features is not self.features:
                raise ValueError('A weak-tree quantization cache crossed feature matrices')
            self.matrix.set_label(residual)
        return self.matrix

    def close(self):
        self.features = self.matrix = None
        gc.collect()

class HistogramResidualTree:
    def __init__(self, cache, max_depth, min_samples_leaf, random_state):
        self.cache = cache
        self.parameters = dict(objective='reg:squarederror', tree_method='hist',
            device=cache.device, max_bin=cache.max_bin, max_depth=max_depth,
            min_child_weight=min_samples_leaf, learning_rate=1., reg_lambda=0.,
            reg_alpha=0., gamma=0., subsample=1., colsample_bytree=1.,
            base_score=0., random_state=random_state, n_jobs=cache.threads,
            validate_parameters=True)

    def fit(self, features, residual):
        matrix = self.cache.update(features, residual)
        self.regressor_ = xgb.XGBRegressor(n_estimators=1, **self.parameters)
        params = self.regressor_.get_xgb_params()
        params['disable_default_eval_metric'] = 1
        self.regressor_._Booster = xgb.train(params, matrix, num_boost_round=1)
        actual_device = json.loads(self.regressor_.get_booster().save_config())['learner']['generic_param']['device']
        if self.cache.device.startswith('cuda') and not actual_device.startswith('cuda'):
            raise RuntimeError('Histogram weak tree silently fell back from CUDA')
        self.cache.actual_device = actual_device
        return self

    def predict(self, features):
        matrix = self.cache.matrix if features is self.cache.features else xgb.DMatrix(
            features, nthread=self.cache.threads)
        return self.regressor_.get_booster().predict(matrix, output_margin=True)

class HistogramCoverageBooster(FirstOrderCoverageBooster):
    def __init__(self, *args, device='cuda', max_bin=256, threads=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.device, self.max_bin, self.threads = device, max_bin, threads

    def fit(self, features, base_margin, context):
        global DecisionTreeRegressor
        started = time.perf_counter()
        cache = ResidualMatrix(self.device, self.max_bin, self.threads)
        tree_class = DecisionTreeRegressor
        def factory(**parameters):
            return HistogramResidualTree(cache, **parameters)
        DecisionTreeRegressor = factory
        try:
            super().fit(features, base_margin, context)
            self.trees_ = [tree.regressor_ for tree in self.trees_]
        finally:
            DecisionTreeRegressor = tree_class
            cache.close()
        return self


    def predict_margin(self, features, base_margin):
        features = np.asarray(features, np.float32)
        if features.ndim != 2 or features.shape[1] != self.feature_count_ or len(features) != len(base_margin):
            raise ValueError('Inference features and expert offsets differ')
        return infer_cpu(self, features, base_margin)

def infer_cpu(model, features, base):
    margin = np.asarray(base, float).copy()
    matrix = xgb.DMatrix(features, nthread=2)
    for tree, step in zip(model.trees_, model.steps_):
        tree.set_params(device='cpu')
        prediction = tree.get_booster().predict(matrix, output_margin=True)
        margin += step * np.clip(prediction, -model.max_direction, model.max_direction)
    return margin

CONTEXT = dict(capacity=.1, temperature=.75, classification_weight=.1,
               anchor_weight=.1, coverage_weight=1., log_fraction=.1)

class JointContext:
    """Exact gradient of (long loss + a * sum(auxiliary losses)) / heads.

    Dividing by heads cancels the pooled learner's row-count rescaling: the
    primary residual scale equals that of the one-head matched control.
    
    """
    def __init__(self, contexts, auxiliary_weight):
        self.contexts = contexts
        self.rows = len(contexts[0].group)
        if any(len(c.group) != self.rows for c in contexts):
            raise ValueError('Head contexts must have the same original candidates')
        self.weights = np.array([1.] + [auxiliary_weight] * (len(contexts) - 1))
        if not np.isfinite(self.weights).all() or (self.weights < 0).any():
            raise ValueError('Invalid joint loss coefficients')

    def value_gradient(self, margin):
        z = np.asarray(margin, float).reshape(len(self.contexts), self.rows)
        states = [context.value_gradient(z[j]) for j, context in enumerate(self.contexts)]
        weights = self.weights / len(states)
        keys = ['loss', 'coverage_loss', 'supported_label_logloss', 'unknown_prior_kl']
        result = {key: float(sum(w * s[key] for w, s in zip(weights, states))) for key in keys}
        result['gradient'] = np.concatenate([w * s['gradient'] for w, s in zip(weights, states)])
        return result

def pooled_features(raw, logits):
    raw = np.asarray(raw, np.float32)
    logits = np.asarray(logits, np.float32)
    if logits.ndim != 2 or len(logits) != len(raw) or not np.isfinite(logits).all():
        raise ValueError('Finite row-local expert logits must align with shared raw inputs')
    common = np.column_stack([raw, logits]).astype(np.float32)
    heads, rows = logits.shape[1], len(raw)
    values = np.empty((heads * rows, common.shape[1] + 1), np.float32)
    for j in range(heads):
        values[j * rows:(j + 1) * rows, :-1] = common
        values[j * rows:(j + 1) * rows, -1] = j
    return values

def head_features(raw, logits, head=0):
    return np.column_stack([np.asarray(raw, np.float32), np.asarray(logits, np.float32),
                            np.full(len(raw), head, np.float32)]).astype(np.float32)

def logits_from(predictions, keys):
    values = predictions[keys].to_numpy(float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('Incomplete or invalid frozen expert predictions')
    return logit(np.clip(values, 1e-6, 1 - 1e-6))

def window_context(frame, episodes, cutoff, label, base):
    known = frame.common_trainable & frame.label_mature_date.lt(cutoff) & frame[label].ge(0)
    labels = np.where(known, frame[label].to_numpy(float), np.nan)
    ri = np.flatnonzero(labels == 1.)
    ids = frame.iloc[ri].next_episode_id
    event_lookup = episodes.set_index('episode_id')
    if not len(ri) or not ids.isin(event_lookup.index).all():
        raise ValueError('Joint head has no supported mature events, or an unmapped event')
    events = event_lookup.loc[ids]
    if not np.array_equal(events.lake_id.to_numpy(), frame.iloc[ri].lake_id.to_numpy()):
        raise ValueError('An auxiliary positive maps to a different lake')
    lower, upper = {'y_31_60': (31, 60), 'y_1_14': (1, 14), 'y_15_30': (15, 30)}[label]
    leads = (events.event_date.to_numpy() - frame.iloc[ri].origin_date.to_numpy()) / np.timedelta64(1, 'D')
    if not ((leads >= lower) & (leads <= upper)).all():
        raise ValueError('A supported head event has an inconsistent lead time')
    if not events.event_date.add(pd.Timedelta(days=7)).lt(cutoff).all():
        raise ValueError('An auxiliary event is not mature at the source cutoff')
    event_index, unique = pd.factorize(ids, sort=False)
    context = LogCoverageContext(pd.factorize(frame.origin_date, sort=False)[0],
        ri, event_index, labels, base, **CONTEXT)
    return context, dict(label=label, supported_labels=int(known.sum()),
        unknown_labels=int((~known).sum()), supported_events=len(unique), opportunities=len(ri))
