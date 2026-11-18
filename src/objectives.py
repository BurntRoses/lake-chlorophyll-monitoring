"""Exact quota and event-coverage objectives and gradients."""
import numpy as np
from .quota_kernel import solve_quota

def quota_gates(scores, group, capacity=.1, temperature=.5, iterations=80):
    """Sigmoid gates with an implicit group intercept enforcing each quota."""
    scores = np.asarray(scores, dtype=np.float64)
    group = np.asarray(group, dtype=np.int64)
    if scores.ndim != 1 or group.shape != scores.shape or not len(scores):
        raise ValueError('Expected nonempty scalar scores and one group per row')
    if not np.isfinite(scores).all() or temperature <= 0 or not 0 <= capacity <= 1:
        raise ValueError('Invalid score, temperature, or capacity')
    if group.min() != 0 or (np.diff(group) < 0).any():
        raise ValueError('Groups must be contiguous, sorted and numbered from zero')
    size = np.bincount(group)
    if not size.all():
        raise ValueError('Group IDs contain gaps')
    quota = np.ceil(capacity * size).astype(int)
    starts = np.r_[0, np.flatnonzero(np.diff(group)) + 1]
    ends = np.r_[starts[1:], len(scores)]
    p = solve_quota(scores, starts, ends, quota, temperature, iterations)
    return p, quota

def coverage_value_gradient(scores, group, row_index, event_index,
                            capacity=.1, temperature=.5, event_weight=None):
    """Negative mean soft event coverage and exact score gradient.

    p_i=sigmoid((score_i-intercept_group)/temperature), sum_group p_i=k.
    coverage_e=1-product_{i in opportunities(e)}(1-p_i).
    The gradient includes the implicit intercept derivative, so unused rows
    still have a capacity opportunity cost. Unknown outcomes are never
    explicitly converted to negative labels by this kernel.
    """
    scores = np.asarray(scores, float)
    group = np.asarray(group, np.int64)
    ri, ei = np.asarray(row_index, np.int64), np.asarray(event_index, np.int64)
    if ri.ndim != 1 or ri.shape != ei.shape or not len(ri):
        raise ValueError('Every eligible event needs at least one opportunity')
    if ri.min() < 0 or ri.max() >= len(scores) or ei.min() != 0:
        raise ValueError('Unmapped event opportunity')
    events = int(ei.max()) + 1
    if not np.bincount(ei, minlength=events).all():
        raise ValueError('Event IDs have gaps or unsupported events')
    if len(np.unique(np.rec.fromarrays([ri, ei]))) != len(ri):
        raise ValueError('An event repeats the same candidate opportunity')
    weight = np.ones(events) if event_weight is None else np.asarray(event_weight, float)
    if weight.shape != (events,) or not np.isfinite(weight).all() or (weight < 0).any() or weight.sum() <= 0:
        raise ValueError('Invalid event weights')
    weight = weight / weight.sum()
    p, quota = quota_gates(scores, group, capacity, temperature)
    failure = 1 - p[ri]
    zero = failure == 0
    log_failure = np.zeros(len(ri))
    log_failure[~zero] = np.log(failure[~zero])
    zeros = np.bincount(ei, weights=zero, minlength=events).astype(int)
    sums = np.bincount(ei, weights=log_failure, minlength=events)
    product = np.where(zeros == 0, np.exp(sums), 0.)
    # Products excluding one factor remain exact at mandatory p=1 gates.
    excluding = np.zeros(len(ri))
    no_zero = zeros[ei] == 0
    excluding[no_zero] = np.exp(sums[ei[no_zero]] - log_failure[no_zero])
    only_zero = (zeros[ei] == 1) & zero
    excluding[only_zero] = np.exp(sums[ei[only_zero]])
    loss_p = np.bincount(ri, weights=-weight[ei] * excluding, minlength=len(scores))
    w = p * (1 - p)
    total_w = np.bincount(group, weights=w)
    weighted_derivative = np.bincount(group, weights=w * loss_p)
    center = np.divide(weighted_derivative, total_w, out=np.zeros_like(total_w), where=total_w > 0)
    gradient = w / temperature * (loss_p - center[group])
    coverage = 1 - product
    return {'loss': float(-np.dot(weight, coverage)), 'gradient': gradient,
            'soft_selection': p, 'event_coverage': coverage, 'quota': quota}

def log_coverage_value_gradient(scores, group, row_index, event_index,
                                capacity=.1, temperature=.5, event_weight=None,
                                log_fraction=.1, epsilon=1e-10):
    """Blend mean event coverage with mean log event coverage, with exact gradient.

    L = -sum_e w_e [(1-a)*C_e + a*log(eps + (1-eps)*C_e)].
    C_e = 1-product_{i in opportunities(e)}(1-p_i), sum_{i in origin}p_i=k.

    The log term prevents an almost-uncovered event from contributing almost
    no learning signal merely because all its scores are below the quota.
    It can also overemphasize inherently unpredictable events. Only later
    source validation can decide whether this tradeoff improves hard capture.
    No first-event or future-event-class flag enters the objective.
    """
    if not np.isfinite(log_fraction) or not 0 <= log_fraction <= 1:
        raise ValueError('log_fraction must be between zero and one')
    if not np.isfinite(epsilon) or not 0 < epsilon < 1:
        raise ValueError('epsilon must be strictly between zero and one')
    # Reuse the established input validation and exact quota calculation.
    state = coverage_value_gradient(scores, group, row_index, event_index,
                                    capacity, temperature, event_weight)
    if log_fraction == 0:
        return state
    group = np.asarray(group, np.int64)
    ri, ei = np.asarray(row_index, np.int64), np.asarray(event_index, np.int64)
    events = len(state['event_coverage'])
    weight = np.ones(events) if event_weight is None else np.asarray(event_weight, float)
    weight = weight / weight.sum()
    p = state['soft_selection']
    mandatory = p[ri] == 1.
    log_failure = np.zeros(len(ri))
    log_failure[~mandatory] = np.log1p(-p[ri[~mandatory]])
    zeros = np.bincount(ei, weights=mandatory, minlength=events).astype(int)
    sums = np.bincount(ei, weights=log_failure, minlength=events)
    # expm1 retains small coverage values which 1-exp(sums) would round away.
    coverage = np.where(zeros == 0, -np.expm1(sums), 1.)
    excluding = np.zeros(len(ri))
    no_zero = zeros[ei] == 0
    excluding[no_zero] = np.exp(sums[ei[no_zero]] - log_failure[no_zero])
    only_zero = (zeros[ei] == 1) & mandatory
    excluding[only_zero] = np.exp(sums[ei[only_zero]])
    stabilized = epsilon + (1 - epsilon) * coverage
    derivative = -(1 - log_fraction) - log_fraction * (1 - epsilon) / stabilized
    loss_p = np.bincount(ri, weights=weight[ei] * derivative[ei] * excluding,
                         minlength=len(p))
    gate_derivative = p * (1 - p)
    total = np.bincount(group, weights=gate_derivative)
    numerator = np.bincount(group, weights=gate_derivative * loss_p)
    center = np.divide(numerator, total, out=np.zeros_like(total), where=total > 0)
    gradient = gate_derivative / temperature * (loss_p - center[group])
    utility = (1 - log_fraction) * coverage + log_fraction * np.log(stabilized)
    return {**state, 'loss': float(-np.dot(weight, utility)),
            'gradient': gradient, 'event_coverage': coverage}
