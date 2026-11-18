"""Bracketed Newton solution of the exact sigmoid monitoring quota."""
import math
import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def solve_quota(scores, starts, ends, quota, temperature, iterations):
    probability = np.empty(len(scores), np.float64)
    for g in range(len(starts)):
        start, end, target = starts[g], ends[g], quota[g]
        if target == 0 or target == end-start:
            probability[start:end] = 1. if target != 0 else 0.
            continue
        lower = np.min(scores[start:end]) - 40*temperature
        upper = np.max(scores[start:end]) + 40*temperature
        threshold = (lower+upper)/2
        for _ in range(iterations):
            mass, derivative = 0., 0.
            for i in range(start, end):
                z = (scores[i]-threshold)/temperature
                if z >= 0:
                    p = 1/(1+math.exp(-z))
                else:
                    ez = math.exp(z)
                    p = ez/(1+ez)
                probability[i] = p
                mass += p
                derivative += p*(1-p)/temperature
            difference = mass-target
            if abs(difference) <= 1e-12:
                break
            if difference > 0:
                lower = threshold
            else:
                upper = threshold
            candidate = threshold + difference/derivative if derivative > 0 else math.inf
            if not lower < candidate < upper:
                candidate = (lower+upper)/2
            if candidate == threshold:
                break
            threshold = candidate
        # Recompute at the final threshold, including the iteration-limit case.
        for i in range(start, end):
            z = (scores[i]-threshold)/temperature
            probability[i] = 1/(1+math.exp(-z)) if z >= 0 else math.exp(z)/(1+math.exp(z))
    return probability
