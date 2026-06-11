import numpy as np

from firm_dynamics.fitting import negative_binomial_log_likelihood


def hazard(ages, rate):
    if rate < 0:
        raise ValueError("rate must be non-negative")
    return np.full_like(np.asarray(ages, dtype=float), rate, dtype=float)


def survival(ages, rate):
    if rate < 0:
        raise ValueError("rate must be non-negative")
    ages = np.asarray(ages, dtype=float)
    return np.exp(-rate * ages)


def neg_log_likelihood(params, ages, survivors, totals):
    (rate,) = params
    if rate < 0:
        return np.inf
    return negative_binomial_log_likelihood(survival, params, ages, survivors, totals)
