import numpy as np

from firm_dynamics.fitting import negative_binomial_log_likelihood


def hazard(ages, lam, alpha):
    if lam <= 0 or alpha <= 0:
        raise ValueError("lam and alpha must be positive")
    ages = np.asarray(ages, dtype=float)
    return lam / (1.0 + alpha * ages)


def survival(ages, lam, alpha):
    if lam <= 0 or alpha <= 0:
        raise ValueError("lam and alpha must be positive")
    ages = np.asarray(ages, dtype=float)
    return (1.0 + alpha * ages) ** (-lam / alpha)


def neg_log_likelihood(params, ages, survivors, totals):
    lam, alpha = params
    if lam <= 0 or alpha <= 0:
        return np.inf
    return negative_binomial_log_likelihood(survival, params, ages, survivors, totals)
