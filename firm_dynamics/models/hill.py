import numpy as np
from scipy.integrate import cumulative_trapezoid

from firm_dynamics.fitting import negative_binomial_log_likelihood


def hazard(ages, mu_ub, mu_lb, k, m):
    """Hill hazard that decreases from mu_ub toward mu_lb with firm age."""
    if mu_lb < 0 or mu_ub < mu_lb or k <= 0 or m <= 0:
        raise ValueError("Require mu_ub >= mu_lb >= 0, k > 0, and m > 0")

    ages = np.asarray(ages, dtype=float)
    return mu_ub - (mu_ub - mu_lb) * (ages**m) / (ages**m + k**m + 1e-10)


def survival(ages, mu_ub, mu_lb, k, m):
    ages_array = np.asarray(ages, dtype=float)
    scalar = ages_array.ndim == 0
    flat_ages = np.atleast_1d(ages_array).astype(float)

    if np.any(flat_ages < 0):
        raise ValueError("ages must be non-negative")

    order = np.argsort(flat_ages)
    sorted_ages = flat_ages[order]
    grid = np.unique(np.concatenate(([0.0], sorted_ages)))
    hazards = hazard(grid, mu_ub, mu_lb, k, m)
    cumulative = cumulative_trapezoid(hazards, grid, initial=0.0)
    sorted_survival = np.exp(-np.interp(sorted_ages, grid, cumulative))

    result = np.empty_like(sorted_survival)
    result[order] = sorted_survival
    return float(result[0]) if scalar else result


def neg_log_likelihood(params, ages, survivors, totals):
    mu_ub, mu_lb, k, m = params
    if mu_lb < 0 or mu_ub < mu_lb or k <= 0 or m <= 0:
        return np.inf
    return negative_binomial_log_likelihood(survival, params, ages, survivors, totals)


hill_hazard = hazard
model_survival_curve_hill = survival
neg_log_likelihood_hill = neg_log_likelihood
hill_survival_function = survival
