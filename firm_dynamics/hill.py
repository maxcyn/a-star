import numpy as np
from scipy.integrate import cumulative_trapezoid


def hill_hazard(a, mu_ub, mu_lb, K, m):
    '''
    Hill function
    Inputs:
    - a: age
    - mu_ub: upper bound of the hazard rate
    - mu_lb: lower bound of the hazard rate
    - K: mid-point
    - m: Hill coefficient
    '''
    return mu_ub - (mu_ub - mu_lb) * (a**m) / (a**m + K**m + 1e-10)


def model_survival_curve_hill(ages, mu_ub, mu_lb, K, m):
    '''
    Compute the survival curve based on the Hill hazard function.
    '''
    hazard_rates = hill_hazard(ages, mu_ub, mu_lb, K, m)
    cumulative_hazard = cumulative_trapezoid(hazard_rates, ages, initial=0.0)
    survival_curve = np.exp(-cumulative_hazard)
    return survival_curve


def neg_log_likelihood_hill(params, ages, survivors, totals):
    '''
    Negative log-likelihood function for the Hill model.
    '''
    import numpy as np

    mu_ub, mu_lb, K, m = params
    if mu_lb < 0 or mu_ub < mu_lb or K <= 0 or m <= 0:
        return np.inf

    S_vals = model_survival_curve_hill(ages, mu_ub, mu_lb, K, m)
    S_vals = np.clip(S_vals, 1e-12, 1 - 1e-12)  # avoid log(0)

    deaths = totals - survivors
    logL = np.sum(survivors * np.log(S_vals) + deaths * np.log(1 - S_vals))
    return -logL  # minimize negative log-likelihood
