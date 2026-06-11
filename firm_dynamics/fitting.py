import numpy as np


def binomial_log_likelihood(probabilities, survivors, totals, *, clip=1e-12):
    """Binomial log-likelihood without the parameter-independent combinatorial term."""
    probabilities = np.asarray(probabilities, dtype=float)
    survivors = np.asarray(survivors, dtype=float)
    totals = np.asarray(totals, dtype=float)
    deaths = totals - survivors

    if np.any(totals < 0) or np.any(survivors < 0) or np.any(deaths < 0):
        return -np.inf

    probabilities = np.clip(probabilities, clip, 1.0 - clip)
    return float(
        np.sum(
            survivors * np.log(probabilities)
            + deaths * np.log(1.0 - probabilities)
        )
    )


def negative_binomial_log_likelihood(model, params, ages, survivors, totals):
    probabilities = model(ages, *params)
    log_likelihood = binomial_log_likelihood(probabilities, survivors, totals)
    if not np.isfinite(log_likelihood):
        return np.inf
    return -log_likelihood


def aic(log_likelihood, n_params):
    return 2 * n_params - 2 * log_likelihood


def bic(log_likelihood, n_params, n_obs):
    return n_params * np.log(n_obs) - 2 * log_likelihood
