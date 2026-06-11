from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad, trapezoid

from firm_dynamics.fitting import binomial_log_likelihood
from firm_dynamics.models import hill


@dataclass(frozen=True)
class PerturbationParams:
    mu_ub: float
    mu_lb: float
    k: float
    m: float
    eps0: float
    tau: float
    lam: float
    event_age: float


def epsilon(s, age, params):
    """Transient event multiplier epsilon(age, s)."""
    if params.eps0 < 0 or params.tau < 0 or params.lam < 0 or params.event_age < 0:
        raise ValueError("perturbation parameters must be non-negative")

    if s < age - params.event_age:
        return 0.0

    cohort_effect = params.eps0 * np.exp(-abs(age - params.event_age) * params.tau)
    elapsed_since_event = params.event_age - (age - s)
    return cohort_effect * np.exp(-params.lam * elapsed_since_event)


def hazard(s, age, params):
    base = hill.hazard(s, params.mu_ub, params.mu_lb, params.k, params.m)
    return (1.0 + epsilon(s, age, params)) * base


def survival(ages, params, *, method="quad", grid_size=300):
    ages = np.asarray(ages, dtype=float)
    scalar = ages.ndim == 0
    flat_ages = np.atleast_1d(ages).astype(float)

    values = np.array(
        [_survival_at_age(age, params, method=method, grid_size=grid_size) for age in flat_ages],
        dtype=float,
    )
    return float(values[0]) if scalar else values


def log_likelihood(params, ages, survivors, totals, *, method="quad", grid_size=300):
    probabilities = survival(ages, params, method=method, grid_size=grid_size)
    return binomial_log_likelihood(probabilities, survivors, totals)


def neg_log_likelihood(param_values, ages, survivors, totals, *, method="quad", grid_size=300):
    params = PerturbationParams(*param_values)
    log_likelihood_value = log_likelihood(
        params,
        ages,
        survivors,
        totals,
        method=method,
        grid_size=grid_size,
    )
    if not np.isfinite(log_likelihood_value):
        return np.inf
    return -log_likelihood_value


def _survival_at_age(age, params, *, method, grid_size):
    if age < 0:
        raise ValueError("ages must be non-negative")
    if age == 0:
        return 1.0

    if method == "quad":
        integral, _ = quad(lambda s: hazard(s, age, params), 0.0, age)
    elif method == "grid":
        if grid_size < 2:
            raise ValueError("grid_size must be at least 2")
        s_grid = np.linspace(0.0, age, grid_size)
        integrand = np.array([hazard(s, age, params) for s in s_grid])
        integral = trapezoid(integrand, s_grid)
    else:
        raise ValueError("method must be 'quad' or 'grid'")

    return float(np.exp(-integral))
