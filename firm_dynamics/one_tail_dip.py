import numpy as np
from scipy.optimize import minimize

from firm_dynamics.models.perturbation import (
    PerturbationParams,
    epsilon as _epsilon,
    hazard as _hazard,
    log_likelihood,
    neg_log_likelihood,
    survival as _survival,
)
from firm_dynamics.parameters import DEFAULT_SECTOR_HILL_PARAMETERS
from firm_dynamics.survival_analysis import (
    obtain_survival_fractions,
    obtain_total_alive_count,
)


def epsilon(s, a, eps0, tau, lam, t_e):
    params = PerturbationParams(0.1, 0.05, 10.0, 5.0, eps0, tau, lam, t_e)
    return _epsilon(s, a, params)


def hazard_with_perturbation(s, a, mu_ub, mu_lb, K, m, eps0, tau, lam, t_e):
    params = PerturbationParams(mu_ub, mu_lb, K, m, eps0, tau, lam, t_e)
    return _hazard(s, a, params)


def survival_hill_with_perturbation(
    ages,
    mu_ub,
    mu_lb,
    K,
    m,
    eps0,
    tau,
    lam,
    t_e,
    *,
    method="quad",
    grid_size=300,
):
    params = PerturbationParams(mu_ub, mu_lb, K, m, eps0, tau, lam, t_e)
    return _survival(ages, params, method=method, grid_size=grid_size)


def model_hill_with_dip(ages, mu_ub, mu_lb, K, m, t_e, eps0, tau, lam):
    return survival_hill_with_perturbation(
        ages,
        mu_ub,
        mu_lb,
        K,
        m,
        eps0,
        tau,
        lam,
        t_e,
    )


def survival_hill_with_perturbation_fast(
    ages,
    mu_ub,
    mu_lb,
    K,
    m,
    eps0,
    lam,
    t_e,
    *,
    tau=0.0,
    n_grid=300,
):
    return survival_hill_with_perturbation(
        ages,
        mu_ub,
        mu_lb,
        K,
        m,
        eps0,
        tau,
        lam,
        t_e,
        method="grid",
        grid_size=n_grid,
    )


def log_likelihood_perturbed(params, ages, survivors, totals):
    perturbation_params = PerturbationParams(*params)
    return log_likelihood(perturbation_params, ages, survivors, totals)


def neg_log_likelihood_perturbed(params, ages, survivors, totals):
    return neg_log_likelihood(params, ages, survivors, totals)


def neg_log_likelihood_perturbed_fast(params, ages, survivors, totals, n_grid=300):
    return neg_log_likelihood(
        _expand_fast_params(params),
        ages,
        survivors,
        totals,
        method="grid",
        grid_size=n_grid,
    )


def find_dip(df_analysis, sector, sector_params=None, *, window=0.5, n_points=9):
    """Estimate dip age from the worst-fitting Hill-only likelihood bins."""
    from firm_dynamics.models.hill import survival

    params_by_sector = sector_params or DEFAULT_SECTOR_HILL_PARAMETERS
    if sector not in params_by_sector:
        raise KeyError(f"No Hill parameters available for sector {sector}")

    _, ages = obtain_survival_fractions(df_analysis, "Sector", sector)
    totals, survivors = obtain_total_alive_count(df_analysis, "Sector", sector)
    mu_ub, mu_lb, K, m = params_by_sector[sector]

    probabilities = np.clip(survival(ages, mu_ub, mu_lb, K, m), 1e-12, 1.0 - 1e-12)
    deaths = totals - survivors
    bin_log_likelihood = survivors * np.log(probabilities) + deaths * np.log(1.0 - probabilities)

    candidate_ages = ages[np.argsort(bin_log_likelihood)[:n_points]]
    best_cluster = np.array([])
    for age in candidate_ages:
        cluster = candidate_ages[(candidate_ages >= age) & (candidate_ages <= age + window)]
        if len(cluster) > len(best_cluster):
            best_cluster = cluster

    return float(np.mean(best_cluster)) if len(best_cluster) > 0 else None


def mle_sector_perturbed(
    sector,
    df_analysis,
    initial_guess=(0.1, 0.05, 10.0, 5.0, 1.0, 1.0, 1.0, 7.0),
):
    totals, survivors, ages = _sector_counts(df_analysis, sector)
    bounds = [
        (1e-6, 0.3),
        (1e-6, 0.15),
        (0.1, 50.0),
        (0.1, 100.0),
        (0.0, 5.0),
        (0.0, 5.0),
        (0.0, 5.0),
        (0.0, float(ages.max())),
    ]
    return minimize(
        neg_log_likelihood_perturbed,
        initial_guess,
        args=(ages, survivors, totals),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500},
    )


def mle_sector_perturbed_fast(
    sector,
    df_analysis,
    initial_guess=(0.1, 0.05, 10.0, 5.0, 1.0, 1.0, 7.0),
    n_grid=300,
):
    totals, survivors, ages = _sector_counts(df_analysis, sector)
    bounds = [
        (1e-6, 0.3),
        (1e-6, 0.15),
        (0.1, 50.0),
        (0.1, 100.0),
        (0.0, 10.0),
        (0.0, 30.0),
        (0.0, 10.0),
    ]
    return minimize(
        neg_log_likelihood_perturbed_fast,
        initial_guess,
        args=(ages, survivors, totals, n_grid),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500},
    )


def _sector_counts(df_analysis, sector):
    totals, survivors = obtain_total_alive_count(df_analysis, "Sector", sector)
    ages = obtain_survival_fractions(df_analysis, "Sector", sector)[1]
    valid_mask = totals > 0
    totals = totals[valid_mask]
    survivors = survivors[valid_mask]
    ages = ages[valid_mask]

    if len(survivors) == 0:
        raise ValueError(f"No valid data for sector {sector}")

    return totals, survivors, ages


def _expand_fast_params(params):
    if len(params) == 8:
        return params
    if len(params) != 7:
        raise ValueError("fast perturbation params must have 7 or 8 values")

    mu_ub, mu_lb, K, m, eps0, lam, t_e = params
    return (mu_ub, mu_lb, K, m, eps0, 0.0, lam, t_e)


__all__ = [
    "PerturbationParams",
    "epsilon",
    "find_dip",
    "hazard_with_perturbation",
    "log_likelihood_perturbed",
    "mle_sector_perturbed",
    "mle_sector_perturbed_fast",
    "model_hill_with_dip",
    "neg_log_likelihood_perturbed",
    "neg_log_likelihood_perturbed_fast",
    "survival_hill_with_perturbation",
    "survival_hill_with_perturbation_fast",
]
